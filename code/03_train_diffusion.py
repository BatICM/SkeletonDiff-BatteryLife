import argparse
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from common import (
    MODEL2_DIR,
    OUTPUT2_DIR,
    PLOT_MAIN_DIR,
    ensure_dirs,
    inv_softplus,
    set_global_seed,
)
from models_diffusion import ConditionalDiffusionModel, GroupClassifier

"""
Code2 训练脚本
=============
本脚本实现“先稳定、后生成”的两阶段训练流程：

1) 训练 GroupClassifier（辅助评估分组可分性）
2) Stage-1: deterministic 预训练（学习主退化骨架）
3) Stage-2: diffusion 训练（学习残差随机性）

输入：
- Output2/code2_dataset_seed{seed}.pkl

输出：
- Model2/code2_model_seed{seed}.pth
- Output2/code2_training_curve_seed{seed}.png
- Output2/code2_train_log_seed{seed}.pkl
"""


plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def apply_condition_ablation(feature_matrix, life_features, mode="full"):
    """
    条件输入消融：
    - full: 原始输入
    - no_life: life_features 全零
    - life_only: feature_matrix 全零（保留 early_soh + life）
    """
    mode = str(mode).strip().lower()
    if mode == "no_life":
        life_features = np.zeros_like(life_features, dtype=np.float32)
    elif mode == "life_only":
        feature_matrix = np.zeros_like(feature_matrix, dtype=np.float32)
    elif mode != "full":
        raise ValueError(f"Unsupported condition_ablation mode: {mode}")
    return feature_matrix, life_features


class Code2Dataset(Dataset):
    """
    把预处理后的 sample 转成训练张量。
    返回项顺序与模型 forward 接口一一对应，减少训练时字段错位风险。
    """

    def __init__(self, package, indices, condition_ablation="full"):
        self.package = package
        self.samples = package["samples"]
        self.indices = list(indices)
        self.max_len = int(package["max_future_length_padded"])
        self.delta_scale = float(package["delta_scale"])
        self.raw_mean = float(package["raw_mean"])
        self.raw_std = float(package["raw_std"])
        self.max_target_len = float(max(package["target_lengths"].values()))
        self.life_dim = int(package.get("life_feature_dim", 14))
        self.condition_ablation = str(condition_ablation).strip().lower()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # 未来段目标长度（stride=5 token 数）
        sample = self.samples[self.indices[i]]
        future_len = int(sample["future_length"])
        observed_future_len = int(sample["observed_future_length"])

        # future_delta 在预处理阶段已是非负量；这里转回实数域再标准化。
        raw = inv_softplus(sample["future_delta"] / self.delta_scale)
        x_norm = (raw - self.raw_mean) / self.raw_std

        x_padded = np.zeros(self.max_len, dtype=np.float32)
        x_padded[:future_len] = x_norm[:future_len]

        valid_mask = np.zeros(self.max_len, dtype=np.float32)
        valid_mask[:future_len] = 1.0

        observed_mask = np.zeros(self.max_len, dtype=np.float32)
        observed_mask[:observed_future_len] = 1.0

        early_soh = np.asarray(sample["early_soh_100"], dtype=np.float32)
        early_norm = (early_soh / 100.0) * 2.0 - 1.0
        feature_matrix = np.asarray(sample["feature_matrix"], dtype=np.float32)

        life_features = np.asarray(sample["life_features"], dtype=np.float32)
        feature_matrix, life_features = apply_condition_ablation(
            feature_matrix, life_features, mode=self.condition_ablation
        )

        return (
            torch.tensor(x_padded, dtype=torch.float32).view(1, -1),
            torch.tensor(observed_mask, dtype=torch.float32).view(1, -1),
            torch.tensor(valid_mask, dtype=torch.float32).view(1, -1),
            torch.tensor(feature_matrix, dtype=torch.float32).unsqueeze(0),
            torch.tensor(early_norm, dtype=torch.float32).view(1, -1),
            torch.tensor(life_features, dtype=torch.float32),
            torch.tensor(int(sample["group_id"]), dtype=torch.long),
            torch.tensor(float(sample["start_soh_100"]), dtype=torch.float32),
            torch.tensor(float(sample["eol_exists"]), dtype=torch.float32),
            torch.tensor(float(sample["eol_fraction"]), dtype=torch.float32),
            torch.tensor(float(sample["censor_fraction"]), dtype=torch.float32),
            torch.tensor(float(sample["target_length"]), dtype=torch.float32),
        )


def build_weighted_sampler(
    dataset,
    short_boost=1.0,
    medium_boost=1.0,
    long_boost=1.5,
    short_hard_boost=0.0,
    short_outlier_boost=0.0,
):
    """构建类别重加权采样器，可分别调节短/中/长寿命组采样强度。"""
    labels = []
    hardness = []
    outlier_raw = []
    short_feats = []
    for idx in dataset.indices:
        s = dataset.samples[idx]
        gid = int(s["group_id"])
        labels.append(gid)
        if gid == 0:
            short_feats.append(np.asarray(s.get("life_features", []), dtype=np.float32))
        if gid == 0 and float(short_hard_boost) > 0.0:
            eol_exists = float(s.get("eol_exists", 0.0))
            frac = float(s.get("eol_fraction", 0.0))
            if eol_exists > 0.5:
                h = float(np.clip((frac - 0.50) / 0.40, 0.0, 1.0))
            else:
                h = 0.35
            hardness.append(h)
        else:
            hardness.append(0.0)
        outlier_raw.append(0.0)
    labels = np.asarray(labels, dtype=np.int64)
    hardness = np.asarray(hardness, dtype=np.float32)
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    counts[counts == 0] = 1.0
    class_weights = 1.0 / counts
    class_weights[0] *= float(max(short_boost, 0.2))
    class_weights[1] *= float(max(medium_boost, 0.2))
    class_weights[2] *= float(max(long_boost, 0.2))
    weights = class_weights[labels] * (1.0 + float(max(short_hard_boost, 0.0)) * hardness)

    if float(short_outlier_boost) > 0.0 and len(short_feats) >= 4:
        feat_mat = np.stack(short_feats, axis=0).astype(np.float32)
        mu = np.mean(feat_mat, axis=0, keepdims=True)
        sd = np.std(feat_mat, axis=0, keepdims=True) + 1e-6
        # Robust outlierness in standardized life-feature space.
        for i, idx in enumerate(dataset.indices):
            s = dataset.samples[idx]
            if int(s["group_id"]) != 0:
                continue
            f = np.asarray(s.get("life_features", []), dtype=np.float32).reshape(1, -1)
            d = float(np.sqrt(np.mean(((f - mu) / sd) ** 2)))
            outlier_raw[i] = d
        outlier_raw = np.asarray(outlier_raw, dtype=np.float32)
        short_mask = labels == 0
        v = outlier_raw[short_mask]
        if len(v) > 0:
            lo = float(np.percentile(v, 10.0))
            hi = float(np.percentile(v, 90.0))
            denom = max(hi - lo, 1e-6)
            z = np.clip((outlier_raw - lo) / denom, 0.0, 1.0)
            boost = 1.0 + float(short_outlier_boost) * z
            boost[~short_mask] = 1.0
            weights = weights * boost

    return WeightedRandomSampler(weights=weights.tolist(), num_samples=len(weights), replacement=True)


def build_train_loader(
    train_set,
    batch_size,
    sampler_mode="weighted_random",
    short_boost=1.25,
    medium_boost=1.0,
    long_boost=1.5,
    short_hard_boost=0.0,
    short_outlier_boost=0.0,
):
    """
    训练采样策略构建器（训练策略对比用）：
    - weighted_random: 组别重加权随机采样（默认）
    - shuffle: 普通随机打乱
    - sequential: 固定顺序输入
    """
    mode = str(sampler_mode).strip().lower()
    if mode == "weighted_random":
        train_sampler = build_weighted_sampler(
            train_set,
            short_boost=short_boost,
            medium_boost=medium_boost,
            long_boost=long_boost,
            short_hard_boost=short_hard_boost,
            short_outlier_boost=short_outlier_boost,
        )
        loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=0)
        return loader, mode
    if mode == "shuffle":
        loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        return loader, mode
    if mode == "sequential":
        loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
        return loader, mode
    raise ValueError(f"Unsupported sampler_mode: {sampler_mode}")


def evaluate_group_classifier(model, loader, device):
    """评估分组分类器：返回验证损失与准确率。"""
    model.eval()
    losses, preds, gts = [], [], []
    with torch.no_grad():
        for batch in loader:
            (
                _x,
                _obs,
                _valid,
                feature_matrix,
                early_soh,
                life_features,
                group_ids,
                _start_soh_100,
                *_rest,
            ) = batch
            feature_matrix = feature_matrix.to(device)
            early_soh = early_soh.to(device)
            life_features = life_features.to(device)
            group_ids = group_ids.to(device)
            hint = torch.zeros_like(group_ids)
            logits = model(feature_matrix, early_soh, life_features, hint)
            loss = F.cross_entropy(logits, group_ids)
            losses.append(float(loss.item()))
            preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
            gts.extend(group_ids.detach().cpu().tolist())

    acc = float(np.mean(np.asarray(preds) == np.asarray(gts))) if preds else 0.0
    return {"loss": float(np.mean(losses)) if losses else 0.0, "acc": acc}


def train_group_classifier(train_loader, val_loader, device, life_dim, epochs=80, lr=1e-3):
    """训练分组分类器（仅用于辅助，不改变测试协议）。"""
    model = GroupClassifier(cond_dim=128, n_groups=3, life_dim=life_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_state, best_val = None, float("inf")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            (
                _x,
                _obs,
                _valid,
                feature_matrix,
                early_soh,
                life_features,
                group_ids,
                _start_soh_100,
                *_rest,
            ) = batch
            feature_matrix = feature_matrix.to(device)
            early_soh = early_soh.to(device)
            life_features = life_features.to(device)
            group_ids = group_ids.to(device)
            hint = torch.zeros_like(group_ids)
            logits = model(feature_matrix, early_soh, life_features, hint)
            loss = F.cross_entropy(logits, group_ids)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_metrics = evaluate_group_classifier(model, val_loader, device)
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"Classifier Epoch {epoch + 1:03d}/{epochs} | "
                f"Train {train_loss:.4f} | Val {val_metrics['loss']:.4f} | Acc {val_metrics['acc']:.4f}"
            )

    model.load_state_dict(best_state)
    final_val = evaluate_group_classifier(model, val_loader, device)
    print(f"Best classifier val loss: {best_val:.4f}, val acc: {final_val['acc']:.4f}")
    return model, history, final_val


def run_model_epoch(model, loader, device, optimizer=None, stage="diffusion"):
    """
    单个 epoch 执行器。
    - stage='deterministic' 时训练/评估确定性分支
    - stage='diffusion' 时训练/评估扩散分支
    """
    is_train = optimizer is not None
    model.train(is_train)
    metric_buffer = defaultdict(list)

    for batch in loader:
        (
            x_start,
            observed_mask,
            valid_mask,
            feature_matrix,
            early_soh,
            life_features,
            group_ids,
            start_soh_100,
            eol_exists,
            eol_fraction,
            censor_fraction,
            target_lengths,
        ) = batch

        x_start = x_start.to(device)
        observed_mask = observed_mask.to(device)
        valid_mask = valid_mask.to(device)
        feature_matrix = feature_matrix.to(device)
        early_soh = early_soh.to(device)
        life_features = life_features.to(device)
        group_ids = group_ids.to(device)
        start_soh_100 = start_soh_100.to(device)
        eol_exists = eol_exists.to(device)
        eol_fraction = eol_fraction.to(device)
        censor_fraction = censor_fraction.to(device)
        target_lengths = target_lengths.to(device)

        if stage == "deterministic":
            loss, metrics, _aux = model.forward_deterministic(
                x_start=x_start,
                observed_mask=observed_mask,
                valid_mask=valid_mask,
                feature_matrix=feature_matrix,
                early_soh=early_soh,
                life_features=life_features,
                group_ids=group_ids,
                start_soh_100=start_soh_100,
                eol_exists=eol_exists,
                eol_fraction=eol_fraction,
                censor_fraction=censor_fraction,
                target_lengths=target_lengths,
            )
        else:
            loss, metrics, _aux = model.forward_train(
                x_start=x_start,
                observed_mask=observed_mask,
                valid_mask=valid_mask,
                feature_matrix=feature_matrix,
                early_soh=early_soh,
                life_features=life_features,
                group_ids=group_ids,
                start_soh_100=start_soh_100,
                eol_exists=eol_exists,
                eol_fraction=eol_fraction,
                censor_fraction=censor_fraction,
                target_lengths=target_lengths,
            )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        for k, v in metrics.items():
            metric_buffer[k].append(float(v))

    return {k: float(np.mean(v)) for k, v in metric_buffer.items()}


def plot_training_curve(history, output_path):
    """绘制训练曲线（总损失/验证损失/验证噪声损失）。"""
    epochs = np.arange(1, len(history["train_total"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_total"], label="Train Total")
    plt.plot(epochs, history["val_total"], label="Val Total")
    if "val_noise" in history and len(history["val_noise"]) == len(epochs):
        plt.plot(epochs, history["val_noise"], label="Val Noise")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Code2 Diffusion Training Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def load_state_with_legacy_compat(model, state_dict, strict=False):
    """
    Load checkpoint with backward compatibility:
    - if old checkpoint lacks short expert branch, keep that branch disabled by default.
    """
    load_msg = model.load_state_dict(state_dict, strict=bool(strict))
    missing = set(load_msg.missing_keys or [])
    has_short_missing = any(k.startswith("short_det_") or k.startswith("short_gate.") for k in missing)
    if has_short_missing:
        with torch.no_grad():
            if hasattr(model, "short_det_basis"):
                model.short_det_basis.zero_()
            if hasattr(model, "short_det_coeff"):
                for p in model.short_det_coeff.parameters():
                    p.zero_()
            if hasattr(model, "short_gate"):
                for p in model.short_gate.parameters():
                    p.zero_()
                try:
                    model.short_gate[-1].bias.fill_(-4.0)
                except Exception:
                    pass
            if hasattr(model, "det_smooth_kernel"):
                model.det_smooth_kernel = 5
    return load_msg


def main(
    seed=42,
    runtime_seed=None,
    run_tag="",
    init_model_path="",
    det_max_epochs=240,
    diff_max_epochs=800,
    det_min_epochs=120,
    diff_min_epochs=120,
    det_patience=60,
    diff_patience=60,
    classifier_epochs=100,
    batch_size=16,
    lr=8e-4,
    short_boost=1.25,
    medium_boost=1.0,
    long_boost=1.5,
    short_hard_boost=0.0,
    short_outlier_boost=0.0,
    short_group_extra_weight=0.0,
    short_hard_weight=0.0,
    accel_weight=0.14,
    smooth_weight=0.04,
    short_accel_weight=0.12,
    long_accel_weight=0.10,
    curve_eol_weight=0.08,
    eol_consistency_weight=0.06,
    knee_weight=0.06,
    group_loss_short=1.30,
    group_loss_medium=1.00,
    group_loss_long=1.55,
    short_censored_weight=1.0,
    short_censored_use_observed_mask=False,
    short_use_observed_mask=False,
    det_rank=48,
    short_rank=24,
    long_rank=24,
    tail_weight_short=0.0,
    tail_weight_medium=0.0,
    tail_weight_long=0.0,
    tail_censored_scale=0.0,
    denoiser_type="transformer",
    condition_ablation="full",
    sampler_mode="weighted_random",
    train_stage_mode="two_stage",
    diffusion_target_mode="residual",
    use_group_experts=True,
):
    """
    主训练入口（完整训练流程）。
    参数可通过命令行覆盖，用于控制 epoch/patience/batch/lr 等超参数。
    """
    ensure_dirs()
    if runtime_seed is None:
        runtime_seed = int(seed)
    run_tag = str(run_tag).strip()
    set_global_seed(runtime_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"Code2 training (seed={seed})")
    print(f"Device: {device}")
    print("=" * 70)

    dataset_path = os.path.join(OUTPUT2_DIR, f"code2_dataset_seed{seed}.pkl")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing dataset file: {dataset_path}. Run 01_data_processing_code2.py first.")
    with open(dataset_path, "rb") as f:
        package = pickle.load(f)

    train_set = Code2Dataset(package, package["train_indices"], condition_ablation=condition_ablation)
    val_set = Code2Dataset(package, package["val_indices"], condition_ablation=condition_ablation)
    test_set = Code2Dataset(package, package["test_indices"], condition_ablation=condition_ablation)
    print(
        f"Samples: train={len(train_set)}, val={len(val_set)}, test={len(test_set)} | "
        f"future_len_max={package['max_future_length_padded']}"
    )
    print(f"Condition ablation mode: {condition_ablation}")
    print(f"Sampler mode: {sampler_mode}")
    print(f"Train stage mode: {train_stage_mode}")
    print(f"Diffusion target mode: {diffusion_target_mode}")
    print(f"Use group experts: {bool(use_group_experts)}")

    # 训练采样策略可配置：顺序 / 打乱 / 加权随机
    train_loader, sampler_mode = build_train_loader(
        train_set=train_set,
        batch_size=batch_size,
        sampler_mode=sampler_mode,
        short_boost=short_boost,
        medium_boost=medium_boost,
        long_boost=long_boost,
        short_hard_boost=short_hard_boost,
        short_outlier_boost=short_outlier_boost,
    )
    val_loader = DataLoader(val_set, batch_size=min(batch_size, max(1, len(val_set))), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=min(batch_size, max(1, len(test_set))), shuffle=False, num_workers=0)
    life_dim = int(package.get("life_feature_dim", 14))

    # 先训练一个轻量分组分类器，作为条件表征质量的辅助检查。
    classifier, clf_history, clf_val = train_group_classifier(
        train_loader, val_loader, device, life_dim=life_dim, epochs=classifier_epochs, lr=1e-3
    )
    clf_test = evaluate_group_classifier(classifier, test_loader, device)
    print(f"Classifier test acc: {clf_test['acc']:.4f}")

    model = ConditionalDiffusionModel(
        n_groups=3,
        cond_dim=128,
        life_dim=life_dim,
        seq_len=int(package["max_future_length_padded"]),
        timesteps=400,
        raw_mean=float(package["raw_mean"]),
        raw_std=float(package["raw_std"]),
        delta_scale=float(package["delta_scale"]),
        denoiser_type=str(denoiser_type),
        cfg_dropout=0.10,
        accel_weight=float(accel_weight),
        smooth_weight=float(smooth_weight),
        short_accel_weight=short_accel_weight,
        long_accel_weight=long_accel_weight,
        curve_eol_weight=curve_eol_weight,
        eol_consistency_weight=eol_consistency_weight,
        knee_weight=knee_weight,
        group_loss_weights=(group_loss_short, group_loss_medium, group_loss_long),
        short_group_extra_weight=short_group_extra_weight,
        short_hard_weight=short_hard_weight,
        short_censored_weight=short_censored_weight,
        short_censored_use_observed_mask=short_censored_use_observed_mask,
        short_use_observed_mask=short_use_observed_mask,
        det_rank=int(det_rank),
        short_rank=int(short_rank),
        long_rank=int(long_rank),
        tail_supervision_weights=(float(tail_weight_short), float(tail_weight_medium), float(tail_weight_long)),
        tail_censored_scale=float(tail_censored_scale),
        diffusion_target_mode=str(diffusion_target_mode),
        use_group_experts=bool(use_group_experts),
    ).to(device)
    model.set_schedule_device(device)

    init_path = str(init_model_path).strip()
    if init_path:
        if not os.path.exists(init_path):
            raise FileNotFoundError(f"init_model_path not found: {init_path}")
        init_ckpt = torch.load(init_path, map_location=device)
        state = init_ckpt.get("model_state", init_ckpt)
        init_msg = load_state_with_legacy_compat(model, state, strict=False)
        print(f"Initialized from: {init_path}")
        if init_msg.missing_keys or init_msg.unexpected_keys:
            print("[Warning] Non-strict init loading:")
            if init_msg.missing_keys:
                print("  missing:", init_msg.missing_keys)
            if init_msg.unexpected_keys:
                print("  unexpected:", init_msg.unexpected_keys)

    stage_mode = str(train_stage_mode).strip().lower()
    if stage_mode not in {"two_stage", "det_only", "diff_only"}:
        raise ValueError(f"Unsupported train_stage_mode: {train_stage_mode}")

    # ------------------------------ Stage 1 ------------------------------
    # deterministic pretraining: first learn trajectory backbone, then diffusion residual.
    best_det_val = float("inf")
    best_det_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    history_det = {"train_total": [], "val_total": []}
    det_epochs_run = 0
    det_min_epochs = min(int(det_max_epochs), max(int(det_min_epochs), int(det_patience)))

    if stage_mode != "diff_only":
        optimizer_det = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler_det = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_det, T_max=max(det_max_epochs, 1), eta_min=lr * 0.3
        )
        det_no_improve = 0

        print("\n[Stage 1] Deterministic trajectory pretraining")
        for epoch in range(det_max_epochs):
            train_metrics = run_model_epoch(model, train_loader, device, optimizer=optimizer_det, stage="deterministic")
            val_metrics = run_model_epoch(model, val_loader, device, optimizer=None, stage="deterministic")
            scheduler_det.step()
            det_epochs_run = epoch + 1

            history_det["train_total"].append(train_metrics.get("total_loss", 0.0))
            history_det["val_total"].append(val_metrics.get("total_loss", 0.0))

            if val_metrics.get("total_loss", 1e9) < (best_det_val - 1e-6):
                best_det_val = val_metrics["total_loss"]
                best_det_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                det_no_improve = 0
            else:
                det_no_improve += 1

            if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == det_max_epochs - 1:
                print(
                    f"Det Epoch {epoch + 1:03d}/{det_max_epochs} | "
                    f"Train {train_metrics.get('total_loss', 0.0):.4f} | "
                    f"Val {val_metrics.get('total_loss', 0.0):.4f} | "
                    f"Accel {val_metrics.get('accel_loss', 0.0):.4f} | "
                    f"ShortAccel {val_metrics.get('short_accel_loss', 0.0):.4f} | "
                    f"LongAccel {val_metrics.get('long_accel_loss', 0.0):.4f} | "
                    f"CurveEOL {val_metrics.get('curve_cls_loss', 0.0):.4f}/{val_metrics.get('curve_reg_loss', 0.0):.4f} | "
                    f"Knee {val_metrics.get('knee_loss', 0.0):.4f}"
                )
            if (epoch + 1) >= det_min_epochs and det_no_improve >= int(det_patience):
                print(
                    f"[Early stop] deterministic stage stopped at epoch {epoch + 1} "
                    f"(patience={det_patience}, best_val={best_det_val:.4f})"
                )
                break

        if best_det_state is not None:
            model.load_state_dict(best_det_state)
    else:
        print("\n[Stage 1] Skipped (train_stage_mode=diff_only)")

    # ------------------------------ Stage 2 ------------------------------
    best_val = float(best_det_val) if stage_mode != "diff_only" else float("inf")
    best_state = best_det_state if best_det_state is not None else {k: v.detach().cpu() for k, v in model.state_dict().items()}
    history = {"train_total": [], "val_total": [], "val_noise": []}
    diff_epochs_run = 0

    if stage_mode in {"two_stage", "diff_only"}:
        optimizer_diff = AdamW(model.parameters(), lr=lr * 0.7, weight_decay=1e-4)
        scheduler_diff = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_diff, T_max=max(diff_max_epochs, 1), eta_min=lr * 0.2
        )

        diff_no_improve = 0
        diff_min_epochs = min(int(diff_max_epochs), max(int(diff_min_epochs), int(diff_patience)))

        stage2_name = "Direct diffusion training" if str(model.diffusion_target_mode) == "direct" else "Residual diffusion training"
        print(f"\n[Stage 2] {stage2_name}")
        for epoch in range(diff_max_epochs):
            train_metrics = run_model_epoch(model, train_loader, device, optimizer=optimizer_diff, stage="diffusion")
            val_metrics = run_model_epoch(model, val_loader, device, optimizer=None, stage="diffusion")
            scheduler_diff.step()
            diff_epochs_run = epoch + 1

            history["train_total"].append(train_metrics.get("total_loss", 0.0))
            history["val_total"].append(val_metrics.get("total_loss", 0.0))
            history["val_noise"].append(val_metrics.get("noise_loss", 0.0))

            if val_metrics.get("total_loss", 1e9) < (best_val - 1e-6):
                best_val = val_metrics["total_loss"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                diff_no_improve = 0
            else:
                diff_no_improve += 1

            if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == diff_max_epochs - 1:
                print(
                    f"Diff Epoch {epoch + 1:03d}/{diff_max_epochs} | "
                    f"Train {train_metrics.get('total_loss', 0.0):.4f} | "
                    f"Val {val_metrics.get('total_loss', 0.0):.4f} | "
                    f"Noise {val_metrics.get('noise_loss', 0.0):.4f} | "
                    f"X0 {val_metrics.get('x0_loss', 0.0):.4f} | "
                    f"Det {val_metrics.get('det_anchor', 0.0):.4f} | "
                    f"ShortAccel {val_metrics.get('short_accel_loss', 0.0):.4f} | "
                    f"LongAccel {val_metrics.get('long_accel_loss', 0.0):.4f} | "
                    f"Knee {val_metrics.get('knee_loss', 0.0):.4f}"
                )
            if (epoch + 1) >= diff_min_epochs and diff_no_improve >= int(diff_patience):
                print(
                    f"[Early stop] diffusion stage stopped at epoch {epoch + 1} "
                    f"(patience={diff_patience}, best_val={best_val:.4f})"
                )
                break
    else:
        # det_only: keep Stage-1 best state and skip Stage-2.
        print("\n[Stage 2] Skipped (train_stage_mode=det_only)")
        history["train_total"] = list(history_det["train_total"])
        history["val_total"] = list(history_det["val_total"])
        history["val_noise"] = [0.0 for _ in history_det["val_total"]]

    if best_state is not None:
        model.load_state_dict(best_state)

    # 保存模型权重、配置与划分索引，保证实验可复现。
    ckpt = {
        "seed": int(seed),
        "runtime_seed": int(runtime_seed),
        "run_tag": run_tag,
        "init_model_path": init_path,
        "model_state": model.state_dict(),
        "classifier_state": classifier.state_dict(),
        "config": {
            "max_future_length_padded": int(package["max_future_length_padded"]),
            "life_feature_dim": int(life_dim),
            "delta_scale": float(package["delta_scale"]),
            "raw_mean": float(package["raw_mean"]),
            "raw_std": float(package["raw_std"]),
            "early_cycles": int(package["early_cycles"]),
            "stride": int(package["stride"]),
            "target_lengths": package["target_lengths"],
            "best_val": float(best_val),
            "best_det_val": float(best_det_val),
            "det_epochs": int(det_epochs_run),
            "diff_epochs": int(diff_epochs_run),
            "det_patience": int(det_patience),
            "diff_patience": int(diff_patience),
            "det_min_epochs": int(det_min_epochs),
            "diff_min_epochs": int(diff_min_epochs),
            "timesteps": int(model.timesteps),
            "denoiser_type": str(model.denoiser_type),
            "classifier_val_acc": float(clf_val["acc"]),
            "classifier_test_acc": float(clf_test["acc"]),
            "runtime_seed": int(runtime_seed),
            "run_tag": run_tag,
            "short_boost": float(short_boost),
            "medium_boost": float(medium_boost),
            "long_boost": float(long_boost),
            "short_hard_boost": float(short_hard_boost),
            "short_outlier_boost": float(short_outlier_boost),
            "short_group_extra_weight": float(short_group_extra_weight),
            "short_hard_weight": float(short_hard_weight),
            "accel_weight": float(accel_weight),
            "smooth_weight": float(smooth_weight),
            "short_accel_weight": float(short_accel_weight),
            "long_accel_weight": float(long_accel_weight),
            "curve_eol_weight": float(curve_eol_weight),
            "eol_consistency_weight": float(eol_consistency_weight),
            "knee_weight": float(knee_weight),
            "group_loss_short": float(group_loss_short),
            "group_loss_medium": float(group_loss_medium),
            "group_loss_long": float(group_loss_long),
            "short_censored_weight": float(short_censored_weight),
            "short_censored_use_observed_mask": bool(short_censored_use_observed_mask),
            "short_use_observed_mask": bool(short_use_observed_mask),
            "det_rank": int(det_rank),
            "short_rank": int(short_rank),
            "long_rank": int(long_rank),
            "tail_weight_short": float(tail_weight_short),
            "tail_weight_medium": float(tail_weight_medium),
            "tail_weight_long": float(tail_weight_long),
            "tail_censored_scale": float(tail_censored_scale),
            "condition_ablation": str(condition_ablation),
            "sampler_mode": str(sampler_mode),
            "train_stage_mode": str(stage_mode),
            "diffusion_target_mode": str(diffusion_target_mode),
            "use_group_experts": bool(use_group_experts),
        },
        "split_indices": {
            "train": package["train_indices"],
            "val": package["val_indices"],
            "test": package["test_indices"],
        },
    }

    suffix = f"_{run_tag}" if run_tag else ""
    model_path = os.path.join(MODEL2_DIR, f"code2_model_seed{seed}{suffix}.pth")
    torch.save(ckpt, model_path)

    curve_path = os.path.join(PLOT_MAIN_DIR, f"code2_training_curve_seed{seed}{suffix}.png")
    plot_training_curve(history, curve_path)

    # 保存训练日志，供后续画图与论文复盘使用。
    train_log = {
        "seed": int(seed),
        "runtime_seed": int(runtime_seed),
        "run_tag": run_tag,
        "init_model_path": init_path,
        "epochs": int(det_epochs_run + diff_epochs_run),
        "det_epochs": int(det_epochs_run),
        "diff_epochs": int(diff_epochs_run),
        "det_patience": int(det_patience),
        "diff_patience": int(diff_patience),
        "det_min_epochs": int(det_min_epochs),
        "diff_min_epochs": int(diff_min_epochs),
        "classifier_epochs": int(classifier_epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "short_boost": float(short_boost),
        "medium_boost": float(medium_boost),
        "long_boost": float(long_boost),
        "short_hard_boost": float(short_hard_boost),
        "short_outlier_boost": float(short_outlier_boost),
        "short_group_extra_weight": float(short_group_extra_weight),
        "short_hard_weight": float(short_hard_weight),
        "accel_weight": float(accel_weight),
        "smooth_weight": float(smooth_weight),
        "short_accel_weight": float(short_accel_weight),
        "long_accel_weight": float(long_accel_weight),
        "curve_eol_weight": float(curve_eol_weight),
        "eol_consistency_weight": float(eol_consistency_weight),
        "knee_weight": float(knee_weight),
        "group_loss_short": float(group_loss_short),
        "group_loss_medium": float(group_loss_medium),
        "group_loss_long": float(group_loss_long),
        "short_censored_weight": float(short_censored_weight),
        "short_censored_use_observed_mask": bool(short_censored_use_observed_mask),
        "short_use_observed_mask": bool(short_use_observed_mask),
        "det_rank": int(det_rank),
        "short_rank": int(short_rank),
        "long_rank": int(long_rank),
        "tail_weight_short": float(tail_weight_short),
        "tail_weight_medium": float(tail_weight_medium),
        "tail_weight_long": float(tail_weight_long),
        "tail_censored_scale": float(tail_censored_scale),
        "denoiser_type": str(denoiser_type),
        "condition_ablation": str(condition_ablation),
        "sampler_mode": str(sampler_mode),
        "train_stage_mode": str(stage_mode),
        "diffusion_target_mode": str(diffusion_target_mode),
        "use_group_experts": bool(use_group_experts),
        "best_det_val": float(best_det_val),
        "best_val": float(best_val),
        "history_det": history_det,
        "history": history,
        "classifier_history": clf_history,
        "classifier_val": clf_val,
        "classifier_test": clf_test,
    }
    log_path = os.path.join(OUTPUT2_DIR, f"code2_train_log_seed{seed}{suffix}.pkl")
    with open(log_path, "wb") as f:
        pickle.dump(train_log, f)

    print("=" * 70)
    print(f"Saved model: {model_path}")
    print(f"Saved training curve: {curve_path}")
    print(f"Saved train log: {log_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code2 train script")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runtime_seed", type=int, default=None)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--init_model_path", type=str, default="")
    parser.add_argument("--det_max_epochs", type=int, default=240)
    parser.add_argument("--diff_max_epochs", type=int, default=800)
    parser.add_argument("--det_min_epochs", type=int, default=120)
    parser.add_argument("--diff_min_epochs", type=int, default=120)
    parser.add_argument("--det_patience", type=int, default=60)
    parser.add_argument("--diff_patience", type=int, default=60)
    parser.add_argument("--classifier_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--accel_weight", type=float, default=0.14)
    parser.add_argument("--smooth_weight", type=float, default=0.04)
    parser.add_argument("--short_boost", type=float, default=1.25)
    parser.add_argument("--medium_boost", type=float, default=1.0)
    parser.add_argument("--long_boost", type=float, default=1.5)
    parser.add_argument("--short_hard_boost", type=float, default=0.0)
    parser.add_argument("--short_outlier_boost", type=float, default=0.0)
    parser.add_argument("--short_group_extra_weight", type=float, default=0.0)
    parser.add_argument("--short_hard_weight", type=float, default=0.0)
    parser.add_argument("--short_accel_weight", type=float, default=0.12)
    parser.add_argument("--long_accel_weight", type=float, default=0.10)
    parser.add_argument("--curve_eol_weight", type=float, default=0.08)
    parser.add_argument("--eol_consistency_weight", type=float, default=0.06)
    parser.add_argument("--knee_weight", type=float, default=0.06)
    parser.add_argument("--group_loss_short", type=float, default=1.30)
    parser.add_argument("--group_loss_medium", type=float, default=1.00)
    parser.add_argument("--group_loss_long", type=float, default=1.55)
    parser.add_argument("--short_censored_weight", type=float, default=1.0)
    parser.add_argument("--short_censored_use_observed_mask", action="store_true", default=False)
    parser.add_argument("--short_use_observed_mask", action="store_true", default=False)
    parser.add_argument("--det_rank", type=int, default=48)
    parser.add_argument("--short_rank", type=int, default=24)
    parser.add_argument("--long_rank", type=int, default=24)
    parser.add_argument("--tail_weight_short", type=float, default=0.0)
    parser.add_argument("--tail_weight_medium", type=float, default=0.0)
    parser.add_argument("--tail_weight_long", type=float, default=0.0)
    parser.add_argument("--tail_censored_scale", type=float, default=0.0)
    parser.add_argument("--denoiser_type", type=str, default="transformer", choices=["transformer", "unet"])
    parser.add_argument(
        "--condition_ablation",
        type=str,
        default="full",
        choices=["full", "no_life", "life_only"],
        help="Ablation on condition inputs.",
    )
    parser.add_argument(
        "--sampler_mode",
        type=str,
        default="weighted_random",
        choices=["weighted_random", "shuffle", "sequential"],
        help="Training data sampling strategy.",
    )
    parser.add_argument(
        "--train_stage_mode",
        type=str,
        default="two_stage",
        choices=["two_stage", "det_only", "diff_only"],
        help="two_stage: Stage1+Stage2, det_only: Stage1 only, diff_only: Stage2 only.",
    )
    parser.add_argument(
        "--diffusion_target_mode",
        type=str,
        default="residual",
        choices=["residual", "direct"],
        help="residual: diffuse residual on top of deterministic backbone; direct: diffuse full trajectory.",
    )
    parser.add_argument(
        "--disable_group_experts",
        action="store_true",
        default=False,
        help="Disable group-specific deterministic expert branches (group template + short/long experts).",
    )
    args = parser.parse_args()
    main(
        seed=args.seed,
        runtime_seed=args.runtime_seed,
        run_tag=args.run_tag,
        init_model_path=args.init_model_path,
        det_max_epochs=args.det_max_epochs,
        diff_max_epochs=args.diff_max_epochs,
        det_min_epochs=args.det_min_epochs,
        diff_min_epochs=args.diff_min_epochs,
        det_patience=args.det_patience,
        diff_patience=args.diff_patience,
        classifier_epochs=args.classifier_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        accel_weight=args.accel_weight,
        smooth_weight=args.smooth_weight,
        short_boost=args.short_boost,
        medium_boost=args.medium_boost,
        long_boost=args.long_boost,
        short_hard_boost=args.short_hard_boost,
        short_outlier_boost=args.short_outlier_boost,
        short_group_extra_weight=args.short_group_extra_weight,
        short_hard_weight=args.short_hard_weight,
        short_accel_weight=args.short_accel_weight,
        long_accel_weight=args.long_accel_weight,
        curve_eol_weight=args.curve_eol_weight,
        eol_consistency_weight=args.eol_consistency_weight,
        knee_weight=args.knee_weight,
        group_loss_short=args.group_loss_short,
        group_loss_medium=args.group_loss_medium,
        group_loss_long=args.group_loss_long,
        short_censored_weight=args.short_censored_weight,
        short_censored_use_observed_mask=bool(args.short_censored_use_observed_mask),
        short_use_observed_mask=bool(args.short_use_observed_mask),
        det_rank=args.det_rank,
        short_rank=args.short_rank,
        long_rank=args.long_rank,
        tail_weight_short=args.tail_weight_short,
        tail_weight_medium=args.tail_weight_medium,
        tail_weight_long=args.tail_weight_long,
        tail_censored_scale=args.tail_censored_scale,
        denoiser_type=args.denoiser_type,
        condition_ablation=args.condition_ablation,
        sampler_mode=args.sampler_mode,
        train_stage_mode=args.train_stage_mode,
        diffusion_target_mode=args.diffusion_target_mode,
        use_group_experts=not bool(args.disable_group_experts),
    )
