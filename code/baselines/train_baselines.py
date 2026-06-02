import argparse
import os
import pickle
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE2_DIR = os.path.dirname(CURRENT_DIR)
if CODE2_DIR not in sys.path:
    sys.path.insert(0, CODE2_DIR)

from common import (
    MODEL2_DIR,
    OUTPUT2_DIR,
    PLOT_BASELINE_DIR,
    ensure_dirs,
    inv_softplus,
    set_global_seed,
)  # noqa: E402
from baselines.models_baseline import BaselineDeterministicModel  # noqa: E402

"""
Baseline 训练脚本
================
公平对比原则：
- 数据、划分、输入、损失口径与扩散主线保持一致
- 仅替换 deterministic 轨迹生成骨干（LSTM / Transformer）

输出：
- Model2/baseline_{model}_seed{seed}.pth
- Output2/baseline_{model}_training_curve_seed{seed}.png
"""


plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class Code2Dataset(Dataset):
    """与主训练脚本同口径的数据读取器（字段保持一致）。"""

    def __init__(self, package, indices):
        self.samples = package["samples"]
        self.indices = list(indices)
        self.max_len = int(package["max_future_length_padded"])
        self.delta_scale = float(package["delta_scale"])
        self.raw_mean = float(package["raw_mean"])
        self.raw_std = float(package["raw_std"])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        s = self.samples[self.indices[i]]
        future_len = int(s["future_length"])
        observed_len = int(s["observed_future_length"])

        raw = inv_softplus(s["future_delta"] / self.delta_scale)
        x_norm = (raw - self.raw_mean) / self.raw_std

        x_pad = np.zeros(self.max_len, dtype=np.float32)
        x_pad[:future_len] = x_norm[:future_len]

        valid_mask = np.zeros(self.max_len, dtype=np.float32)
        valid_mask[:future_len] = 1.0

        observed_mask = np.zeros(self.max_len, dtype=np.float32)
        observed_mask[:observed_len] = 1.0

        early = np.asarray(s["early_soh_100"], dtype=np.float32)
        early_norm = (early / 100.0) * 2.0 - 1.0
        fm = np.asarray(s["feature_matrix"], dtype=np.float32)
        life = np.asarray(s["life_features"], dtype=np.float32)

        return (
            torch.tensor(x_pad, dtype=torch.float32).view(1, -1),
            torch.tensor(observed_mask, dtype=torch.float32).view(1, -1),
            torch.tensor(valid_mask, dtype=torch.float32).view(1, -1),
            torch.tensor(fm, dtype=torch.float32).unsqueeze(0),
            torch.tensor(early_norm, dtype=torch.float32).view(1, -1),
            torch.tensor(life, dtype=torch.float32),
            torch.tensor(int(s["group_id"]), dtype=torch.long),
            torch.tensor(float(s["start_soh_100"]), dtype=torch.float32),
            torch.tensor(float(s["eol_exists"]), dtype=torch.float32),
            torch.tensor(float(s["eol_fraction"]), dtype=torch.float32),
            torch.tensor(float(s["censor_fraction"]), dtype=torch.float32),
            torch.tensor(float(s["target_length"]), dtype=torch.float32),
        )


def build_weighted_sampler(dataset, long_boost=1.5):
    """组别重加权采样器，提升长寿命组覆盖率。"""
    labels = np.asarray([int(dataset.samples[i]["group_id"]) for i in dataset.indices], dtype=np.int64)
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    counts[counts == 0] = 1.0
    w_cls = 1.0 / counts
    w_cls[2] *= float(max(long_boost, 1.0))
    weights = w_cls[labels]
    return WeightedRandomSampler(weights.tolist(), num_samples=len(weights), replacement=True)


def run_epoch(model, loader, device, optimizer=None):
    """单个 epoch 训练/验证执行器（deterministic 前向）。"""
    is_train = optimizer is not None
    model.train(is_train)
    meter = defaultdict(list)
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

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        for k, v in metrics.items():
            meter[k].append(float(v))
    return {k: float(np.mean(v)) for k, v in meter.items()}


def plot_curve(history, out_path, title):
    """保存 baseline 训练曲线图。"""
    xs = np.arange(1, len(history["train_total"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(xs, history["train_total"], label="Train Total")
    plt.plot(xs, history["val_total"], label="Val Total")
    plt.plot(xs, history["train_rec"], label="Train Rec")
    plt.plot(xs, history["val_rec"], label="Val Rec")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train_one(
    backbone,
    package,
    seed=42,
    batch_size=32,
    epochs=400,
    patience=80,
    lr=8e-4,
    wd=1e-4,
):
    """训练单个 baseline（lstm 或 transformer）。"""
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = Code2Dataset(package, package["train_indices"])
    val_set = Code2Dataset(package, package["val_indices"])
    test_set = Code2Dataset(package, package["test_indices"])

    # 与主线一致：长组重采样，避免训练被短组主导。
    train_sampler = build_weighted_sampler(train_set, long_boost=1.5)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # 关键公平性：复用与扩散模型一致的条件编码和损失，仅替换序列骨干。
    model = BaselineDeterministicModel(
        backbone=backbone,
        n_groups=3,
        cond_dim=128,
        life_dim=int(package.get("life_feature_dim", 14)),
        seq_len=int(package["max_future_length_padded"]),
        raw_mean=float(package["raw_mean"]),
        raw_std=float(package["raw_std"]),
        delta_scale=float(package["delta_scale"]),
        accel_weight=0.14,
        smooth_weight=0.04,
        long_accel_weight=0.10,
        curve_eol_weight=0.08,
        eol_consistency_weight=0.06,
        knee_weight=0.06,
        group_loss_weights=(1.0, 1.05, 1.8),
    ).to(device)
    model.set_schedule_device(device)

    optimizer = AdamW(model.parameters(), lr=float(lr), weight_decay=float(wd))
    best_val = float("inf")
    best_state = None
    bad = 0
    history = {"train_total": [], "val_total": [], "train_rec": [], "val_rec": []}

    print("=" * 70)
    print(f"Training baseline: {backbone} | seed={seed} | device={device}")
    print(f"Train/Val/Test: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    print("=" * 70)

    # 标准早停训练循环
    for ep in range(1, int(epochs) + 1):
        tr = run_epoch(model, train_loader, device, optimizer)
        va = run_epoch(model, val_loader, device, optimizer=None)

        tr_total = float(tr.get("total_loss", 0.0))
        va_total = float(va.get("total_loss", 0.0))
        tr_rec = float(tr.get("rec_loss", 0.0))
        va_rec = float(va.get("rec_loss", 0.0))
        history["train_total"].append(tr_total)
        history["val_total"].append(va_total)
        history["train_rec"].append(tr_rec)
        history["val_rec"].append(va_rec)

        if va_total < best_val:
            best_val = va_total
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep == 1 or ep % 20 == 0:
            print(
                f"[{backbone}] Epoch {ep:03d}/{epochs} | "
                f"Train {tr_total:.4f} (rec {tr_rec:.4f}) | Val {va_total:.4f} (rec {va_rec:.4f})"
            )
        if bad >= int(patience):
            print(f"[{backbone}] Early stop at epoch {ep}, best val={best_val:.4f}")
            break

    if best_state is None:
        raise RuntimeError(f"{backbone} baseline failed to produce a valid checkpoint.")
    model.load_state_dict(best_state)
    te = run_epoch(model, test_loader, device, optimizer=None)
    print(f"[{backbone}] Test total={te.get('total_loss', -1):.4f} | rec={te.get('rec_loss', -1):.4f}")

    ckpt = {
        "model_state": model.state_dict(),
        "config": {
            "backbone": backbone,
            "n_groups": 3,
            "cond_dim": 128,
            "life_dim": int(package.get("life_feature_dim", 14)),
            "seq_len": int(package["max_future_length_padded"]),
            "raw_mean": float(package["raw_mean"]),
            "raw_std": float(package["raw_std"]),
            "delta_scale": float(package["delta_scale"]),
            "seed": int(seed),
        },
        "best_val_total": float(best_val),
        "history": history,
        "split": {
            "train": package["train_indices"],
            "val": package["val_indices"],
            "test": package["test_indices"],
        },
    }

    model_path = os.path.join(MODEL2_DIR, f"baseline_{backbone}_seed{seed}.pth")
    with open(model_path, "wb") as f:
        torch.save(ckpt, f)
    curve_path = os.path.join(PLOT_BASELINE_DIR, f"baseline_{backbone}_training_curve_seed{seed}.png")
    plot_curve(history, curve_path, f"Baseline {backbone.upper()} training (seed={seed})")

    print(f"[{backbone}] Saved model: {model_path}")
    print(f"[{backbone}] Saved curve: {curve_path}")


def main(seed=42, model="both", batch_size=32, epochs=400, patience=80, lr=8e-4, wd=1e-4):
    """支持一次训练单模型或同时训练两个 baseline。"""
    ensure_dirs()
    dataset_path = os.path.join(OUTPUT2_DIR, f"code2_dataset_seed{seed}.pkl")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")
    with open(dataset_path, "rb") as f:
        package = pickle.load(f)

    targets = ["lstm", "transformer"] if model == "both" else [model]
    for m in targets:
        train_one(
            backbone=m,
            package=package,
            seed=seed,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            lr=lr,
            wd=wd,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deterministic baseline models for Code2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="both", choices=["lstm", "transformer", "both"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()
    main(
        seed=args.seed,
        model=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        wd=args.weight_decay,
    )
