import argparse
import os
import pickle
import sys

import numpy as np
import torch
from scipy.stats import pearsonr

"""
Baseline 测试脚本
=================
目标：在“与 Code2 主测试同口径”的条件下评估 baseline。

评估流程：
1) 读取 baseline checkpoint
2) 对每个测试样本预测完整曲线
3) 进行 EOL 融合与终端约束
4) 计算 pre-EOL RMSE/MAE/Corr + EOL误差
5) 保存为与主线一致的 pkl 结构，便于统一画图
"""


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE2_DIR = os.path.dirname(CURRENT_DIR)
if CODE2_DIR not in sys.path:
    sys.path.insert(0, CODE2_DIR)

from common import (  # noqa: E402
    EARLY_CYCLES,
    EOL_THRESHOLD,
    MODEL2_DIR,
    OUTPUT2_DIR,
    SEQ_CYCLE_STRIDE,
    calculate_eol_cycle,
    ensure_dirs,
    monotone_curve,
    safe_interp,
    set_global_seed,
)
from baselines.models_baseline import BaselineDeterministicModel  # noqa: E402


SOH_MIN = 60.0
EOL_EVAL_TOL = 0.1


def reconstruct_full_curve_from_future(early_soh_100, future_soh, target_length):
    """未来段重建全长曲线（与主测试脚本同逻辑）。"""
    early_soh_100 = np.asarray(early_soh_100, dtype=np.float32).flatten()
    future_soh = np.asarray(future_soh, dtype=np.float32).flatten()
    target_length = int(target_length)

    if target_length <= EARLY_CYCLES:
        return monotone_curve(np.clip(early_soh_100[:target_length], SOH_MIN, 100.0))

    early_cycles = np.arange(1, EARLY_CYCLES + 1, dtype=np.float32)
    future_cycles = np.arange(
        EARLY_CYCLES + SEQ_CYCLE_STRIDE,
        target_length + 1,
        SEQ_CYCLE_STRIDE,
        dtype=np.float32,
    )
    n_future = min(len(future_cycles), len(future_soh))
    future_cycles = future_cycles[:n_future]
    future_soh = future_soh[:n_future]

    x_anchor = np.concatenate([early_cycles, future_cycles], axis=0)
    y_anchor = np.concatenate([early_soh_100, future_soh], axis=0)
    y_anchor = monotone_curve(np.clip(y_anchor, SOH_MIN, 100.0))

    full_x = np.arange(1, target_length + 1, dtype=np.float32)
    full = safe_interp(x_anchor, y_anchor, full_x)
    return monotone_curve(np.clip(full, SOH_MIN, 100.0))


def calculate_eol_cycle_eval(soh_values, threshold=EOL_THRESHOLD, tol=EOL_EVAL_TOL):
    """带容差的 EOL 判定，避免 80.00x 边界误判。"""
    soh_values = np.asarray(soh_values, dtype=np.float32).flatten()
    if len(soh_values) == 0:
        return -1
    idx = np.where(soh_values <= float(threshold))[0]
    if len(idx) > 0:
        return int(idx[0] + 1)
    idx_tol = np.where(soh_values <= float(threshold) + float(tol))[0]
    return int(idx_tol[0] + 1) if len(idx_tol) > 0 else -1


def apply_eol_terminal(curve, eol_target):
    """将预测曲线终止于 EOL=80 并返回绘图版曲线（EOL 后置 NaN）。"""
    curve = np.asarray(curve, dtype=np.float32).flatten()
    out = monotone_curve(np.clip(curve, SOH_MIN, 100.0))
    n = len(out)
    if n <= 1:
        return out.astype(np.float32), out.astype(np.float32), int(max(1, n))

    if int(eol_target) <= 0:
        eol_target = int(0.98 * n)
    eol_target = int(np.clip(int(eol_target), EARLY_CYCLES + 1, n))
    idx = int(np.clip(eol_target - 1, 0, n - 1))

    natural = calculate_eol_cycle(out, EOL_THRESHOLD)
    if natural > 0 and abs(int(natural) - int(eol_target)) <= 60:
        out[int(natural) - 1 :] = EOL_THRESHOLD
        out = monotone_curve(np.clip(out, SOH_MIN, 100.0))
        plot = out.copy()
        if natural < n:
            plot[int(natural) :] = np.nan
        return out.astype(np.float32), plot.astype(np.float32), int(natural)

    if idx >= 1:
        start = float(out[0])
        desired_drop = max(start - float(EOL_THRESHOLD), 1e-4)
        d = np.clip(out[:idx] - out[1 : idx + 1], 0.0, None)
        d_sum = float(np.sum(d))
        if d_sum <= 1e-8:
            d_adj = np.full((idx,), desired_drop / max(idx, 1), dtype=np.float32)
        else:
            d_adj = (d / d_sum) * desired_drop
        prefix = np.empty((idx + 1,), dtype=np.float32)
        prefix[0] = start
        for i in range(idx):
            prefix[i + 1] = prefix[i] - d_adj[i]
        out[: idx + 1] = prefix
        out[:idx] = np.maximum(out[:idx], EOL_THRESHOLD + 1e-3)

    out[idx:] = EOL_THRESHOLD
    out = monotone_curve(np.clip(out, SOH_MIN, 100.0))
    eol = calculate_eol_cycle(out, EOL_THRESHOLD)
    if eol <= 0:
        eol = int(np.clip(eol_target, EARLY_CYCLES + 1, n))
        out[eol - 1 :] = EOL_THRESHOLD
        out = monotone_curve(np.clip(out, SOH_MIN, 100.0))

    plot = out.copy()
    if eol > 0 and eol < n:
        plot[eol:] = np.nan
    return out.astype(np.float32), plot.astype(np.float32), int(eol)


def fuse_eol_prediction(curve_eol, aux_prob, aux_eol_pred, group_id, max_len):
    """融合 curve crossing 与辅助头 EOL。"""
    curve_eol = int(curve_eol)
    aux_eol_pred = int(aux_eol_pred)
    max_len = int(max_len)
    g = int(group_id)

    candidates = []
    if curve_eol > 0:
        candidates.append(("curve", float(curve_eol), 0.65))
    if aux_prob >= 0.45 and aux_eol_pred > 0:
        candidates.append(("aux", float(aux_eol_pred), 0.35))
    if not candidates:
        fallback = aux_eol_pred if aux_eol_pred > 0 else int(0.98 * max_len)
        return int(np.clip(fallback, EARLY_CYCLES + 1, max_len))

    if g == 1:
        vals = [v for _, v, _ in candidates]
        fused = float(min(vals))
    else:
        vals = np.asarray([v for _, v, _ in candidates], dtype=np.float32)
        ws = np.asarray([w for _, _, w in candidates], dtype=np.float32)
        fused = float(np.sum(vals * ws) / np.clip(np.sum(ws), 1e-8, None))
    fused = int(round(fused))
    return int(np.clip(fused, EARLY_CYCLES + 1, max_len))


def eval_rmse_mae_corr(true_curve, pred_curve):
    """计算 pre-EOL RMSE/MAE/Corr。"""
    a = np.asarray(true_curve, dtype=np.float32)
    b = np.asarray(pred_curve, dtype=np.float32)
    eol_true = calculate_eol_cycle_eval(a, EOL_THRESHOLD, EOL_EVAL_TOL)
    eol_pred = calculate_eol_cycle_eval(b, EOL_THRESHOLD, EOL_EVAL_TOL)
    if eol_true > 0 and eol_pred > 0:
        end = max(EARLY_CYCLES, min(eol_true, eol_pred))
    elif eol_true > 0:
        end = max(EARLY_CYCLES, eol_true)
    elif eol_pred > 0:
        end = max(EARLY_CYCLES, eol_pred)
    else:
        end = len(a)
    end = int(np.clip(end, 5, len(a)))
    a = a[:end]
    b = b[:end]

    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    mae = float(np.mean(np.abs(a - b)))
    if len(a) > 10:
        corr, _ = pearsonr(a, b)
        corr = float(corr) if not np.isnan(corr) else -1.0
    else:
        corr = -1.0
    return rmse, mae, corr


def calculate_metrics(results):
    """聚合测试集指标并统计各组最差样本。"""
    rmses, maes, cors = [], [], []
    eol_errors = []
    false_cross, missed_cross = 0, 0
    true_eol_positive = 0
    group_worst = {0: {"rmse": [], "eol": []}, 1: {"rmse": [], "eol": []}, 2: {"rmse": [], "eol": []}}

    for item in results:
        true_curve = np.asarray(item["true_curve"], dtype=np.float32)
        pred_curve = np.asarray(item["pred_curve"], dtype=np.float32)
        g = int(item["true_group"])

        rmse, mae, corr = eval_rmse_mae_corr(true_curve, pred_curve)
        rmses.append(rmse)
        maes.append(mae)
        group_worst[g]["rmse"].append(rmse)
        if corr >= -0.5:
            cors.append(corr)

        eol_true = int(item["eol_true"])
        eol_pred = int(item["eol_pred"])
        if eol_true > 0:
            true_eol_positive += 1
            if eol_pred > 0:
                err = abs(eol_true - eol_pred)
                eol_errors.append(err)
                group_worst[g]["eol"].append(err)
            else:
                missed_cross += 1
        elif eol_pred > 0:
            false_cross += 1

    valid_ratio = float(len(eol_errors) / max(true_eol_positive, 1))
    out = {
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "corr_mean": float(np.mean(cors)) if cors else -1.0,
        "corr_std": float(np.std(cors)) if cors else -1.0,
        "eol_error_mean": float(np.mean(eol_errors)) if eol_errors else -1.0,
        "eol_error_std": float(np.std(eol_errors)) if eol_errors else -1.0,
        "n_valid_eol": int(len(eol_errors)),
        "n_true_eol_positive": int(true_eol_positive),
        "valid_eol_ratio": float(valid_ratio),
        "false_cross": int(false_cross),
        "missed_cross": int(missed_cross),
        "group_worst": {},
    }
    for g in [0, 1, 2]:
        rmse_max = max(group_worst[g]["rmse"]) if group_worst[g]["rmse"] else -1.0
        eol_max = max(group_worst[g]["eol"]) if group_worst[g]["eol"] else -1.0
        pass_rmse = rmse_max <= 3.0 if rmse_max >= 0 else False
        pass_eol = eol_max <= 150.0 if eol_max >= 0 else True
        out["group_worst"][g] = {
            "rmse_max": float(rmse_max),
            "eol_max": float(eol_max),
            "pass_rmse<=3": bool(pass_rmse),
            "pass_eol<=150": bool(pass_eol),
        }
    return out


def print_metrics(metrics, tag):
    """打印结构化指标报告。"""
    print("\n" + "=" * 70)
    print(f"Baseline summary ({tag})")
    print("=" * 70)
    print(f"RMSE(pre-EOL): {metrics['rmse_mean']:.4f} +/- {metrics['rmse_std']:.4f}")
    print(f"MAE (pre-EOL): {metrics['mae_mean']:.4f} +/- {metrics['mae_std']:.4f}")
    print(f"Corr(pre-EOL): {metrics['corr_mean']:.4f} +/- {metrics['corr_std']:.4f}")
    print(f"EOL mean error: {metrics['eol_error_mean']:.2f} +/- {metrics['eol_error_std']:.2f}")
    print(f"Valid EOL: {metrics['n_valid_eol']} / {metrics['n_true_eol_positive']} (ratio={metrics['valid_eol_ratio']:.3f})")
    print(f"False crossings: {metrics['false_cross']} | Missed crossings: {metrics['missed_cross']}")
    for g in [0, 1, 2]:
        s = metrics["group_worst"][g]
        print(
            f"  Group {g}: RMSE_max={s['rmse_max']:.3f}, EOL_max={s['eol_max']:.1f}, "
            f"pass(RMSE<=3)={s['pass_rmse<=3']}, pass(EOL<=150)={s['pass_eol<=150']}"
        )
    print("=" * 70)


def main(seed=42, model="lstm"):
    """baseline 测试入口。"""
    ensure_dirs()
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = os.path.join(OUTPUT2_DIR, f"code2_dataset_seed{seed}.pkl")
    model_path = os.path.join(MODEL2_DIR, f"baseline_{model}_seed{seed}.pth")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    with open(dataset_path, "rb") as f:
        package = pickle.load(f)
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["config"]

    target_lengths = {int(k): int(v) for k, v in package["target_lengths"].items()}
    # 与训练时同配置重建模型。
    model_net = BaselineDeterministicModel(
        backbone=str(cfg["backbone"]),
        n_groups=int(cfg["n_groups"]),
        cond_dim=int(cfg["cond_dim"]),
        life_dim=int(cfg["life_dim"]),
        seq_len=int(cfg["seq_len"]),
        raw_mean=float(cfg["raw_mean"]),
        raw_std=float(cfg["raw_std"]),
        delta_scale=float(cfg["delta_scale"]),
        accel_weight=0.14,
        smooth_weight=0.04,
        long_accel_weight=0.10,
        curve_eol_weight=0.08,
        eol_consistency_weight=0.06,
        knee_weight=0.06,
        group_loss_weights=(1.0, 1.05, 1.8),
    ).to(device)
    msg = model_net.load_state_dict(ckpt["model_state"], strict=False)
    if len(msg.missing_keys) > 0 or len(msg.unexpected_keys) > 0:
        print("[Warning] Non-strict model loading:")
        if msg.missing_keys:
            print("  missing:", msg.missing_keys)
        if msg.unexpected_keys:
            print("  unexpected:", msg.unexpected_keys)
    model_net.eval()
    model_net.set_schedule_device(device)

    print("=" * 70)
    print(f"Baseline testing ({model}) | seed={seed} | device={device}")
    print(f"Test samples: {len(package['test_indices'])}")
    print("=" * 70)

    results = []
    group_acc_hits = 0
    for i, idx in enumerate(package["test_indices"], start=1):
        s = package["samples"][idx]
        true_group = int(s["group_id"])
        pred_group = int(s["group_id"])
        group_acc_hits += int(pred_group == true_group)

        pred_target_len = int(target_lengths[pred_group])
        true_target_len = int(s["target_length"])
        true_curve = np.asarray(s["true_curve_full"], dtype=np.float32)

        early_soh = np.asarray(s["early_soh_100"], dtype=np.float32)
        early_norm = (early_soh / 100.0) * 2.0 - 1.0
        fm = np.asarray(s["feature_matrix"], dtype=np.float32)
        life = np.asarray(s["life_features"], dtype=np.float32)
        start_soh = float(s["start_soh_100"])

        fm_t = torch.tensor(fm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        early_t = torch.tensor(early_norm, dtype=torch.float32, device=device).view(1, 1, -1)
        life_t = torch.tensor(life, dtype=torch.float32, device=device).unsqueeze(0)
        gid_t = torch.tensor([pred_group], dtype=torch.long, device=device)
        start_t = torch.tensor([start_soh], dtype=torch.float32, device=device)

        # 前向预测：cond -> x_norm -> future_soh -> full curve
        with torch.no_grad():
            cond = model_net.cond_encoder(fm_t, early_t, life_t, gid_t)
            x_norm = model_net.predict_deterministic_x(cond, gid_t)
            aux_pack = model_net.predict_eol_head(fm_t, early_t, life_t, gid_t)
            aux_prob = float(aux_pack["eol_prob"].detach().cpu().numpy().reshape(-1)[0])
            aux_frac = float(aux_pack["eol_fraction"].detach().cpu().numpy().reshape(-1)[0])
            aux_eol_pred = int(round(aux_frac * pred_target_len)) if aux_prob >= 0.30 else -1
            future_soh_all, _delta = model_net.build_future_curve(x_norm, start_t)

        future_len = int((pred_target_len - EARLY_CYCLES) // SEQ_CYCLE_STRIDE)
        future_soh = future_soh_all.detach().cpu().numpy().reshape(-1)[:future_len]
        curve = reconstruct_full_curve_from_future(early_soh, future_soh, pred_target_len)
        if pred_target_len != true_target_len:
            x_src = np.arange(1, pred_target_len + 1, dtype=np.float32)
            x_dst = np.arange(1, true_target_len + 1, dtype=np.float32)
            curve = monotone_curve(np.clip(safe_interp(x_src, curve, x_dst), SOH_MIN, 100.0))
        else:
            curve = monotone_curve(np.clip(curve, SOH_MIN, 100.0))

        curve_eol = int(calculate_eol_cycle(curve, EOL_THRESHOLD))
        eol_pred = fuse_eol_prediction(
            curve_eol=curve_eol,
            aux_prob=aux_prob,
            aux_eol_pred=aux_eol_pred,
            group_id=pred_group,
            max_len=true_target_len,
        )
        pred_curve, pred_curve_plot, eol_pred = apply_eol_terminal(curve, eol_pred)
        eol_true = int(calculate_eol_cycle_eval(true_curve, EOL_THRESHOLD, EOL_EVAL_TOL))
        eol_true_strict = int(calculate_eol_cycle(true_curve, EOL_THRESHOLD))

        rmse, _mae, _corr = eval_rmse_mae_corr(true_curve, pred_curve)
        status = f"RMSE={rmse:.3f}, trueG={true_group}, predG={pred_group}"
        if eol_true > 0 and eol_pred > 0:
            status += f", EOLerr={abs(eol_true - eol_pred)}"
        print(f"[{i:03d}/{len(package['test_indices']):03d}] {s['battery_id']:<12} | {status}")

        item = {
            "battery_id": s["battery_id"],
            "true_group": true_group,
            "pred_group": pred_group,
            "group_probs": np.eye(3, dtype=np.float32)[pred_group],
            "true_curve": true_curve.astype(np.float32),
            "aux_eol_prob": float(aux_prob),
            "aux_eol_cycle_pred_group_len": int(aux_eol_pred),
            "knn_eol_pred": -1,
            "knn_eol_conf": 0.0,
            "eol_true_strict": int(eol_true_strict),
            "eol_true": int(eol_true),
            "pred_target_length": int(pred_target_len),
            "true_target_length": int(true_target_len),
            "rep_idx": 0,
            "run_curves_true_len": [pred_curve.astype(np.float32)],
            "pred_curve_plot_n1": pred_curve_plot.astype(np.float32),
            "pred_curve_plot_n10": pred_curve_plot.astype(np.float32),
            "pred_curve": pred_curve.astype(np.float32),
            "pred_curve_plot": pred_curve_plot.astype(np.float32),
            "eol_pred": int(eol_pred),
        }
        results.append(item)

    metrics = calculate_metrics(results)
    metrics["group_acc"] = float(group_acc_hits / max(len(package["test_indices"]), 1))
    print(f"\nPaper K-means group classification accuracy on test: {metrics['group_acc']:.4f}")
    print_metrics(metrics, tag=f"baseline={model}")

    out_path = os.path.join(OUTPUT2_DIR, f"baseline_{model}_test_predictions_seed{seed}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"results": results, "metrics": metrics}, f)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test deterministic baseline models for Code2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "transformer"])
    args = parser.parse_args()
    main(seed=args.seed, model=args.model)
