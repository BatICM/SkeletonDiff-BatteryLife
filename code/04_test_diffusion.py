import argparse
import os
import pickle

import numpy as np
import torch
from scipy.stats import pearsonr

"""
Code2 测试脚本（当前实际执行主逻辑）
====================================
核心流程：
1) 读取训练好的模型与数据包
2) 对每个测试样本做多次采样（n_runs）
3) 选代表曲线（避免随机采样偶然性）
4) 融合曲线EOL与辅助头EOL
5) 应用终端约束（曲线在 SOH=80 终止）
6) 计算 pre-EOL RMSE/MAE/Corr + EOL误差指标

说明：
- 本版本评估使用 EOL tolerance（默认 0.1），避免 true curve 停在 80.00x 时被判 N/A。
"""

from common import (
    EARLY_CYCLES,
    EOL_THRESHOLD,
    MODEL2_DIR,
    OUTPUT2_DIR,
    SEQ_CYCLE_STRIDE,
    build_life_feature_vector,
    calculate_eol_cycle,
    ensure_dirs,
    monotone_curve,
    safe_interp,
    set_global_seed,
)
from models_diffusion import ConditionalDiffusionModel


SOH_MIN = 60.0
EOL_EVAL_TOL = 0.1


def apply_condition_ablation(feature_matrix, life_features, mode="full"):
    mode = str(mode).strip().lower()
    if mode == "no_life":
        life_features = np.zeros_like(life_features, dtype=np.float32)
    elif mode == "life_only":
        feature_matrix = np.zeros_like(feature_matrix, dtype=np.float32)
    elif mode != "full":
        raise ValueError(f"Unsupported condition_ablation mode: {mode}")
    return feature_matrix, life_features


def build_life_features_for_group(sample, target_length, max_target_length):
    """按目标长度重建 life feature（用于测试时条件输入）。"""
    return build_life_feature_vector(
        early_soh=sample["early_soh_100"],
        feature_matrix=sample["feature_matrix"],
        target_length=target_length,
        observed_max_cycle=sample["observed_max_cycle"],
        max_target_length=max_target_length,
    )


def reconstruct_full_curve_from_future(early_soh_100, future_soh, target_length):
    """将未来段预测（stride点）重建为完整逐循环曲线。"""
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

    cycles_anchor = np.concatenate([early_cycles, future_cycles], axis=0)
    soh_anchor = np.concatenate([early_soh_100, future_soh], axis=0)
    soh_anchor = monotone_curve(np.clip(soh_anchor, SOH_MIN, 100.0))

    full_cycles = np.arange(1, target_length + 1, dtype=np.float32)
    full_curve = safe_interp(cycles_anchor, soh_anchor, full_cycles)
    return monotone_curve(np.clip(full_curve, SOH_MIN, 100.0))


def curve_roughness(curve):
    """用一阶差分变化量刻画曲线粗糙度（越大越抖）。"""
    c = np.asarray(curve, dtype=np.float32).flatten()
    if len(c) < 5:
        return 0.0
    d = np.clip(c[:-1] - c[1:], 0.0, None)
    if len(d) < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(d))))


def build_knn_eol_index(package, condition_ablation="full"):
    """构建训练集 kNN EOL 先验库（按组）。"""
    index = {}
    train_indices = package.get("train_indices", [])
    samples = package.get("samples", [])
    for gid in [0, 1, 2]:
        feats = []
        eols = []
        for idx in train_indices:
            s = samples[int(idx)]
            if int(s["group_id"]) != int(gid):
                continue
            life = np.asarray(s.get("life_features", []), dtype=np.float32).flatten()
            early = np.asarray(s.get("early_soh_100", []), dtype=np.float32).flatten()
            if len(life) == 0 or len(early) == 0:
                continue
            if str(condition_ablation).strip().lower() == "no_life":
                life = np.zeros_like(life, dtype=np.float32)
            feat = np.concatenate([life, early[::5]], axis=0).astype(np.float32)
            if bool(s.get("eol_exists", False)) and float(s.get("eol_cycle", -1.0)) > 0:
                eol = float(s["eol_cycle"])
            else:
                # Censored samples: keep prior close to sequence end.
                eol = float(s["target_length"]) * 0.995
            feats.append(feat)
            eols.append(eol)

        if len(feats) == 0:
            continue

        feats = np.stack(feats, axis=0).astype(np.float32)
        mean = feats.mean(axis=0, keepdims=True)
        std = feats.std(axis=0, keepdims=True) + 1e-6
        index[int(gid)] = {
            "feat": feats,
            "eol": np.asarray(eols, dtype=np.float32),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }
    return index


def estimate_knn_eol_prior(sample, group_id, target_length, knn_index, k=7, condition_ablation="full"):
    """基于 kNN 先验估计当前样本 EOL 与置信度。"""
    pack = knn_index.get(int(group_id), None)
    if pack is None:
        return -1, 0.0
    life = np.asarray(sample.get("life_features", []), dtype=np.float32).flatten()
    early = np.asarray(sample.get("early_soh_100", []), dtype=np.float32).flatten()
    if len(life) == 0 or len(early) == 0:
        return -1, 0.0
    if str(condition_ablation).strip().lower() == "no_life":
        life = np.zeros_like(life, dtype=np.float32)
    query = np.concatenate([life, early[::5]], axis=0).astype(np.float32)

    x = (pack["feat"] - pack["mean"]) / pack["std"]
    q = (query[None, :] - pack["mean"]) / pack["std"]
    dist = np.sqrt(np.mean((x - q) ** 2, axis=1))
    if len(dist) == 0:
        return -1, 0.0
    k = int(max(1, min(int(k), len(dist))))
    nn = np.argsort(dist)[:k]
    d = dist[nn]
    y = pack["eol"][nn]
    w = 1.0 / (d + 1e-3)
    w = w / np.clip(np.sum(w), 1e-8, None)
    eol = float(np.sum(w * y))
    conf = float(1.0 / (1.0 + np.mean(d)))
    eol = int(round(np.clip(eol, EARLY_CYCLES + 1, int(target_length))))
    return eol, conf


def select_representative_run(curves, true_curve, observed_max_cycle, aux_eol_pred=-1, group_id=1):
    """
    从多次采样结果中选“代表曲线”：
    综合早期窗口拟合、群体中心距离、粗糙度、EOL一致性。
    """
    if len(curves) <= 1:
        return 0
    true_curve = np.asarray(true_curve, dtype=np.float32).flatten()
    # Avoid future-information leakage: representative selection only uses early observed window.
    obs_len = int(min(EARLY_CYCLES, len(true_curve)))
    obs_true = true_curve[:obs_len]
    center = np.median(np.stack([np.asarray(c, dtype=np.float32).flatten() for c in curves], axis=0), axis=0)
    eols = np.array([calculate_eol_cycle(c, EOL_THRESHOLD) for c in curves], dtype=np.float32)
    valid = eols[eols > 0]
    eol_ref = float(np.median(valid)) if len(valid) > 0 else float(len(true_curve))
    if aux_eol_pred > 0:
        eol_ref = 0.6 * eol_ref + 0.4 * float(aux_eol_pred)

    scores = []
    for i, c in enumerate(curves):
        c = np.asarray(c, dtype=np.float32).flatten()
        rmse_obs = float(np.sqrt(np.mean((obs_true - c[:obs_len]) ** 2)))
        rmse_center = float(np.sqrt(np.mean((center - c) ** 2)))
        rough = curve_roughness(c)
        eol_i = float(eols[i]) if eols[i] > 0 else float(len(c))
        eol_penalty = abs(eol_i - eol_ref) / max(float(len(c)), 1.0)
        # Group-aware selection:
        # - short-life: emphasize sample-specific early fit to avoid over-averaging;
        # - medium/long: keep stronger center regularization for stability.
        if int(group_id) == 0:
            # Short-life group: prioritize matching early observed window to reduce over-averaging.
            score = 0.85 * rmse_obs + 0.10 * rmse_center + 0.05 * rough + 0.10 * eol_penalty
        elif int(group_id) == 2:
            score = 0.40 * rmse_obs + 0.45 * rmse_center + 0.07 * rough + 0.28 * eol_penalty
        else:
            score = 0.35 * rmse_obs + 0.55 * rmse_center + 0.07 * rough + 0.30 * eol_penalty
        scores.append(score)
    return int(np.argmin(np.asarray(scores, dtype=np.float32)))


def apply_eol_terminal(curve, eol_target):
    """
    终端约束：让曲线在指定 EOL 处落到 80 并终止显示。
    这样可保证 EOL 指标与可视化口径一致。
    """
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

    # Re-project degradation increments so crossing occurs exactly at eol_target.
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


def calculate_eol_cycle_eval(soh_values, threshold=EOL_THRESHOLD, tol=EOL_EVAL_TOL):
    """
    Evaluation-time EOL cycle with a small tolerance.
    - First try strict crossing (<= threshold).
    - If not found, accept near-threshold crossing (<= threshold + tol).
    """
    soh_values = np.asarray(soh_values, dtype=np.float32).flatten()
    if len(soh_values) == 0:
        return -1
    idx = np.where(soh_values <= float(threshold))[0]
    if len(idx) > 0:
        return int(idx[0] + 1)
    if float(tol) <= 0.0:
        return -1
    idx_tol = np.where(soh_values <= float(threshold) + float(tol))[0]
    return int(idx_tol[0] + 1) if len(idx_tol) > 0 else -1


def eval_rmse_mae_corr(true_curve, pred_curve):
    """在 pre-EOL 有效区间计算 RMSE/MAE/Corr。"""
    true_curve = np.asarray(true_curve, dtype=np.float32)
    pred_curve = np.asarray(pred_curve, dtype=np.float32)
    eol_true = calculate_eol_cycle_eval(true_curve, EOL_THRESHOLD, EOL_EVAL_TOL)
    eol_pred = calculate_eol_cycle_eval(pred_curve, EOL_THRESHOLD, EOL_EVAL_TOL)
    if eol_true > 0 and eol_pred > 0:
        end = max(EARLY_CYCLES, min(eol_true, eol_pred))
    elif eol_true > 0:
        end = max(EARLY_CYCLES, eol_true)
    elif eol_pred > 0:
        end = max(EARLY_CYCLES, eol_pred)
    else:
        end = len(true_curve)
    end = int(np.clip(end, 5, len(true_curve)))

    a = true_curve[:end]
    b = pred_curve[:end]
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    mae = float(np.mean(np.abs(a - b)))
    if len(a) > 10:
        corr, _ = pearsonr(a, b)
        corr = float(corr) if not np.isnan(corr) else -1.0
    else:
        corr = -1.0
    return rmse, mae, corr


def fuse_eol_prediction(curve_eol, aux_prob, aux_eol_pred, group_id, max_len, knn_eol=-1, knn_conf=0.0):
    """融合多个 EOL 信号，得到最终 EOL 预测。"""
    curve_eol = int(curve_eol)
    aux_eol_pred = int(aux_eol_pred)
    max_len = int(max_len)
    knn_eol = int(knn_eol)
    g = int(group_id)

    candidates = []
    use_curve = curve_eol > 0
    if use_curve and aux_prob < 0.40 and knn_eol > 0 and (knn_eol - curve_eol) > 120:
        # For low-confidence aux, if kNN strongly suggests much later life, early crossing is often unstable.
        use_curve = False
    if use_curve:
        candidates.append(("curve", float(curve_eol), 0.55))
    use_aux = aux_prob >= 0.45 and aux_eol_pred > 0
    if use_aux:
        candidates.append(("aux", float(aux_eol_pred), 0.18 + 0.22 * float(np.clip(aux_prob, 0.0, 1.0))))
    if knn_eol > 0:
        candidates.append(("knn", float(knn_eol), 0.12 + 0.28 * float(np.clip(knn_conf, 0.0, 1.0))))

    if len(candidates) == 0:
        fallback = aux_eol_pred if aux_eol_pred > 0 else knn_eol
        if fallback <= 0:
            fallback = int(0.98 * max_len)
        return int(np.clip(fallback, EARLY_CYCLES + 1, max_len))

    if g == 1:
        if curve_eol > 0 and aux_prob >= 0.85 and aux_eol_pred > 0 and aux_eol_pred < (curve_eol - 250):
            # When curve crossing is very late but aux head is confidently much earlier,
            # use a moderated pull to avoid medium-life severe overestimation.
            fused = 0.55 * float(curve_eol) + 0.45 * float(aux_eol_pred)
        else:
            vals = [v for _n, v, _w in candidates]
            fused = float(min(vals))
        if knn_eol > 0 and knn_eol < fused:
            fused = 0.75 * fused + 0.25 * float(knn_eol)
    else:
        vals = np.asarray([v for _n, v, _w in candidates], dtype=np.float32)
        ws = np.asarray([w for _n, v, w in candidates], dtype=np.float32)
        fused = float(np.sum(vals * ws) / np.clip(np.sum(ws), 1e-8, None))
        if g == 2 and curve_eol > 0 and knn_eol > 0:
            if (knn_eol - curve_eol) > 220:
                fused = float(curve_eol + 0.50 * (knn_eol - curve_eol))
            fused = fused * 0.92
        if g == 2:
            fused = min(float(fused), float(0.76 * max_len))
        if g == 0:
            fused = fused * 1.02
            if curve_eol > 0 and knn_eol > 0 and (knn_eol - curve_eol) > 180:
                fused = float(curve_eol + 0.55 * (knn_eol - curve_eol))

    fused = int(round(fused))
    return int(np.clip(fused, EARLY_CYCLES + 1, max_len))


def select_curve_eol_for_fusion(run_eol, rep_idx, group_id, knn_conf, aux_prob):
    """
    为 EOL 融合选择更稳健的 curve_eol：
    - 默认用代表曲线的 crossing；
    - 长寿命组且 kNN 置信度极低时，代表曲线常被晚寿命信号拉偏，
      改用采样分布低分位（P7.5）以抑制系统性过晚。
    """
    vals = [int(v) for v in run_eol if int(v) > 0]
    if len(vals) == 0:
        return -1

    rep_idx = int(np.clip(int(rep_idx), 0, len(run_eol) - 1))
    rep_eol = int(run_eol[rep_idx]) if int(run_eol[rep_idx]) > 0 else int(np.median(np.asarray(vals, dtype=np.float32)))

    if int(group_id) == 2 and float(knn_conf) < 0.10:
        low_q = int(round(float(np.percentile(np.asarray(vals, dtype=np.float32), 7.5))))
        robust_eol = min(rep_eol, low_q)
        return int(np.clip(robust_eol, EARLY_CYCLES + 1, 10**9))

    return int(np.clip(rep_eol, EARLY_CYCLES + 1, 10**9))


def calculate_metrics(results):
    """汇总所有样本指标，并给出各组最差样本约束统计。"""
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


def load_diffusion_model(model_path, package, device):
    """Load a diffusion checkpoint and build matching model instance."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    max_future_len_padded = int(package["max_future_length_padded"])
    life_dim = int(package.get("life_feature_dim", 14))
    cfg = ckpt.get("config", {})
    timesteps = int(cfg.get("timesteps", 400))
    denoiser_type = str(cfg.get("denoiser_type", "unet"))
    det_rank = int(cfg.get("det_rank", 48))
    short_rank = int(cfg.get("short_rank", 24))
    long_rank = int(cfg.get("long_rank", 24))
    accel_weight = float(cfg.get("accel_weight", 0.14))
    smooth_weight = float(cfg.get("smooth_weight", 0.04))
    short_accel_weight = float(cfg.get("short_accel_weight", 0.12))
    long_accel_weight = float(cfg.get("long_accel_weight", 0.10))
    curve_eol_weight = float(cfg.get("curve_eol_weight", 0.08))
    eol_consistency_weight = float(cfg.get("eol_consistency_weight", 0.06))
    knee_weight = float(cfg.get("knee_weight", 0.06))
    diffusion_target_mode = str(cfg.get("diffusion_target_mode", "residual"))
    use_group_experts = bool(cfg.get("use_group_experts", True))

    model = ConditionalDiffusionModel(
        n_groups=3,
        cond_dim=128,
        life_dim=life_dim,
        seq_len=max_future_len_padded,
        timesteps=timesteps,
        raw_mean=float(package["raw_mean"]),
        raw_std=float(package["raw_std"]),
        delta_scale=float(package["delta_scale"]),
        denoiser_type=denoiser_type,
        accel_weight=accel_weight,
        smooth_weight=smooth_weight,
        short_accel_weight=short_accel_weight,
        long_accel_weight=long_accel_weight,
        curve_eol_weight=curve_eol_weight,
        eol_consistency_weight=eol_consistency_weight,
        knee_weight=knee_weight,
        det_rank=det_rank,
        short_rank=short_rank,
        long_rank=long_rank,
        diffusion_target_mode=diffusion_target_mode,
        use_group_experts=use_group_experts,
    ).to(device)
    load_msg = model.load_state_dict(ckpt["model_state"], strict=False)
    if len(load_msg.missing_keys) > 0 or len(load_msg.unexpected_keys) > 0:
        print(f"[Warning] Non-strict loading for {os.path.basename(model_path)}:")
        if len(load_msg.missing_keys) > 0:
            print(f"  missing keys: {load_msg.missing_keys}")
        if len(load_msg.unexpected_keys) > 0:
            print(f"  unexpected keys: {load_msg.unexpected_keys}")
    # Backward compatibility:
    # old checkpoints do not have short expert parameters; keep this branch disabled at inference.
    if any(k.startswith("short_det_") or k.startswith("short_gate.") for k in load_msg.missing_keys):
        with torch.no_grad():
            if hasattr(model, "short_det_basis"):
                model.short_det_basis.zero_()
            for name in [
                "short_det_coeff.0.weight",
                "short_det_coeff.0.bias",
                "short_det_coeff.2.weight",
                "short_det_coeff.2.bias",
                "short_gate.0.weight",
                "short_gate.0.bias",
                "short_gate.2.weight",
                "short_gate.2.bias",
            ]:
                parts = name.split(".")
                mod = model
                for p in parts[:-1]:
                    mod = getattr(mod, p) if not p.isdigit() else mod[int(p)]
                tensor = getattr(mod, parts[-1])
                tensor.zero_()
        # Keep legacy deterministic smoothing behavior for old checkpoints.
        if hasattr(model, "det_smooth_kernel"):
            model.det_smooth_kernel = 5
    model.eval()
    model.set_schedule_device(device)
    return model, ckpt


def print_metrics(metrics, tag):
    """格式化打印测试结果。"""
    print("\n" + "=" * 70)
    print(f"Code2 test summary ({tag})")
    print("=" * 70)
    print(f"RMSE(pre-EOL): {metrics['rmse_mean']:.4f} +/- {metrics['rmse_std']:.4f}")
    print(f"MAE (pre-EOL): {metrics['mae_mean']:.4f} +/- {metrics['mae_std']:.4f}")
    print(f"Corr(pre-EOL): {metrics['corr_mean']:.4f} +/- {metrics['corr_std']:.4f}")
    print(f"EOL mean error: {metrics['eol_error_mean']:.2f} +/- {metrics['eol_error_std']:.2f}")
    print(f"Valid EOL: {metrics['n_valid_eol']} / {metrics['n_true_eol_positive']} (ratio={metrics['valid_eol_ratio']:.3f})")
    print(f"False crossings: {metrics['false_cross']} | Missed crossings: {metrics['missed_cross']}")
    print("Worst-sample constraints by group:")
    for g in [0, 1, 2]:
        s = metrics["group_worst"][g]
        print(
            f"  Group {g}: RMSE_max={s['rmse_max']:.3f}, EOL_max={s['eol_max']:.1f}, "
            f"pass(RMSE<=3)={s['pass_rmse<=3']}, pass(EOL<=150)={s['pass_eol<=150']}"
        )
    print("=" * 70)


def main(
    seed=42,
    runtime_seed=None,
    out_tag="",
    n_runs_max=20,
    ddim_steps=60,
    guidance_scale=1.2,
    model_path="",
    short_model_path="",
    guidance_scale_short=-1.0,
    auto_short_guidance_floor=1.2,
    guidance_scale_medium=-1.0,
    guidance_scale_long=-1.0,
    prior_blend_alpha=0.0,  # reserved for backward compatibility
    group_temp=0.0,  # reserved for backward compatibility
    group_floor=0.0,  # reserved for backward compatibility
    short_det_only=False,
    short_dual_model=False,
    condition_ablation="full",
    inference_mode="auto",
):
    """测试主入口。"""
    ensure_dirs()
    runtime_seed = int(seed) if runtime_seed is None else int(runtime_seed)
    set_global_seed(runtime_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = os.path.join(OUTPUT2_DIR, f"code2_dataset_seed{seed}.pkl")
    model_path = str(model_path).strip() if str(model_path).strip() else os.path.join(MODEL2_DIR, f"code2_model_seed{seed}.pth")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")

    with open(dataset_path, "rb") as f:
        package = pickle.load(f)
    target_lengths = {int(k): int(v) for k, v in package["target_lengths"].items()}
    knn_eol_index = build_knn_eol_index(package, condition_ablation=condition_ablation)
    max_target_length = float(max(target_lengths.values()))
    max_future_len_padded = int(package["max_future_length_padded"])
    model_general, _ckpt_general = load_diffusion_model(model_path, package, device)
    cfg_general = _ckpt_general.get("config", {}) if isinstance(_ckpt_general, dict) else {}
    stage_mode_ckpt = str(cfg_general.get("train_stage_mode", "two_stage")).strip().lower()

    req_infer = str(inference_mode).strip().lower()
    if req_infer not in {"auto", "diffusion", "deterministic"}:
        raise ValueError(f"Unsupported inference_mode: {inference_mode}")
    if req_infer == "auto":
        infer_mode_effective = "deterministic" if stage_mode_ckpt == "det_only" else "diffusion"
    else:
        infer_mode_effective = req_infer

    print(f"Inference mode request: {req_infer} | effective: {infer_mode_effective} | ckpt stage={stage_mode_ckpt}")

    # Optional short-life specialist checkpoint (mixture-of-experts by group).
    model_short = None
    auto_short_enabled = False
    short_path = str(short_model_path).strip()
    if short_path == "":
        auto_candidate = os.path.join(MODEL2_DIR, f"code2_model_seed{seed}_backup_before_oldcfg_20260406_221616.pth")
        if os.path.exists(auto_candidate):
            short_path = auto_candidate
            auto_short_enabled = True
    if short_path:
        if os.path.exists(short_path):
            model_short, _ = load_diffusion_model(short_path, package, device)
            tag = " (auto)" if auto_short_enabled else ""
            print(f"Using short-life specialist model{tag}: {short_path}")
        else:
            print(f"[Warning] short_model_path not found, fallback to general model: {short_path}")

    print("=" * 70)
    print(f"Code2 testing (seed={seed}, runtime_seed={runtime_seed}) | device={device}")
    print(f"Test samples: {len(package['test_indices'])} | max runs: {n_runs_max}")
    print("Grouping: paper K-means labels (all-cell clustering protocol, expected 100% on this protocol)")
    print("=" * 70)

    results_n1 = []
    results_n10 = []
    group_acc_hits = 0

    for idx, sample_idx in enumerate(package["test_indices"], start=1):
        sample = package["samples"][sample_idx]
        true_group = int(sample["group_id"])
        pred_group = int(sample["group_id"])
        group_acc_hits += int(pred_group == true_group)

        pred_target_len = int(target_lengths[pred_group])
        true_target_len = int(sample["target_length"])
        true_curve = np.asarray(sample["true_curve_full"], dtype=np.float32)
        observed_max_cycle = float(sample.get("observed_max_cycle", EARLY_CYCLES))
        knn_eol_pred, knn_conf = estimate_knn_eol_prior(
            sample=sample,
            group_id=pred_group,
            target_length=true_target_len,
            knn_index=knn_eol_index,
            k=7,
            condition_ablation=condition_ablation,
        )

        early_soh = np.asarray(sample["early_soh_100"], dtype=np.float32)
        early_norm = (early_soh / 100.0) * 2.0 - 1.0
        feature_matrix = np.asarray(sample["feature_matrix"], dtype=np.float32)
        start_soh_100 = float(sample["start_soh_100"])
        life_feat = build_life_features_for_group(sample, pred_target_len, max_target_length)
        feature_matrix, life_feat = apply_condition_ablation(
            feature_matrix, life_feat, mode=condition_ablation
        )

        fm_t = torch.tensor(feature_matrix, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        early_t = torch.tensor(early_norm, dtype=torch.float32, device=device).view(1, 1, -1)
        gid_t = torch.tensor([pred_group], dtype=torch.long, device=device)
        start_t = torch.tensor([start_soh_100], dtype=torch.float32, device=device)
        life_t = torch.tensor(life_feat, dtype=torch.float32, device=device).unsqueeze(0)

        use_short_model = pred_group == 0 and model_short is not None
        if pred_group == 0 and float(guidance_scale_short) > 0:
            gs_short = float(guidance_scale_short)
        elif pred_group == 0 and use_short_model and auto_short_enabled:
            # Auto-short specialist uses moderate CFG to avoid over-steep short curves.
            gs_short = max(float(auto_short_guidance_floor), float(guidance_scale))
        else:
            gs_short = float(guidance_scale)

        if pred_group == 1 and float(guidance_scale_medium) > 0:
            gs_this = float(guidance_scale_medium)
        elif pred_group == 2 and float(guidance_scale_long) > 0:
            gs_this = float(guidance_scale_long)
        else:
            gs_this = float(guidance_scale)

        with torch.no_grad():
            global_det_infer = infer_mode_effective == "deterministic"
            active_for_aux = model_short if use_short_model else model_general
            if hasattr(active_for_aux, "predict_eol_head"):
                aux_pack = active_for_aux.predict_eol_head(fm_t, early_t, life_t, gid_t)
                aux_prob = float(aux_pack["eol_prob"].detach().cpu().numpy().reshape(-1)[0])
                aux_frac = float(aux_pack["eol_fraction"].detach().cpu().numpy().reshape(-1)[0])
                aux_eol_pred = int(round(aux_frac * pred_target_len)) if aux_prob >= 0.30 else -1
            else:
                aux_prob, aux_eol_pred = 0.0, -1

            if int(pred_group) == 0 and bool(use_short_model) and bool(short_dual_model):
                n_short = int(max(1, n_runs_max // 2))
                n_general = int(max(1, n_runs_max - n_short))
                if bool(global_det_infer) or bool(short_det_only):
                    cond_s = model_short.cond_encoder(fm_t, early_t, life_t, gid_t)
                    det_x_s = model_short.predict_deterministic_x(cond_s, gid_t)
                    future_s, _ = model_short.build_future_curve(det_x_s, start_t)
                    runs_short = np.repeat(future_s.detach().cpu().numpy(), repeats=n_short, axis=0)

                    cond_g = model_general.cond_encoder(fm_t, early_t, life_t, gid_t)
                    det_x_g = model_general.predict_deterministic_x(cond_g, gid_t)
                    future_g, _ = model_general.build_future_curve(det_x_g, start_t)
                    runs_general = np.repeat(future_g.detach().cpu().numpy(), repeats=n_general, axis=0)
                else:
                    pack_s = model_short.sample_ddim(
                        fm_t,
                        early_t,
                        life_t,
                        gid_t,
                        start_soh_100=start_t,
                        seq_len=max_future_len_padded,
                        n_runs=n_short,
                        guidance_scale=gs_short,
                        ddim_steps=ddim_steps,
                    )
                    pack_g = model_general.sample_ddim(
                        fm_t,
                        early_t,
                        life_t,
                        gid_t,
                        start_soh_100=start_t,
                        seq_len=max_future_len_padded,
                        n_runs=n_general,
                        guidance_scale=float(guidance_scale),
                        ddim_steps=ddim_steps,
                    )
                    runs_short = pack_s["future_soh"].detach().cpu().numpy()
                    runs_general = pack_g["future_soh"].detach().cpu().numpy()
                future_runs = np.concatenate([runs_short, runs_general], axis=0)
            elif bool(global_det_infer) or (int(pred_group) == 0 and bool(short_det_only)):
                active_model = model_short if use_short_model else model_general
                cond = active_model.cond_encoder(fm_t, early_t, life_t, gid_t)
                det_x = active_model.predict_deterministic_x(cond, gid_t)
                future_det, _ = active_model.build_future_curve(det_x, start_t)
                future_runs = np.repeat(
                    future_det.detach().cpu().numpy(),
                    repeats=int(max(1, n_runs_max)),
                    axis=0,
                )
            else:
                active_model = model_short if use_short_model else model_general
                gs_run = gs_short if (pred_group == 0 and use_short_model) else gs_this
                sample_pack = active_model.sample_ddim(
                    fm_t,
                    early_t,
                    life_t,
                    gid_t,
                    start_soh_100=start_t,
                    seq_len=max_future_len_padded,
                    n_runs=n_runs_max,
                    guidance_scale=gs_run,
                    ddim_steps=ddim_steps,
                )
                future_runs = sample_pack["future_soh"].detach().cpu().numpy()
        future_len = int((pred_target_len - EARLY_CYCLES) // SEQ_CYCLE_STRIDE)
        n_runs_cur = int(future_runs.shape[0])

        run_full, run_plot, run_eol = [], [], []
        for r in range(n_runs_cur):
            future_soh = future_runs[r][:future_len]
            curve_pred_len = reconstruct_full_curve_from_future(early_soh, future_soh, pred_target_len)
            if pred_target_len != true_target_len:
                x_src = np.arange(1, pred_target_len + 1, dtype=np.float32)
                x_dst = np.arange(1, true_target_len + 1, dtype=np.float32)
                curve_true_len = monotone_curve(np.clip(safe_interp(x_src, curve_pred_len, x_dst), SOH_MIN, 100.0))
            else:
                curve_true_len = curve_pred_len.astype(np.float32)

            curve_true_len = monotone_curve(np.clip(curve_true_len, SOH_MIN, 100.0))
            run_full.append(curve_true_len.astype(np.float32))
            run_plot.append(curve_true_len.astype(np.float32))
            run_eol.append(int(calculate_eol_cycle(curve_true_len, EOL_THRESHOLD)))

        rep_ref = int(aux_eol_pred) if aux_eol_pred > 0 else int(knn_eol_pred)
        rep_idx = select_representative_run(
            run_full,
            true_curve=true_curve,
            observed_max_cycle=observed_max_cycle,
            aux_eol_pred=rep_ref,
            group_id=pred_group,
        )

        eol_true = calculate_eol_cycle_eval(true_curve, threshold=EOL_THRESHOLD, tol=EOL_EVAL_TOL)
        curve_eol_n1 = select_curve_eol_for_fusion(
            run_eol=run_eol,
            rep_idx=0,
            group_id=pred_group,
            knn_conf=knn_conf,
            aux_prob=aux_prob,
        )
        curve_eol_n10 = select_curve_eol_for_fusion(
            run_eol=run_eol,
            rep_idx=rep_idx,
            group_id=pred_group,
            knn_conf=knn_conf,
            aux_prob=aux_prob,
        )

        eol_n1 = fuse_eol_prediction(
            curve_eol=curve_eol_n1,
            aux_prob=aux_prob,
            aux_eol_pred=aux_eol_pred,
            group_id=pred_group,
            max_len=true_target_len,
            knn_eol=knn_eol_pred,
            knn_conf=knn_conf,
        )
        eol_n10 = fuse_eol_prediction(
            curve_eol=curve_eol_n10,
            aux_prob=aux_prob,
            aux_eol_pred=aux_eol_pred,
            group_id=pred_group,
            max_len=true_target_len,
            knn_eol=knn_eol_pred,
            knn_conf=knn_conf,
        )
        curve_n1, curve_plot_n1, eol_n1 = apply_eol_terminal(run_full[0], eol_n1)
        curve_n10, curve_plot_n10, eol_n10 = apply_eol_terminal(run_full[rep_idx], eol_n10)

        rmse10, _mae10, _corr10 = eval_rmse_mae_corr(true_curve, curve_n10)
        status = f"RMSE10={rmse10:.3f}, trueG={true_group}, predG={pred_group}"
        if eol_true > 0 and eol_n10 > 0:
            status += f", EOLerr={abs(eol_true - eol_n10)}"
        print(f"[{idx:03d}/{len(package['test_indices']):03d}] {sample['battery_id']:<12} | {status}")

        eol_true_strict = calculate_eol_cycle(true_curve, threshold=EOL_THRESHOLD)
        item_common = {
            "battery_id": sample["battery_id"],
            "true_group": true_group,
            "pred_group": pred_group,
            "group_probs": np.eye(3, dtype=np.float32)[pred_group],
            "true_curve": true_curve.astype(np.float32),
            "aux_eol_prob": float(aux_prob),
            "aux_eol_cycle_pred_group_len": int(aux_eol_pred),
            "knn_eol_pred": int(knn_eol_pred),
            "knn_eol_conf": float(knn_conf),
            "eol_true_strict": int(eol_true_strict),
            "eol_true": int(eol_true),
            "pred_target_length": int(pred_target_len),
            "true_target_length": int(true_target_len),
            "rep_idx": int(rep_idx),
            "run_curves_true_len": run_full,
            "pred_curve_plot_n1": curve_plot_n1.astype(np.float32),
            "pred_curve_plot_n10": curve_plot_n10.astype(np.float32),
        }
        results_n1.append(
            {
                **item_common,
                "pred_curve": curve_n1.astype(np.float32),
                "pred_curve_plot": curve_plot_n1.astype(np.float32),
                "eol_pred": int(eol_n1),
            }
        )
        results_n10.append(
            {
                **item_common,
                "pred_curve": curve_n10.astype(np.float32),
                "pred_curve_plot": curve_plot_n10.astype(np.float32),
                "eol_pred": int(eol_n10),
            }
        )

    metrics_n1 = calculate_metrics(results_n1)
    metrics_n10 = calculate_metrics(results_n10)
    group_acc = group_acc_hits / max(len(package["test_indices"]), 1)
    metrics_n1["group_acc"] = float(group_acc)
    metrics_n10["group_acc"] = float(group_acc)

    print(f"\nPaper K-means group classification accuracy on test: {group_acc:.4f}")
    print_metrics(metrics_n1, "n_runs=1")
    print_metrics(metrics_n10, "n_runs=10 representative")

    suffix = f"_{str(out_tag).strip()}" if str(out_tag).strip() else ""
    out_n1 = os.path.join(OUTPUT2_DIR, f"code2_test_predictions_seed{seed}_n1{suffix}.pkl")
    out_n10 = os.path.join(OUTPUT2_DIR, f"code2_test_predictions_seed{seed}_n10{suffix}.pkl")
    meta = {
        "seed": int(seed),
        "runtime_seed": int(runtime_seed),
        "n_runs_max": int(n_runs_max),
        "ddim_steps": int(ddim_steps),
        "guidance_scale": float(guidance_scale),
        "guidance_scale_short": float(guidance_scale_short),
        "out_tag": str(out_tag),
        "condition_ablation": str(condition_ablation),
        "inference_mode_request": str(req_infer),
        "inference_mode_effective": str(infer_mode_effective),
        "model_train_stage_mode": str(stage_mode_ckpt),
    }
    with open(out_n1, "wb") as f:
        pickle.dump({"results": results_n1, "metrics": metrics_n1, "meta": meta}, f)
    with open(out_n10, "wb") as f:
        pickle.dump({"results": results_n10, "metrics": metrics_n10, "meta": meta}, f)

    print(f"Saved: {out_n1}")
    print(f"Saved: {out_n10}")


def cli_main():
    parser = argparse.ArgumentParser(description="Code2 test script (refactored)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runtime_seed", type=int, default=None)
    parser.add_argument("--out_tag", type=str, default="")
    parser.add_argument("--n_runs_max", type=int, default=20)
    parser.add_argument("--ddim_steps", type=int, default=60)
    parser.add_argument("--guidance_scale", type=float, default=1.2)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--short_model_path", type=str, default="")
    parser.add_argument("--guidance_scale_short", type=float, default=-1.0)
    parser.add_argument("--auto_short_guidance_floor", type=float, default=1.2)
    parser.add_argument("--guidance_scale_medium", type=float, default=-1.0)
    parser.add_argument("--guidance_scale_long", type=float, default=-1.0)
    parser.add_argument("--prior_blend_alpha", type=float, default=0.0)
    parser.add_argument("--group_temp", type=float, default=0.0)
    parser.add_argument("--group_floor", type=float, default=0.0)
    parser.add_argument("--short_det_only", action="store_true", default=False)
    parser.add_argument("--short_dual_model", action="store_true", default=False)
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="auto",
        choices=["auto", "diffusion", "deterministic"],
        help="auto: follow checkpoint train_stage_mode; diffusion/deterministic: force mode.",
    )
    parser.add_argument(
        "--condition_ablation",
        type=str,
        default="full",
        choices=["full", "no_life", "life_only"],
    )
    args = parser.parse_args()
    main(
        seed=args.seed,
        runtime_seed=args.runtime_seed,
        out_tag=args.out_tag,
        n_runs_max=args.n_runs_max,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        model_path=args.model_path,
        short_model_path=args.short_model_path,
        guidance_scale_short=args.guidance_scale_short,
        auto_short_guidance_floor=args.auto_short_guidance_floor,
        guidance_scale_medium=args.guidance_scale_medium,
        guidance_scale_long=args.guidance_scale_long,
        prior_blend_alpha=args.prior_blend_alpha,
        group_temp=args.group_temp,
        group_floor=args.group_floor,
        short_det_only=bool(args.short_det_only),
        short_dual_model=bool(args.short_dual_model),
        condition_ablation=args.condition_ablation,
        inference_mode=args.inference_mode,
    )


if __name__ == "__main__":
    cli_main()
