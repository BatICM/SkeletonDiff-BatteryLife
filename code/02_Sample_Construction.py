import argparse
import datetime
import os
import pickle
import runpy
import sys

import numpy as np

"""
Code2 数据构建脚本
==================
输入：
- Output/grouped_data.pkl（已按寿命组整理的原始数据）

输出：
- Output/code2_dataset_seed{seed}.pkl（训练/验证/测试统一数据包）

核心思想：
1) 固定早期窗口（前100循环）作为条件输入
2) 把未来段曲线转换为“非负退化增量 future_delta”
3) 生成 observed/valid 掩码，区分真实观测与插值外推部分
4) 生成 life features 与 EOL 标签
5) 按组做可复现分层划分（train/val/test）
"""

from common import (
    EARLY_CYCLES,
    OUTPUT_DIR,
    OUTPUT2_DIR,
    PAD_MULTIPLE,
    SEQ_CYCLE_STRIDE,
    build_life_feature_vector,
    calculate_eol_cycle,
    ensure_dirs,
    get_future_length,
    get_group_target_lengths,
    inv_softplus,
    monotone_curve,
    round_up_multiple,
    safe_interp,
    set_global_seed,
)


# Robust console output on Windows terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def ensure_grouped_data(seed, rebuild_grouped=False):
    """
    Ensure Output/grouped_data.pkl exists.
    If missing (or forced), rebuild from raw Data/batch*.pkl by invoking legacy preprocessing.
    """
    grouped_path = os.path.join(OUTPUT_DIR, "grouped_data.pkl")
    if os.path.exists(grouped_path) and not rebuild_grouped:
        mtime = os.path.getmtime(grouped_path)
        mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[OK] Using existing grouped_data: {grouped_path}")
        print(f"     Last modified: {mtime_str}")
        return grouped_path

    legacy_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "00_data_processing.py")
    if not os.path.exists(legacy_script):
        raise FileNotFoundError(
            "Missing grouped_data.pkl and preprocessing script 00_data_processing.py. "
            "Cannot build grouped data from raw batches."
        )

    print("=" * 70)
    print("grouped_data.pkl not found (or rebuild requested).")
    print("Running legacy preprocessing from raw Data/batch1-3.pkl ...")
    print(f"Legacy script: {legacy_script}")
    print("=" * 70)
    runpy.run_path(legacy_script, run_name="__main__")

    if not os.path.exists(grouped_path):
        raise RuntimeError(
            f"Legacy preprocessing finished but grouped_data.pkl was not created: {grouped_path}"
        )
    print(f"[OK] grouped_data prepared: {grouped_path}")
    return grouped_path


def split_indices(n, rng, train_ratio=0.7, val_ratio=0.15):
    """给定样本数，返回随机划分后的 train/val/test 下标。"""
    idx = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def extract_true_eol(cycles, soh, threshold=80.0):
    """
    从原始循环点中提取真值 EOL。
    返回 (是否存在EOL, EOL循环位置)。
    """
    cycles = np.asarray(cycles, dtype=np.float32).flatten()
    soh = np.asarray(soh, dtype=np.float32).flatten()
    if len(cycles) == 0:
        return False, -1.0
    idx = np.where(soh <= threshold)[0]
    if len(idx) == 0:
        return False, -1.0
    return True, float(cycles[idx[0]])


def build_samples(grouped_data):
    """
    把 grouped_data 转为统一训练样本列表。
    每个样本包含：
    - 早期 SOH（100点）
    - 未来退化增量序列
    - 掩码（observed/valid）
    - life_features
    - 全长真值曲线与EOL标签
    """
    target_lengths = get_group_target_lengths(grouped_data)
    max_target_length = max(target_lengths.values())
    samples = []
    dropped = 0

    for group_id, group in grouped_data.items():
        group_id = int(group_id)
        target_length = int(target_lengths[group_id])

        for item in group["data"]:
            if item.get("is_augmented", False):
                # Augmented long-life samples in the old pipeline often miss raw observed curves.
                continue

            cycles = np.asarray(item.get("observed_raw_cycles", []), dtype=np.float32).flatten()
            soh = np.asarray(item.get("observed_raw_soh", []), dtype=np.float32).flatten()
            valid = ~(np.isnan(cycles) | np.isnan(soh))
            cycles = cycles[valid]
            soh = soh[valid]
            if len(cycles) < EARLY_CYCLES + 20:
                dropped += 1
                continue

            soh = monotone_curve(np.clip(soh, 0.0, 100.0))
            # 先对原始 SOH 投影为单调不增，减少异常上升噪声。

            early_grid = np.arange(1, EARLY_CYCLES + 1, dtype=np.float32)
            early_soh = monotone_curve(safe_interp(cycles, soh, early_grid))
            start_soh_100 = float(early_soh[-1])

            future_grid = np.arange(
                EARLY_CYCLES + SEQ_CYCLE_STRIDE,
                target_length + 1,
                SEQ_CYCLE_STRIDE,
                dtype=np.float32,
            )
            if len(future_grid) == 0:
                dropped += 1
                continue

            future_soh = safe_interp(cycles, soh, future_grid)
            future_soh = np.minimum(future_soh, start_soh_100)
            future_soh = monotone_curve(np.clip(future_soh, 0.0, 100.0))
            # 将未来段转为“相邻差分增量”，用于后续模型输出建模。

            prev = np.concatenate([[start_soh_100], future_soh[:-1]])
            future_delta = np.clip(prev - future_soh, 0.0, None).astype(np.float32)

            observed_max_cycle = int(item.get("observed_max_cycle", int(cycles[-1])))
            observed_future_mask = (future_grid <= float(observed_max_cycle)).astype(np.float32)
            observed_future_len = int(observed_future_mask.sum())
            if observed_future_len < 5:
                dropped += 1
                continue

            full_grid = np.arange(1, target_length + 1, dtype=np.float32)
            full_true_curve = monotone_curve(np.clip(safe_interp(cycles, soh, full_grid), 0.0, 100.0))
            # eol_fraction：EOL 在目标长度中的相对位置（0~1）
            eol_exists, eol_cycle = extract_true_eol(cycles, soh, threshold=80.0)
            eol_fraction = float(np.clip(eol_cycle / max(target_length, 1), 0.0, 1.0)) if eol_exists else 1.0
            censor_fraction = float(np.clip(observed_max_cycle / max(target_length, 1), 0.0, 1.0))

            sample = {
                "battery_id": item["battery_id"],
                "source_battery_id": item.get("source_battery_id", item["battery_id"]),
                "group_id": group_id,
                "group_name": group["name"],
                "target_length": target_length,
                "future_length": int(len(future_grid)),
                "observed_future_length": observed_future_len,
                "observed_max_cycle": observed_max_cycle,
                "early_soh_100": early_soh.astype(np.float32),
                "start_soh_100": start_soh_100,
                "future_delta": future_delta.astype(np.float32),
                "observed_future_mask": observed_future_mask.astype(np.float32),
                "feature_matrix": np.asarray(item["feature_matrix"], dtype=np.float32),
                "life_features": build_life_feature_vector(
                    early_soh=early_soh,
                    feature_matrix=np.asarray(item["feature_matrix"], dtype=np.float32),
                    target_length=target_length,
                    observed_max_cycle=observed_max_cycle,
                    max_target_length=max_target_length,
                ),
                "true_curve_full": full_true_curve.astype(np.float32),
                "eol_exists": bool(eol_exists),
                "eol_cycle": float(eol_cycle),
                "eol_fraction": float(eol_fraction),
                "censor_fraction": float(censor_fraction),
            }
            samples.append(sample)

    return samples, target_lengths, dropped


def compute_delta_stats(samples, train_indices):
    """
    仅用训练集统计归一化参数：
    - delta_scale: 观测增量 99 分位数（缩放因子）
    - raw_mean/raw_std: inv_softplus 后的均值方差
    """
    obs_delta_values = []
    for idx in train_indices:
        sample = samples[idx]
        mask = sample["observed_future_mask"] > 0.5
        if np.any(mask):
            obs_delta_values.append(sample["future_delta"][mask])

    if not obs_delta_values:
        raise RuntimeError("No observed delta values found in training split.")

    obs_delta = np.concatenate(obs_delta_values, axis=0).astype(np.float32)
    delta_scale = float(np.percentile(obs_delta, 99.0))
    delta_scale = max(delta_scale, 0.02)

    raw_values = []
    for idx in train_indices:
        sample = samples[idx]
        mask = sample["observed_future_mask"] > 0.5
        if not np.any(mask):
            continue
        raw = inv_softplus(sample["future_delta"][mask] / delta_scale)
        raw_values.append(raw.astype(np.float32))

    raw_all = np.concatenate(raw_values, axis=0).astype(np.float32)
    raw_mean = float(np.mean(raw_all))
    raw_std = float(np.std(raw_all) + 1e-6)

    return {
        "delta_scale": delta_scale,
        "raw_mean": raw_mean,
        "raw_std": raw_std,
    }


def main(seed=42, rebuild_grouped=False):
    """主函数：读取 grouped_data，构建并保存 Code2 数据包。"""
    ensure_dirs()
    set_global_seed(seed)

    grouped_path = ensure_grouped_data(seed=seed, rebuild_grouped=rebuild_grouped)
    with open(grouped_path, "rb") as f:
        grouped_pack = pickle.load(f)
    grouped_data = grouped_pack["grouped_data"]

    samples, target_lengths, dropped = build_samples(grouped_data)
    if len(samples) == 0:
        raise RuntimeError("No valid samples found for Code2 pipeline.")

    rng = np.random.default_rng(seed)
    train_indices, val_indices, test_indices = [], [], []
    split_stats = {}

    for gid in sorted(target_lengths.keys()):
        # 按组划分，保持不同寿命组在 train/val/test 中均有样本。
        group_sample_ids = [i for i, s in enumerate(samples) if int(s["group_id"]) == int(gid)]
        g_train, g_val, g_test = split_indices(len(group_sample_ids), rng)

        train_indices.extend([group_sample_ids[i] for i in g_train])
        val_indices.extend([group_sample_ids[i] for i in g_val])
        test_indices.extend([group_sample_ids[i] for i in g_test])

        split_stats[int(gid)] = {
            "group_name": next(s["group_name"] for s in samples if int(s["group_id"]) == int(gid)),
            "n_total": len(group_sample_ids),
            "n_train": len(g_train),
            "n_val": len(g_val),
            "n_test": len(g_test),
            "target_length": int(target_lengths[gid]),
        }

    max_future_len = max(get_future_length(v) for v in target_lengths.values())
    max_future_len_padded = round_up_multiple(max_future_len, PAD_MULTIPLE)
    # pad 到固定倍数，保证网络输入长度对齐。

    stats = compute_delta_stats(samples, train_indices)

    package = {
        "seed": int(seed),
        "samples": samples,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "target_lengths": target_lengths,
        "max_future_length": int(max_future_len),
        "max_future_length_padded": int(max_future_len_padded),
        "split_stats": split_stats,
        "delta_scale": float(stats["delta_scale"]),
        "raw_mean": float(stats["raw_mean"]),
        "raw_std": float(stats["raw_std"]),
        "early_cycles": int(EARLY_CYCLES),
        "stride": int(SEQ_CYCLE_STRIDE),
        "life_feature_dim": int(len(samples[0]["life_features"])),
    }

    out_path = os.path.join(OUTPUT2_DIR, f"code2_dataset_seed{seed}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(package, f)

    print("=" * 70)
    print(f"Code2 data prepared (seed={seed})")
    print(f"Saved dataset: {out_path}")
    print(f"Total samples: {len(samples)} (dropped: {dropped})")
    print(f"Max future len: {max_future_len} -> padded {max_future_len_padded}")
    print(
        f"Delta stats: scale={package['delta_scale']:.6f}, "
        f"raw_mean={package['raw_mean']:.4f}, raw_std={package['raw_std']:.4f}"
    )
    print("Split stats:")
    for gid in sorted(split_stats.keys()):
        s = split_stats[gid]
        print(
            f"  Group {gid} ({s['group_name']}): total={s['n_total']}, "
            f"train={s['n_train']}, val={s['n_val']}, test={s['n_test']}, "
            f"target={s['target_length']}"
        )
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code2 data preparation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rebuild_grouped",
        action="store_true",
        help="Force rebuilding grouped_data.pkl from raw Data/batch*.pkl before creating code2 dataset.",
    )
    args = parser.parse_args()
    main(seed=args.seed, rebuild_grouped=args.rebuild_grouped)
