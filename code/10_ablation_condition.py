import argparse
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from common import EARLY_CYCLES, EOL_THRESHOLD, OUTPUT2_DIR

"""
条件输入消融汇总脚本
==================
目标：
1) 统一口径评估 full / no_life / life_only 三种条件输入模式
2) 输出整体指标、分组指标
3) 输出每组“最差RMSE样本”和“最差EOL误差样本”（允许为不同电池）

说明：
- 评估口径与 03_test_code2_refactor.py 保持一致（pre-EOL RMSE/MAE/Corr）
- 默认读取 n10 结果文件，优先 *_fair.pkl，其次普通 pkl
"""


EOL_EVAL_TOL = 0.1


def calculate_eol_cycle_eval(soh_values, threshold=EOL_THRESHOLD, tol=EOL_EVAL_TOL):
    """与 03_test_code2_refactor.py 对齐的评估期 EOL 判定。"""
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
    """与 03_test_code2_refactor.py 对齐：在 pre-EOL 有效段计算。"""
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


def resolve_pred_path(seed: int, mode_name: str, n_mode: str = "n10", runtime_tag: str = "r82") -> str:
    """按命名规则自动定位对应的消融结果文件。"""
    candidates = [
        os.path.join(OUTPUT2_DIR, f"code2_test_predictions_seed{seed}_{n_mode}_ab_{mode_name}_{runtime_tag}_fair.pkl"),
        os.path.join(OUTPUT2_DIR, f"code2_test_predictions_seed{seed}_{n_mode}_ab_{mode_name}_{runtime_tag}.pkl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Cannot find ablation prediction file for mode={mode_name}, seed={seed}, n_mode={n_mode}, runtime_tag={runtime_tag}"
    )


def summarize_pack(pack: Dict) -> Tuple[Dict, pd.DataFrame]:
    """汇总单个模式结果。"""
    results = pack["results"]
    rows: List[Dict] = []
    for item in results:
        true_curve = np.asarray(item["true_curve"], dtype=np.float32)
        pred_curve = np.asarray(item["pred_curve"], dtype=np.float32)
        rmse, mae, corr = eval_rmse_mae_corr(true_curve, pred_curve)
        eol_true = int(item.get("eol_true", -1))
        eol_pred = int(item.get("eol_pred", -1))
        eol_err = abs(eol_true - eol_pred) if (eol_true > 0 and eol_pred > 0) else np.nan
        rows.append(
            {
                "battery_id": item.get("battery_id", "N/A"),
                "group": int(item.get("true_group", -1)),
                "rmse": rmse,
                "mae": mae,
                "corr": corr,
                "eol_true": eol_true,
                "eol_pred": eol_pred,
                "eol_err": eol_err,
            }
        )

    df = pd.DataFrame(rows)

    def _agg(sub: pd.DataFrame) -> Dict:
        corr_valid = sub.loc[sub["corr"] >= -0.5, "corr"]
        eol_valid = sub["eol_err"].dropna()
        n_true_eol_pos = int((sub["eol_true"] > 0).sum())
        false_cross = int(((sub["eol_true"] <= 0) & (sub["eol_pred"] > 0)).sum())
        missed_cross = int(((sub["eol_true"] > 0) & (sub["eol_pred"] <= 0)).sum())
        return {
            "n": int(len(sub)),
            "rmse_mean": float(sub["rmse"].mean()),
            "rmse_std": float(sub["rmse"].std(ddof=0)),
            "mae_mean": float(sub["mae"].mean()),
            "mae_std": float(sub["mae"].std(ddof=0)),
            "corr_mean": float(corr_valid.mean()) if len(corr_valid) else -1.0,
            "corr_std": float(corr_valid.std(ddof=0)) if len(corr_valid) else -1.0,
            "eol_error_mean": float(eol_valid.mean()) if len(eol_valid) else -1.0,
            "eol_error_std": float(eol_valid.std(ddof=0)) if len(eol_valid) else -1.0,
            "n_valid_eol": int(len(eol_valid)),
            "n_true_eol_positive": n_true_eol_pos,
            "valid_eol_ratio": float(len(eol_valid) / max(n_true_eol_pos, 1)),
            "false_cross": false_cross,
            "missed_cross": missed_cross,
        }

    out = {
        "overall": _agg(df),
        "group": {g: _agg(df[df["group"] == g]) for g in [0, 1, 2]},
        "worst_by_group": {},
    }

    for g in [0, 1, 2]:
        sub = df[df["group"] == g]
        if len(sub) == 0:
            out["worst_by_group"][g] = {
                "worst_rmse_id": "N/A",
                "worst_rmse": np.nan,
                "worst_eol_id": "N/A",
                "worst_eol": np.nan,
            }
            continue

        idx_rmse = sub["rmse"].idxmax()
        wr = sub.loc[idx_rmse]
        eol_sub = sub.dropna(subset=["eol_err"])
        if len(eol_sub) > 0:
            idx_eol = eol_sub["eol_err"].idxmax()
            we = eol_sub.loc[idx_eol]
            worst_eol_id, worst_eol = str(we["battery_id"]), float(we["eol_err"])
        else:
            worst_eol_id, worst_eol = "N/A", np.nan

        out["worst_by_group"][g] = {
            "worst_rmse_id": str(wr["battery_id"]),
            "worst_rmse": float(wr["rmse"]),
            "worst_eol_id": worst_eol_id,
            "worst_eol": worst_eol,
        }

    if isinstance(pack, dict) and isinstance(pack.get("metrics", None), dict):
        if "group_acc" in pack["metrics"]:
            out["overall"]["group_acc"] = float(pack["metrics"]["group_acc"])
        else:
            out["overall"]["group_acc"] = np.nan
    else:
        out["overall"]["group_acc"] = np.nan

    return out, df


def run(seed=42, n_mode="n10", runtime_tag="r82"):
    mode_alias = [("full", "full"), ("no_life", "nolife"), ("life_only", "lifeonly")]
    packs = {}
    for mode_name, file_mode_name in mode_alias:
        p = resolve_pred_path(seed=seed, mode_name=file_mode_name, n_mode=n_mode, runtime_tag=runtime_tag)
        with open(p, "rb") as f:
            packs[mode_name] = (p, pickle.load(f))

    overall_rows = []
    group_rows = []
    worst_rows = []

    for mode_name in ["full", "no_life", "life_only"]:
        pred_path, pack = packs[mode_name]
        summary, _df = summarize_pack(pack)

        o = summary["overall"]
        overall_rows.append(
            {
                "mode": mode_name,
                "pred_path": pred_path,
                "rmse_mean": o["rmse_mean"],
                "rmse_std": o["rmse_std"],
                "mae_mean": o["mae_mean"],
                "mae_std": o["mae_std"],
                "corr_mean": o["corr_mean"],
                "corr_std": o["corr_std"],
                "eol_error_mean": o["eol_error_mean"],
                "eol_error_std": o["eol_error_std"],
                "valid_eol_ratio": o["valid_eol_ratio"],
                "n_valid_eol": o["n_valid_eol"],
                "n_true_eol_positive": o["n_true_eol_positive"],
                "false_cross": o["false_cross"],
                "missed_cross": o["missed_cross"],
                "group_acc": o.get("group_acc", np.nan),
            }
        )

        for g in [0, 1, 2]:
            gg = summary["group"][g]
            group_rows.append(
                {
                    "mode": mode_name,
                    "group": g,
                    "rmse_mean": gg["rmse_mean"],
                    "mae_mean": gg["mae_mean"],
                    "corr_mean": gg["corr_mean"],
                    "eol_error_mean": gg["eol_error_mean"],
                    "valid_eol_ratio": gg["valid_eol_ratio"],
                }
            )
            w = summary["worst_by_group"][g]
            worst_rows.append(
                {
                    "mode": mode_name,
                    "group": g,
                    "worst_rmse_id": w["worst_rmse_id"],
                    "worst_rmse": w["worst_rmse"],
                    "worst_eol_id": w["worst_eol_id"],
                    "worst_eol": w["worst_eol"],
                }
            )

    df_overall = pd.DataFrame(overall_rows)
    df_group = pd.DataFrame(group_rows)
    df_worst = pd.DataFrame(worst_rows)

    stem = f"condition_input_ablation_seed{seed}_{runtime_tag}_{n_mode}"
    out_csv_overall = os.path.join(OUTPUT2_DIR, f"{stem}_overall.csv")
    out_csv_group = os.path.join(OUTPUT2_DIR, f"{stem}_group.csv")
    out_csv_worst = os.path.join(OUTPUT2_DIR, f"{stem}_worst.csv")
    out_md = os.path.join(OUTPUT2_DIR, f"{stem}.md")

    df_overall.to_csv(out_csv_overall, index=False, encoding="utf-8-sig")
    df_group.to_csv(out_csv_group, index=False, encoding="utf-8-sig")
    df_worst.to_csv(out_csv_worst, index=False, encoding="utf-8-sig")

    def _to_md_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    if np.isnan(v):
                        vals.append("N/A")
                    else:
                        vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# 条件输入消融汇总 (seed={seed}, tag={runtime_tag}, {n_mode})\n\n")
        f.write("## 1) 整体指标\n\n")
        f.write(_to_md_table(df_overall))
        f.write("\n\n## 2) 分组指标\n\n")
        f.write(_to_md_table(df_group))
        f.write("\n\n## 3) 每组最差样本（RMSE最差 & EOL误差最差）\n\n")
        f.write(_to_md_table(df_worst))
        f.write("\n")

    print("=" * 88)
    print(f"Condition input ablation summary | seed={seed}, tag={runtime_tag}, {n_mode}")
    print("=" * 88)
    print("\n[Overall]")
    print(df_overall.to_string(index=False))
    print("\n[Per-group]")
    print(df_group.to_string(index=False))
    print("\n[Worst by group]")
    print(df_worst.to_string(index=False))
    print("\nSaved files:")
    print(f"- {out_csv_overall}")
    print(f"- {out_csv_group}")
    print(f"- {out_csv_worst}")
    print(f"- {out_md}")


def main():
    parser = argparse.ArgumentParser(description="Condition input ablation summary")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_mode", type=str, default="n10", choices=["n1", "n10"])
    parser.add_argument("--runtime_tag", type=str, default="r82")
    args = parser.parse_args()
    run(seed=int(args.seed), n_mode=str(args.n_mode), runtime_tag=str(args.runtime_tag))


if __name__ == "__main__":
    main()
