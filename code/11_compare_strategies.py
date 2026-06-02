import argparse
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from common import EARLY_CYCLES, EOL_THRESHOLD, OUTPUT2_DIR


EOL_EVAL_TOL = 0.1


def calculate_eol_cycle_eval(soh_values, threshold=EOL_THRESHOLD, tol=EOL_EVAL_TOL):
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


def to_markdown_table(df: pd.DataFrame) -> str:
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


def summarize_one(pred_path: str) -> Dict:
    with open(pred_path, "rb") as f:
        pack = pickle.load(f)
    metrics = pack["metrics"]
    rows: List[Dict] = []
    for item in pack["results"]:
        g = int(item["true_group"])
        rmse, mae, corr = eval_rmse_mae_corr(item["true_curve"], item["pred_curve"])
        eol_true = int(item.get("eol_true", -1))
        eol_pred = int(item.get("eol_pred", -1))
        eol_err = abs(eol_true - eol_pred) if (eol_true > 0 and eol_pred > 0) else np.nan
        rows.append(
            {
                "battery_id": item["battery_id"],
                "group": g,
                "rmse": rmse,
                "mae": mae,
                "corr": corr,
                "eol_err": eol_err,
            }
        )
    df = pd.DataFrame(rows)
    worst = {}
    for g in [0, 1, 2]:
        sub = df[df["group"] == g]
        if len(sub) == 0:
            worst[g] = {
                "worst_rmse_id": "N/A",
                "worst_rmse": np.nan,
                "worst_eol_id": "N/A",
                "worst_eol": np.nan,
            }
            continue
        wr = sub.loc[sub["rmse"].idxmax()]
        eol_sub = sub.dropna(subset=["eol_err"])
        if len(eol_sub) > 0:
            we = eol_sub.loc[eol_sub["eol_err"].idxmax()]
            worst_eol_id = str(we["battery_id"])
            worst_eol = float(we["eol_err"])
        else:
            worst_eol_id = "N/A"
            worst_eol = np.nan
        worst[g] = {
            "worst_rmse_id": str(wr["battery_id"]),
            "worst_rmse": float(wr["rmse"]),
            "worst_eol_id": worst_eol_id,
            "worst_eol": worst_eol,
        }
    return {"metrics": metrics, "worst": worst}


def run(seed=42):
    sampler_exps = [
        ("weighted_random", f"code2_test_predictions_seed{seed}_n10_stratA_weighted_r82.pkl"),
        ("shuffle", f"code2_test_predictions_seed{seed}_n10_stratA_shuffle_r82.pkl"),
        ("sequential", f"code2_test_predictions_seed{seed}_n10_stratA_sequential_r82.pkl"),
    ]
    stage_exps = [
        ("two_stage", f"code2_test_predictions_seed{seed}_n10_stratA_weighted_r82.pkl"),
        ("two_stage_long", f"code2_test_predictions_seed{seed}_n10_stratB_twostage_long_r82.pkl"),
        ("det_only", f"code2_test_predictions_seed{seed}_n10_stratB_detonly_r82.pkl"),
    ]

    def make_tables(exp_list, compare_name):
        rows_overall = []
        rows_worst = []
        for name, fn in exp_list:
            pred_path = os.path.join(OUTPUT2_DIR, fn)
            if not os.path.exists(pred_path):
                raise FileNotFoundError(f"Missing result file: {pred_path}")
            s = summarize_one(pred_path)
            m = s["metrics"]
            rows_overall.append(
                {
                    "strategy": name,
                    "rmse_mean": float(m["rmse_mean"]),
                    "rmse_std": float(m["rmse_std"]),
                    "mae_mean": float(m["mae_mean"]),
                    "mae_std": float(m["mae_std"]),
                    "corr_mean": float(m["corr_mean"]),
                    "corr_std": float(m["corr_std"]),
                    "eol_error_mean": float(m["eol_error_mean"]),
                    "eol_error_std": float(m["eol_error_std"]),
                    "valid_eol_ratio": float(m["valid_eol_ratio"]),
                    "group_acc": float(m.get("group_acc", np.nan)),
                    "path": pred_path,
                }
            )
            for g in [0, 1, 2]:
                w = s["worst"][g]
                rows_worst.append(
                    {
                        "strategy": name,
                        "group": g,
                        "worst_rmse_id": w["worst_rmse_id"],
                        "worst_rmse": w["worst_rmse"],
                        "worst_eol_id": w["worst_eol_id"],
                        "worst_eol": w["worst_eol"],
                    }
                )

        df_o = pd.DataFrame(rows_overall)
        df_w = pd.DataFrame(rows_worst)
        stem = f"strategy_compare_{compare_name}_seed{seed}"
        out_o = os.path.join(OUTPUT2_DIR, f"{stem}_overall.csv")
        out_w = os.path.join(OUTPUT2_DIR, f"{stem}_worst.csv")
        out_md = os.path.join(OUTPUT2_DIR, f"{stem}.md")
        df_o.to_csv(out_o, index=False, encoding="utf-8-sig")
        df_w.to_csv(out_w, index=False, encoding="utf-8-sig")
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(f"# {compare_name} strategy comparison (seed={seed})\n\n")
            f.write("## Overall metrics\n\n")
            f.write(to_markdown_table(df_o))
            f.write("\n\n## Worst sample in each group (worst RMSE / worst EOL)\n\n")
            f.write(to_markdown_table(df_w))
            f.write("\n")
        return df_o, df_w, out_o, out_w, out_md

    df_sampler_o, df_sampler_w, so, sw, smd = make_tables(sampler_exps, "sampler")
    df_stage_o, df_stage_w, to, tw, tmd = make_tables(stage_exps, "stage")

    print("=" * 88)
    print("Sampler strategy comparison")
    print("=" * 88)
    print(df_sampler_o.to_string(index=False))
    print("\nWorst samples by group:")
    print(df_sampler_w.to_string(index=False))

    print("\n" + "=" * 88)
    print("Stage strategy comparison")
    print("=" * 88)
    print(df_stage_o.to_string(index=False))
    print("\nWorst samples by group:")
    print(df_stage_w.to_string(index=False))

    print("\nSaved files:")
    print(f"- {so}")
    print(f"- {sw}")
    print(f"- {smd}")
    print(f"- {to}")
    print(f"- {tw}")
    print(f"- {tmd}")


def main():
    parser = argparse.ArgumentParser(description="Training strategy comparison summary")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(seed=int(args.seed))


if __name__ == "__main__":
    main()
