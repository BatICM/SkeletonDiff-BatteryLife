import argparse
import os
import pickle
import sys

"""
对比汇总脚本
============
读取：
- 扩散模型测试结果（code2_test_predictions_seed*_n10.pkl）
- baseline 测试结果（lstm/transformer）

输出：
- 终端表格打印
- Output2/baseline_comparison_seed*.md（论文可直接引用）
"""


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE2_DIR = os.path.dirname(CURRENT_DIR)
if CODE2_DIR not in sys.path:
    sys.path.insert(0, CODE2_DIR)

from common import OUTPUT2_DIR  # noqa: E402


def load_metrics(path):
    """从结果 pkl 中安全提取 metrics。"""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "metrics" not in obj:
        return None
    return obj["metrics"]


def row(name, m):
    """把指标字典格式化为单行字符串。"""
    if m is None:
        return [name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]
    return [
        name,
        f"{m.get('rmse_mean', -1):.4f}",
        f"{m.get('mae_mean', -1):.4f}",
        f"{m.get('corr_mean', -1):.4f}",
        f"{m.get('eol_error_mean', -1):.2f}",
        f"{m.get('valid_eol_ratio', -1):.3f}",
        f"{m.get('group_acc', -1):.3f}",
    ]


def main(seed=42):
    """对比表主入口。"""
    diff_path = os.path.join(OUTPUT2_DIR, f"code2_test_predictions_seed{seed}_n10.pkl")
    lstm_path = os.path.join(OUTPUT2_DIR, f"baseline_lstm_test_predictions_seed{seed}.pkl")
    tfm_path = os.path.join(OUTPUT2_DIR, f"baseline_transformer_test_predictions_seed{seed}.pkl")

    m_diff = load_metrics(diff_path)
    m_lstm = load_metrics(lstm_path)
    m_tfm = load_metrics(tfm_path)

    headers = ["Model", "RMSE(pre-EOL)", "MAE(pre-EOL)", "Corr(pre-EOL)", "EOL Error", "Valid EOL Ratio", "Group Acc"]
    rows = [
        row("Diffusion(Code2 n10)", m_diff),
        row("Baseline-LSTM", m_lstm),
        row("Baseline-Transformer", m_tfm),
    ]

    line = "-" * 110
    print(line)
    print(f"{headers[0]:<24} {headers[1]:>14} {headers[2]:>14} {headers[3]:>14} {headers[4]:>10} {headers[5]:>16} {headers[6]:>10}")
    print(line)
    for r in rows:
        print(f"{r[0]:<24} {r[1]:>14} {r[2]:>14} {r[3]:>14} {r[4]:>10} {r[5]:>16} {r[6]:>10}")
    print(line)

    out_md = os.path.join(OUTPUT2_DIR, f"baseline_comparison_seed{seed}.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Diffusion vs baseline metrics")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)
