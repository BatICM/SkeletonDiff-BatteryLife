import os
import random

import numpy as np
import torch

"""
Code2 公共工具模块（被训练/测试/可视化/基线脚本复用）
======================================================
本文件集中放“不会因模型类型变化而改变”的基础能力：

1) 路径与全局常量
2) 复现实验所需随机种子设置
3) 时序曲线处理（单调化、插值、EOL计算）
4) 特征工程（早期曲线形态特征 + life feature 向量）

阅读建议：
- 先看常量区，明确时间窗口与阈值定义
- 再看曲线处理函数（monotone/safe_interp/calculate_eol）
- 最后看 build_life_feature_vector，它定义了模型条件输入
"""

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
OUTPUT2_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL2_DIR = os.path.join(PROJECT_ROOT, "weights")
PLOT_DIR = os.path.join(PROJECT_ROOT, "results", "plots")
PLOT_DATASET_DIR = os.path.join(PLOT_DIR, "dataset")
PLOT_FEATURE_DIR = os.path.join(PLOT_DIR, "features")
PLOT_MAIN_DIR = os.path.join(PLOT_DIR, "main_experiment")
PLOT_BASELINE_DIR = os.path.join(PLOT_DIR, "baselines")

EARLY_CYCLES = 100
SEQ_CYCLE_STRIDE = 5
EOL_THRESHOLD = 80.0
PAD_MULTIPLE = 8


def ensure_dirs():
    """创建输出目录（幂等操作，可重复调用）。"""
    os.makedirs(OUTPUT2_DIR, exist_ok=True)
    os.makedirs(MODEL2_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(PLOT_DATASET_DIR, exist_ok=True)
    os.makedirs(PLOT_FEATURE_DIR, exist_ok=True)
    os.makedirs(PLOT_MAIN_DIR, exist_ok=True)
    os.makedirs(PLOT_BASELINE_DIR, exist_ok=True)


def set_global_seed(seed, deterministic=False):
    """
    设置 Python / NumPy / Torch 的随机种子。
    目的：保证同一 seed 下数据划分与训练过程尽可能可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def monotone_curve(values):
    """
    对序列做“单调不增”投影。
    电池 SOH 理论上应随循环下降，因此这里用 cumulative minimum 消除上升噪声。
    """
    values = np.asarray(values, dtype=np.float32).flatten()
    if len(values) == 0:
        return values
    return np.minimum.accumulate(values)


def safe_interp(x_src, y_src, x_dst):
    """
    安全插值函数（1D）。
    与 np.interp 不同点：
    - 自动排序/去重 x_src
    - 对边界外点执行“端点延拓”（而不是报错）
    """
    x_src = np.asarray(x_src, dtype=np.float32).flatten()
    y_src = np.asarray(y_src, dtype=np.float32).flatten()
    x_dst = np.asarray(x_dst, dtype=np.float32).flatten()
    if len(x_src) == 0:
        return np.zeros_like(x_dst, dtype=np.float32)

    order = np.argsort(x_src)
    x_src = x_src[order]
    y_src = y_src[order]

    x_src, unique_idx = np.unique(x_src, return_index=True)
    y_src = y_src[unique_idx]

    if len(x_src) == 1:
        return np.full_like(x_dst, float(y_src[0]), dtype=np.float32)

    y = np.interp(x_dst, x_src, y_src).astype(np.float32)
    y[x_dst < x_src[0]] = y_src[0]
    y[x_dst > x_src[-1]] = y_src[-1]
    return y


def calculate_eol_cycle(soh_values, threshold=EOL_THRESHOLD):
    """
    返回第一次达到 EOL 阈值的循环位置（1-based）。
    若曲线未触达阈值，返回 -1。
    """
    soh_values = np.asarray(soh_values, dtype=np.float32).flatten()
    idx = np.where(soh_values <= float(threshold))[0]
    return int(idx[0] + 1) if len(idx) > 0 else -1


def get_group_target_lengths(grouped_data):
    """从 grouped_data 中提取每组统一目标长度（cycle 数）。"""
    out = {}
    for gid, group in grouped_data.items():
        out[int(gid)] = int(group["target_length"])
    return out


def get_future_length(target_length):
    """
    给定目标总长度，计算“未来段（stride=5）”的 token 长度。
    未来段从 EARLY_CYCLES 之后开始。
    """
    target_length = int(target_length)
    if target_length <= EARLY_CYCLES:
        return 1
    return int((target_length - EARLY_CYCLES) // SEQ_CYCLE_STRIDE)


def round_up_multiple(value, multiple):
    """将长度向上补齐到指定倍数（用于网络输入维度对齐）。"""
    value = int(value)
    multiple = int(max(1, multiple))
    return ((value + multiple - 1) // multiple) * multiple


def inv_softplus(y):
    """
    softplus 的近似逆变换。
    在数据预处理阶段用于把非负增量映射回实数域，便于标准化建模。
    """
    y = np.asarray(y, dtype=np.float32)
    return np.log(np.expm1(np.maximum(y, 1e-6)) + 1e-8).astype(np.float32)


def softplus_numpy(x):
    """NumPy 版本 softplus（与 Torch 对齐）。"""
    x = np.asarray(x, dtype=np.float32)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def extract_early_shape_features(early_soh):
    """
    从前100循环 SOH 提取形态特征（8维）。
    这些特征描述了“早期退化速度、曲率、加速度”等信息。
    """
    early_soh = np.asarray(early_soh, dtype=np.float32).flatten()
    if len(early_soh) == 0:
        return np.zeros(8, dtype=np.float32)

    n = len(early_soh)
    x = np.arange(1, n + 1, dtype=np.float32)
    x_norm = (x - x.min()) / max(float(x.max() - x.min()), 1e-6)

    d_1_20 = float(early_soh[0] - early_soh[min(19, n - 1)])
    d_1_50 = float(early_soh[0] - early_soh[min(49, n - 1)])
    d_1_end = float(early_soh[0] - early_soh[-1])

    if n >= 3:
        p2 = np.polyfit(x_norm, early_soh, deg=2)
        quad = float(p2[0])
        lin = float(p2[1])
    else:
        quad = 0.0
        lin = float((early_soh[-1] - early_soh[0]) / max(n - 1, 1))

    slope = np.diff(early_soh) if n >= 2 else np.array([0.0], dtype=np.float32)
    slope_abs_mean = float(np.mean(np.abs(slope)))
    slope_end = float(-slope[-1]) if len(slope) > 0 else 0.0
    slope_change = np.diff(slope) if len(slope) >= 2 else np.array([0.0], dtype=np.float32)
    accel_mean = float(np.mean(slope_change))

    return np.array(
        [
            d_1_20,
            d_1_50,
            d_1_end,
            lin,
            quad,
            slope_abs_mean,
            slope_end,
            accel_mean,
        ],
        dtype=np.float32,
    )


def build_life_feature_vector(
    early_soh,
    feature_matrix,
    target_length,
    observed_max_cycle,
    max_target_length,
):
    """
    构建 life feature 向量（最终供条件编码器使用）。

    组成：
    - base 特征：目标长度比例、观测比例、末端早期SOH、feature矩阵统计量
    - shape 特征：extract_early_shape_features 返回的 8 维形态特征

    关键约束：
    - observed_max_cycle 在这里会被截断到 EARLY_CYCLES，
      目的是避免未来信息泄漏到条件向量中。
    """
    early_soh = np.asarray(early_soh, dtype=np.float32).flatten()
    feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
    max_target_length = float(max(max_target_length, 1.0))
    target_length = float(max(target_length, 1.0))
    # Prevent future-information leakage: only use information available at/within early window.
    observed_for_feature = float(min(float(observed_max_cycle), float(EARLY_CYCLES)))

    base = np.array(
        [
            target_length / max_target_length,
            observed_for_feature / target_length,
            float(early_soh[-1] if len(early_soh) > 0 else 100.0) / 100.0,
            float(np.mean(feature_matrix)),
            float(np.std(feature_matrix)),
            float(np.mean(feature_matrix[:, -10:])),
        ],
        dtype=np.float32,
    )
    shape = extract_early_shape_features(early_soh)
    return np.concatenate([base, shape], axis=0).astype(np.float32)
