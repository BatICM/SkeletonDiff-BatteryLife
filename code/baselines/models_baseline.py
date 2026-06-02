import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE2_DIR = os.path.dirname(CURRENT_DIR)
if CODE2_DIR not in sys.path:
    sys.path.insert(0, CODE2_DIR)

from models_diffusion import ConditionalDiffusionModel  # noqa: E402

"""
Baseline 模型定义（公平对比版）
==============================
目标：只替换“轨迹生成骨干网络”，其余协议保持与扩散模型一致。

做法：
- 继承 ConditionalDiffusionModel 复用同一条件编码器与同一损失体系
- 覆盖 predict_deterministic_x：
  - backbone=lstm        -> LSTMSequenceHead
  - backbone=transformer -> TransformerSequenceHead

这样可保证“对比只来自模型结构差异”。
"""


class LSTMSequenceHead(nn.Module):
    """把 cond 向量解码为整段序列（LSTM 版本）。"""

    def __init__(self, cond_dim=128, seq_len=432, hidden_dim=256, layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = int(seq_len)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)

        self.init_h = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * layers),
            nn.SiLU(),
        )
        self.init_c = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * layers),
            nn.SiLU(),
        )
        self.step_token = nn.Parameter(torch.randn(1, self.seq_len, hidden_dim) * 0.01)
        self.cond_gate = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            dropout=float(dropout) if layers > 1 else 0.0,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, cond):
        # cond -> 初始隐状态 + 步进token门控
        bsz = cond.shape[0]
        gate = self.cond_gate(cond).unsqueeze(1)
        token = self.step_token.expand(bsz, -1, -1) * gate

        h0 = self.init_h(cond).view(bsz, self.layers, self.hidden_dim).transpose(0, 1).contiguous()
        c0 = self.init_c(cond).view(bsz, self.layers, self.hidden_dim).transpose(0, 1).contiguous()

        out, _ = self.rnn(token, (h0, c0))
        return self.out(out).transpose(1, 2)


class TransformerSequenceHead(nn.Module):
    """把 cond 向量解码为整段序列（Transformer 版本）。"""

    def __init__(self, cond_dim=128, seq_len=432, model_dim=256, depth=4, nhead=8, dropout=0.1):
        super().__init__()
        self.seq_len = int(seq_len)
        self.model_dim = int(model_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, self.seq_len, self.model_dim) * 0.01)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(nhead),
            dim_feedforward=self.model_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(depth))
        self.norm = nn.LayerNorm(self.model_dim)
        self.out = nn.Linear(self.model_dim, 1)

    def _pos(self, length):
        """长度自适应位置编码（插值扩展）。"""
        if length == self.pos_emb.shape[1]:
            return self.pos_emb
        if length < self.pos_emb.shape[1]:
            return self.pos_emb[:, :length, :]
        pe = F.interpolate(self.pos_emb.transpose(1, 2), size=length, mode="linear", align_corners=False)
        return pe.transpose(1, 2)

    def forward(self, cond):
        # 以“位置编码 + 条件偏置”作为输入序列，输出整段 x_norm。
        bsz = cond.shape[0]
        pos = self._pos(self.seq_len).expand(bsz, -1, -1)
        c = self.cond_proj(cond).unsqueeze(1)
        h = self.encoder(pos + c)
        return self.out(self.norm(h)).transpose(1, 2)


class BaselineDeterministicModel(ConditionalDiffusionModel):
    """
    Keep the same conditioning + curve/EOL losses as Code2 diffusion model,
    and only replace the deterministic sequence generator backbone.
    """

    def __init__(
        self,
        backbone="lstm",
        n_groups=3,
        cond_dim=128,
        life_dim=14,
        seq_len=432,
        raw_mean=0.0,
        raw_std=1.0,
        delta_scale=1.0,
        accel_weight=0.14,
        smooth_weight=0.04,
        long_accel_weight=0.10,
        curve_eol_weight=0.08,
        eol_consistency_weight=0.06,
        knee_weight=0.06,
        group_loss_weights=(1.0, 1.05, 1.8),
    ):
        super().__init__(
            n_groups=n_groups,
            cond_dim=cond_dim,
            life_dim=life_dim,
            seq_len=seq_len,
            timesteps=16,
            raw_mean=raw_mean,
            raw_std=raw_std,
            delta_scale=delta_scale,
            denoiser_type="unet",
            accel_weight=accel_weight,
            smooth_weight=smooth_weight,
            long_accel_weight=long_accel_weight,
            curve_eol_weight=curve_eol_weight,
            eol_consistency_weight=eol_consistency_weight,
            knee_weight=knee_weight,
            group_loss_weights=group_loss_weights,
        )
        self.backbone = str(backbone).strip().lower()
        if self.backbone not in {"lstm", "transformer"}:
            raise ValueError(f"Unsupported baseline backbone: {backbone}")

        if self.backbone == "lstm":
            self.seq_head = LSTMSequenceHead(cond_dim=cond_dim, seq_len=seq_len, hidden_dim=256, layers=2, dropout=0.1)
        else:
            self.seq_head = TransformerSequenceHead(
                cond_dim=cond_dim,
                seq_len=seq_len,
                model_dim=256,
                depth=4,
                nhead=8,
                dropout=0.1,
            )
        self.group_bias = nn.Embedding(n_groups, int(seq_len))

        # Keep diffusion submodules out of training to make the baseline clean and faster.
        self.unet = None
        self.transformer_denoiser = None

    def predict_deterministic_x(self, cond, group_ids):
        """统一确定性预测接口：seq_head + group_bias + 平滑池化。"""
        x = self.seq_head(cond) + self.group_bias(group_ids).unsqueeze(1)
        x = F.avg_pool1d(x, kernel_size=5, stride=1, padding=2)
        return torch.clamp(x, min=-6.0, max=3.0)
