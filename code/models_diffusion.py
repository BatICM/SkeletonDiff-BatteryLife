import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import EOL_THRESHOLD

"""
Code2 模型定义文件
==================
文件包含四层结构：

1) 条件编码器（ConditionEncoder）
   - 融合 feature_matrix、early_soh、life_features、group embedding
2) 去噪器（UNet1D / Transformer1D）
   - 用于扩散训练阶段的噪声预测
3) 确定性轨迹分支（predict_deterministic_x）
   - 先给出稳定的退化骨架
4) 物理/任务约束
   - 非负增量解码、单调/平滑/长寿命加速约束
   - EOL 辅助头 + 曲线EOL一致性 + knee 辅助约束

阅读建议：
- 先看 ConditionEncoder
- 再看 ConditionalDiffusionModel.__init__
- 然后看 forward_deterministic / forward_train / sample_ddim
"""


def masked_mean(values, mask, eps=1e-8):
    """掩码均值：仅在 mask=1 的位置统计。"""
    mask = mask.float()
    denom = mask.sum().clamp(min=eps)
    return (values * mask).sum() / denom


def masked_mean_weighted(values, mask, sample_weights, eps=1e-8):
    """带样本权重的掩码均值（用于组别重加权训练）。"""
    # sample_weights: [B], values/mask: [B, 1, L]
    w = sample_weights.view(-1, 1, 1).float()
    wm = mask.float() * w
    denom = wm.sum().clamp(min=eps)
    return (values * wm).sum() / denom


class TimeEmbedding(nn.Module):
    """标准正余弦时间步嵌入（扩散步 t -> embedding）。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(0, half).float() / max(half - 1, 1))
        self.register_buffer("freq", freq)

    def forward(self, t):
        t = t.float().unsqueeze(1)
        emb = t * self.freq.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class ConditionEncoder(nn.Module):
    """将多源条件信息编码为统一 cond 向量。"""

    def __init__(self, cond_dim=128, n_groups=3, life_dim=14):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.early_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.life_net = nn.Sequential(
            nn.Linear(life_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
        )
        self.group_emb = nn.Embedding(n_groups, 32)
        self.fuse = nn.Sequential(
            nn.Linear(32 + 32 + 32 + 32, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, feature_matrix, early_soh, life_features, group_ids):
        feat = self.feature_net(feature_matrix).flatten(1)
        early = self.early_net(early_soh).flatten(1)
        life = self.life_net(life_features)
        g = self.group_emb(group_ids)
        cond = self.fuse(torch.cat([feat, early, life, g], dim=1))
        return cond


class GroupClassifier(nn.Module):
    """基于同一条件编码器的分组分类头。"""

    def __init__(self, cond_dim=128, n_groups=3, life_dim=14):
        super().__init__()
        self.encoder = ConditionEncoder(cond_dim=cond_dim, n_groups=n_groups, life_dim=life_dim)
        self.head = nn.Sequential(
            nn.Linear(cond_dim, cond_dim // 2),
            nn.SiLU(),
            nn.Linear(cond_dim // 2, n_groups),
        )

    def forward(self, feature_matrix, early_soh, life_features, group_ids_hint):
        cond = self.encoder(feature_matrix, early_soh, life_features, group_ids_hint)
        logits = self.head(cond)
        return logits


class ResBlock1D(nn.Module):
    """1D 残差块（条件通过 scale/shift 注入）。"""

    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch * 2)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale_shift = self.emb(F.silu(emb)).unsqueeze(-1)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class Downsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ConditionalUNet1D(nn.Module):
    """扩散去噪 UNet（1D 序列版）。"""

    def __init__(self, cond_dim=128, base_ch=32):
        super().__init__()
        self.time_emb = TimeEmbedding(cond_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.in_conv = nn.Conv1d(1, base_ch, kernel_size=7, padding=3)

        self.down1 = ResBlock1D(base_ch, base_ch * 2, cond_dim)
        self.ds1 = Downsample1D(base_ch * 2)
        self.down2 = ResBlock1D(base_ch * 2, base_ch * 4, cond_dim)
        self.ds2 = Downsample1D(base_ch * 4)

        self.mid1 = ResBlock1D(base_ch * 4, base_ch * 4, cond_dim)
        self.mid2 = ResBlock1D(base_ch * 4, base_ch * 4, cond_dim)

        self.up2 = Upsample1D(base_ch * 4)
        self.res_up2 = ResBlock1D(base_ch * 8, base_ch * 2, cond_dim)
        self.up1 = Upsample1D(base_ch * 2)
        self.res_up1 = ResBlock1D(base_ch * 4, base_ch, cond_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv1d(base_ch, 1, kernel_size=3, padding=1)

    def forward(self, x, t, cond):
        t_emb = self.time_mlp(self.time_emb(t))
        emb = t_emb + cond

        x0 = self.in_conv(x)
        d1 = self.down1(x0, emb)
        d1_ds = self.ds1(d1)
        d2 = self.down2(d1_ds, emb)
        d2_ds = self.ds2(d2)

        mid = self.mid1(d2_ds, emb)
        mid = self.mid2(mid, emb)

        u2 = self.up2(mid)
        if u2.shape[-1] != d2.shape[-1]:
            u2 = F.interpolate(u2, size=d2.shape[-1], mode="nearest")
        u2 = self.res_up2(torch.cat([u2, d2], dim=1), emb)

        u1 = self.up1(u2)
        if u1.shape[-1] != d1.shape[-1]:
            u1 = F.interpolate(u1, size=d1.shape[-1], mode="nearest")
        u1 = self.res_up1(torch.cat([u1, d1], dim=1), emb)

        out = self.out_conv(F.silu(self.out_norm(u1)))
        return out


class ConditionalTransformer1D(nn.Module):
    """扩散去噪 Transformer（1D 序列版）。"""

    def __init__(
        self,
        cond_dim=128,
        seq_len=432,
        model_dim=192,
        depth=4,
        nhead=8,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.model_dim = int(model_dim)
        self.time_emb = TimeEmbedding(cond_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.in_proj = nn.Conv1d(1, self.model_dim, kernel_size=1)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, self.seq_len, self.model_dim) * 0.01)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(nhead),
            dim_feedforward=self.model_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(depth))
        self.out_norm = nn.LayerNorm(self.model_dim)
        self.out_proj = nn.Linear(self.model_dim, 1)

    def _get_pos_emb(self, length):
        if length == self.pos_emb.shape[1]:
            return self.pos_emb
        if length < self.pos_emb.shape[1]:
            return self.pos_emb[:, :length, :]
        pe = F.interpolate(self.pos_emb.transpose(1, 2), size=length, mode="linear", align_corners=False)
        return pe.transpose(1, 2)

    def forward(self, x, t, cond):
        # x: [B, 1, L], t: [B], cond: [B, C]
        h = self.in_proj(x).transpose(1, 2)  # [B, L, D]
        length = h.shape[1]
        pos = self._get_pos_emb(length)

        temb = self.time_mlp(self.time_emb(t))
        c = self.cond_proj(cond + temb).unsqueeze(1)  # [B, 1, D]
        h = h + pos + c
        h = self.encoder(h)
        out = self.out_proj(self.out_norm(h)).transpose(1, 2)  # [B, 1, L]
        return out


class DiffusionSchedule:
    """扩散前向/反向过程所需的 alpha/beta 调度与提取函数。"""

    def __init__(self, timesteps=200, device="cpu"):
        self.timesteps = int(timesteps)
        betas = torch.linspace(1e-4, 0.02, self.timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.alpha_bar_prev = alpha_bar_prev
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        self.sqrt_recip_alpha_bar = torch.sqrt(1.0 / alpha_bar)
        self.sqrt_recipm1_alpha_bar = torch.sqrt(1.0 / alpha_bar - 1.0)

    def extract(self, arr, t, x_shape):
        out = arr.gather(0, t)
        return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))

    def q_sample(self, x0, t, noise):
        return self.extract(self.sqrt_alpha_bar, t, x0.shape) * x0 + self.extract(
            self.sqrt_one_minus_alpha_bar, t, x0.shape
        ) * noise

    def predict_x0(self, x_t, t, pred_noise):
        return self.extract(self.sqrt_recip_alpha_bar, t, x_t.shape) * x_t - self.extract(
            self.sqrt_recipm1_alpha_bar, t, x_t.shape
        ) * pred_noise


class ConditionalDiffusionModel(nn.Module):
    """
    Code2 主模型：
    - 条件编码 + 确定性轨迹分支 + 扩散残差分支
    - 统一在“非负退化增量”空间建模
    - 带 EOL 与 knee 辅助任务约束
    """

    def __init__(
        self,
        n_groups=3,
        cond_dim=128,
        life_dim=14,
        seq_len=432,
        timesteps=400,
        raw_mean=0.0,
        raw_std=1.0,
        delta_scale=1.0,
        denoiser_type="unet",
        cfg_dropout=0.1,
        accel_weight=0.14,
        smooth_weight=0.04,
        short_accel_weight=0.12,
        long_accel_weight=0.10,
        curve_eol_weight=0.08,
        eol_consistency_weight=0.06,
        knee_weight=0.06,
        group_loss_weights=(1.0, 1.0, 1.8),
        short_group_extra_weight=0.0,
        short_hard_weight=0.0,
        short_censored_weight=1.0,
        short_censored_use_observed_mask=False,
        short_use_observed_mask=False,
        det_rank=48,
        short_rank=24,
        long_rank=24,
        tail_supervision_weights=(0.0, 0.0, 0.0),
        tail_censored_scale=0.0,
        diffusion_target_mode="residual",
        use_group_experts=True,
    ):
        super().__init__()
        self.cond_encoder = ConditionEncoder(cond_dim=cond_dim, n_groups=n_groups, life_dim=life_dim)
        self.denoiser_type = str(denoiser_type).strip().lower()
        if self.denoiser_type == "transformer":
            self.transformer_denoiser = ConditionalTransformer1D(
                cond_dim=cond_dim,
                seq_len=int(seq_len),
                model_dim=192,
                depth=4,
                nhead=8,
                dropout=0.10,
            )
            self.unet = None
        else:
            self.unet = ConditionalUNet1D(cond_dim=cond_dim, base_ch=32)
            self.transformer_denoiser = None
        self.schedule = DiffusionSchedule(timesteps=timesteps, device="cpu")
        self.timesteps = int(timesteps)
        self.cfg_dropout = float(cfg_dropout)
        self.seq_len = int(seq_len)
        self.det_rank = int(det_rank)
        self.short_rank = int(short_rank)
        self.long_rank = int(long_rank)
        # Deterministic branch smoothing kernel (legacy checkpoints use 5).
        self.det_smooth_kernel = 3

        # Stage-1 deterministic trajectory head.
        self.det_coeff = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.SiLU(),
            nn.Linear(128, self.det_rank),
        )
        self.det_basis = nn.Parameter(torch.randn(self.det_rank, self.seq_len) * 0.02)
        self.group_template = nn.Embedding(n_groups, self.seq_len)
        # Short-life expert branch: captures steep degradation diversity in short group.
        self.short_det_coeff = nn.Sequential(
            nn.Linear(cond_dim, 96),
            nn.SiLU(),
            nn.Linear(96, self.short_rank),
        )
        self.short_det_basis = nn.Parameter(torch.randn(self.short_rank, self.seq_len) * 0.02)
        self.short_gate = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        # Long-life expert branch: only activated on long-life samples.
        self.long_det_coeff = nn.Sequential(
            nn.Linear(cond_dim, 96),
            nn.SiLU(),
            nn.Linear(96, self.long_rank),
        )
        self.long_det_basis = nn.Parameter(torch.randn(self.long_rank, self.seq_len) * 0.02)
        self.long_gate = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # EOL auxiliary head.
        self.eol_head = nn.Sequential(
            nn.Linear(cond_dim, cond_dim // 2),
            nn.SiLU(),
            nn.Linear(cond_dim // 2, cond_dim // 2),
            nn.SiLU(),
        )
        self.eol_cls = nn.Linear(cond_dim // 2, 1)
        self.eol_reg = nn.Linear(cond_dim // 2, 1)
        self.knee_reg = nn.Linear(cond_dim // 2, 1)

        self.raw_mean = float(raw_mean)
        self.raw_std = float(raw_std)
        self.delta_scale = float(delta_scale)
        self.accel_weight = float(accel_weight)
        self.smooth_weight = float(smooth_weight)
        self.short_accel_weight = float(short_accel_weight)
        self.long_accel_weight = float(long_accel_weight)
        self.curve_eol_weight = float(curve_eol_weight)
        self.eol_consistency_weight = float(eol_consistency_weight)
        self.knee_weight = float(knee_weight)
        self.short_group_extra_weight = float(short_group_extra_weight)
        self.short_hard_weight = float(short_hard_weight)
        self.short_censored_weight = float(short_censored_weight)
        self.short_censored_use_observed_mask = bool(short_censored_use_observed_mask)
        self.short_use_observed_mask = bool(short_use_observed_mask)
        self.register_buffer("group_loss_weights", torch.tensor(group_loss_weights, dtype=torch.float32))
        self.register_buffer("tail_supervision_weights", torch.tensor(tail_supervision_weights, dtype=torch.float32))
        self.tail_censored_scale = float(tail_censored_scale)
        mode = str(diffusion_target_mode).strip().lower()
        if mode not in {"residual", "direct"}:
            raise ValueError(f"Unsupported diffusion_target_mode: {diffusion_target_mode}")
        self.diffusion_target_mode = mode
        self.use_group_experts = bool(use_group_experts)

    def set_schedule_device(self, device):
        """将扩散调度器缓存迁移到指定设备。"""
        self.schedule = DiffusionSchedule(timesteps=self.timesteps, device=device)

    def _denoise(self, x_t, t, cond):
        """统一调用当前去噪器（UNet 或 Transformer）。"""
        if self.denoiser_type == "transformer":
            return self.transformer_denoiser(x_t, t, cond)
        return self.unet(x_t, t, cond)

    def _sample_weights(self, group_ids):
        """根据组别返回样本权重。"""
        g = group_ids.long().clamp(min=0, max=self.group_loss_weights.numel() - 1)
        return self.group_loss_weights[g]

    def _focus_sample_weights(self, sample_weights, group_ids, eol_exists, eol_fraction):
        """
        短组优先权重：
        - 对所有短组样本增加基础权重；
        - 对“短组中的晚退化样本”再增重，缓解短组内部平均化。
        """
        short_mask = (group_ids == 0).float()
        hard = torch.clamp((eol_fraction.float() - 0.55) / 0.35, min=0.0, max=1.0)
        # 删失样本也保留一定 hard 权重，避免只学“很快掉到 80”的短组样本。
        hard = torch.where(eol_exists.float() > 0.5, hard, torch.full_like(hard, 0.35))
        gain = 1.0 + short_mask * (self.short_group_extra_weight + self.short_hard_weight * hard)
        w = sample_weights * gain
        if self.short_censored_weight < 0.999:
            short_censored = short_mask * (eol_exists.float() < 0.5).float()
            w = w * (1.0 - short_censored + short_censored * float(self.short_censored_weight))
        return w

    def _effective_valid_mask(self, valid_mask, observed_mask, group_ids, eol_exists):
        """
        对短组删失样本可选仅用已观测窗口做形状约束，避免把不确定尾部当作真值强监督。
        """
        if self.short_use_observed_mask:
            short_mask = (group_ids == 0).float().view(-1, 1, 1)
            return valid_mask * (1.0 - short_mask) + observed_mask * short_mask
        if not self.short_censored_use_observed_mask:
            return valid_mask
        short_censored = ((group_ids == 0) & (eol_exists.float() < 0.5)).float().view(-1, 1, 1)
        return valid_mask * (1.0 - short_censored) + observed_mask * short_censored

    def _reconstruction_mask(self, observed_mask, valid_mask, group_ids, eol_exists):
        """
        Main supervision mask for rec/noise losses.
        - Default: observed_mask only (legacy behavior).
        - Optional: add weak tail supervision on (valid_mask - observed_mask),
          with per-group weights and reduced factor for censored samples.
        """
        base = observed_mask.float()
        tail = (valid_mask.float() - base).clamp(min=0.0)
        if float(self.tail_supervision_weights.abs().sum().item()) <= 1e-8:
            return base
        g = group_ids.long().clamp(min=0, max=self.tail_supervision_weights.numel() - 1)
        w = self.tail_supervision_weights[g].view(-1, 1, 1).float()
        if self.tail_censored_scale >= 0.0:
            s = torch.where(
                eol_exists.float().view(-1, 1, 1) > 0.5,
                torch.ones_like(w),
                torch.full_like(w, float(self.tail_censored_scale)),
            )
            w = w * s
        return base + tail * w

    def predict_deterministic_x(self, cond, group_ids):
        """
        预测确定性轨迹（x_norm 域）。
        包含：
        - 共享低秩基底
        - group template
        - long-life expert 分支（仅长寿命组激活）
        """
        coeff = self.det_coeff(cond)
        base = torch.matmul(coeff, self.det_basis)
        if not self.use_group_experts:
            x = base
            k = int(max(1, self.det_smooth_kernel))
            if k % 2 == 0:
                k += 1
            x = F.avg_pool1d(x.unsqueeze(1), kernel_size=k, stride=1, padding=k // 2)
            return torch.clamp(x, min=-6.0, max=3.0)

        g = self.group_template(group_ids)
        short_coeff = self.short_det_coeff(cond)
        short_component = torch.matmul(short_coeff, self.short_det_basis)
        short_indicator = (group_ids == 0).float().unsqueeze(1)
        short_gate = torch.sigmoid(self.short_gate(cond)) * short_indicator
        long_coeff = self.long_det_coeff(cond)
        long_component = torch.matmul(long_coeff, self.long_det_basis)
        long_indicator = (group_ids == 2).float().unsqueeze(1)
        long_gate = torch.sigmoid(self.long_gate(cond)) * long_indicator

        x = base + g + short_gate * short_component + long_gate * long_component
        # Keep only light smoothing for new models; legacy checkpoints can override kernel.
        k = int(max(1, self.det_smooth_kernel))
        if k % 2 == 0:
            k += 1
        x = F.avg_pool1d(x.unsqueeze(1), kernel_size=k, stride=1, padding=k // 2)
        return torch.clamp(x, min=-6.0, max=3.0)

    def decode_delta(self, x_norm):
        """
        x_norm -> 非负退化增量 delta。
        softplus 保证增量非负（物理约束核心）。
        """
        raw = x_norm * self.raw_std + self.raw_mean
        raw = torch.clamp(raw, min=-12.0, max=2.0)
        delta = F.softplus(raw) * self.delta_scale
        delta = torch.clamp(delta, min=0.0, max=1.8)
        return delta

    def build_future_curve(self, x_norm, start_soh_100):
        """由增量累计构建未来 SOH 曲线。"""
        delta = self.decode_delta(x_norm)
        cum = torch.cumsum(delta, dim=-1)
        future_soh = start_soh_100.view(-1, 1, 1) - cum
        return torch.clamp(future_soh, min=0.0, max=100.0), delta

    def derive_curve_eol(self, future_soh, target_lengths):
        """从预测曲线直接推导 EOL 存在性与相对位置。"""
        cross = (future_soh <= EOL_THRESHOLD).float()
        has_cross = cross.max(dim=-1).values.squeeze(1)
        first_idx = torch.argmax(cross, dim=-1).float().squeeze(1)
        pred_cycle = 100.0 + (first_idx + 1.0) * 5.0
        pred_cycle = torch.where(has_cross > 0.5, pred_cycle, target_lengths.float().view(-1))
        pred_frac = (pred_cycle / target_lengths.float().clamp(min=1.0)).clamp(0.0, 1.0)
        return has_cross, pred_frac

    def derive_knee_fraction(self, x_norm, valid_mask, target_lengths):
        """估计 knee 位置（加速退化拐点）相对比例。"""
        # x_norm -> decoded degradation increments.
        delta = self.decode_delta(x_norm)
        delta_step = delta[:, :, 1:] - delta[:, :, :-1]
        valid_pair = valid_mask[:, :, 1:] * valid_mask[:, :, :-1]
        bsz, _, length = delta_step.shape
        if length <= 2:
            frac = torch.full((bsz,), 0.5, device=x_norm.device)
            valid = torch.zeros((bsz,), device=x_norm.device)
            return frac, valid

        # Focus on middle-late phase to capture accelerated degradation knee.
        start = max(4, int(0.25 * length))
        win = torch.zeros_like(delta_step)
        win[:, :, start:] = 1.0
        score_mask = valid_pair * win
        score = delta_step.clone()
        score = torch.where(score_mask > 0.5, score, torch.full_like(score, -1e6))
        idx = torch.argmax(score, dim=-1).squeeze(1).float()

        cycle = 100.0 + (idx + 1.0) * 5.0
        frac = (cycle / target_lengths.float().clamp(min=1.0)).clamp(0.0, 1.0)
        has_valid = (score_mask.sum(dim=-1).squeeze(1) > 0).float()
        fallback = torch.full_like(frac, 0.5)
        frac = torch.where(has_valid > 0.5, frac, fallback)
        return frac, has_valid

    def _shape_losses(self, delta, valid_mask, sample_weights, group_ids, eol_exists=None, eol_fraction=None):
        """
        形状约束损失：
        - accel_loss: 抑制不合理的“减速”（鼓励中后期退化加速）
        - smooth_loss: 抑制高频抖动
        - short_accel_loss: 短寿命组按样本 EOL 位置自适应的加速约束
        - long_accel_loss: 对长寿命组额外强化中后期加速约束
        """
        delta_step = delta[:, :, 1:] - delta[:, :, :-1]
        valid_pair_mask = valid_mask[:, :, 1:] * valid_mask[:, :, :-1]
        progress = torch.linspace(0.0, 1.0, delta_step.shape[-1], device=delta.device).view(1, 1, -1)
        accel_gate = torch.sigmoid((progress - 0.35) / 0.10)
        accel_loss = masked_mean_weighted(F.relu(-delta_step) * accel_gate, valid_pair_mask, sample_weights)

        second_diff = delta_step[:, :, 1:] - delta_step[:, :, :-1]
        valid_second_mask = valid_pair_mask[:, :, 1:] * valid_pair_mask[:, :, :-1]
        smooth_loss = masked_mean_weighted(second_diff.pow(2), valid_second_mask, sample_weights)

        short_mask = (group_ids.view(-1, 1, 1) == 0).float()
        if (eol_exists is not None) and (eol_fraction is not None):
            eol_pos = (eol_exists.view(-1, 1, 1) > 0.5).float()
            frac = eol_fraction.view(-1, 1, 1).clamp(0.0, 1.0)
            # Earlier-EOL samples should accelerate earlier; later-EOL samples later.
            knee_prog = torch.clamp(frac - 0.22, min=0.18, max=0.78)
            short_gate = torch.sigmoid((progress - knee_prog) / 0.07)
        else:
            eol_pos = torch.ones_like(short_mask)
            short_gate = torch.sigmoid((progress - 0.35) / 0.09)
        target_margin_short = 0.0025 + 0.0075 * progress
        short_accel_penalty = F.relu(target_margin_short - delta_step)
        short_accel_loss = masked_mean_weighted(
            short_accel_penalty * short_gate * short_mask * eol_pos,
            valid_pair_mask,
            sample_weights,
        )

        long_mask = (group_ids.view(-1, 1, 1) == 2).float()
        long_gate = torch.sigmoid((progress - 0.45) / 0.09)
        # Encourage accelerating degradation in middle/late phase for long-life cells.
        target_margin = 0.0035 + 0.0075 * progress
        long_accel_penalty = F.relu(target_margin - delta_step)
        long_accel_loss = masked_mean_weighted(
            long_accel_penalty * long_gate * long_mask,
            valid_pair_mask,
            sample_weights,
        )
        return accel_loss, smooth_loss, short_accel_loss, long_accel_loss

    def _eol_losses(
        self,
        cond,
        future_soh,
        target_lengths,
        eol_exists,
        eol_fraction,
        censor_fraction,
        sample_weights,
        x_ref,
        valid_mask,
    ):
        """
        EOL/knee 多任务损失集合：
        - EOL 是否存在（二分类）
        - EOL 相对位置（回归）
        - curve-derived EOL 与 head 预测一致性
        - knee 位置回归
        """
        device = cond.device
        w = sample_weights.view(-1).float()
        hidden = self.eol_head(cond)
        logits = self.eol_cls(hidden).squeeze(-1)
        pred_prob = torch.sigmoid(logits)
        pred_frac = torch.sigmoid(self.eol_reg(hidden).squeeze(-1))
        pred_knee = torch.sigmoid(self.knee_reg(hidden).squeeze(-1))

        eol_exists = eol_exists.view(-1).float()
        eol_fraction = eol_fraction.view(-1).float().clamp(0.0, 1.0)
        censor_fraction = censor_fraction.view(-1).float().clamp(0.0, 1.0)

        positive = eol_exists > 0.5
        clearly_negative = (~positive) & (censor_fraction >= 0.98)
        useful = positive | clearly_negative

        if useful.any():
            cls_target = torch.where(positive, torch.ones_like(pred_prob), torch.zeros_like(pred_prob))
            cls_weight = torch.where(positive, torch.full_like(pred_prob, 1.0), torch.full_like(pred_prob, 0.5))
            cls_loss_raw = F.binary_cross_entropy(pred_prob, cls_target, reduction="none")
            cls_w = cls_weight * useful.float() * w
            eol_cls_loss = (cls_loss_raw * cls_w).sum() / (
                cls_w.sum().clamp(min=1e-8)
            )
        else:
            eol_cls_loss = torch.zeros((), device=device)

        if positive.any():
            pw = w[positive]
            eol_reg_loss = (((pred_frac[positive] - eol_fraction[positive]) ** 2) * pw).sum() / pw.sum().clamp(min=1e-8)
        else:
            eol_reg_loss = torch.zeros((), device=device)

        censored = ~positive
        if censored.any():
            margin = 0.03
            cw = w[censored]
            eol_censor_loss = (F.relu(censor_fraction[censored] + margin - pred_frac[censored]).pow(2) * cw).sum() / cw.sum().clamp(min=1e-8)
        else:
            eol_censor_loss = torch.zeros((), device=device)

        curve_cross, curve_frac = self.derive_curve_eol(future_soh, target_lengths)
        if useful.any():
            curve_target = torch.where(positive, torch.ones_like(curve_cross), torch.zeros_like(curve_cross))
            curve_weight = torch.where(positive, torch.full_like(curve_cross, 1.0), torch.full_like(curve_cross, 0.5))
            curve_cls_raw = F.binary_cross_entropy(curve_cross, curve_target, reduction="none")
            cw = curve_weight * useful.float() * w
            curve_cls_loss = (curve_cls_raw * cw).sum() / (
                cw.sum().clamp(min=1e-8)
            )
        else:
            curve_cls_loss = torch.zeros((), device=device)

        if positive.any():
            pw = w[positive]
            curve_reg_loss = (((curve_frac[positive] - eol_fraction[positive]) ** 2) * pw).sum() / pw.sum().clamp(min=1e-8)
        else:
            curve_reg_loss = torch.zeros((), device=device)

        if censored.any():
            margin = 0.03
            cw = w[censored]
            curve_censor_loss = (F.relu(censor_fraction[censored] + margin - curve_frac[censored]).pow(2) * cw).sum() / cw.sum().clamp(min=1e-8)
        else:
            curve_censor_loss = torch.zeros((), device=device)

        consistency = ((pred_prob - curve_cross).pow(2) * w).sum() / w.sum().clamp(min=1e-8)
        consistency = consistency + ((pred_frac - curve_frac).pow(2) * w).sum() / w.sum().clamp(min=1e-8)

        knee_tgt, knee_valid = self.derive_knee_fraction(x_ref, valid_mask, target_lengths)
        if knee_valid.any():
            kw = w * knee_valid
            knee_loss = (((pred_knee - knee_tgt) ** 2) * kw).sum() / kw.sum().clamp(min=1e-8)
        else:
            knee_loss = torch.zeros((), device=device)

        losses = {
            "eol_cls_loss": eol_cls_loss,
            "eol_reg_loss": eol_reg_loss,
            "eol_censor_loss": eol_censor_loss,
            "curve_cls_loss": curve_cls_loss,
            "curve_reg_loss": curve_reg_loss,
            "curve_censor_loss": curve_censor_loss,
            "consistency_loss": consistency,
            "knee_loss": knee_loss,
        }
        aux = {"eol_prob": pred_prob.detach(), "eol_fraction": pred_frac.detach(), "knee_fraction": pred_knee.detach()}
        return losses, aux

    def forward_deterministic(
        self,
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
    ):
        """Stage-1 前向：仅确定性分支训练。"""
        cond = self.cond_encoder(feature_matrix, early_soh, life_features, group_ids)
        sample_weights = self._sample_weights(group_ids)
        sample_weights = self._focus_sample_weights(sample_weights, group_ids, eol_exists, eol_fraction)
        rec_mask = self._reconstruction_mask(observed_mask, valid_mask, group_ids, eol_exists)
        eff_valid_mask = self._effective_valid_mask(valid_mask, observed_mask, group_ids, eol_exists)
        det_x = self.predict_deterministic_x(cond, group_ids)
        rec_loss = masked_mean_weighted((det_x - x_start).pow(2), rec_mask, sample_weights)

        future_soh, delta = self.build_future_curve(det_x, start_soh_100)
        accel_loss, smooth_loss, short_accel_loss, long_accel_loss = self._shape_losses(
            delta,
            eff_valid_mask,
            sample_weights,
            group_ids,
            eol_exists=eol_exists,
            eol_fraction=eol_fraction,
        )
        eol_losses, aux = self._eol_losses(
            cond=cond,
            future_soh=future_soh,
            target_lengths=target_lengths,
            eol_exists=eol_exists,
            eol_fraction=eol_fraction,
            censor_fraction=censor_fraction,
            sample_weights=sample_weights,
            x_ref=x_start,
            valid_mask=eff_valid_mask,
        )

        total = (
            rec_loss
            + self.accel_weight * accel_loss
            + self.smooth_weight * smooth_loss
            + self.short_accel_weight * short_accel_loss
            + self.long_accel_weight * long_accel_loss
            + 0.10 * eol_losses["eol_cls_loss"]
            + 0.10 * eol_losses["eol_reg_loss"]
            + self.curve_eol_weight
            * (eol_losses["curve_cls_loss"] + eol_losses["curve_reg_loss"] + eol_losses["curve_censor_loss"])
            + 0.08 * eol_losses["eol_censor_loss"]
            + self.eol_consistency_weight * eol_losses["consistency_loss"]
            + self.knee_weight * eol_losses["knee_loss"]
        )
        metrics = {
            "total_loss": float(total.detach().item()),
            "rec_loss": float(rec_loss.detach().item()),
            "accel_loss": float(accel_loss.detach().item()),
            "smooth_loss": float(smooth_loss.detach().item()),
            "short_accel_loss": float(short_accel_loss.detach().item()),
            "long_accel_loss": float(long_accel_loss.detach().item()),
            "eol_cls_loss": float(eol_losses["eol_cls_loss"].detach().item()),
            "eol_reg_loss": float(eol_losses["eol_reg_loss"].detach().item()),
            "eol_censor_loss": float(eol_losses["eol_censor_loss"].detach().item()),
            "curve_cls_loss": float(eol_losses["curve_cls_loss"].detach().item()),
            "curve_reg_loss": float(eol_losses["curve_reg_loss"].detach().item()),
            "curve_censor_loss": float(eol_losses["curve_censor_loss"].detach().item()),
            "consistency_loss": float(eol_losses["consistency_loss"].detach().item()),
            "knee_loss": float(eol_losses["knee_loss"].detach().item()),
        }
        aux["det_x"] = det_x.detach()
        return total, metrics, aux

    def forward_train(
        self,
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
    ):
        """Stage-2 前向：扩散残差训练（带 CFG dropout）。"""
        device = x_start.device
        if self.schedule.betas.device != device:
            self.set_schedule_device(device)

        bsz = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (bsz,), device=device).long()

        cond = self.cond_encoder(feature_matrix, early_soh, life_features, group_ids)
        sample_weights = self._sample_weights(group_ids)
        sample_weights = self._focus_sample_weights(sample_weights, group_ids, eol_exists, eol_fraction)
        rec_mask = self._reconstruction_mask(observed_mask, valid_mask, group_ids, eol_exists)
        eff_valid_mask = self._effective_valid_mask(valid_mask, observed_mask, group_ids, eol_exists)
        det_x = self.predict_deterministic_x(cond, group_ids)
        if self.diffusion_target_mode == "direct":
            target_x = x_start
        else:
            target_x = x_start - det_x

        noise = torch.randn_like(target_x)
        x_t = self.schedule.q_sample(target_x, t, noise)

        if self.training and self.cfg_dropout > 0:
            drop_mask = (torch.rand(bsz, device=device) < self.cfg_dropout).float().view(-1, 1)
            cond_noise = cond * (1.0 - drop_mask)
        else:
            cond_noise = cond

        pred_noise = self._denoise(x_t, t, cond_noise)
        noise_loss = masked_mean_weighted((pred_noise - noise).pow(2), rec_mask, sample_weights)

        x0_base = self.schedule.predict_x0(x_t, t, pred_noise)
        x0_base = torch.clamp(x0_base, min=-4.0, max=4.0)
        if self.diffusion_target_mode == "direct":
            x0_pred = torch.clamp(x0_base, min=-6.0, max=3.0)
            det_anchor_weight = 0.0
        else:
            x0_pred = torch.clamp(det_x + x0_base, min=-6.0, max=3.0)
            det_anchor_weight = 0.25
        x0_loss = masked_mean_weighted((x0_pred - x_start).pow(2), rec_mask, sample_weights)
        det_anchor = masked_mean_weighted((det_x - x_start).pow(2), rec_mask, sample_weights)

        future_soh, delta = self.build_future_curve(x0_pred, start_soh_100)
        accel_loss, smooth_loss, short_accel_loss, long_accel_loss = self._shape_losses(
            delta,
            eff_valid_mask,
            sample_weights,
            group_ids,
            eol_exists=eol_exists,
            eol_fraction=eol_fraction,
        )
        eol_losses, aux = self._eol_losses(
            cond=cond,
            future_soh=future_soh,
            target_lengths=target_lengths,
            eol_exists=eol_exists,
            eol_fraction=eol_fraction,
            censor_fraction=censor_fraction,
            sample_weights=sample_weights,
            x_ref=x_start,
            valid_mask=eff_valid_mask,
        )

        total_loss = (
            noise_loss
            + 0.70 * x0_loss
            + det_anchor_weight * det_anchor
            + self.accel_weight * accel_loss
            + self.smooth_weight * smooth_loss
            + self.short_accel_weight * short_accel_loss
            + self.long_accel_weight * long_accel_loss
            + 0.08 * eol_losses["eol_cls_loss"]
            + 0.08 * eol_losses["eol_reg_loss"]
            + self.curve_eol_weight
            * (eol_losses["curve_cls_loss"] + eol_losses["curve_reg_loss"] + eol_losses["curve_censor_loss"])
            + 0.08 * eol_losses["eol_censor_loss"]
            + self.eol_consistency_weight * eol_losses["consistency_loss"]
            + self.knee_weight * eol_losses["knee_loss"]
        )
        metrics = {
            "total_loss": float(total_loss.detach().item()),
            "noise_loss": float(noise_loss.detach().item()),
            "x0_loss": float(x0_loss.detach().item()),
            "det_anchor": float(det_anchor.detach().item()),
            "accel_loss": float(accel_loss.detach().item()),
            "smooth_loss": float(smooth_loss.detach().item()),
            "short_accel_loss": float(short_accel_loss.detach().item()),
            "long_accel_loss": float(long_accel_loss.detach().item()),
            "eol_cls_loss": float(eol_losses["eol_cls_loss"].detach().item()),
            "eol_reg_loss": float(eol_losses["eol_reg_loss"].detach().item()),
            "eol_censor_loss": float(eol_losses["eol_censor_loss"].detach().item()),
            "curve_cls_loss": float(eol_losses["curve_cls_loss"].detach().item()),
            "curve_reg_loss": float(eol_losses["curve_reg_loss"].detach().item()),
            "curve_censor_loss": float(eol_losses["curve_censor_loss"].detach().item()),
            "consistency_loss": float(eol_losses["consistency_loss"].detach().item()),
            "knee_loss": float(eol_losses["knee_loss"].detach().item()),
        }
        aux["det_x"] = det_x.detach()
        return total_loss, metrics, aux

    @torch.no_grad()
    def predict_eol_head(self, feature_matrix, early_soh, life_features, group_ids):
        """推理时调用的 EOL/knee 辅助头。"""
        cond = self.cond_encoder(feature_matrix, early_soh, life_features, group_ids)
        hidden = self.eol_head(cond)
        logits = self.eol_cls(hidden).squeeze(-1)
        frac = torch.sigmoid(self.eol_reg(hidden).squeeze(-1))
        knee_frac = torch.sigmoid(self.knee_reg(hidden).squeeze(-1))
        return {"eol_prob": torch.sigmoid(logits), "eol_fraction": frac, "knee_fraction": knee_frac}

    @torch.no_grad()
    def sample_ddim(
        self,
        feature_matrix,
        early_soh,
        life_features,
        group_ids,
        start_soh_100,
        seq_len,
        n_runs=1,
        guidance_scale=1.2,
        ddim_steps=40,
    ):
        """
        DDIM 推理采样。
        输出 future_soh/future_delta，可在测试脚本中重建全长曲线并计算指标。
        """
        device = feature_matrix.device
        if self.schedule.betas.device != device:
            self.set_schedule_device(device)

        bsz = feature_matrix.shape[0]
        total = bsz * n_runs
        fm = feature_matrix.repeat_interleave(n_runs, dim=0)
        es = early_soh.repeat_interleave(n_runs, dim=0)
        lf = life_features.repeat_interleave(n_runs, dim=0)
        gid = group_ids.repeat_interleave(n_runs, dim=0)

        cond = self.cond_encoder(fm, es, lf, gid)
        det = self.predict_deterministic_x(cond, gid)
        uncond = torch.zeros_like(cond)

        x = torch.randn(total, 1, seq_len, device=device)
        steps = np.linspace(self.timesteps - 1, 0, ddim_steps, dtype=np.int64)

        for i, t_val in enumerate(steps):
            t = torch.full((total,), int(t_val), device=device, dtype=torch.long)
            eps_cond = self._denoise(x, t, cond)
            if guidance_scale > 0:
                eps_uncond = self._denoise(x, t, uncond)
                eps = (1.0 + guidance_scale) * eps_cond - guidance_scale * eps_uncond
            else:
                eps = eps_cond

            x0 = self.schedule.predict_x0(x, t, eps)
            x0 = torch.clamp(x0, min=-4.0, max=4.0)
            if i == len(steps) - 1:
                x = x0
                break
            t_prev = torch.full((total,), int(steps[i + 1]), device=device, dtype=torch.long)
            alpha_prev = self.schedule.extract(self.schedule.alpha_bar, t_prev, x.shape)
            x = torch.sqrt(alpha_prev) * x0 + torch.sqrt(torch.clamp(1.0 - alpha_prev, min=1e-8)) * eps

        if self.diffusion_target_mode == "direct":
            x_norm = torch.clamp(x, min=-6.0, max=3.0)
        else:
            x_norm = torch.clamp(det + x, min=-6.0, max=3.0)
        start_soh = start_soh_100.view(-1).repeat_interleave(n_runs)
        future_soh, delta = self.build_future_curve(x_norm, start_soh)
        return {
            "x0_norm": x_norm.view(total, seq_len),
            "future_delta": delta.view(total, seq_len),
            "future_soh": future_soh.view(total, seq_len),
            "det_x": det.view(total, seq_len),
        }
