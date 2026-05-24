"""Backbone probe: fixed head for benchmarking visual backbones.

Pipeline
--------
    image  [B, 3, H, W]
      ↓  backbone (variable — the thing being benchmarked)
    spatial features  [B, C, H', W']
      ↓  ProbeHead (fixed across all backbone variants)
         1. 1×1 conv projection  →  [B, D, H', W']
         2. sinusoidal 2D PE added at full resolution
         3. num_conv_layers × ConvNeXtBlock(D) with conditional AvgPool2d(2,2)
            (pool applied after a block only when max(H,W) > target_size)
         4. flatten → [B, H*W, D], add sinusoidal 2D PE
         5. num_attn_layers × TransformerBlock (pre-norm, MHSA + FFN)
         6. mean-pool over tokens  →  [B, D]
      ↓  PoseSlotDecoder (fixed)
         Linear(D, 1024) → ReLU → Linear(1024, N*(1 + K*2 + K))
    pred dict  {keypoints, scores, scores_logits, vis_logits, keypoint_scores}

The backbone is the only component that varies between benchmark variants.
ProbeHead and PoseSlotDecoder are constructed internally and kept identical
across all variants regardless of input spatial resolution.
"""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.embedders.spatial_attention import make_2d_sinusoidal_pe, make_pe_from_centers
from src.features.base import BaseFeatureExtractor
from src.training.augmentations import add_gaussian_noise, color_jitter
from src.metrics import CROWDPOSE_SIGMAS, KeypointAPMetric, canonical_to_pixel
from src.training.diagnostics import compute_probe_diagnostics
from src.training.losses import (
    SetPredictionLoss,
    SetPredictionLossWeights,
    compute_oks,
    hungarian_match,
)
from src.training.base_module import BaseTrainingModule
from src.training.optim import build_adamw_with_schedule
from src.vis import save_prediction_plot


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvNeXtBlock(nn.Module):
    """Depthwise ConvNeXt-style block.

    DWConv(7×7) → LayerNorm → Linear(D→ffn_mult*D) → GELU → Linear(ffn_mult*D→D) → Dropout + residual
    """

    def __init__(self, dim: int, ffn_mult: int = 4, dropout: float = 0.5) -> None:
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * ffn_mult)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(dim * ffn_mult, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dw_conv(x)              # [B, D, H, W]
        x = x.permute(0, 2, 3, 1)       # [B, H, W, D]
        x = self.norm(x)
        x = self.act(self.pw1(x))
        x = self.drop(x)
        x = self.pw2(x)
        x = x.permute(0, 3, 1, 2)       # [B, D, H, W]
        return x + residual


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: MHSA + Add, FFN + Add."""

    def __init__(self, dim: int, num_heads: int, ffn_mult: int = 4, attn_dropout: float = 0.0, ffn_dropout: float = 0.5) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# ProbeHead
# ---------------------------------------------------------------------------


class ConvProbe(nn.Module):
    """Self-contained spatial feature map → pose prediction dict.

    Two-stage projection funnels backbone channels to a narrow conv_dim, where
    the cheap ConvNeXt stack runs.  A readout_proj lifts to feature_dim before
    the Transformer blocks.  After mean-pool, a shared linear projection feeds
    three separate head linears (scores, keypoints, visibility).

        proj1(1×1): in_ch → proj_dim
        proj2(1×1): proj_dim → conv_dim
        + sinusoidal 2D PE at conv_dim
        num_conv_layers × ConvNeXtBlock(conv_dim, dropout=conv_dropout)
            [if max(H, W) > target_size: AvgPool2d(2,2)]
        readout_proj(1×1): conv_dim → feature_dim
        flatten → [B, H*W, feature_dim]
        + sinusoidal 2D PE at feature_dim
        num_attn_layers × TransformerBlock(feature_dim)
        mean-pool → [B, feature_dim]
        shared_proj: Linear(feature_dim → hidden_dim) + ReLU + Dropout(mlp_dropout)
        scores_head:    Linear(hidden_dim → num_slots)
        keypoints_head: Linear(hidden_dim → num_slots * num_keypoints * 2)
        vis_head:       Linear(hidden_dim → num_slots * num_keypoints)

    Returns a pred dict with keys:
        keypoints       [B, N, K, 2]  canonical (−1, 1)
        scores          [B, N]
        scores_logits   [B, N]
        keypoint_scores [B, N, K]
        vis_logits      [B, N, K]

    Args:
        in_channels:     backbone output channels (varies per variant).
        feature_dim:     readout/attention dimension D.  Must be divisible by 4.
        proj_dim:        intermediate wide projection (adapter width).
        conv_dim:        narrow working dimension for the ConvNeXt stack.
                         Must be divisible by 4.
        target_size:     spatial size below which no further pooling is applied.
        num_conv_layers: number of ConvNeXtBlocks. 0 skips them entirely.
        num_attn_layers: number of TransformerBlocks in the readout.
        num_heads:       attention heads (feature_dim must be divisible by num_heads).
        ffn_mult:        hidden expansion factor in both ConvNeXt and Transformer FFNs.
        conv_dropout:    dropout inside each ConvNeXtBlock.
        attn_dropout:    dropout inside MultiheadAttention.
        ffn_dropout:     dropout inside Transformer FFN.
        num_slots:       number of pose slots N.
        num_keypoints:   keypoints per slot K.
        hidden_dim:      shared projection width before the three head linears.
        mlp_dropout:     dropout after shared_proj.
        proj_dropout:    dropout inside shared_proj.
        feature_key:     key in the backbone feature dict to consume.
    """

    def __init__(
        self,
        in_channels: int,
        feature_dim: int = 256,
        proj_dim: int = 768,
        conv_dim: int = 96,
        target_size: int = 16,
        num_conv_layers: int = 5,
        num_attn_layers: int = 2,
        num_heads: int = 4,
        ffn_mult: int = 4,
        conv_dropout: float = 0.5,
        attn_dropout: float = 0.3,
        ffn_dropout: float = 0.5,
        num_slots: int = 20,
        num_keypoints: int = 14,
        hidden_dim: int = 256,
        proj_dropout: float = 0.2,
        feature_key: str = "spatial",
    ) -> None:
        super().__init__()
        if feature_dim % 4 != 0:
            raise ValueError(f"feature_dim must be divisible by 4; got {feature_dim}")
        if conv_dim % 4 != 0:
            raise ValueError(f"conv_dim must be divisible by 4; got {conv_dim}")
        if feature_dim % num_heads != 0:
            raise ValueError(f"feature_dim must be divisible by num_heads; got {feature_dim} % {num_heads}")

        self.feature_key = feature_key
        self.feature_dim = feature_dim
        self.conv_dim = conv_dim
        self.target_size = target_size
        self.num_slots = num_slots
        self.num_keypoints = num_keypoints

        self.proj1 = nn.Conv2d(in_channels, proj_dim, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.proj2 = nn.Conv2d(proj_dim, conv_dim, kernel_size=1)
        self.conv_layers = nn.ModuleList([
            ConvNeXtBlock(conv_dim, ffn_mult, conv_dropout) for _ in range(num_conv_layers)
        ])
        self.readout_proj = nn.Conv2d(conv_dim, feature_dim, kernel_size=1)
        self.attn_layers = nn.ModuleList([
            TransformerBlock(feature_dim, num_heads, ffn_mult, attn_dropout=attn_dropout, ffn_dropout=ffn_dropout)
            for _ in range(num_attn_layers)
        ])

        self.shared_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(proj_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.scores_head    = nn.Linear(hidden_dim, num_slots)
        self.keypoints_head = nn.Linear(hidden_dim, num_slots * num_keypoints * 2)
        self.vis_head       = nn.Linear(hidden_dim, num_slots * num_keypoints)

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        feat = features[self.feature_key]          # [B, C, H, W]
        x = self.proj2(self.act(self.proj1(feat))) # [B, conv_dim, H, W]
        _, _, H, W = x.shape
        pe = make_2d_sinusoidal_pe(H, W, self.conv_dim, x.device, x.dtype)
        x = x + pe.view(H, W, self.conv_dim).permute(2, 0, 1).unsqueeze(0)

        for block in self.conv_layers:
            x = block(x)
            _, _, H, W = x.shape
            if max(H, W) > self.target_size:
                x = F.avg_pool2d(x, 2)

        x = self.readout_proj(x)                   # [B, feature_dim, H, W]
        _, _, H, W = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)    # [B, H*W, feature_dim]
        pe = make_2d_sinusoidal_pe(H, W, self.feature_dim, x.device, x.dtype)
        tokens = tokens + pe.unsqueeze(0)

        for block in self.attn_layers:
            tokens = block(tokens)

        g = self.shared_proj(tokens.mean(dim=1))   # [B, hidden_dim]
        B = g.shape[0]
        N, K = self.num_slots, self.num_keypoints

        scores_logits = self.scores_head(g)                            # [B, N]
        kpt_logits    = self.keypoints_head(g).view(B, N, K, 2)       # [B, N, K, 2]
        vis_logits    = self.vis_head(g).view(B, N, K)                 # [B, N, K]

        return {
            "keypoints":       torch.sigmoid(kpt_logits) * 2.0 - 1.0,
            "scores":          torch.sigmoid(scores_logits),
            "scores_logits":   scores_logits,
            "keypoint_scores": torch.sigmoid(vis_logits),
            "vis_logits":      vis_logits,
        }


# ---------------------------------------------------------------------------
# CrossAttentionBlock
# ---------------------------------------------------------------------------


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block: Q attends to KV, then FFN + residual."""

    def __init__(self, dim: int, num_heads: int, ffn_mult: int = 2,
                 attn_dropout: float = 0.0, ffn_dropout: float = 0.1) -> None:
        super().__init__()
        self.q_norm  = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        kv_n = self.kv_norm(kv)
        attn_out, _ = self.attn(self.q_norm(q), kv_n, kv_n)
        q = q + attn_out
        q = q + self.ffn(self.norm2(q))
        return q


# ---------------------------------------------------------------------------
# QueryProbe
# ---------------------------------------------------------------------------


class QueryProbe(nn.Module):
    """Query-based probe: DETR-style cross-attention over backbone patch tokens.

    MLP projection funnels backbone channels to feature_dim.  Learned instance
    queries and joint queries are then applied in five steps:

        proj MLP: in_ch → proj_dim (GroupNorm+GELU+Dropout) → feature_dim
        flatten + sinusoidal 2D PE → patch_tokens [B, HW, D]

        Step A: instance_queries [B,N,D] cross-attend to patch_tokens [B,HW,D]
        Step B: instance_latents [B,N,D] self-attend (slot interaction)
        objectness_head: Linear(D→1) → scores [B,N]

        Step C: joint_queries [B*N,K,D] cross-attend to patch_tokens [B*N,HW,D]
        Step D: joint_latents [B*N,K,D] cross-attend to instance_latents [B*N,1,D]
        Step E: joint_latents [B*N,K,D] self-attend (joint interaction within slot)
        → reshape [B,N,K,D]
        vis_head:       Linear(D→1) → vis_logits  [B,N,K]
        keypoints_head: Linear(D→2) → keypoints   [B,N,K,2]

    Returns the same pred dict as ConvProbe — drop-in for BackboneProbeModule.

    Args:
        in_channels:   backbone output channels.
        feature_dim:   attention dimension D. Must be divisible by 4 and num_heads.
        proj_dim:      intermediate width of the MLP projection.
        num_slots:     number of instance slots N.
        num_keypoints: keypoints per slot K.
        num_heads:     attention heads.
        ffn_mult:      FFN hidden expansion factor.
        attn_dropout:  dropout inside MultiheadAttention.
        ffn_dropout:   dropout inside FFN.
        proj_dropout:  dropout inside the MLP projection.
        feature_key:   key in the backbone feature dict to consume.
    """

    def __init__(
        self,
        in_channels: int,
        feature_dim: int = 256,
        num_slots: int = 20,
        num_keypoints: int = 14,
        num_heads: int = 4,
        ffn_mult: int = 2,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        feature_key: str = "spatial",
        pe_scale: float = 1000.0,
        predict_visibility: bool = False,
    ) -> None:
        super().__init__()
        D = feature_dim

        if D % 4 != 0:
            raise ValueError(f"feature_dim must be divisible by 4; got {feature_dim}")
        if D % num_heads != 0:
            raise ValueError(f"feature_dim must be divisible by num_heads; got {feature_dim} % {num_heads}")

        self.feature_key   = feature_key
        self.feature_dim   = D
        self.pe_scale      = pe_scale
        self.num_slots     = num_slots
        self.num_keypoints = num_keypoints
        self.predict_visibility = predict_visibility

        self.proj = nn.Linear(in_channels, D)
        self.instance_queries    = nn.Embedding(num_slots, D)
        self.inst_cross_attn     = CrossAttentionBlock(D, num_heads, ffn_mult, attn_dropout, ffn_dropout)
        self.inst_self_attn      = TransformerBlock(D, num_heads, ffn_mult, attn_dropout=attn_dropout, ffn_dropout=ffn_dropout)

        self.objectness_head     = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(D // 2, 1),
        )

        self.joint_queries       = nn.Embedding(num_keypoints, D)

        self.joint_cross_inst    = CrossAttentionBlock(D, num_heads, ffn_mult, attn_dropout, ffn_dropout)
        self.joint_self_attn     = TransformerBlock(D, num_heads, ffn_mult, attn_dropout=attn_dropout, ffn_dropout=ffn_dropout)

        self.final_norm = nn.LayerNorm(D)
        self.vis_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.Dropout(proj_dropout),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(D // 2, 1),
        ) if predict_visibility else None
        self.keypoints_head      = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, 2),
        )

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        N, K, D = self.num_slots, self.num_keypoints, self.feature_dim

        feat = features[self.feature_key]
        centers = features["patch_centers"].reshape(-1, 2)         # [T/HW, 2]
        B = feat.shape[0]
        if feat.ndim == 3:
            tokens = self.proj(feat)                               # [B, T, D]
        else:
            tokens = self.proj(feat.permute(0, 2, 3, 1)).flatten(1, 2)  # [B, HW, D]
        pe = make_pe_from_centers(centers, D, self.pe_scale)       # [T/HW, D]
        patch_tokens = tokens + pe.unsqueeze(0)                   # [B, T/HW, D]

        # Instance path
        inst_q = self.instance_queries.weight.unsqueeze(0).expand(B, -1, -1)   # [B, N, D]
        inst_latents = self.inst_cross_attn(inst_q, patch_tokens)              # [B, N, D]
        inst_latents = self.inst_self_attn(inst_latents)                       # [B, N, D]
        scores_logits = self.objectness_head(inst_latents).squeeze(-1)         # [B, N]

        # Joint path: queries attend directly to instance latents, then self-attend
        joint_q  = self.joint_queries.weight.unsqueeze(0).expand(B * N, -1, -1)  # [B*N, K, D]
        inst_kv = inst_latents.reshape(B * N, 1, D)
        joint_latents = self.joint_cross_inst(joint_q, inst_kv)                # [B*N, K, D]

        joint_latents = self.joint_self_attn(joint_latents)                    # [B*N, K, D]
        joint_latents = joint_latents.view(B, N, K, D)                         # [B, N, K, D]
        
        joint_latents = self.final_norm(joint_latents)

        vis_logits = self.vis_head(joint_latents).squeeze(-1) if self.predict_visibility else torch.zeros(B, N, K, device=joint_latents.device, dtype=joint_latents.dtype)  # [B, N, K]
        kpt_logits = self.keypoints_head(joint_latents)                        # [B, N, K, 2]

        return {
            "keypoints":       torch.sigmoid(kpt_logits) * 2.0 - 1.0,
            "scores":          torch.sigmoid(scores_logits),
            "scores_logits":   scores_logits,
            "keypoint_scores": torch.sigmoid(vis_logits),
            "vis_logits":      vis_logits,
        }


# ---------------------------------------------------------------------------
# BackboneProbeModule
# ---------------------------------------------------------------------------


class BackboneProbeModule(BaseTrainingModule):
    """Lightning module for the backbone probe benchmark.

    The probe head and slot decoder are constructed internally with fixed
    hyperparameters.  Only the backbone varies between runs.

    Args:
        backbone:               feature extractor being benchmarked.
        probe_head:             ProbeHead instance (fixed across runs).
        weights:                loss component multipliers.
        lr / weight_decay:      AdamW hyperparameters.
        lr_T_max:               cosine annealing period (None = no cosine).
        lr_warmup_epochs:       linear warmup duration.
        lr_min_frac:            cosine floor as fraction of peak lr.
        val_ann_file:           annotation JSON for val AP metric (or None).
        test_ann_file:          annotation JSON for test AP metric (or None).
        num_keypoints:          K.
        grad_clip:              gradient clip norm (0 = disabled).
        val_ap_every_n_epochs:  run full AP on val set every N epochs.
        gt_sanity_check:        if True, forward returns GT-filled predictions.
        plot_dir:               directory to save prediction images on test.
        max_plot_images:        max images to save.
        max_persons:            passed to KeypointAPMetric denominator.
    """

    def __init__(
        self,
        backbone: BaseFeatureExtractor,
        probe_head: ConvProbe,
        weights: SetPredictionLossWeights | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_T_max: int | None = None,
        lr_warmup_epochs: int = 0,
        lr_min_frac: float = 1e-2,
        val_ann_file: str | None = None,
        test_ann_file: str | None = None,
        num_keypoints: int = 14,
        grad_clip: float = 0.0,
        val_ap_every_n_epochs: int = 10,
        gt_sanity_check: bool = False,
        plot_dir: str | None = None,
        max_plot_images: int = 5,
        max_persons: int | None = None,
        noise_std: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone   = backbone
        self.probe_head = probe_head

        resolved_weights = weights or SetPredictionLossWeights()
        self.weights = resolved_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_T_max = lr_T_max
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_min_frac = lr_min_frac
        self.num_keypoints = num_keypoints
        self.grad_clip = grad_clip
        self.val_ap_every_n_epochs = val_ap_every_n_epochs
        self.gt_sanity_check = gt_sanity_check
        self.plot_dir = plot_dir
        self.max_plot_images = max_plot_images
        self._plots_saved = 0
        self.noise_std = noise_std
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

        sigmas = torch.from_numpy(CROWDPOSE_SIGMAS).float()
        self._loss = SetPredictionLoss(resolved_weights, sigmas)

        self._val_metric  = KeypointAPMetric(val_ann_file,  max_persons=max_persons) if val_ann_file  else None
        self._test_metric = KeypointAPMetric(test_ann_file, max_persons=max_persons) if test_ann_file else None

        _vis = getattr(probe_head, "predict_visibility", True)
        self._log_loss_names = (
            "oks_loss", "objectness_loss",
            *(("visibility_loss",) if _vis else ()),
            "ranking_loss", "mean_oks",
        )

        self.save_hyperparameters(ignore=["backbone", "probe_head"])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(self, batch: dict) -> dict[str, Tensor]:
        if self.gt_sanity_check:
            return self._make_gt_pred(batch)
        image = batch["image"].to(self.device)
        features = self.backbone(image=image)
        return self.probe_head(features)

    def _make_gt_pred(self, batch: dict) -> dict[str, Tensor]:
        """GT-filled predictions for AP sanity-checking."""
        gt_kp  = batch["gt_keypoints"].to(self.device)  # [B, P_max, K, 2]
        gt_vis = batch["gt_visible"].to(self.device)     # [B, P_max, K]
        mask   = batch["person_mask"].to(self.device)    # [B, P_max]
        B, P_max, K = gt_kp.shape[:3]
        N = self.probe_head.num_slots

        kp_out  = torch.full((B, N, K, 2), -1.0, device=self.device)
        vis_out = torch.zeros(B, N, K, device=self.device)
        obj_out = torch.full((B, N), -10.0, device=self.device)

        for b in range(B):
            P = int(mask[b].sum().item())
            slots = min(P, N)
            kp_out[b, :slots]  = gt_kp[b, :slots]
            vis_out[b, :slots] = gt_vis[b, :slots]
            obj_out[b, :slots] = 10.0

        return {
            "keypoints":       kp_out,
            "scores":          torch.sigmoid(obj_out),
            "scores_logits":   obj_out,
            "keypoint_scores": torch.sigmoid(vis_out),
            "vis_logits":      vis_out,
        }

    def _compute_loss(self, pred: dict[str, Tensor], batch: dict) -> dict[str, Tensor]:
        targets = {
            "gt_keypoints": batch["gt_keypoints"],
            "gt_labeled":   batch["gt_labeled"],
            "gt_visible":   batch["gt_visible"],
            "person_scale": batch["person_scale"],
            "person_mask":  batch["person_mask"],
        }
        return self._loss(pred, targets)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        image = batch["image"]
        if self.noise_std > 0:
            image = add_gaussian_noise(image, self.noise_std)
        if self.brightness > 0 or self.contrast > 0 or self.saturation > 0:
            image = color_jitter(image, self.brightness, self.contrast, self.saturation)
        batch = {**batch, "image": image}
        pred   = self._forward(batch)
        losses = self._compute_loss(pred, batch)
        B = batch["image"].shape[0]
        kw = dict(batch_size=B, on_step=True, on_epoch=True)
        self.log("train/loss", losses["total"], prog_bar=True, **kw)
        for name in self._log_loss_names:
            self.log(f"train/{name}", losses[name], **kw)
        return losses["total"]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        pred   = self._forward(batch)
        losses = self._compute_loss(pred, batch)
        B = batch["image"].shape[0]
        kw = dict(batch_size=B, on_step=False, on_epoch=True)
        self.log("val/loss", losses["total"], prog_bar=True, **kw)
        for name in self._log_loss_names:
            self.log(f"val/{name}", losses[name], **kw)

        # Diagnostic metrics
        with torch.no_grad():
            diag = compute_probe_diagnostics(
                pred, batch, self._loss.sigmas,
                matcher_lambda_oks=self.weights.matcher_lambda_oks,
                matcher_lambda_obj=self.weights.matcher_lambda_obj,
            )
        _debug_keys = frozenset({"vis_tp", "vis_fp", "vis_tn", "vis_fn"})
        for key, val in diag.items():
            if key.startswith("per_joint_oks") or key in _debug_keys:
                self.log(f"debug/val/{key}", val.to(self.device), batch_size=B, on_step=False, on_epoch=True)
            else:
                self.log(f"val/{key}", val.to(self.device), batch_size=B, on_step=False, on_epoch=True)

        # Accumulate for periodic AP
        if self._val_metric is not None:
            self._update_ap_metric(self._val_metric, pred, batch)

    def on_validation_epoch_end(self) -> None:
        if self._val_metric is None:
            return
        run_ap = (self.current_epoch + 1) % self.val_ap_every_n_epochs == 0
        if run_ap:
            for key, val in self._val_metric.compute().items():
                self.log(f"val/{key}", val, sync_dist=True)
        self._val_metric.reset()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def on_test_start(self) -> None:
        self._plots_saved = 0

    def test_step(self, batch: dict, batch_idx: int) -> None:
        pred   = self._forward(batch)
        losses = self._compute_loss(pred, batch)
        B = batch["image"].shape[0]
        kw = dict(batch_size=B, on_epoch=True)
        self.log("test/loss", losses["total"], **kw)
        for name in self._log_loss_names:
            self.log(f"test/{name}", losses[name], **kw)

        with torch.no_grad():
            diag = compute_probe_diagnostics(
                pred, batch, self._loss.sigmas,
                matcher_lambda_oks=self.weights.matcher_lambda_oks,
                matcher_lambda_obj=self.weights.matcher_lambda_obj,
            )
        _debug_keys = frozenset({"vis_tp", "vis_fp", "vis_tn", "vis_fn"})
        for key, val in diag.items():
            prefix = "debug/test" if (key.startswith("per_joint_oks") or key in _debug_keys) else "test"
            self.log(f"{prefix}/{key}", val.to(self.device), **kw)

        if self._test_metric is not None:
            self._update_ap_metric(self._test_metric, pred, batch)

        if self.plot_dir is not None and self._plots_saved < self.max_plot_images:
            self._save_plots(pred, batch)

    def on_test_epoch_end(self) -> None:
        if self._test_metric is None:
            return
        for key, val in self._test_metric.compute().items():
            self.log(f"test/{key}", val, sync_dist=True)
        self._test_metric.reset()

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        return build_adamw_with_schedule(
            self,
            lr=self.lr,
            weight_decay=self.weight_decay,
            lr_T_max=self.lr_T_max,
            lr_warmup_epochs=self.lr_warmup_epochs,
            lr_min_frac=self.lr_min_frac,
        )

    # ------------------------------------------------------------------
    # AP wiring
    # ------------------------------------------------------------------

    def _update_ap_metric(
        self,
        metric: KeypointAPMetric,
        pred: dict[str, Tensor],
        batch: dict,
    ) -> None:
        img_hw = batch["img_hw"].to(self.device)
        img_wh = img_hw[:, [1, 0]]
        pred_pixel    = canonical_to_pixel(pred["keypoints"], img_wh).float()
        kpt_scores    = pred["keypoint_scores"].float()
        bbox_scores   = pred["scores"].float()
        image_ids     = batch["image_id"].to(self.device)
        metric.update(
            pred_kpts=pred_pixel,
            pred_scores=kpt_scores,
            image_ids=image_ids,
            bbox_scores=bbox_scores,
        )

    def _save_plots(self, pred: dict[str, Tensor], batch: dict) -> None:
        img_hw = batch["img_hw"].to(self.device)
        img_wh = img_hw[:, [1, 0]]
        pred_pixel = canonical_to_pixel(pred["keypoints"], img_wh).float()

        gt_canon = batch["gt_keypoints"].to(self.device)
        B, P_max, K, _ = gt_canon.shape
        gt_pixel = canonical_to_pixel(
            gt_canon.view(B, P_max * K, 2).unsqueeze(1),
            img_wh,
        ).squeeze(1).view(B, P_max, K, 2).float()

        for b in range(B):
            if self._plots_saved >= self.max_plot_images:
                break
            P = int(batch["person_mask"][b].sum().item())
            fname = Path(self.plot_dir) / f"pred_{self._plots_saved:04d}_img{int(batch['image_id'][b])}.png"
            save_prediction_plot(
                out_path=fname,
                image_chw=batch["image"][b].cpu(),
                pred_kpts_pixel=pred_pixel[b].cpu(),
                pred_scores=pred["scores"][b].float().cpu(),
                gt_kpts_pixel=gt_pixel[b, :P].cpu(),
                gt_labeled=batch["gt_labeled"][b, :P].cpu(),
                top_k=5,
            )
            self._plots_saved += 1
