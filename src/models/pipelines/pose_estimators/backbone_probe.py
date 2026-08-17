"""Backbone-probe benchmark pipeline: fixed head, swappable backbone.

Pipeline
--------
    image  [B, 3, H, W]
      -> backbone (variable -- the thing being benchmarked)
    spatial features  [B, C, H', W']
      -> ConvProbe (fixed across all backbone variants)
         1. 1x1 conv projection  ->  [B, D, H', W']
         2. sinusoidal 2D PE added at full resolution
         3. num_conv_layers x ConvNeXtBlock(D) with conditional AvgPool2d(2,2)
            (pool applied after a block only when max(H,W) > target_size)
         4. flatten -> [B, H*W, D], add sinusoidal 2D PE
         5. num_attn_layers x TransformerBlock (pre-norm, MHSA + FFN)
         6. mean-pool over tokens  ->  [B, D]
    -> fixed-slot prediction head
         Linear(D, 1024) -> ReLU -> Linear(1024, N*(1 + K*2 + K))
    pred dict  {keypoints, scores, scores_logits, vis_logits, keypoint_scores}

``QueryProbe`` is an alternate DETR-style head selectable via config.
The backbone is the only component that varies between benchmark variants.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field
from torch import Tensor

from src.configs.base_config import BaseConfig
from src.data.augmentations import add_gaussian_noise, color_jitter
from src.data.data_module import PoseDataModule
from src.eval.diagnostics import compute_probe_diagnostics
from src.training.losses import SetPredictionLoss, SetPredictionLossWeights
from src.eval.metrics import CROWDPOSE_SIGMAS, KeypointAPMetric, canonical_to_pixel
from src.models.features.base import BaseFeatureExtractor
from src.models.common.positional import make_2d_sinusoidal_pe, make_pe_from_centers
from src.training.optim import build_adamw_with_schedule
from src.eval.vis import save_prediction_plot


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvNeXtBlock(nn.Module):
    """Depthwise ConvNeXt-style block."""

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
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.act(self.pw1(x))
        x = self.drop(x)
        x = self.pw2(x)
        x = x.permute(0, 3, 1, 2)
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


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block: Q attends to KV, then FFN + residual."""

    def __init__(self, dim: int, num_heads: int, ffn_mult: int = 2,
                 attn_dropout: float = 0.0, ffn_dropout: float = 0.1) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
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
# ConvProbe
# ---------------------------------------------------------------------------


class ConvProbe(nn.Module):
    """Self-contained spatial feature map -> pose prediction dict."""

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
        self.scores_head = nn.Linear(hidden_dim, num_slots)
        self.keypoints_head = nn.Linear(hidden_dim, num_slots * num_keypoints * 2)
        self.vis_head = nn.Linear(hidden_dim, num_slots * num_keypoints)

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        feat = features[self.feature_key]
        x = self.proj2(self.act(self.proj1(feat)))
        _, _, H, W = x.shape
        pe = make_2d_sinusoidal_pe(H, W, self.conv_dim, x.device, x.dtype)
        x = x + pe.view(H, W, self.conv_dim).permute(2, 0, 1).unsqueeze(0)

        for block in self.conv_layers:
            x = block(x)
            _, _, H, W = x.shape
            if max(H, W) > self.target_size:
                x = F.avg_pool2d(x, 2)

        x = self.readout_proj(x)
        _, _, H, W = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)
        pe = make_2d_sinusoidal_pe(H, W, self.feature_dim, x.device, x.dtype)
        tokens = tokens + pe.unsqueeze(0)

        for block in self.attn_layers:
            tokens = block(tokens)

        g = self.shared_proj(tokens.mean(dim=1))
        B = g.shape[0]
        N, K = self.num_slots, self.num_keypoints

        scores_logits = self.scores_head(g)
        kpt_logits = self.keypoints_head(g).view(B, N, K, 2)
        vis_logits = self.vis_head(g).view(B, N, K)

        return {
            "keypoints": torch.sigmoid(kpt_logits) * 2.0 - 1.0,
            "scores": torch.sigmoid(scores_logits),
            "scores_logits": scores_logits,
            "keypoint_scores": torch.sigmoid(vis_logits),
            "vis_logits": vis_logits,
        }


# ---------------------------------------------------------------------------
# QueryProbe
# ---------------------------------------------------------------------------


class QueryProbe(nn.Module):
    """Query-based probe: DETR-style cross-attention over backbone patch tokens.

    Returns the same pred dict as :class:`ConvProbe` -- drop-in for
    :class:`BackboneProbePipeline`.
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

        self.feature_key = feature_key
        self.feature_dim = D
        self.pe_scale = pe_scale
        self.num_slots = num_slots
        self.num_keypoints = num_keypoints
        self.predict_visibility = predict_visibility

        self.proj = nn.Linear(in_channels, D)
        self.instance_queries = nn.Embedding(num_slots, D)
        self.inst_cross_attn = CrossAttentionBlock(D, num_heads, ffn_mult, attn_dropout, ffn_dropout)
        self.inst_self_attn = TransformerBlock(D, num_heads, ffn_mult, attn_dropout=attn_dropout, ffn_dropout=ffn_dropout)

        self.objectness_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(D // 2, 1),
        )

        self.joint_queries = nn.Embedding(num_keypoints, D)
        self.joint_cross_inst = CrossAttentionBlock(D, num_heads, ffn_mult, attn_dropout, ffn_dropout)
        self.joint_self_attn = TransformerBlock(D, num_heads, ffn_mult, attn_dropout=attn_dropout, ffn_dropout=ffn_dropout)

        self.final_norm = nn.LayerNorm(D)
        self.vis_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.Dropout(proj_dropout),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(D // 2, 1),
        ) if predict_visibility else None
        self.keypoints_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, 2),
        )

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        N, K, D = self.num_slots, self.num_keypoints, self.feature_dim

        feat = features[self.feature_key]
        centers = features["patch_centers"].reshape(-1, 2)
        B = feat.shape[0]
        if feat.ndim == 3:
            tokens = self.proj(feat)
        else:
            tokens = self.proj(feat.permute(0, 2, 3, 1)).flatten(1, 2)
        pe = make_pe_from_centers(centers, D, self.pe_scale)
        patch_tokens = tokens + pe.unsqueeze(0)

        inst_q = self.instance_queries.weight.unsqueeze(0).expand(B, -1, -1)
        inst_latents = self.inst_cross_attn(inst_q, patch_tokens)
        inst_latents = self.inst_self_attn(inst_latents)
        scores_logits = self.objectness_head(inst_latents).squeeze(-1)

        joint_q = self.joint_queries.weight.unsqueeze(0).expand(B * N, -1, -1)
        inst_kv = inst_latents.reshape(B * N, 1, D)
        joint_latents = self.joint_cross_inst(joint_q, inst_kv)

        joint_latents = self.joint_self_attn(joint_latents)
        joint_latents = joint_latents.view(B, N, K, D)
        joint_latents = self.final_norm(joint_latents)

        vis_logits = (
            self.vis_head(joint_latents).squeeze(-1)
            if self.predict_visibility
            else torch.zeros(B, N, K, device=joint_latents.device, dtype=joint_latents.dtype)
        )
        kpt_logits = self.keypoints_head(joint_latents)

        return {
            "keypoints": torch.sigmoid(kpt_logits) * 2.0 - 1.0,
            "scores": torch.sigmoid(scores_logits),
            "scores_logits": scores_logits,
            "keypoint_scores": torch.sigmoid(vis_logits),
            "vis_logits": vis_logits,
        }


# ---------------------------------------------------------------------------
# BackboneProbePipeline
# ---------------------------------------------------------------------------


class BackboneProbePipeline(pl.LightningModule):
    """Lightning pipeline for the backbone-probe benchmark.

    The probe head is fixed per run (:class:`ConvProbe` or :class:`QueryProbe`);
    only the backbone varies between benchmark variants.
    """

    def __init__(
        self,
        backbone: BaseFeatureExtractor,
        probe_head: nn.Module,
        weights: SetPredictionLossWeights | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_T_max: int | None = None,
        lr_warmup_epochs: int = 0,
        lr_min_frac: float = 1e-2,
        val_ann_file: str | None = None,
        test_ann_file: str | None = None,
        num_keypoints: int = 14,
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
        self.backbone = backbone
        self.probe_head = probe_head

        resolved_weights = weights or SetPredictionLossWeights()
        self.weights = resolved_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_T_max = lr_T_max
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_min_frac = lr_min_frac
        self.num_keypoints = num_keypoints
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

        self._val_metric = KeypointAPMetric(val_ann_file, max_persons=max_persons) if val_ann_file else None
        self._test_metric = KeypointAPMetric(test_ann_file, max_persons=max_persons) if test_ann_file else None

        _vis = getattr(probe_head, "predict_visibility", True)
        self._log_loss_names = (
            "oks_loss", "objectness_loss",
            *(("visibility_loss",) if _vis else ()),
            "ranking_loss", "mean_oks",
        )

        self.save_hyperparameters(ignore=["backbone", "probe_head"])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        from pytorch_lightning.utilities.model_summary import ModelSummary
        log = logging.getLogger(__name__)
        log.info("Model summary:\n%s", ModelSummary(self, max_depth=3))

    def on_train_epoch_start(self) -> None:
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        self.log(
            "train/lr", optimizer.param_groups[0]["lr"],
            prog_bar=False, on_step=False, on_epoch=True,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        features = self.backbone(image=images)
        return self.probe_head(features)

    def _forward(self, batch: dict) -> dict[str, Tensor]:
        if self.gt_sanity_check:
            return self._make_gt_pred(batch)
        return self.forward(batch["image"].to(self.device))

    def _make_gt_pred(self, batch: dict) -> dict[str, Tensor]:
        """GT-filled predictions for AP sanity-checking."""
        gt_kp = batch["gt_keypoints"].to(self.device)
        gt_vis = batch["gt_visible"].to(self.device)
        mask = batch["person_mask"].to(self.device)
        B, P_max, K = gt_kp.shape[:3]
        N = self.probe_head.num_slots

        kp_out = torch.full((B, N, K, 2), -1.0, device=self.device)
        vis_out = torch.zeros(B, N, K, device=self.device)
        obj_out = torch.full((B, N), -10.0, device=self.device)

        for b in range(B):
            P = int(mask[b].sum().item())
            slots = min(P, N)
            kp_out[b, :slots] = gt_kp[b, :slots]
            vis_out[b, :slots] = gt_vis[b, :slots]
            obj_out[b, :slots] = 10.0

        return {
            "keypoints": kp_out,
            "scores": torch.sigmoid(obj_out),
            "scores_logits": obj_out,
            "keypoint_scores": torch.sigmoid(vis_out),
            "vis_logits": vis_out,
        }

    def _compute_loss(self, pred: dict[str, Tensor], batch: dict) -> dict[str, Tensor]:
        targets = {
            "gt_keypoints": batch["gt_keypoints"],
            "gt_labeled": batch["gt_labeled"],
            "gt_visible": batch["gt_visible"],
            "person_scale": batch["person_scale"],
            "person_mask": batch["person_mask"],
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
        pred = self._forward(batch)
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
        pred = self._forward(batch)
        losses = self._compute_loss(pred, batch)
        B = batch["image"].shape[0]
        kw = dict(batch_size=B, on_step=False, on_epoch=True)
        self.log("val/loss", losses["total"], prog_bar=True, **kw)
        for name in self._log_loss_names:
            self.log(f"val/{name}", losses[name], **kw)

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

        if self._val_metric is not None:
            self._update_ap_metric(self._val_metric, pred, batch)

    def on_validation_epoch_end(self) -> None:
        if self._val_metric is None:
            return
        if (self.current_epoch + 1) % self.val_ap_every_n_epochs == 0:
            for key, val in self._val_metric.compute().items():
                self.log(f"val/{key}", val, sync_dist=True)
        self._val_metric.reset()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def on_test_start(self) -> None:
        self._plots_saved = 0

    def test_step(self, batch: dict, batch_idx: int) -> None:
        pred = self._forward(batch)
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
    # Optimizer
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

    def _update_ap_metric(self, metric: KeypointAPMetric, pred: dict[str, Tensor], batch: dict) -> None:
        img_hw = batch["img_hw"].to(self.device)
        img_wh = img_hw[:, [1, 0]]
        pred_pixel = canonical_to_pixel(pred["keypoints"], img_wh).float()
        kpt_scores = pred["keypoint_scores"].float()
        bbox_scores = pred["scores"].float()
        image_ids = batch["image_id"].to(self.device)
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


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


class SetPredictionLossWeightsConfig(BaseConfig):
    """Pydantic mirror of :class:`SetPredictionLossWeights`."""

    oks: float = 1.0
    obj: float = 1.0
    vis: float = 0.5
    rank: float = 1.0
    matcher_lambda_oks: float = 1.0
    matcher_lambda_obj: float = 0.1

    def to_dataclass(self) -> SetPredictionLossWeights:
        return SetPredictionLossWeights(
            oks=self.oks,
            obj=self.obj,
            vis=self.vis,
            rank=self.rank,
            matcher_lambda_oks=self.matcher_lambda_oks,
            matcher_lambda_obj=self.matcher_lambda_obj,
        )


class BackboneProbeConfig(BaseConfig):
    """Builds :class:`BackboneProbePipeline` with a :class:`ConvProbe` head."""

    lr: float = 1e-3
    weight_decay: float = 1e-3
    cosine_lr: bool = True
    lr_warmup_epochs: int = 0
    lr_min_frac: float = 1e-1

    gt_sanity_check: bool = False

    feature_dim: int = 256
    proj_dim: int = 768
    conv_dim: int = 96
    target_size: int = 16
    num_conv_layers: int = 5
    num_attn_layers: int = 2
    num_heads: int = 4
    ffn_mult: int = 4
    conv_dropout: float = 0.1
    attn_dropout: float = 0.3
    ffn_dropout: float = 0.5
    feature_key: str = "spatial"

    hidden_dim: int = 256
    num_slots: int = 20
    proj_dropout: float = 0.2

    noise_std: float = 0.05
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.2

    val_ap_every_n_epochs: int = 10

    weights: SetPredictionLossWeightsConfig = Field(default_factory=SetPredictionLossWeightsConfig)

    def build(
        self,
        backbone: BaseFeatureExtractor,
        data: PoseDataModule,
        *,
        max_epochs: int,
    ) -> BackboneProbePipeline:
        probe_head = ConvProbe(
            in_channels=backbone.out_channels,
            feature_dim=self.feature_dim,
            proj_dim=self.proj_dim,
            conv_dim=self.conv_dim,
            target_size=self.target_size,
            num_conv_layers=self.num_conv_layers,
            num_attn_layers=self.num_attn_layers,
            num_heads=self.num_heads,
            ffn_mult=self.ffn_mult,
            conv_dropout=self.conv_dropout,
            attn_dropout=self.attn_dropout,
            ffn_dropout=self.ffn_dropout,
            num_slots=self.num_slots,
            num_keypoints=data.num_keypoints,
            hidden_dim=self.hidden_dim,
            proj_dropout=self.proj_dropout,
            feature_key=self.feature_key,
        )
        return BackboneProbePipeline(
            backbone=backbone,
            probe_head=probe_head,
            weights=self.weights.to_dataclass(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            lr_T_max=max_epochs if self.cosine_lr else None,
            lr_warmup_epochs=self.lr_warmup_epochs,
            lr_min_frac=self.lr_min_frac,
            val_ann_file=str(data.val_ann_path),
            test_ann_file=str(data.test_ann_path),
            num_keypoints=data.num_keypoints,
            val_ap_every_n_epochs=self.val_ap_every_n_epochs,
            gt_sanity_check=self.gt_sanity_check,
            max_persons=getattr(data, "max_persons", None),
            noise_std=self.noise_std,
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
        )


class QueryBackboneProbeConfig(BaseConfig):
    """Builds :class:`BackboneProbePipeline` with a :class:`QueryProbe` head."""

    lr: float = 4e-4
    weight_decay: float = 1e-3
    cosine_lr: bool = True
    lr_warmup_epochs: int = 1
    lr_min_frac: float = 1e-1

    gt_sanity_check: bool = False

    feature_dim: int = 256
    num_slots: int = 20
    num_heads: int = 4
    ffn_mult: int = 2
    attn_dropout: float = 0.0
    ffn_dropout: float = 0.1
    proj_dropout: float = 0.1
    feature_key: str = "spatial"
    pe_scale: float = 1000.0
    predict_visibility: bool = False

    noise_std: float = 0.05
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.2

    val_ap_every_n_epochs: int = 10

    weights: SetPredictionLossWeightsConfig = Field(default_factory=SetPredictionLossWeightsConfig)

    def build(
        self,
        backbone: BaseFeatureExtractor,
        data: PoseDataModule,
        *,
        max_epochs: int,
    ) -> BackboneProbePipeline:
        probe_head = QueryProbe(
            in_channels=backbone.out_channels,
            feature_dim=self.feature_dim,
            num_slots=self.num_slots,
            num_keypoints=data.num_keypoints,
            num_heads=self.num_heads,
            ffn_mult=self.ffn_mult,
            attn_dropout=self.attn_dropout,
            ffn_dropout=self.ffn_dropout,
            proj_dropout=self.proj_dropout,
            feature_key=self.feature_key,
        )
        return BackboneProbePipeline(
            backbone=backbone,
            probe_head=probe_head,
            weights=self.weights.to_dataclass(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            lr_T_max=max_epochs if self.cosine_lr else None,
            lr_warmup_epochs=self.lr_warmup_epochs,
            lr_min_frac=self.lr_min_frac,
            val_ann_file=str(data.val_ann_path),
            test_ann_file=str(data.test_ann_path),
            num_keypoints=data.num_keypoints,
            val_ap_every_n_epochs=self.val_ap_every_n_epochs,
            gt_sanity_check=self.gt_sanity_check,
            max_persons=getattr(data, "max_persons", None),
            noise_std=self.noise_std,
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
        )
