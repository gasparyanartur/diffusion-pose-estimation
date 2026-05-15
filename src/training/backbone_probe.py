"""Backbone probe: fixed head for benchmarking visual backbones.

Pipeline
--------
    image  [B, 3, H, W]
      ↓  backbone (variable — the thing being benchmarked)
    spatial features  [B, C, H', W']
      ↓  ProbeHead (fixed across all backbone variants)
         1. 1×1 conv projection  →  [B, D, H', W']
         2. sinusoidal 2D PE added at full resolution
         3. conv_depth × (3×3 conv + BN + ReLU)
         4. adaptive avg-pool   →  [B, D, pool_h, pool_w]
         5. sinusoidal 2D PE + single self-attention + LayerNorm
         6. mean-pool over tokens  →  [B, D]
      ↓  PoseSlotDecoder (fixed)
         Linear(D, 1024) → ReLU → Linear(1024, N*(1 + K*2 + K))
    pred dict  {keypoints, scores, scores_logits, vis_logits, keypoint_scores}

The backbone is the only component that varies between benchmark variants.
ProbeHead and PoseSlotDecoder are constructed internally and kept identical
across all variants.
"""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.decoders.pose_slot import PoseSlotDecoder
from src.embedders.spatial_attention import make_2d_sinusoidal_pe
from src.features.base import BaseFeatureExtractor
from src.metrics import CROWDPOSE_SIGMAS, KeypointAPMetric, canonical_to_pixel
from src.training.diagnostics import compute_probe_diagnostics
from src.training.losses import (
    SetPredictionLoss,
    SetPredictionLossWeights,
    compute_oks,
    hungarian_match,
)
from src.training.optim import build_adamw_with_schedule
from src.vis import save_prediction_plot


# ---------------------------------------------------------------------------
# ProbeHead
# ---------------------------------------------------------------------------


class ProbeHead(nn.Module):
    """Fixed projection + pool + attention head.

    Takes a spatial feature map of arbitrary channel count and spatial size,
    and produces a single [B, feature_dim] vector.

    Steps:
        1x1 conv: in_channels → feature_dim
        sinusoidal 2D PE added at full spatial resolution
        conv_depth × (3×3 conv + BN + ReLU): feature_dim → feature_dim
        adaptive avg-pool: → [B, feature_dim, pool_h, pool_w]
        sinusoidal 2D PE + single self-attention + LayerNorm
        mean-pool over tokens → [B, feature_dim]

    Args:
        in_channels:  channel count from the backbone (varies per variant).
        feature_dim:  fixed internal dimension D.  Must be divisible by 4.
        pool_h:       height after adaptive pool.
        pool_w:       width after adaptive pool.
        num_heads:    attention heads.
        conv_depth:   number of 3×3 conv blocks after the 1×1 projection.
                      0 means skip the conv blocks entirely.
        feature_key:  key in the backbone feature dict to consume.
    """

    def __init__(
        self,
        in_channels: int,
        feature_dim: int = 256,
        pool_h: int = 32,
        pool_w: int = 32,
        num_heads: int = 8,
        conv_depth: int = 1,
        feature_key: str = "spatial",
    ) -> None:
        super().__init__()
        if feature_dim % 4 != 0:
            raise ValueError(f"feature_dim must be divisible by 4; got {feature_dim}")

        self.feature_key = feature_key
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.feature_dim = feature_dim

        # 1×1 projection (channel adapter — the only backbone-dependent part)
        self.proj = nn.Conv2d(in_channels, feature_dim, kernel_size=1)

        # optional conv blocks
        conv_blocks: list[nn.Module] = []
        for _ in range(conv_depth):
            conv_blocks += [
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True),
            ]
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # attention
        self.attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

        # cached PE for the fixed pooled grid (moves with the module via register_buffer)
        pe_pool = make_2d_sinusoidal_pe(pool_h, pool_w, feature_dim, device=torch.device("cpu"))
        self.register_buffer("_pe_pool", pe_pool)  # [pool_h*pool_w, D]

    def forward(self, features: dict[str, Tensor]) -> Tensor:
        feat = features[self.feature_key]                    # [B, C, H', W']
        x = self.proj(feat)                                  # [B, D, H', W']
        _, _, H, W = x.shape
        pe_spatial = make_2d_sinusoidal_pe(
            H, W, self.feature_dim, x.device, x.dtype,
        )  # [H'*W', D]
        x = x + pe_spatial.view(H, W, self.feature_dim).permute(2, 0, 1).unsqueeze(0)
        x = self.conv_blocks(x)                              # [B, D, H', W']
        x = F.adaptive_avg_pool2d(x, (self.pool_h, self.pool_w))  # [B, D, ph, pw]
        tokens = x.flatten(2).permute(0, 2, 1)              # [B, ph*pw, D]
        tokens = tokens + self._pe_pool.to(dtype=x.dtype).unsqueeze(0)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        return tokens.mean(dim=1)                            # [B, D]


# ---------------------------------------------------------------------------
# BackboneProbeModule
# ---------------------------------------------------------------------------


class BackboneProbeModule(pl.LightningModule):
    """Lightning module for the backbone probe benchmark.

    The probe head and slot decoder are constructed internally with fixed
    hyperparameters.  Only the backbone varies between runs.

    Args:
        backbone:               feature extractor being benchmarked.
        probe_head:             ProbeHead instance (fixed across runs).
        slot_decoder:           PoseSlotDecoder instance (fixed across runs).
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
        probe_head: ProbeHead,
        slot_decoder: PoseSlotDecoder,
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
    ) -> None:
        super().__init__()
        self.backbone     = backbone
        self.probe_head   = probe_head
        self.slot_decoder = slot_decoder

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

        sigmas = torch.from_numpy(CROWDPOSE_SIGMAS).float()
        self._loss = SetPredictionLoss(resolved_weights, sigmas)

        self._val_metric  = KeypointAPMetric(val_ann_file,  max_persons=max_persons) if val_ann_file  else None
        self._test_metric = KeypointAPMetric(test_ann_file, max_persons=max_persons) if test_ann_file else None

        self.save_hyperparameters(ignore=["backbone", "probe_head", "slot_decoder"])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        frozen, trainable = [], []
        for name, param in self.named_parameters():
            (frozen if not param.requires_grad else trainable).append(name)
        import logging
        log = logging.getLogger(__name__)
        log.info("Trainable parameters (%d):", len(trainable))
        for name in trainable:
            log.info("  + %s", name)
        if frozen:
            log.info("Frozen parameters (%d):", len(frozen))
            for name in frozen:
                log.info("  - %s", name)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(self, batch: dict) -> dict[str, Tensor]:
        if self.gt_sanity_check:
            return self._make_gt_pred(batch)
        image = batch["image"].to(self.device)
        features = self.backbone(image=image)
        global_feat = self.probe_head(features)            # [B, D]
        return self.slot_decoder({"feature": global_feat})

    def _make_gt_pred(self, batch: dict) -> dict[str, Tensor]:
        """GT-filled predictions for AP sanity-checking."""
        gt_kp  = batch["gt_keypoints"].to(self.device)  # [B, P_max, K, 2]
        gt_vis = batch["gt_visible"].to(self.device)     # [B, P_max, K]
        mask   = batch["person_mask"].to(self.device)    # [B, P_max]
        B, P_max, K = gt_kp.shape[:3]
        N = self.slot_decoder.num_slots

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
        pred   = self._forward(batch)
        losses = self._compute_loss(pred, batch)
        B = batch["image"].shape[0]
        kw = dict(batch_size=B, on_step=True, on_epoch=True)
        self.log("train/loss", losses["total"], prog_bar=True, **kw)
        for name in ("oks_loss", "objectness_loss", "visibility_loss", "ranking_loss", "mean_oks"):
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
        for name in ("oks_loss", "objectness_loss", "visibility_loss", "ranking_loss", "mean_oks"):
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
        for name in ("oks_loss", "objectness_loss", "visibility_loss", "ranking_loss", "mean_oks"):
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
