"""PyTorch-Lightning module for set-prediction pose estimators.

Pairs with set-format DataModule batches (see ``src.data.datasets.SetPoseDataset``).
The pose estimator must produce decoder output keys:
    keypoints       [B, N, K, 2]   canonical (-1, 1)
    scores_logits   [B, N]
    vis_logits      [B, N, K]

AP convention (see ``docs/assumptions.md``): submit ALL N slots per image
to KeypointAPMetric, scored by ``sigmoid(scores_logits)``; per-keypoint
score is ``sigmoid(vis_logits)``.
"""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import Tensor

from src.metrics import KeypointAPMetric, canonical_to_pixel
from src.metrics import CROWDPOSE_SIGMAS
from src.vis import save_prediction_plot
from src.pose_estimator import PoseEstimator
from src.training.losses import (
    SetPredictionLoss,
    SetPredictionLossWeights,
    compute_oks,
    hungarian_match,
)
from src.training.optim import build_adamw_with_schedule


class SetPredictionTrainingModule(pl.LightningModule):
    """Lightning wrapper for set-prediction pose training."""

    def __init__(
        self,
        pose_estimator: PoseEstimator,
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
        gt_sanity_check: bool = False,
        plot_dir: str | None = None,
        max_plot_images: int = 5,
        max_persons: int | None = None,
    ) -> None:
        super().__init__()
        self.pose_estimator = pose_estimator
        resolved_weights = weights or SetPredictionLossWeights()
        self.weights = resolved_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_T_max = lr_T_max
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_min_frac = lr_min_frac
        self.num_keypoints = num_keypoints
        self.grad_clip = grad_clip
        self.gt_sanity_check = gt_sanity_check
        self.plot_dir = plot_dir
        self.max_plot_images = max_plot_images
        self._plots_saved = 0

        sigmas = torch.from_numpy(CROWDPOSE_SIGMAS).float()
        self._loss = SetPredictionLoss(resolved_weights, sigmas)

        self._val_metric = KeypointAPMetric(val_ann_file, max_persons=max_persons) if val_ann_file else None
        self._test_metric = KeypointAPMetric(test_ann_file, max_persons=max_persons) if test_ann_file else None

        self.save_hyperparameters(ignore=["pose_estimator"])

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def _forward(self, batch: dict) -> dict[str, Tensor]:
        if self.gt_sanity_check:
            return self._make_gt_pred(batch)
        image = batch["image"].to(self.device)
        return self.pose_estimator(image=image)

    def _make_gt_pred(self, batch: dict) -> dict[str, Tensor]:
        """Return a fake prediction that copies GT into the first P slots.

        Objectness logits: +10 for real slots, -10 for empty ones.
        Keypoint scores come from gt_visible; keypoints copy gt_keypoints.
        Used to verify that AP eval is correct when given perfect input.
        """
        gt_kp = batch["gt_keypoints"].to(self.device)    # [B, P_max, K, 2]
        gt_vis = batch["gt_visible"].to(self.device)      # [B, P_max, K]
        mask = batch["person_mask"].to(self.device)       # [B, P_max]
        B, P_max, K = gt_kp.shape[:3]
        N = self.pose_estimator.decoder.num_slots

        # Pad / truncate to N slots
        kp = torch.zeros(B, N, K, 2, device=self.device)
        vis_logits = torch.full((B, N, K), -10.0, device=self.device)
        obj_logits = torch.full((B, N), -10.0, device=self.device)

        n_fill = min(N, P_max)
        kp[:, :n_fill] = gt_kp[:, :n_fill]
        vis_logits[:, :n_fill] = gt_vis[:, :n_fill] * 20.0 - 10.0  # {0,1} -> {-10, +10}
        for b in range(B):
            for p in range(n_fill):
                obj_logits[b, p] = 10.0 if mask[b, p] > 0 else -10.0

        return {
            "keypoints":       kp,
            "scores":          torch.sigmoid(obj_logits),
            "scores_logits":   obj_logits,
            "keypoint_scores": torch.sigmoid(vis_logits),
            "vis_logits":      vis_logits,
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

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        pred = self._forward(batch)
        losses = self._compute_loss(pred, batch)
        B = batch["image"].shape[0]
        log_kwargs = dict(batch_size=B, on_step=True, on_epoch=True)
        self.log("train/loss", losses["total"], prog_bar=True, **log_kwargs)
        for name in ("oks_loss", "objectness_loss", "visibility_loss", "ranking_loss", "mean_oks"):
            self.log(f"train/{name}", losses[name], **log_kwargs)
        return losses["total"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        pred = self._forward(batch)
        losses = self._compute_loss(pred, batch)
        B = batch["image"].shape[0]
        log_kwargs = dict(batch_size=B, on_step=False, on_epoch=True)
        self.log("val/loss", losses["total"], prog_bar=True, **log_kwargs)
        for name in ("oks_loss", "objectness_loss", "visibility_loss", "ranking_loss", "mean_oks"):
            self.log(f"val/{name}", losses[name], **log_kwargs)
        # mean OKS over the matched slots, computed inside set_prediction_loss
        # is the canonical val signal; AP eval runs only at test time.

    def on_test_start(self) -> None:
        self._plots_saved = 0

    def test_step(self, batch: dict, batch_idx: int) -> None:
        pred = self._forward(batch)
        losses = self._compute_loss(pred, batch)
        B = batch["image"].shape[0]
        self.log("test/loss", losses["total"], batch_size=B, on_epoch=True)
        self.log("test/mean_oks",        losses["mean_oks"],        batch_size=B, on_epoch=True)
        self.log("test/oks_loss",        losses["oks_loss"],        batch_size=B, on_epoch=True)
        self.log("test/objectness_loss", losses["objectness_loss"], batch_size=B, on_epoch=True)
        self.log("test/ranking_loss",    losses["ranking_loss"],    batch_size=B, on_epoch=True)

        if self._test_metric is not None:
            self._update_ap_metric(self._test_metric, pred, batch)

        if self.plot_dir is not None and self._plots_saved < self.max_plot_images:
            self._save_plots(pred, batch)

    # ------------------------------------------------------------------
    # AP wiring
    # ------------------------------------------------------------------
    def _update_ap_metric(
        self,
        metric: KeypointAPMetric,
        pred: dict[str, Tensor],
        batch: dict,
    ) -> None:
        # img_hw is [H, W]; canonical_to_pixel expects [W, H]
        img_hw = batch["img_hw"].to(self.device)
        img_wh = img_hw[:, [1, 0]]
        pred_pixel = canonical_to_pixel(pred["keypoints"], img_wh).float()  # [B, N, K, 2]
        keypoint_scores = pred["keypoint_scores"].float()                    # [B, N, K]
        bbox_scores = pred["scores"].float()                                 # [B, N]
        image_ids = batch["image_id"].to(self.device)
        metric.update(
            pred_kpts=pred_pixel,
            pred_scores=keypoint_scores,
            image_ids=image_ids,
            bbox_scores=bbox_scores,
        )

    def _save_plots(self, pred: dict[str, Tensor], batch: dict) -> None:
        """Save per-image prediction plots; stops once max_plot_images reached."""
        img_hw = batch["img_hw"].to(self.device)
        img_wh = img_hw[:, [1, 0]]                                       # [B, 2]
        pred_pixel = canonical_to_pixel(pred["keypoints"], img_wh).float()  # [B, N, K, 2]

        # canonical_to_pixel broadcasts img_wh as [B, 1, 1, 2] via its internal op
        gt_canon = batch["gt_keypoints"].to(self.device)                 # [B, P_max, K, 2]
        B, P_max, K, _ = gt_canon.shape
        gt_pixel_flat = canonical_to_pixel(
            gt_canon.view(B, P_max * K, 2).unsqueeze(1),                # [B, 1, P_max*K, 2]
            img_wh,
        ).squeeze(1).view(B, P_max, K, 2).float()                       # [B, P_max, K, 2]

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
                gt_kpts_pixel=gt_pixel_flat[b, :P].cpu(),
                gt_labeled=batch["gt_labeled"][b, :P].cpu(),
                top_k=5,
            )
            self._plots_saved += 1

    def on_test_epoch_end(self) -> None:
        if self._test_metric is None:
            return
        for key, val in self._test_metric.compute().items():
            self.log(f"test/{key}", val, sync_dist=True)
        self._test_metric.reset()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @torch.no_grad()
    def matched_oks(self, pred: dict[str, Tensor], batch: dict) -> Tensor:
        """Mean OKS over Hungarian-matched (slot, gt) pairs across the batch."""
        oks_vals: list[Tensor] = []
        B = pred["keypoints"].shape[0]
        for b in range(B):
            P = int(batch["person_mask"][b].sum().item())
            if P == 0:
                continue
            oks = compute_oks(
                pred["keypoints"][b],
                batch["gt_keypoints"][b, :P].to(self.device),
                batch["gt_labeled"][b, :P].to(self.device),
                batch["person_scale"][b, :P].to(self.device),
                self._loss.sigmas,
            )
            slot_idx, gt_idx = hungarian_match(
                oks, pred["scores_logits"][b],
                lambda_oks=self.weights.matcher_lambda_oks,
                lambda_obj=self.weights.matcher_lambda_obj,
            )
            if slot_idx.numel() > 0:
                oks_vals.append(oks[slot_idx, gt_idx])
        if not oks_vals:
            return torch.tensor(0.0, device=self.device)
        return torch.cat(oks_vals).mean()

    # ------------------------------------------------------------------
    # Optimisation
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
