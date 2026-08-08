"""Lightning module for the graph-diffusion pipeline (boxes + keypoints).

Wires together:
- a frozen RADIO backbone (RadioFeatureExtractor) producing image tokens,
  patch centres, and a spatial feature map;
- GraphDiffusionPipelineDecoder (Stage 1 attention + Stage 2 GNN refiner);
- GraphPipelineLoss (Hungarian box + keypoint loss) for training;
- KeypointAPMetric for periodic validation / test AP.

Training protocol
-----------------
For each batch:
1. Sample σ ~ EDM log-normal.
2. Pad GT boxes (cx,cy,w,h) to N slots with random noise; shuffle.
3. Compute GT joints_rel = gt_joints_canon − gt_box_centres.
4. Pad gt_joints_rel to N slots with zeros; apply same shuffle.
5. Add σ·ε noise to both boxes and joints.
6. Run decoder → predicted (x₀_boxes, x₀_joints_rel).
7. Compute GraphPipelineLoss (Hungarian end-to-end).

Batch dict keys required
------------------------
    image           [B, 3, H, W]
    gt_boxes_canon  [B, P_max, 4]    (cx, cy, w, h) canonical
    gt_joints_canon [B, P_max, K, 2] absolute canonical keypoints
    gt_box_labeled  [B, P_max]       {0, 1} — padded slots = 0
    gt_visibility   [B, P_max, K]    {0, 1, 2}
    img_hw          [B, 2]           (H, W) in pixels — for AP metric
    image_id        [B]              COCO image IDs

Batch keys are produced by a CrowdPose dataset that includes both bounding
boxes and keypoints.  See docs/architecture.md for coordinate conventions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor
from pydantic import Field

from src.base_config import BaseConfig
from src.features.base import BaseFeatureExtractor
from src.metrics import KeypointAPMetric, canonical_to_pixel
from src.training.augmentations import add_gaussian_noise, color_jitter
from src.training.base_module import BaseTrainingModule
from src.training.losses import GraphPipelineLoss, GraphPipelineLossWeights
from src.training.optim import build_adamw_with_schedule

if TYPE_CHECKING:
    from src.decoders.diffusion.graph_pipeline import GraphDiffusionPipelineDecoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _boxes_pixel_to_canonical(
    boxes_xywh: Tensor,  # [B, P, 4]  x, y, w, h in pixels
    img_hw: Tensor,      # [B, 2]     (H, W)
) -> Tensor:
    """Convert pixel (x,y,w,h) to canonical (cx,cy,cw,ch) using per-axis scaling.

    canonical_x = (x + w/2) / W * 2 − 1,  canonical_y = (y + h/2) / H * 2 − 1.
    canonical_w = w / W * 2,               canonical_h = h / H * 2.

    This is the inverse of ``canonical_to_pixel`` and is consistent with
    ``pixel_to_canonical`` used for keypoints.
    """
    H = img_hw[:, 0:1].unsqueeze(1)   # [B, 1, 1]
    W = img_hw[:, 1:2].unsqueeze(1)   # [B, 1, 1]
    x, y, w, h = boxes_xywh.unbind(-1)
    cx = (x + w / 2.0) / W.squeeze(-1) * 2.0 - 1.0
    cy = (y + h / 2.0) / H.squeeze(-1) * 2.0 - 1.0
    cw = w / W.squeeze(-1) * 2.0
    ch = h / H.squeeze(-1) * 2.0
    return torch.stack([cx, cy, cw, ch], dim=-1)


def _normalize_batch(batch: dict) -> dict:
    """Map dataset batch keys to the names expected by GraphDiffusionPipelineModule.

    The standard CrowdPose set-prediction batch provides:
        gt_keypoints   [B, P, K, 2]  per-axis canonical keypoints
        gt_labeled     [B, P, K]     {0,1} — joint is annotated
        gt_boxes_pixel [B, P, 4]     (x, y, w, h) in pixels

    This function adds:
        gt_joints_canon [B, P, K, 2]  (alias for gt_keypoints)
        gt_visibility   [B, P, K]     (alias for gt_labeled; >0 means labeled)
        gt_boxes_canon  [B, P, 4]     per-axis canonical boxes (cx,cy,cw,ch)

    The per-axis canonical coordinate system is used throughout so that box
    centres and keypoint positions are directly comparable.  This is consistent
    with ``canonical_to_pixel`` / ``pixel_to_canonical`` and with the AP metric.
    """
    out = dict(batch)
    if "gt_joints_canon" not in out:
        out["gt_joints_canon"] = out["gt_keypoints"]
    if "gt_visibility" not in out:
        out["gt_visibility"] = out["gt_labeled"]
    if "gt_boxes_canon" not in out or True:
        # Recompute using per-axis scaling regardless of what on_after_batch_transfer set.
        if "gt_boxes_pixel" in out and "img_hw" in out:
            device = out["gt_boxes_pixel"].device
            img_hw = out["img_hw"].to(device).float()
            out["gt_boxes_canon"] = _boxes_pixel_to_canonical(
                out["gt_boxes_pixel"].float(), img_hw,
            )
    return out


def _radio_features(
    backbone: BaseFeatureExtractor,
    image: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run backbone and extract image_tokens, patch_centers, spatial_feat.

    Returns:
        image_tokens  [B, T, C]
        patch_centers [T, 2]   canonical (−1, 1)
        spatial_feat  [B, C, H', W']
    """
    features = backbone(image=image)
    spatial = features["spatial"]                               # [B, C, H', W']
    centers = features["patch_centers"].reshape(-1, 2)          # [T, 2]
    B, C, H, W = spatial.shape
    tokens = spatial.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, T, C]
    if centers.shape[0] != tokens.shape[1]:
        raise ValueError(
            f"patch_centers length {centers.shape[0]} != token count "
            f"{tokens.shape[1]}"
        )
    return tokens, centers, spatial


def _build_noisy_input(
    gt_boxes_canon: Tensor,   # [B, P_max, 4]
    gt_joints_canon: Tensor,  # [B, P_max, K, 2]
    gt_box_labeled: Tensor,   # [B, P_max]  {0,1}
    sigma: Tensor,            # [B]
    num_slots: int,
    num_keypoints: int,
) -> tuple[Tensor, Tensor]:
    """Pad GT to N slots, shuffle, add σ·ε noise.

    Padding strategy:
    - Real GT (P slots): use GT values.
    - Padding (N-P slots): cx,cy ~ U(-1,1); w,h ~ U(0.05, 0.5);
      joints_rel = 0.

    Shuffling ensures the model cannot exploit GT slot ordering.

    Returns:
        noisy_boxes      [B, N, 4]
        noisy_joints_rel [B, N, K, 2]
    """
    B, P_max, _ = gt_boxes_canon.shape
    K = num_keypoints
    N = num_slots
    device = gt_boxes_canon.device
    dtype  = gt_boxes_canon.dtype

    # Build padded GT boxes [B, N, 4].
    pad_cx = torch.rand(B, N, device=device, dtype=dtype) * 2.0 - 1.0
    pad_cy = torch.rand(B, N, device=device, dtype=dtype) * 2.0 - 1.0
    pad_w  = torch.rand(B, N, device=device, dtype=dtype) * 0.45 + 0.05
    pad_h  = torch.rand(B, N, device=device, dtype=dtype) * 0.45 + 0.05
    x0_boxes = torch.stack([pad_cx, pad_cy, pad_w, pad_h], dim=-1)  # [B, N, 4]

    # Compute GT joints_rel = gt_joints - gt_box_centre.
    gt_ctr = gt_boxes_canon[:, :, :2]                          # [B, P_max, 2]
    gt_jrel = gt_joints_canon - gt_ctr.unsqueeze(2)            # [B, P_max, K, 2]

    x0_joints = torch.zeros(B, N, K, 2, device=device, dtype=dtype)

    # Fill real GT slots (per image).
    for b in range(B):
        real = int((gt_box_labeled[b] > 0).sum().item())
        real = min(real, N, P_max)
        if real > 0:
            x0_boxes[b, :real]  = gt_boxes_canon[b, :real]
            x0_joints[b, :real] = gt_jrel[b, :real]

    # Random slot shuffle (prevent trivial ordering).
    for b in range(B):
        perm = torch.randperm(N, device=device)
        x0_boxes[b]  = x0_boxes[b, perm]
        x0_joints[b] = x0_joints[b, perm]
        # NOTE: Hungarian matching in the loss handles the shuffled order.

    # Add noise.
    eps_boxes  = torch.randn_like(x0_boxes)
    eps_joints = torch.randn_like(x0_joints)
    sigma_b = sigma.view(B, 1, 1).to(dtype)
    noisy_boxes  = x0_boxes  + sigma_b * eps_boxes
    sigma_bk = sigma.view(B, 1, 1, 1).to(dtype)
    noisy_joints = x0_joints + sigma_bk * eps_joints
    return noisy_boxes, noisy_joints


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class GraphDiffusionOnestageModule(BaseTrainingModule):
    """Lightning module training GraphDiffusionPipelineDecoder.

    Args:
        backbone:               Visual feature extractor (RADIO, frozen).
        decoder:                GraphDiffusionPipelineDecoder.
        weights:                Loss component multipliers.
        lr / weight_decay:      AdamW hyperparameters.
        lr_T_max:               Cosine annealing period (None = disabled).
        lr_warmup_epochs:       Linear warmup epochs.
        lr_min_frac:            Cosine floor as fraction of peak LR.
        sigma_min/max/rho:      EDM σ-schedule parameters.
        sigma_p_mean/std:       Log-normal training σ-sampling parameters.
        num_inference_steps:    Heun steps for multi-step AP.
        val_ap_every_n_epochs:  Run AP every N epochs.
        val_ann_file:           COCO annotation JSON for val AP (None = off).
        test_ann_file:          COCO annotation JSON for test AP.
        noise_std/brightness/contrast/saturation:  Image augmentations.
        grad_clip:              Gradient clip norm (0 = disabled).
    """

    def __init__(
        self,
        backbone: BaseFeatureExtractor,
        decoder: "GraphDiffusionPipelineDecoder",
        weights: GraphPipelineLossWeights | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_T_max: int | None = None,
        lr_warmup_epochs: int = 0,
        lr_min_frac: float = 1e-2,
        sigma_min: float = 0.002,
        sigma_max: float = 10.0,
        rho: float = 7.0,
        sigma_p_mean: float = -1.2,
        sigma_p_std: float = 1.2,
        num_inference_steps: int = 5,
        val_ap_every_n_epochs: int = 10,
        val_ann_file: str | None = None,
        test_ann_file: str | None = None,
        noise_std: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        grad_clip: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder  = decoder

        self._loss = GraphPipelineLoss(weights)

        self.lr               = lr
        self.weight_decay     = weight_decay
        self.lr_T_max         = lr_T_max
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_min_frac      = lr_min_frac

        self.sigma_min    = float(sigma_min)
        self.sigma_max    = float(sigma_max)
        self.rho          = float(rho)
        self.sigma_p_mean = float(sigma_p_mean)
        self.sigma_p_std  = float(sigma_p_std)
        self.num_inference_steps  = int(num_inference_steps)
        self.val_ap_every_n_epochs = int(val_ap_every_n_epochs)

        self.noise_std  = float(noise_std)
        self.brightness = float(brightness)
        self.contrast   = float(contrast)
        self.saturation = float(saturation)
        self.grad_clip  = float(grad_clip)

        self._val_ap   = KeypointAPMetric(ann_file=val_ann_file)  if val_ann_file  else None
        self._val_ap1  = KeypointAPMetric(ann_file=val_ann_file)  if val_ann_file  else None
        self._test_ap  = KeypointAPMetric(ann_file=test_ann_file) if test_ann_file else None
        self._test_ap1 = KeypointAPMetric(ann_file=test_ann_file) if test_ann_file else None

        self._log_loss_names = (
            "box_l1", "box_giou", "box_l2", "kpt_l2", "obj_loss", "ranking_loss", "matched",
        )
        self.save_hyperparameters(ignore=["backbone", "decoder", "weights"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def num_slots(self) -> int:
        return self.decoder.num_slots

    @property
    def num_keypoints(self) -> int:
        return self.decoder.num_keypoints

    def _sample_sigma(self, B: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        log_sigma = self.sigma_p_mean + self.sigma_p_std * torch.randn(
            B, device=device, dtype=dtype,
        )
        return log_sigma.exp().clamp(self.sigma_min, self.sigma_max)

    def _forward_decoder(
        self,
        batch: dict,
        noisy_boxes: Tensor,
        noisy_joints_rel: Tensor,
        sigma: Tensor,
    ) -> dict[str, Tensor]:
        tokens, centers, spatial = _radio_features(self.backbone, batch["image"])
        return self.decoder(
            noisy_boxes=noisy_boxes,
            noisy_joints_rel=noisy_joints_rel,
            sigma=sigma,
            image_tokens=tokens,
            patch_centers=centers,
            spatial_feat=spatial,
        )

    @torch.no_grad()
    def _sample_decoder(self, batch: dict, num_steps: int) -> dict[str, Tensor]:
        tokens, centers, spatial = _radio_features(self.backbone, batch["image"])
        return self.decoder.sample(
            image_tokens=tokens,
            patch_centers=centers,
            spatial_feat=spatial,
            num_inference_steps=num_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        batch = _normalize_batch(batch)
        image = batch["image"]
        if self.noise_std > 0:
            image = add_gaussian_noise(image, self.noise_std)
        if self.brightness > 0 or self.contrast > 0 or self.saturation > 0:
            image = color_jitter(image, self.brightness, self.contrast, self.saturation)
        batch = {**batch, "image": image}

        B = image.shape[0]
        sigma = self._sample_sigma(B, image.device, image.dtype)

        noisy_boxes, noisy_joints = _build_noisy_input(
            batch["gt_boxes_canon"],
            batch["gt_joints_canon"],
            batch["gt_box_labeled"],
            sigma,
            num_slots=self.num_slots,
            num_keypoints=self.num_keypoints,
        )

        pred = self._forward_decoder(batch, noisy_boxes, noisy_joints, sigma)

        losses = self._loss(pred, {
            "gt_boxes_canon":  batch["gt_boxes_canon"],
            "gt_joints_canon": batch["gt_joints_canon"],
            "gt_box_labeled":  batch["gt_box_labeled"],
            "gt_visibility":   batch["gt_visibility"],
        })

        kw = dict(batch_size=B, on_step=True, on_epoch=True)
        self.log("train/loss", losses["total"], prog_bar=True, **kw)
        for name in self._log_loss_names:
            self.log(f"train/{name}", losses[name], **kw)
        self.log("debug/sigma_mean", sigma.mean(), batch_size=B, on_step=True, on_epoch=False)
        return losses["total"]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _ap_epoch_active(self) -> bool:
        is_period = (self.current_epoch + 1) % self.val_ap_every_n_epochs == 0
        is_last = (
            self._trainer is not None
            and (self.current_epoch + 1) == self.trainer.max_epochs
        )
        return is_period or is_last

    def _update_kpt_ap(
        self,
        metric: KeypointAPMetric,
        pred: dict[str, Tensor],
        batch: dict,
    ) -> None:
        """Convert decoder output to pixel-space keypoints and update metric."""
        pred_boxes    = pred["pred_boxes"].detach().float()       # [B, N, 4]
        pred_jrel     = pred["pred_joints_rel"].detach().float()  # [B, N, K, 2]
        obj_scores    = torch.sigmoid(pred["obj_logits"]).detach().float()  # [B, N]
        # canonical_to_pixel expects img_wh = [W, H]; batch provides img_hw = [H, W].
        img_hw        = batch["img_hw"].to(pred_boxes.device).float()  # [B, 2] (H, W)
        img_wh        = img_hw[:, [1, 0]]                              # [B, 2] (W, H)

        # Absolute canonical keypoints [B, N, K, 2].
        box_ctr   = pred_boxes[:, :, :2]
        kpt_canon = box_ctr.unsqueeze(2) + pred_jrel

        # Convert to pixel space.
        kpt_pixel = canonical_to_pixel(kpt_canon, img_wh)         # [B, N, K, 2]

        # Per-keypoint scores: broadcast objectness over all keypoints.
        B, N, K = kpt_pixel.shape[:3]
        kpt_scores = obj_scores.unsqueeze(-1).expand(B, N, K)     # [B, N, K]

        metric.update(
            pred_kpts=kpt_pixel,
            pred_scores=kpt_scores,
            image_ids=batch["image_id"].to(pred_boxes.device),
        )

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        batch = _normalize_batch(batch)
        B = batch["image"].shape[0]
        sigma = self._sample_sigma(B, batch["image"].device, batch["image"].dtype)
        noisy_boxes, noisy_joints = _build_noisy_input(
            batch["gt_boxes_canon"], batch["gt_joints_canon"],
            batch["gt_box_labeled"], sigma,
            self.num_slots, self.num_keypoints,
        )
        pred = self._forward_decoder(batch, noisy_boxes, noisy_joints, sigma)
        losses = self._loss(pred, {
            "gt_boxes_canon":  batch["gt_boxes_canon"],
            "gt_joints_canon": batch["gt_joints_canon"],
            "gt_box_labeled":  batch["gt_box_labeled"],
            "gt_visibility":   batch["gt_visibility"],
        })
        kw = dict(batch_size=B, on_step=False, on_epoch=True)
        self.log("val/loss", losses["total"], prog_bar=True, **kw)
        for name in self._log_loss_names:
            self.log(f"val/{name}", losses[name], **kw)

        if not self._ap_epoch_active():
            return
        if self._val_ap is not None:
            multi = self._sample_decoder(batch, self.num_inference_steps)
            self._update_kpt_ap(self._val_ap, multi, batch)
        if self._val_ap1 is not None:
            single = self._sample_decoder(batch, 1)
            self._update_kpt_ap(self._val_ap1, single, batch)

    def on_validation_epoch_end(self) -> None:
        if not self._ap_epoch_active():
            return
        n = self.num_inference_steps
        if self._val_ap is not None:
            for k, v in self._val_ap.compute().items():
                self.log(f"ap-val/{n}-step/{k}", v, sync_dist=True)
            self._val_ap.reset()
        if self._val_ap1 is not None:
            for k, v in self._val_ap1.compute().items():
                self.log(f"ap-val/1-step/{k}", v, sync_dist=True)
            self._val_ap1.reset()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch: dict, batch_idx: int) -> None:
        batch = _normalize_batch(batch)
        B = batch["image"].shape[0]
        sigma = self._sample_sigma(B, batch["image"].device, batch["image"].dtype)
        noisy_boxes, noisy_joints = _build_noisy_input(
            batch["gt_boxes_canon"], batch["gt_joints_canon"],
            batch["gt_box_labeled"], sigma,
            self.num_slots, self.num_keypoints,
        )
        pred = self._forward_decoder(batch, noisy_boxes, noisy_joints, sigma)
        losses = self._loss(pred, {
            "gt_boxes_canon":  batch["gt_boxes_canon"],
            "gt_joints_canon": batch["gt_joints_canon"],
            "gt_box_labeled":  batch["gt_box_labeled"],
            "gt_visibility":   batch["gt_visibility"],
        })
        kw = dict(batch_size=B, on_step=False, on_epoch=True)
        self.log("test/loss", losses["total"], **kw)
        for name in self._log_loss_names:
            self.log(f"test/{name}", losses[name], **kw)

        if self._test_ap is not None:
            multi = self._sample_decoder(batch, self.num_inference_steps)
            self._update_kpt_ap(self._test_ap, multi, batch)
        if self._test_ap1 is not None:
            single = self._sample_decoder(batch, 1)
            self._update_kpt_ap(self._test_ap1, single, batch)

    def on_test_epoch_end(self) -> None:
        n = self.num_inference_steps
        if self._test_ap is not None:
            for k, v in self._test_ap.compute().items():
                self.log(f"ap-test/{n}-step/{k}", v, sync_dist=True)
            self._test_ap.reset()
        if self._test_ap1 is not None:
            for k, v in self._test_ap1.compute().items():
                self.log(f"ap-test/1-step/{k}", v, sync_dist=True)
            self._test_ap1.reset()

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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class GraphPipelineLossWeightsConfig(BaseConfig):
    """Pydantic mirror of :class:`GraphPipelineLossWeights`."""

    box_l1: float = 5.0
    box_giou: float = 2.0
    box_l2: float = 1.0
    kpt_l2: float = 5.0
    obj_bce: float = 1.0
    rank: float = 1.0
    matcher_lambda_l1: float = 5.0
    matcher_lambda_giou: float = 2.0
    matcher_lambda_kpt: float = 1.0
    matcher_lambda_obj: float = 0.1

    def to_dataclass(self) -> GraphPipelineLossWeights:
        return GraphPipelineLossWeights(
            box_l1=self.box_l1,
            box_giou=self.box_giou,
            box_l2=self.box_l2,
            kpt_l2=self.kpt_l2,
            obj_bce=self.obj_bce,
            rank=self.rank,
            matcher_lambda_l1=self.matcher_lambda_l1,
            matcher_lambda_giou=self.matcher_lambda_giou,
            matcher_lambda_kpt=self.matcher_lambda_kpt,
            matcher_lambda_obj=self.matcher_lambda_obj,
        )


class GraphDiffusionOnestageModuleConfig(BaseConfig):
    """Hydra-instantiable config for :class:`GraphDiffusionOnestageModule`."""

    # optimiser / schedule
    lr: float = 1e-3
    weight_decay: float = 1e-4
    cosine_lr: bool = True
    lr_warmup_epochs: int = 0
    lr_min_frac: float = 1e-2

    # trainer
    max_epochs: int = 50
    seed: int = 42
    grad_clip: float = 0.0
    resume: Optional[str] = None

    # diffusion
    sigma_min: float = 0.002
    sigma_max: float = 10.0
    rho: float = 7.0
    sigma_p_mean: float = -1.2
    sigma_p_std: float = 1.2
    num_inference_steps: int = 5
    val_ap_every_n_epochs: int = 10

    # augmentations
    noise_std: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0

    weights: GraphPipelineLossWeightsConfig = Field(
        default_factory=GraphPipelineLossWeightsConfig,
    )

    def build(
        self,
        backbone: BaseFeatureExtractor,
        decoder: "GraphDiffusionOnestageDecoder",
        val_ann_file: str | None = None,
        test_ann_file: str | None = None,
    ) -> GraphDiffusionOnestageModule:
        return GraphDiffusionOnestageModule(
            backbone=backbone,
            decoder=decoder,
            weights=self.weights.to_dataclass(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            lr_T_max=self.max_epochs if self.cosine_lr else None,
            lr_warmup_epochs=self.lr_warmup_epochs,
            lr_min_frac=self.lr_min_frac,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            sigma_p_mean=self.sigma_p_mean,
            sigma_p_std=self.sigma_p_std,
            num_inference_steps=self.num_inference_steps,
            val_ap_every_n_epochs=self.val_ap_every_n_epochs,
            val_ann_file=val_ann_file,
            test_ann_file=test_ann_file,
            noise_std=self.noise_std,
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            grad_clip=self.grad_clip,
        )


# Compatibility names retained for existing scripts, checkpoints, and imports.
GraphDiffusionPipelineModule = GraphDiffusionOnestageModule
GraphDiffusionPipelineModuleConfig = GraphDiffusionOnestageModuleConfig
