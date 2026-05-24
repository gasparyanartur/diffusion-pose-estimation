"""Lightning module for the box-set diffusion pipeline (I6).

Wires together:
- a frozen / trainable visual backbone (``BaseFeatureExtractor``) producing
  image tokens + patch centres;
- the EDM box-set diffusion decoder (:class:`BoxSetDiffusionDecoder`);
- :class:`SetBoxDiffusionLoss` for training;
- :class:`BoxAPMetric` for periodic validation / test bbox-AP.

Mirrors :class:`BackboneProbeModule` in spirit: the backbone and decoder are
constructed externally and passed in; the module owns σ-sampling, the loss,
augmentations, AP wiring, and optimiser.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.base_config import BaseConfig
from src.data.transforms import boxes_padded_canonical_to_pixel_xywh
from src.features.base import BaseFeatureExtractor
from src.metrics import BoxAPMetric
from src.training.augmentations import (
    add_gaussian_noise,
    color_jitter,
    random_hflip_image_and_boxes,
)
from src.training.base_module import BaseTrainingModule
from src.training.losses import SetBoxDiffusionLoss, SetBoxDiffusionLossWeights
from src.training.optim import build_adamw_with_schedule

from pydantic import Field

if TYPE_CHECKING:
    from src.decoders.diffusion.box_set import BoxSetDiffusionDecoder


def _image_tokens_and_centers(
    features: dict[str, Tensor],
    feature_key: str,
) -> tuple[Tensor, Tensor]:
    """Reshape backbone outputs into ``(image_tokens [B,T,C], centers [T,2])``.

    Accepts ``features[feature_key]`` as either ``[B, C, H, W]`` (spatial map)
    or ``[B, T, C]`` (token sequence) and ``features["patch_centers"]`` as
    either ``[H, W, 2]`` or ``[T, 2]``.
    """
    feat = features[feature_key]
    centers = features["patch_centers"].reshape(-1, 2)
    if feat.ndim == 4:
        B, C, H, W = feat.shape
        tokens = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
    elif feat.ndim == 3:
        tokens = feat
    else:
        raise ValueError(f"unexpected feature shape {tuple(feat.shape)}")
    if centers.shape[0] != tokens.shape[1]:
        raise ValueError(
            f"patch_centers length {centers.shape[0]} != token count "
            f"{tokens.shape[1]}"
        )
    return tokens, centers


class BoxSetDiffusionModule(BaseTrainingModule):
    """Lightning module training :class:`BoxSetDiffusionDecoder`.

    Args:
        backbone:               visual feature extractor.
        decoder:                box-set diffusion decoder.
        weights:                loss component multipliers.
        lr / weight_decay:      AdamW hyperparameters.
        lr_T_max:               cosine annealing period (None disables cosine).
        lr_warmup_epochs:       linear warmup duration.
        lr_min_frac:            cosine floor fraction.
        sigma_min/sigma_max/rho: EDM σ-schedule parameters used both for
                                training σ sampling and inference.
        num_inference_steps:    Heun steps for multi-step validation AP.
                                AP is logged under ``ap-val/{n}-step/`` and
                                ``ap-test/{n}-step/`` (n = this value) and the
                                single-step variant under ``ap-val/1-step/`` /
                                ``ap-test/1-step/``.
        val_ap_every_n_epochs:  run AP every N epochs (epochs are 0-indexed;
                                triggers when ``(current_epoch + 1) % N == 0``).
        val_ann_file:           COCO annotation JSON for val AP (None = off).
        test_ann_file:          COCO annotation JSON for test AP.
        feature_key:            key in the backbone feature dict to consume.
        noise_std/brightness/contrast/saturation/hflip_p:
                                training augmentation hyperparameters.
        grad_clip:              gradient-clip norm passed through to Lightning.
    """

    def __init__(
        self,
        backbone: BaseFeatureExtractor,
        decoder: BoxSetDiffusionDecoder,
        weights: SetBoxDiffusionLossWeights | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_T_max: int | None = None,
        lr_warmup_epochs: int = 0,
        lr_min_frac: float = 1e-2,
        sigma_min: float = 0.002,
        sigma_max: float = 10.0,
        rho: float = 7.0,
        num_inference_steps: int = 5,
        sigma_p_mean: float = -1.2,
        sigma_p_std: float = 1.2,
        val_ap_every_n_epochs: int = 10,
        val_ann_file: str | None = None,
        test_ann_file: str | None = None,
        feature_key: str = "spatial",
        noise_std: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hflip_p: float = 0.0,
        grad_clip: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

        self.weights = weights or SetBoxDiffusionLossWeights()
        self._loss = SetBoxDiffusionLoss(self.weights)

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_T_max = lr_T_max
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_min_frac = lr_min_frac

        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_p_mean = float(sigma_p_mean)
        self.sigma_p_std = float(sigma_p_std)
        self.val_ap_every_n_epochs = int(val_ap_every_n_epochs)

        self.feature_key = feature_key
        self.noise_std = float(noise_std)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hflip_p = float(hflip_p)
        self.grad_clip = float(grad_clip)

        self._val_ap_metric = BoxAPMetric(ann_file=val_ann_file) if val_ann_file else None
        self._val_ap_metric_1step = BoxAPMetric(ann_file=val_ann_file) if val_ann_file else None
        self._test_ap_metric = BoxAPMetric(ann_file=test_ann_file) if test_ann_file else None
        self._test_ap_metric_1step = BoxAPMetric(ann_file=test_ann_file) if test_ann_file else None

        self._log_loss_names = ("l1_loss", "giou_loss", "diff_l2_loss", "obj_loss", "ranking_loss", "matched")

        self.save_hyperparameters(ignore=["backbone", "decoder"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def num_slots(self) -> int:
        return self.decoder.num_slots

    def _sample_sigma(self, B: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """EDM log-normal sigma sampling (Karras et al. 2022)."""
        log_sigma = self.sigma_p_mean + self.sigma_p_std * torch.randn(
            B, device=device, dtype=dtype
        )
        return log_sigma.exp().clamp(self.sigma_min, self.sigma_max)

    def _build_noisy_input(
        self,
        gt_boxes_canon: Tensor,    # [B, P_max, 4]
        gt_box_labeled: Tensor,    # [B, P_max]
        sigma: Tensor,             # [B]
    ) -> Tensor:
        """Pad GT boxes to ``N`` slots with random targets, then add σ·ε noise.

        Real GTs are placed in the leading slots of each row (in the order
        produced by the collate function). Padded slots are filled with
        ``cx, cy ∼ U(-1, 1)`` and ``w = h = 1`` so the decoder always sees
        plausible inputs even when ``P < N``.
        """
        B, _, _ = gt_boxes_canon.shape
        N = self.num_slots
        device = gt_boxes_canon.device
        dtype = gt_boxes_canon.dtype

        pad_cx = torch.rand(B, N, device=device, dtype=dtype) * 2.0 - 1.0
        pad_cy = torch.rand(B, N, device=device, dtype=dtype) * 2.0 - 1.0
        pad_w = torch.ones(B, N, device=device, dtype=dtype)
        pad_h = torch.ones(B, N, device=device, dtype=dtype)
        x0 = torch.stack([pad_cx, pad_cy, pad_w, pad_h], dim=-1)  # [B, N, 4]

        P_max = gt_boxes_canon.shape[1]
        for b in range(B):
            real = int(gt_box_labeled[b].sum().item())
            real = min(real, N, P_max)
            if real > 0:
                x0[b, :real] = gt_boxes_canon[b, :real]

        eps = torch.randn_like(x0)
        sigma_b = sigma.view(B, 1, 1).to(dtype)
        return x0 + sigma_b * eps

    def _forward_decoder(
        self,
        batch: dict,
        noisy_boxes: Tensor,
        sigma: Tensor,
    ) -> dict[str, Tensor]:
        features = self.backbone(image=batch["image"])
        tokens, centers = _image_tokens_and_centers(features, self.feature_key)
        return self.decoder(
            noisy_boxes=noisy_boxes,
            sigma=sigma,
            image_tokens=tokens,
            patch_centers=centers,
        )

    @torch.no_grad()
    def _sample_decoder(
        self,
        batch: dict,
        num_inference_steps: int,
    ) -> dict[str, Tensor]:
        features = self.backbone(image=batch["image"])
        tokens, centers = _image_tokens_and_centers(features, self.feature_key)
        return self.decoder.sample(
            image_tokens=tokens,
            patch_centers=centers,
            num_inference_steps=num_inference_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        image = batch["image"]
        boxes_canon = batch["gt_boxes_canon"]
        if self.noise_std > 0:
            image = add_gaussian_noise(image, self.noise_std)
        if self.brightness > 0 or self.contrast > 0 or self.saturation > 0:
            image = color_jitter(image, self.brightness, self.contrast, self.saturation)
        if self.hflip_p > 0:
            image, boxes_canon = random_hflip_image_and_boxes(
                image, boxes_canon, self.hflip_p,
            )
        batch = {**batch, "image": image, "gt_boxes_canon": boxes_canon}

        B = image.shape[0]
        sigma = self._sample_sigma(B, image.device, image.dtype)
        noisy = self._build_noisy_input(
            batch["gt_boxes_canon"], batch["gt_box_labeled"], sigma,
        )

        # Register a hook to capture ∂loss/∂noisy_boxes for debug logging.
        noisy = noisy.detach().requires_grad_(True)
        self._dbg_grad: list[Tensor] = []
        self._dbg_handle = noisy.register_hook(
            lambda g: self._dbg_grad.append(g.detach().abs().mean())
        )

        pred = self._forward_decoder(batch, noisy, sigma)
        losses = self._loss(pred, {
            "gt_boxes_canon": batch["gt_boxes_canon"],
            "gt_box_labeled": batch["gt_box_labeled"],
        })

        kw = dict(batch_size=B, on_step=True, on_epoch=True)
        self.log("train/loss", losses["total"], prog_bar=True, **kw)
        for name in self._log_loss_names:
            self.log(f"train/{name}", losses[name], **kw)
        self.log("debug/sigma_mean", sigma.mean(), batch_size=B, on_step=True, on_epoch=False)
        return losses["total"]

    def on_after_backward(self) -> None:
        handle = getattr(self, "_dbg_handle", None)
        if handle is not None:
            handle.remove()
            self._dbg_handle = None
        grads = getattr(self, "_dbg_grad", [])
        if grads:
            self.log(
                "debug/noisy_boxes_grad_norm", grads[0],
                on_step=True, on_epoch=False,
            )
            self._dbg_grad = []

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

    def _update_ap_metric(
        self,
        metric: BoxAPMetric,
        pred: dict[str, Tensor],
        batch: dict,
    ) -> None:
        scores = torch.sigmoid(pred["obj_logits"]).detach().float()  # [B, N]
        pixel_xywh = boxes_padded_canonical_to_pixel_xywh(
            pred["pred_boxes"].detach(), batch["img_hw"].to(pred["pred_boxes"].device),
        ).float()
        metric.update(
            pred_boxes_xywh=pixel_xywh,
            pred_scores=scores,
            image_ids=batch["image_id"].to(pred["pred_boxes"].device),
        )

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        B = batch["image"].shape[0]
        sigma = self._sample_sigma(B, batch["image"].device, batch["image"].dtype)
        noisy = self._build_noisy_input(
            batch["gt_boxes_canon"], batch["gt_box_labeled"], sigma,
        )
        pred = self._forward_decoder(batch, noisy, sigma)
        losses = self._loss(pred, {
            "gt_boxes_canon": batch["gt_boxes_canon"],
            "gt_box_labeled": batch["gt_box_labeled"],
        })
        kw = dict(batch_size=B, on_step=False, on_epoch=True)
        self.log("val/loss", losses["total"], prog_bar=True, **kw)
        for name in self._log_loss_names:
            self.log(f"val/{name}", losses[name], **kw)

        if not self._ap_epoch_active():
            return
        if self._val_ap_metric is not None:
            multi = self._sample_decoder(batch, self.num_inference_steps)
            self._update_ap_metric(self._val_ap_metric, multi, batch)
        if self._val_ap_metric_1step is not None:
            single = self._sample_decoder(batch, 1)
            self._update_ap_metric(self._val_ap_metric_1step, single, batch)

    def on_validation_epoch_end(self) -> None:
        if not self._ap_epoch_active():
            return
        if self._val_ap_metric is not None:
            n = self.num_inference_steps
            for k, v in self._val_ap_metric.compute().items():
                self.log(f"ap-val/{n}-step/{k}", v, sync_dist=True)
            self._val_ap_metric.reset()
        if self._val_ap_metric_1step is not None:
            for k, v in self._val_ap_metric_1step.compute().items():
                self.log(f"ap-val/1-step/{k}", v, sync_dist=True)
            self._val_ap_metric_1step.reset()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch: dict, batch_idx: int) -> None:
        B = batch["image"].shape[0]
        sigma = self._sample_sigma(B, batch["image"].device, batch["image"].dtype)
        noisy = self._build_noisy_input(
            batch["gt_boxes_canon"], batch["gt_box_labeled"], sigma,
        )
        pred = self._forward_decoder(batch, noisy, sigma)
        losses = self._loss(pred, {
            "gt_boxes_canon": batch["gt_boxes_canon"],
            "gt_box_labeled": batch["gt_box_labeled"],
        })
        kw = dict(batch_size=B, on_step=False, on_epoch=True)
        self.log("test/loss", losses["total"], **kw)
        for name in self._log_loss_names:
            self.log(f"test/{name}", losses[name], **kw)

        if self._test_ap_metric is not None:
            multi = self._sample_decoder(batch, self.num_inference_steps)
            self._update_ap_metric(self._test_ap_metric, multi, batch)
        if self._test_ap_metric_1step is not None:
            single = self._sample_decoder(batch, 1)
            self._update_ap_metric(self._test_ap_metric_1step, single, batch)

    def on_test_epoch_end(self) -> None:
        if self._test_ap_metric is not None:
            n = self.num_inference_steps
            for k, v in self._test_ap_metric.compute().items():
                self.log(f"ap-test/{n}-step/{k}", v, sync_dist=True)
            self._test_ap_metric.reset()
        if self._test_ap_metric_1step is not None:
            for k, v in self._test_ap_metric_1step.compute().items():
                self.log(f"ap-test/1-step/{k}", v, sync_dist=True)
            self._test_ap_metric_1step.reset()

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


class SetBoxDiffusionLossWeightsConfig(BaseConfig):
    """Pydantic mirror of :class:`SetBoxDiffusionLossWeights`."""

    l1: float = 5.0
    giou: float = 2.0
    diff_l2: float = 1.0
    obj_bce: float = 1.0
    rank: float = 1.0
    matcher_lambda_l1: float = 5.0
    matcher_lambda_giou: float = 2.0
    matcher_lambda_obj: float = 0.1

    def to_dataclass(self) -> SetBoxDiffusionLossWeights:
        return SetBoxDiffusionLossWeights(
            l1=self.l1,
            giou=self.giou,
            diff_l2=self.diff_l2,
            obj_bce=self.obj_bce,
            rank=self.rank,
            matcher_lambda_l1=self.matcher_lambda_l1,
            matcher_lambda_giou=self.matcher_lambda_giou,
            matcher_lambda_obj=self.matcher_lambda_obj,
        )


class BoxSetDiffusionModuleConfig(BaseConfig):
    """Hydra-instantiable config for :class:`BoxSetDiffusionModule`.

    The ``backbone`` and ``decoder`` are wired by the caller (see the entry
    script in I7); this config only owns the module-level scalars.
    """

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
    log_every_n_steps: int = 50
    resume: Optional[str] = None

    # diffusion
    sigma_min: float = 0.002
    sigma_max: float = 10.0
    rho: float = 7.0
    num_inference_steps: int = 5
    val_ap_every_n_epochs: int = 10
    sigma_p_mean: float = -1.2
    sigma_p_std: float = 1.2

    feature_key: str = "spatial"

    # augmentation
    noise_std: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hflip_p: float = 0.0

    weights: SetBoxDiffusionLossWeightsConfig = Field(
        default_factory=SetBoxDiffusionLossWeightsConfig,
    )

    def build(
        self,
        backbone: BaseFeatureExtractor,
        decoder: "BoxSetDiffusionDecoder",
        val_ann_file: str | None = None,
        test_ann_file: str | None = None,
    ) -> BoxSetDiffusionModule:
        return BoxSetDiffusionModule(
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
            num_inference_steps=self.num_inference_steps,
            val_ap_every_n_epochs=self.val_ap_every_n_epochs,
            val_ann_file=val_ann_file,
            test_ann_file=test_ann_file,
            feature_key=self.feature_key,
            noise_std=self.noise_std,
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hflip_p=self.hflip_p,
            grad_clip=self.grad_clip,
            sigma_p_mean=self.sigma_p_mean,
            sigma_p_std=self.sigma_p_std,
        )
