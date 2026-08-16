"""Tensor-native EDM box-diffusion object-detection pipeline."""

from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch
from pydantic import Field
from torch import Tensor, nn

from src.configs.base_config import BaseConfig
from src.data.augmentations import (
    add_gaussian_noise,
    color_jitter,
    random_hflip_image_and_boxes,
)
from src.data.transforms import boxes_padded_canonical_to_pixel_xywh
from src.training.losses import SetBoxDiffusionLoss, SetBoxDiffusionLossWeights
from src.eval.metrics import BoxAPMetric
from src.models.common.attention import (
    FourierFeatures,
    TimeFiLMTransformerBlock,
)
from src.models.features.base import (
    BaseFeatureExtractor,
    BaseFeatureExtractorConfig,
)
from src.models.common.positional import make_pe_from_centers
from src.training.optim import build_adamw_with_schedule


def _image_tokens_and_centers(
    features: dict[str, Tensor],
    feature_key: str,
) -> tuple[Tensor, Tensor]:
    """Return image tokens ``[B, T, C]`` and their canonical centres ``[T, 2]``."""
    feature_map = features[feature_key]
    centers = features["patch_centers"].reshape(-1, 2)
    if feature_map.ndim == 4:
        batch_size, channels, height, width = feature_map.shape
        tokens = feature_map.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channels
        )
    elif feature_map.ndim == 3:
        tokens = feature_map
    else:
        raise ValueError(f"unexpected feature shape {tuple(feature_map.shape)}")
    if centers.shape[0] != tokens.shape[1]:
        raise ValueError(
            f"patch_centers length {centers.shape[0]} != token count {tokens.shape[1]}"
        )
    return tokens, centers


class _BoxDiffusionDenoiser(nn.Module):
    """Image-conditioned denoiser for a fixed set of canonical box slots."""

    def __init__(
        self,
        *,
        num_slots: int,
        dim: int,
        num_heads: int,
        num_blocks: int,
        num_obj_blocks: int,
        num_box_branch_blocks: int,
        num_box_refine_blocks: int,
        obj_dim: int,
        box_dim: int,
        time_dim: int,
        ffn_mult: float,
        image_token_dim: int,
        fourier_box_bands: int,
        fourier_box_scale: float,
        fourier_sigma_bands: int,
        fourier_sigma_scale: float,
        sigma_log_offset: float,
        pe_scale: float,
        self_cond: bool,
    ) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"dim must be divisible by 4; got {dim}")
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1; got {num_slots}")

        self.num_slots = num_slots
        self.dim = dim
        self.sigma_log_offset = float(sigma_log_offset)
        self.pe_scale = float(pe_scale)
        self.self_cond = self_cond

        self.box_in = FourierFeatures(
            in_dim=4,
            out_dim=dim,
            num_bands=fourier_box_bands,
            scale=fourier_box_scale,
        )
        if self_cond:
            self.self_cond_in = FourierFeatures(
                in_dim=4,
                out_dim=dim,
                num_bands=fourier_box_bands,
                scale=fourier_box_scale,
            )
        self.slot_query = nn.Embedding(num_slots, dim)
        self.time_in = FourierFeatures(
            in_dim=1,
            out_dim=time_dim,
            num_bands=fourier_sigma_bands,
            scale=fourier_sigma_scale,
        )
        self.image_proj = nn.Linear(image_token_dim, dim)

        self.blocks = nn.ModuleList(
            TimeFiLMTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                time_dim=time_dim,
                ffn_mult=ffn_mult,
                cross_dim=dim,
                use_cross_attn=True,
            )
            for _ in range(num_blocks)
        )
        self.obj_proj = nn.Linear(dim, obj_dim)
        self.box_proj = nn.Linear(dim, box_dim)
        self.obj_blocks = nn.ModuleList(
            TimeFiLMTransformerBlock(
                dim=obj_dim,
                num_heads=num_heads,
                time_dim=time_dim,
                ffn_mult=ffn_mult,
                cross_dim=dim,
                use_cross_attn=True,
            )
            for _ in range(num_obj_blocks)
        )
        self.obj_head = nn.Linear(obj_dim, 1)
        self.box_blocks = nn.ModuleList(
            TimeFiLMTransformerBlock(
                dim=box_dim,
                num_heads=num_heads,
                time_dim=time_dim,
                ffn_mult=ffn_mult,
                cross_dim=dim,
                use_cross_attn=True,
            )
            for _ in range(num_box_branch_blocks)
        )
        self.box_obj_cross = TimeFiLMTransformerBlock(
            dim=box_dim,
            num_heads=num_heads,
            time_dim=time_dim,
            ffn_mult=ffn_mult,
            cross_dim=obj_dim,
            use_cross_attn=True,
        )
        self.box_refine_blocks = nn.ModuleList(
            TimeFiLMTransformerBlock(
                dim=box_dim,
                num_heads=num_heads,
                time_dim=time_dim,
                ffn_mult=ffn_mult,
                cross_dim=dim,
                use_cross_attn=True,
            )
            for _ in range(num_box_refine_blocks)
        )
        self.box_head = nn.Linear(box_dim, 4)

    def forward(
        self,
        noisy_boxes: Tensor,
        sigma: Tensor,
        image_tokens: Tensor,
        patch_centers: Tensor,
        image_key_padding_mask: Optional[Tensor] = None,
        x0_prev: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        if noisy_boxes.shape[1] != self.num_slots:
            raise ValueError(
                f"expected N={self.num_slots} slots, got {noisy_boxes.shape[1]}"
            )

        box_tokens = self.box_in(noisy_boxes)
        if self.self_cond:
            previous_boxes = x0_prev if x0_prev is not None else torch.zeros_like(noisy_boxes)
            box_tokens = box_tokens + self.self_cond_in(previous_boxes)
        slot_tokens = box_tokens + self.slot_query.weight.unsqueeze(0)

        log_sigma = torch.log(
            sigma.clamp_min(self.sigma_log_offset).to(noisy_boxes.dtype)
        ).unsqueeze(-1)
        time_embedding = self.time_in(log_sigma)
        image_context = self.image_proj(image_tokens)
        image_context = image_context + make_pe_from_centers(
            patch_centers.to(image_context.dtype), self.dim, scale=self.pe_scale
        ).unsqueeze(0)

        for block in self.blocks:
            slot_tokens = block(
                slot_tokens,
                context=image_context,
                time_emb=time_embedding,
                context_key_padding_mask=image_key_padding_mask,
            )

        object_tokens = self.obj_proj(slot_tokens)
        for block in self.obj_blocks:
            object_tokens = block(
                object_tokens,
                context=image_context,
                time_emb=time_embedding,
                context_key_padding_mask=image_key_padding_mask,
            )
        obj_logits = self.obj_head(object_tokens).squeeze(-1)

        box_tokens = self.box_proj(slot_tokens)
        for block in self.box_blocks:
            box_tokens = block(
                box_tokens,
                context=image_context,
                time_emb=time_embedding,
                context_key_padding_mask=image_key_padding_mask,
            )
        box_tokens = self.box_obj_cross(
            box_tokens,
            context=object_tokens.detach(),
            time_emb=time_embedding,
        )
        for block in self.box_refine_blocks:
            box_tokens = block(
                box_tokens,
                context=image_context,
                time_emb=time_embedding,
                context_key_padding_mask=image_key_padding_mask,
            )
        return {"pred_boxes": self.box_head(box_tokens), "obj_logits": obj_logits}


class BoxDiffusionPipeline(pl.LightningModule):
    """End-to-end tensor-native Lightning pipeline for box-set diffusion."""

    def __init__(
        self,
        *,
        backbone: BaseFeatureExtractor,
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
        num_slots: int = 20,
        dim: int = 256,
        num_heads: int = 8,
        num_blocks: int = 3,
        num_obj_blocks: int = 2,
        num_box_branch_blocks: int = 2,
        num_box_refine_blocks: int = 2,
        obj_dim: int = 256,
        box_dim: int = 256,
        time_dim: int = 256,
        ffn_mult: float = 4.0,
        image_token_dim: int = 1280,
        fourier_box_bands: int = 16,
        fourier_box_scale: float = 1000.0,
        fourier_sigma_bands: int = 16,
        fourier_sigma_scale: float = 1.0,
        sigma_log_offset: float = 1e-4,
        pe_scale: float = 1000.0,
        self_cond: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.denoiser = _BoxDiffusionDenoiser(
            num_slots=num_slots,
            dim=dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            num_obj_blocks=num_obj_blocks,
            num_box_branch_blocks=num_box_branch_blocks,
            num_box_refine_blocks=num_box_refine_blocks,
            obj_dim=obj_dim,
            box_dim=box_dim,
            time_dim=time_dim,
            ffn_mult=ffn_mult,
            image_token_dim=image_token_dim,
            fourier_box_bands=fourier_box_bands,
            fourier_box_scale=fourier_box_scale,
            fourier_sigma_bands=fourier_sigma_bands,
            fourier_sigma_scale=fourier_sigma_scale,
            sigma_log_offset=sigma_log_offset,
            pe_scale=pe_scale,
            self_cond=self_cond,
        )
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
        self._val_ap_metric = BoxAPMetric(ann_file=val_ann_file) if val_ann_file else None
        self._val_ap_metric_1step = BoxAPMetric(ann_file=val_ann_file) if val_ann_file else None
        self._test_ap_metric = BoxAPMetric(ann_file=test_ann_file) if test_ann_file else None
        self._test_ap_metric_1step = BoxAPMetric(ann_file=test_ann_file) if test_ann_file else None
        self._log_loss_names = (
            "l1_loss", "giou_loss", "diff_l2_loss", "obj_loss", "ranking_loss", "matched"
        )
        self.save_hyperparameters(ignore=["backbone", "weights"])

    @property
    def num_slots(self) -> int:
        return self.denoiser.num_slots

    def _image_condition(self, images: Tensor) -> tuple[Tensor, Tensor]:
        return _image_tokens_and_centers(self.backbone(image=images), self.feature_key)

    def forward(
        self,
        images: Tensor,
        noisy_boxes: Tensor,
        sigma: Tensor,
        image_key_padding_mask: Optional[Tensor] = None,
        x0_prev: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Predict clean canonical boxes and objectness from image tensors."""
        image_tokens, patch_centers = self._image_condition(images)
        return self.denoiser(
            noisy_boxes,
            sigma,
            image_tokens,
            patch_centers,
            image_key_padding_mask=image_key_padding_mask,
            x0_prev=x0_prev,
        )

    @staticmethod
    def edm_sigma_schedule(
        num_steps: int,
        sigma_min: float = 0.002,
        sigma_max: float = 1.0,
        rho: float = 7.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1; got {num_steps}")
        step_indices = torch.arange(num_steps, device=device, dtype=dtype)
        inverse_rho = 1.0 / rho
        sigmas = (
            sigma_max ** inverse_rho
            + step_indices / max(num_steps - 1, 1)
            * (sigma_min ** inverse_rho - sigma_max ** inverse_rho)
        ) ** rho
        return torch.cat([sigmas, sigmas.new_zeros(1)])

    @torch.no_grad()
    def sample(
        self,
        images: Tensor,
        *,
        num_inference_steps: int | None = None,
        image_key_padding_mask: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
        x_init: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Run EDM Heun sampling and return canonical boxes plus objectness logits."""
        steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        if steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1; got {steps}")
        image_tokens, patch_centers = self._image_condition(images)
        batch_size = images.shape[0]
        dtype = image_tokens.dtype
        if x_init is None:
            boxes = torch.empty(
                batch_size, self.num_slots, 4, device=images.device, dtype=dtype
            ).normal_(generator=generator) * self.sigma_max
        else:
            boxes = x_init.to(device=images.device, dtype=dtype)

        if steps == 1:
            sigma = torch.full(
                (batch_size,), self.sigma_max, device=images.device, dtype=dtype
            )
            return self.denoiser(
                boxes, sigma, image_tokens, patch_centers, image_key_padding_mask
            )

        sigmas = self.edm_sigma_schedule(
            steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            device=images.device,
            dtype=dtype,
        )
        if x_init is None:
            boxes = boxes * (sigmas[0] / max(self.sigma_max, 1e-12))

        x0_prev: Tensor | None = None
        last_obj_logits: Tensor | None = None
        for step_index in range(steps):
            sigma_current = sigmas[step_index]
            sigma_next = sigmas[step_index + 1]
            current = self.denoiser(
                boxes,
                sigma_current.expand(batch_size),
                image_tokens,
                patch_centers,
                image_key_padding_mask,
                x0_prev,
            )
            x0 = current["pred_boxes"]
            last_obj_logits = current["obj_logits"]
            derivative = (boxes - x0) / sigma_current.clamp_min(
                self.denoiser.sigma_log_offset
            )
            delta_sigma = sigma_next - sigma_current
            euler_boxes = boxes + delta_sigma * derivative
            if sigma_next > 0:
                corrected = self.denoiser(
                    euler_boxes,
                    sigma_next.expand(batch_size),
                    image_tokens,
                    patch_centers,
                    image_key_padding_mask,
                    x0_prev,
                )
                next_x0 = corrected["pred_boxes"]
                last_obj_logits = corrected["obj_logits"]
                next_derivative = (euler_boxes - next_x0) / sigma_next
                boxes = boxes + 0.5 * delta_sigma * (derivative + next_derivative)
                x0_prev = next_x0.detach()
            else:
                boxes = euler_boxes
                x0_prev = x0.detach()
        assert last_obj_logits is not None
        return {"pred_boxes": boxes, "obj_logits": last_obj_logits}

    def _sample_sigma(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        log_sigma = self.sigma_p_mean + self.sigma_p_std * torch.randn(
            batch_size, device=device, dtype=dtype
        )
        return log_sigma.exp().clamp(self.sigma_min, self.sigma_max)

    def _build_noisy_input(
        self,
        gt_boxes_canon: Tensor,
        gt_box_labeled: Tensor,
        sigma: Tensor,
    ) -> Tensor:
        batch_size, max_people, _ = gt_boxes_canon.shape
        random_boxes = torch.stack(
            [
                torch.rand(batch_size, self.num_slots, device=gt_boxes_canon.device, dtype=gt_boxes_canon.dtype) * 2.0 - 1.0,
                torch.rand(batch_size, self.num_slots, device=gt_boxes_canon.device, dtype=gt_boxes_canon.dtype) * 2.0 - 1.0,
                torch.ones(batch_size, self.num_slots, device=gt_boxes_canon.device, dtype=gt_boxes_canon.dtype),
                torch.ones(batch_size, self.num_slots, device=gt_boxes_canon.device, dtype=gt_boxes_canon.dtype),
            ],
            dim=-1,
        )
        for batch_index in range(batch_size):
            real_count = min(
                int(gt_box_labeled[batch_index].sum().item()), self.num_slots, max_people
            )
            if real_count:
                random_boxes[batch_index, :real_count] = gt_boxes_canon[batch_index, :real_count]
        return random_boxes + sigma.view(batch_size, 1, 1).to(random_boxes.dtype) * torch.randn_like(random_boxes)

    def _losses_for_batch(self, batch: dict, images: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        batch_size = images.shape[0]
        sigma = self._sample_sigma(batch_size, images.device, images.dtype)
        noisy_boxes = self._build_noisy_input(
            batch["gt_boxes_canon"], batch["gt_box_labeled"], sigma
        ).detach().requires_grad_(True)
        x0_prev: Tensor | None = None
        if self.denoiser.self_cond:
            self_condition_mask = torch.rand(batch_size, device=images.device) < 0.5
            if self_condition_mask.any():
                with torch.no_grad():
                    previous = self(images, noisy_boxes.detach(), sigma)["pred_boxes"]
                x0_prev = previous * self_condition_mask.view(batch_size, 1, 1).to(previous.dtype)
        predictions = self(images, noisy_boxes, sigma, x0_prev=x0_prev)
        return self._loss(
            predictions,
            {
                "gt_boxes_canon": batch["gt_boxes_canon"],
                "gt_box_labeled": batch["gt_box_labeled"],
            },
        ), sigma

    def _log_losses(self, prefix: str, losses: dict[str, Tensor], batch_size: int, *, on_step: bool) -> None:
        self.log(f"{prefix}/loss", losses["total"], prog_bar=True, batch_size=batch_size, on_step=on_step, on_epoch=True)
        for name in self._log_loss_names:
            self.log(f"{prefix}/{name}", losses[name], batch_size=batch_size, on_step=on_step, on_epoch=True)

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        del batch_idx
        images = batch["image"]
        boxes_canon = batch["gt_boxes_canon"]
        if self.noise_std:
            images = add_gaussian_noise(images, self.noise_std)
        if self.brightness or self.contrast or self.saturation:
            images = color_jitter(images, self.brightness, self.contrast, self.saturation)
        if self.hflip_p:
            images, boxes_canon = random_hflip_image_and_boxes(images, boxes_canon, self.hflip_p)
        losses, sigma = self._losses_for_batch(
            {**batch, "gt_boxes_canon": boxes_canon}, images
        )
        self._log_losses("train", losses, images.shape[0], on_step=True)
        self.log("debug/sigma_mean", sigma.mean(), batch_size=images.shape[0], on_step=True, on_epoch=False)
        return losses["total"]

    def _ap_epoch_active(self) -> bool:
        is_period = (self.current_epoch + 1) % self.val_ap_every_n_epochs == 0
        is_last = self._trainer is not None and (self.current_epoch + 1) == self.trainer.max_epochs
        return is_period or is_last

    def _update_ap_metric(self, metric: BoxAPMetric, predictions: dict[str, Tensor], batch: dict) -> None:
        pred_boxes = predictions["pred_boxes"].detach()
        metric.update(
            pred_boxes_xywh=boxes_padded_canonical_to_pixel_xywh(
                pred_boxes, batch["img_hw"].to(pred_boxes.device)
            ).float(),
            pred_scores=torch.sigmoid(predictions["obj_logits"]).detach().float(),
            image_ids=batch["image_id"].to(pred_boxes.device),
        )

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        del batch_idx
        losses, _ = self._losses_for_batch(batch, batch["image"])
        self._log_losses("val", losses, batch["image"].shape[0], on_step=False)
        if self._ap_epoch_active():
            if self._val_ap_metric is not None:
                self._update_ap_metric(self._val_ap_metric, self.sample(batch["image"]), batch)
            if self._val_ap_metric_1step is not None:
                self._update_ap_metric(self._val_ap_metric_1step, self.sample(batch["image"], num_inference_steps=1), batch)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        del batch_idx
        losses, _ = self._losses_for_batch(batch, batch["image"])
        self._log_losses("test", losses, batch["image"].shape[0], on_step=False)
        if self._test_ap_metric is not None:
            self._update_ap_metric(self._test_ap_metric, self.sample(batch["image"]), batch)
        if self._test_ap_metric_1step is not None:
            self._update_ap_metric(self._test_ap_metric_1step, self.sample(batch["image"], num_inference_steps=1), batch)

    def _log_and_reset_metric(self, metric: BoxAPMetric | None, prefix: str) -> None:
        if metric is not None:
            for name, value in metric.compute().items():
                self.log(f"{prefix}/{name}", value, sync_dist=True)
            metric.reset()

    def on_validation_epoch_end(self) -> None:
        if self._ap_epoch_active():
            self._log_and_reset_metric(self._val_ap_metric, f"ap-val/{self.num_inference_steps}-step")
            self._log_and_reset_metric(self._val_ap_metric_1step, "ap-val/1-step")

    def on_test_epoch_end(self) -> None:
        self._log_and_reset_metric(self._test_ap_metric, f"ap-test/{self.num_inference_steps}-step")
        self._log_and_reset_metric(self._test_ap_metric_1step, "ap-test/1-step")

    def configure_optimizers(self):
        return build_adamw_with_schedule(
            self,
            lr=self.lr,
            weight_decay=self.weight_decay,
            lr_T_max=self.lr_T_max,
            lr_warmup_epochs=self.lr_warmup_epochs,
            lr_min_frac=self.lr_min_frac,
        )


class BoxDiffusionLossWeightsConfig(BaseConfig):
    l1: float = 5.0
    giou: float = 2.0
    diff_l2: float = 1.0
    obj_bce: float = 1.0
    rank: float = 1.0
    matcher_lambda_l1: float = 5.0
    matcher_lambda_giou: float = 2.0
    matcher_lambda_obj: float = 0.1

    def to_dataclass(self) -> SetBoxDiffusionLossWeights:
        return SetBoxDiffusionLossWeights(**self.model_dump())


class BoxDiffusionPipelineConfig(BaseConfig):
    backbone: BaseFeatureExtractorConfig
    lr: float = 2e-4
    weight_decay: float = 1e-4
    cosine_lr: bool = True
    lr_warmup_epochs: int = 1
    lr_min_frac: float = 1e-2
    max_epochs: int = 100
    seed: int = 42
    grad_clip: float = 1.0
    log_every_n_steps: int = 50
    resume: str | None = None
    sigma_min: float = 0.002
    sigma_max: float = 10.0
    rho: float = 7.0
    num_inference_steps: int = 5
    sigma_p_mean: float = -1.2
    sigma_p_std: float = 1.2
    val_ap_every_n_epochs: int = 10
    feature_key: str = "spatial"
    noise_std: float = 0.02
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.1
    hflip_p: float = 0.5
    num_slots: int = 20
    dim: int = 512
    num_heads: int = 8
    num_blocks: int = 3
    num_obj_blocks: int = 1
    num_box_branch_blocks: int = 1
    num_box_refine_blocks: int = 1
    obj_dim: int = 256
    box_dim: int = 256
    time_dim: int = 256
    ffn_mult: float = 4.0
    fourier_box_bands: int = 16
    fourier_box_scale: float = 1000.0
    fourier_sigma_bands: int = 16
    fourier_sigma_scale: float = 100.0
    sigma_log_offset: float = 1e-4
    pe_scale: float = 1000.0
    self_cond: bool = True
    weights: BoxDiffusionLossWeightsConfig = Field(default_factory=BoxDiffusionLossWeightsConfig)

    def build(
        self,
        *,
        val_ann_file: str | None = None,
        test_ann_file: str | None = None,
    ) -> BoxDiffusionPipeline:
        backbone = self.backbone.build()
        try:
            image_token_dim = backbone.out_channels
        except AttributeError as exc:
            raise ValueError("box-diffusion backbones must expose `out_channels`") from exc
        return BoxDiffusionPipeline(
            backbone=backbone,
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
            sigma_p_mean=self.sigma_p_mean,
            sigma_p_std=self.sigma_p_std,
            val_ap_every_n_epochs=self.val_ap_every_n_epochs,
            val_ann_file=val_ann_file,
            test_ann_file=test_ann_file,
            feature_key=self.feature_key,
            noise_std=self.noise_std,
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hflip_p=self.hflip_p,
            num_slots=self.num_slots,
            dim=self.dim,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            num_obj_blocks=self.num_obj_blocks,
            num_box_branch_blocks=self.num_box_branch_blocks,
            num_box_refine_blocks=self.num_box_refine_blocks,
            obj_dim=self.obj_dim,
            box_dim=self.box_dim,
            time_dim=self.time_dim,
            ffn_mult=self.ffn_mult,
            image_token_dim=image_token_dim,
            fourier_box_bands=self.fourier_box_bands,
            fourier_box_scale=self.fourier_box_scale,
            fourier_sigma_bands=self.fourier_sigma_bands,
            fourier_sigma_scale=self.fourier_sigma_scale,
            sigma_log_offset=self.sigma_log_offset,
            pe_scale=self.pe_scale,
            self_cond=self.self_cond,
        )
