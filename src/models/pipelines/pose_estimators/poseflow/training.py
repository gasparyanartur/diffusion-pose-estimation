"""Continuous endpoint-flow training and sampling for PoseFlow."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Literal

import torch
from pydantic import field_validator
from torch import Tensor, nn

from src.models.pipelines.pose_estimators.diffusionregpose.training import (
    CROWDPOSE_PERSON_LABEL,
    DiffusionRegPosePipeline,
    DiffusionRegPosePipelineConfig,
    _keypoints_to_xyxyzz,
    _postprocess_diffusionregpose_ap,
    diffusionregpose_ap_epoch_active,
    update_diffusionregpose_ap_metric,
    update_diffusionregpose_bbox_ap_metric,
)
from .model import PoseFlowModel


def _flow_generator(
    tensor: Tensor, seed: int | None
) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(seed)
    return generator


def _transport_to_decoder_coordinates(state: Tensor, scale: float) -> Tensor:
    return ((state / scale) + 1.0).div(2.0).clamp(0.0, 1.0)


def build_poseflow_targets(
    batch: dict[str, Tensor],
    model: nn.Module,
    *,
    num_proposals: int,
    num_body_points: int,
    padding_policy: Literal["source_prior", "diffusionregpose"],
    noise_seed: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, list[dict[str, Tensor]]]:
    """Build endpoint-flow decoder inputs and existing semantic task targets."""
    keypoints = batch["gt_keypoints"]
    proposal_keypoints = batch.get("gt_diffusion_keypoints", keypoints)
    labeled = batch["gt_labeled"].bool()
    boxes = batch["gt_boxes_canon"]
    batch_size, person_count, point_count, _ = keypoints.shape
    if point_count != num_body_points:
        raise ValueError("batch keypoint count does not match the model")
    if boxes.shape[:2] != (batch_size, person_count):
        raise ValueError("box and keypoint target counts must agree")
    if proposal_keypoints.shape[:1] != (batch_size,) or proposal_keypoints.shape[2:] != (
        point_count,
        2,
    ):
        raise ValueError("proposal keypoints must match batch and keypoint dimensions")
    if num_proposals < 1:
        raise ValueError("num_proposals must be positive")

    scale = float(model.scale)
    generator = _flow_generator(keypoints, noise_seed)
    source = torch.randn(
        batch_size,
        num_proposals,
        point_count * 2,
        device=keypoints.device,
        dtype=keypoints.dtype,
        generator=generator,
    )
    endpoint = torch.empty_like(source)
    targets: list[dict[str, Tensor]] = []
    for batch_index in range(batch_size):
        valid_indices = labeled[batch_index].any(dim=-1).nonzero(as_tuple=False).flatten()
        target_keypoints = (keypoints[batch_index].index_select(0, valid_indices) + 1.0) / 2.0
        target_visibility = labeled[batch_index].index_select(0, valid_indices).float()
        target_boxes = boxes[batch_index].index_select(0, valid_indices)
        targets.append({
            "labels": torch.full(
                (valid_indices.numel(),),
                CROWDPOSE_PERSON_LABEL,
                dtype=torch.long,
                device=keypoints.device,
            ),
            "boxes": target_boxes,
            "keypoints": _keypoints_to_xyxyzz(
                torch.cat((target_keypoints, target_visibility.unsqueeze(-1)), dim=-1)
            ),
            "area": (target_boxes[:, 2] * target_boxes[:, 3]).clamp_min(1e-6),
        })

        proposal_indices = batch.get("diffusion_person_mask", batch.get("person_mask"))
        if proposal_indices is None:
            proposal_indices = torch.arange(proposal_keypoints.shape[1], device=keypoints.device)
        else:
            proposal_indices = proposal_indices[batch_index].bool().nonzero(as_tuple=False).flatten()
        clean = proposal_keypoints[batch_index].index_select(0, proposal_indices)
        clean = ((clean + 1.0) / 2.0).flatten(1)
        count = min(clean.shape[0], num_proposals)
        if count:
            endpoint[batch_index, :count] = (clean[:count] * 2.0 - 1.0) * scale
        if count < num_proposals:
            if padding_policy == "source_prior":
                endpoint[batch_index, count:] = source[batch_index, count:]
            else:
                endpoint[batch_index, count:] = torch.randn(
                    num_proposals - count,
                    point_count * 2,
                    device=keypoints.device,
                    dtype=keypoints.dtype,
                    generator=generator,
                ) * scale

    times = torch.rand(
        batch_size, device=keypoints.device, dtype=keypoints.dtype, generator=generator
    )
    state = (1.0 - times[:, None, None]) * source + times[:, None, None] * endpoint
    return _transport_to_decoder_coordinates(state, scale), source, times, targets


@torch.no_grad()
def flow_sample_keypoints(
    model: nn.Module,
    images: Tensor,
    *,
    padding_mask: Tensor | None = None,
    num_proposals: int | None = None,
    sampling_steps: int,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor | list[dict[str, Tensor]]]:
    """Run fixed-source forward-Euler endpoint flow sampling."""
    proposal_count = model.num_proposals if num_proposals is None else num_proposals
    if proposal_count < 1:
        raise ValueError(f"num_proposals must be positive; got {proposal_count}")
    if proposal_count != model.transformer.decoder.num_group:
        raise ValueError("num_proposals must equal the configured decoder group count")
    if sampling_steps < 1:
        raise ValueError(f"sampling_steps must be positive; got {sampling_steps}")

    source = torch.randn(
        images.shape[0],
        proposal_count,
        model.transformer.decoder.num_body_points * 2,
        device=images.device,
        dtype=images.dtype,
        generator=generator,
    )
    state = source.clone()
    final_output: dict[str, Tensor | list[dict[str, Tensor]]] | None = None
    for step in range(sampling_steps):
        times = torch.full(
            (images.shape[0],), step / sampling_steps, device=images.device, dtype=images.dtype
        )
        final_output = model(
            images,
            state,
            times,
            padding_mask=padding_mask,
        )
        predicted_keypoints = final_output["pred_keypoints"]
        if not isinstance(predicted_keypoints, Tensor):
            raise RuntimeError("model forward returned an invalid keypoint prediction")
        coordinate_count = model.transformer.decoder.num_body_points * 2
        endpoint = (predicted_keypoints[..., :coordinate_count] * 2.0 - 1.0) * model.scale
        state = state + (endpoint - source) / sampling_steps
    if final_output is None:
        raise RuntimeError("flow sampler did not execute")
    final_output["flow_keypoints"] = state
    return final_output


class PoseFlowPipeline(DiffusionRegPosePipeline):
    """DiffusionRegPose architecture trained and sampled as endpoint flow."""

    def __init__(self, *args, flow_padding_policy: Literal["source_prior", "diffusionregpose"], flow_sampling_steps: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flow_padding_policy = flow_padding_policy
        self.flow_sampling_steps = flow_sampling_steps

    def _losses_for_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        decoder_state, _, times, targets = build_poseflow_targets(
            batch,
            self.model,
            num_proposals=self.model.num_proposals,
            num_body_points=self.model.transformer.decoder.num_body_points,
            padding_policy=self.flow_padding_policy,
            noise_seed=self.diffusion_noise_seed,
        )
        rng_context = nullcontext() if self.deterministic_seed is None else torch.random.fork_rng(
            devices=[batch["image"].device] if batch["image"].is_cuda else []
        )
        with rng_context:
            if self.deterministic_seed is not None:
                torch.manual_seed(self.deterministic_seed)
            outputs = self(
                batch["image"], (decoder_state * 2.0 - 1.0) * self.model.scale, times,
                targets, batch.get("padding_mask"),
            )
        return self.criterion(outputs, targets)

    def sample_keypoints(
        self,
        images: Tensor,
        *,
        padding_mask: Tensor | None = None,
        num_proposals: int | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        """Sample endpoint-flow pose predictions with the configured solver."""
        return self.model.sample_keypoints(
            images,
            padding_mask=padding_mask,
            num_proposals=num_proposals,
            generator=generator,
        )

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        del batch_idx
        image = batch["image"]
        devices = [image.get_device()] if image.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            losses = self._losses_for_batch(batch)
            self._log_losses("val", losses, image.shape[0])
            if self._ap_epoch_active():
                generator = None
                if self.deterministic_seed is not None:
                    generator = torch.Generator(device=image.device)
                    generator.manual_seed(self.deterministic_seed)
                sampled_outputs = self.sample_keypoints(
                    image,
                    padding_mask=batch.get("padding_mask"),
                    generator=generator,
                )
                results = _postprocess_diffusionregpose_ap(
                    sampled_outputs, batch,
                    num_body_points=self.model.transformer.decoder.num_body_points,
                )
                update_diffusionregpose_ap_metric(
                    self._val_ap, sampled_outputs, batch,
                    num_body_points=self.model.transformer.decoder.num_body_points,
                    results=results,
                )
                assert self._val_bbox_ap is not None
                update_diffusionregpose_bbox_ap_metric(self._val_bbox_ap, results, batch)


class PoseFlowPipelineConfig(DiffusionRegPosePipelineConfig):
    """PoseFlow configuration; OTA is the default task-loss assignment."""

    matching_strategy: Literal["hungarian", "ota"] = "ota"
    flow_padding_policy: Literal["source_prior", "diffusionregpose"] = "source_prior"
    flow_sampling_steps: int = 1

    @field_validator("flow_sampling_steps")
    @classmethod
    def _validate_flow_sampling_steps(cls, value: int) -> int:
        if value < 1:
            raise ValueError("flow_sampling_steps must be positive")
        return value

    def _build_poseflow_model(self) -> PoseFlowModel:
        return PoseFlowModel(
            backbone_pretrained=self.backbone_pretrained,
            timesteps=self.timesteps,
            sampling_timesteps=self.sampling_timesteps,
            num_queries=self.num_queries,
            num_proposals=self.num_proposals,
            num_classes=self.num_classes,
            num_feature_levels=self.num_feature_levels,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            hidden_dim=self.hidden_dim,
            feedforward_dim=self.feedforward_dim,
            num_heads=self.num_heads,
            num_body_points=self.num_body_points,
            num_box_decoder_layers=self.num_box_decoder_layers,
            num_group=self.num_group,
            dn_number=self.dn_number,
            dn_box_noise_scale=self.dn_box_noise_scale,
            dn_label_noise_ratio=self.dn_label_noise_ratio,
            freeze_backbone_stem_and_layer1=self.freeze_backbone_stem_and_layer1,
            freeze_backbone=self.freeze_backbone,
            freeze_backbone_and_transformer=self.freeze_backbone_and_transformer,
            separate_expansion_pose_head=self.pretrained_keep_decoder_layers is not None,
            flow_sampling_steps=self.flow_sampling_steps,
        )

    def build(self, **kwargs) -> PoseFlowPipeline:
        diffusion_pipeline = super().build(**kwargs)
        model = self._build_poseflow_model()
        model.load_state_dict(diffusion_pipeline.model.state_dict(), strict=True)
        return PoseFlowPipeline(
            model,
            diffusion_pipeline.criterion,
            lr=diffusion_pipeline.lr,
            lr_backbone=diffusion_pipeline.lr_backbone,
            weight_decay=diffusion_pipeline.weight_decay,
            max_epochs=diffusion_pipeline.max_epochs,
            lr_warmup_epochs=diffusion_pipeline.lr_warmup_epochs,
            lr_min_frac=diffusion_pipeline.lr_min_frac,
            scheduler_type=diffusion_pipeline.scheduler_type,
            lr_decay_milestones=diffusion_pipeline.lr_decay_milestones,
            split_weight_decay=diffusion_pipeline.split_weight_decay,
            val_ap_every_n_epochs=diffusion_pipeline.val_ap_every_n_epochs,
            val_ann_file=kwargs.get("val_ann_file"),
            diffusion_noise_seed=diffusion_pipeline.diffusion_noise_seed,
            deterministic_seed=diffusion_pipeline.deterministic_seed,
            flow_padding_policy=self.flow_padding_policy,
            flow_sampling_steps=self.flow_sampling_steps,
        )
