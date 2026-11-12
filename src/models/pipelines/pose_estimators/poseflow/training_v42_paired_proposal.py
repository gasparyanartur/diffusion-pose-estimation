from typing import ClassVar, Literal

import torch
from torch import Tensor

from ..diffusionregpose.training import (
    DiffusionRegPoseMatcher, build_diffusionregpose_clean_targets,
)
from .training_v42 import PoseFlowV42Pipeline, PoseFlowV42PipelineConfig


@torch.no_grad()
def build_proposal_paired_pose_inputs(
    selected_logits: Tensor,
    reference_boxes: Tensor,
    timesteps: Tensor,
    targets: list[dict[str, Tensor]],
    *,
    diffusion,
    matcher: DiffusionRegPoseMatcher,
    noise: Tensor | None = None,
    noise_seed: int | None = None,
) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
    # B = images, N = selected persons, K = joints.
    if reference_boxes.ndim != 3 or reference_boxes.shape[-1] != 4:
        raise ValueError('Reference boxes must have shape [B, N, 4]')
    batch_size, capacity, _ = reference_boxes.shape
    if selected_logits.ndim != 3 or selected_logits.shape[:2] != (batch_size, capacity):
        raise ValueError('Selected logits must have shape [B, N, C]')
    if len(targets) != batch_size or capacity < 1:
        raise ValueError('Targets must match the proposal batch with positive capacity')
    if any(len(target['labels']) > capacity for target in targets):
        raise ValueError('Valid person count exceeds selected proposal capacity')
    if timesteps.dtype != torch.long:
        raise TypeError('Timesteps must have dtype torch.long')
    if timesteps.shape != (batch_size,) or timesteps.device != reference_boxes.device:
        raise ValueError('Timesteps must have shape [B] on the proposal device')
    if ((timesteps < 0) | (timesteps >= diffusion.num_timesteps)).any():
        raise ValueError('Timesteps must lie in the diffusion schedule')
    matches = matcher(dict(pred_logits=selected_logits, pred_boxes=reference_boxes),
                      targets, include_keypoints=False)
    generator = None
    if noise_seed is not None:
        generator = torch.Generator(device=reference_boxes.device).manual_seed(noise_seed)
    coordinate_count = 2 * matcher.num_body_points
    clean = reference_boxes.new_empty(batch_size, capacity, coordinate_count)
    for batch_index, (queries, people) in enumerate(matches):
        assigned = (targets[batch_index]['keypoints'][people, :coordinate_count] * 2 - 1) * diffusion.scale
        clean[batch_index, queries] = assigned
        background = torch.ones(capacity, device=reference_boxes.device, dtype=torch.bool)
        background[queries] = False
        clean[batch_index, background] = torch.randn(
            (capacity - len(queries), coordinate_count), device=clean.device,
            dtype=clean.dtype, generator=generator,
        ) * diffusion.scale
    if noise is None:
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    elif noise.shape != clean.shape or noise.device != clean.device:
        raise ValueError('Noise must match the paired coordinate shape and device')
    noisy = diffusion.q_sample(clean, timesteps, noise=noise).to(clean)
    return noisy.clamp(-diffusion.scale, diffusion.scale), matches


class PoseFlowV42PairedProposalPipeline(PoseFlowV42Pipeline):
    def _additional_pose_losses(self, outputs, targets, timesteps, losses):
        return losses

    def _build_paired_pose_inputs(self, selected_logits, reference_boxes, timesteps, targets):
        return build_proposal_paired_pose_inputs(
            selected_logits, reference_boxes, timesteps, targets,
            diffusion=self.model, matcher=self.criterion.matcher, noise_seed=self.diffusion_noise_seed,
        )

    def forward(self, images, diffusion_keypoints, timesteps, targets=None, padding_mask=None):
        return self.model(images, diffusion_keypoints, timesteps, targets, padding_mask,
            pose_input_builder=self._build_paired_pose_inputs if targets is not None else None)

    def _losses_for_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        keypoints = batch['gt_keypoints']
        targets = build_diffusionregpose_clean_targets(
            batch, num_body_points=self.model.transformer.decoder.num_body_points,
        )
        generator = None
        if self.diffusion_noise_seed is not None:
            generator = torch.Generator(device=keypoints.device).manual_seed(self.diffusion_noise_seed)
        timesteps = torch.randint(self.model.num_timesteps, (keypoints.shape[0],),
            device=keypoints.device, generator=generator)
        placeholder = keypoints.new_zeros(keypoints.shape[0], self.model.num_proposals,
                                         self.model.transformer.decoder.num_body_points * 2)
        outputs = self._forward_diffusion_batch(batch, placeholder, timesteps, targets)
        matches = outputs['pose_target_matches']
        auxiliary_matches = [
            matches if index >= self.criterion.num_box_decoder_layers else None
            for index, _ in enumerate(outputs.get('aux_outputs', []))
        ]
        losses = self.criterion(outputs, targets, final_matches=matches,
                                auxiliary_matches=auxiliary_matches)
        matched_count = sum(len(people) for _, people in matches)
        target_count = sum(len(target['labels']) for target in targets)
        if matched_count != target_count:
            raise RuntimeError('Proposal pairing must cover every valid clean target')
        losses.update({
            'diffusion/timestep_mean': timesteps.float().mean(),
            'diffusion/timestep_min': timesteps.min().float(),
            'diffusion/timestep_max': timesteps.max().float(),
            'paired_proposal/matched_count': keypoints.new_tensor(matched_count),
            'paired_proposal/background_count': keypoints.new_tensor(placeholder.shape[0] * placeholder.shape[1] - matched_count),
            'paired_proposal/target_coverage': keypoints.new_tensor(matched_count / max(target_count, 1)),
        })
        return self._additional_pose_losses(outputs, targets, timesteps, losses)


class PoseFlowV42PairedProposalPipelineConfig(PoseFlowV42PipelineConfig):
    pipeline_class: ClassVar[type[PoseFlowV42Pipeline]] = PoseFlowV42PairedProposalPipeline
    target_assignment: Literal['proposal_hungarian_fixed'] = 'proposal_hungarian_fixed'
    training_timestep_policy: Literal['uniform'] = 'uniform'

    def build(self, **kwargs):
        if self.matching_strategy != 'hungarian':
            raise ValueError('V42 paired proposals require Hungarian matching')
        pipeline = super().build(**kwargs)
        pipeline.save_hyperparameters({'target_assignment': self.target_assignment})
        return pipeline
