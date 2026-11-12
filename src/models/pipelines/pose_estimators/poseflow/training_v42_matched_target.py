from typing import ClassVar, Literal

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from ..diffusionregpose.training import (
    DiffusionRegPoseMatcher,
    build_diffusionregpose_targets,
    pairwise_pose_matching_costs,
)
from .training_v42 import PoseFlowV42Pipeline, PoseFlowV42PipelineConfig


@torch.no_grad()
def noisy_pose_matching_costs(
    noisy_poses: Tensor,
    targets: list[dict[str, Tensor]],
    *,
    scale: float,
    matcher: DiffusionRegPoseMatcher,
) -> list[Tensor]:
    # B = images, N = noisy person slots, K = joints; input is <B, N, 2*K>.
    if scale <= 0 or noisy_poses.ndim != 3 or noisy_poses.shape[-1] != 2 * matcher.num_body_points:
        raise ValueError('Noisy poses must have shape [B, N, 2*K] and positive scale')
    if noisy_poses.shape[0] != len(targets):
        raise ValueError('Noisy poses and targets must have the same batch size')
    normalized_xy = (noisy_poses / scale + 1) / 2
    costs = []
    for poses, target in zip(normalized_xy, targets):
        keypoint_cost, oks = pairwise_pose_matching_costs(
            poses, target, num_body_points=matcher.num_body_points, sigmas=matcher.sigmas,
        )
        costs.append(matcher.weights.set_cost_keypoints * keypoint_cost
                     + matcher.weights.set_cost_oks * (1 - oks.clamp_min(1e-6)))
    return costs


def _assign_pose_costs(costs: list[Tensor]) -> list[tuple[Tensor, Tensor]]:
    matches = []
    for cost in costs:
        query_indices, target_indices = linear_sum_assignment(cost.cpu())
        matches.append((torch.as_tensor(query_indices, device=cost.device, dtype=torch.long),
                        torch.as_tensor(target_indices, device=cost.device, dtype=torch.long)))
    return matches


def match_noisy_pose_targets(
    noisy_poses: Tensor,
    targets: list[dict[str, Tensor]],
    *,
    scale: float,
    matcher: DiffusionRegPoseMatcher,
) -> list[tuple[Tensor, Tensor]]:
    return _assign_pose_costs(noisy_pose_matching_costs(
        noisy_poses, targets, scale=scale, matcher=matcher,
    ))


@torch.no_grad()
def input_assignment_diagnostics(
    costs: list[Tensor],
    matches: list[tuple[Tensor, Tensor]],
    output_matches: list[tuple[Tensor, Tensor]],
    targets: list[dict[str, Tensor]],
    reference: Tensor,
) -> dict[str, Tensor]:
    matched_costs = []
    agreements = []
    matched_count = 0
    for cost, (queries, people), (output_queries, output_people) in zip(costs, matches, output_matches):
        matched_costs.append(cost[queries, people])
        assigned_people = torch.full((cost.shape[0],), -1, device=cost.device, dtype=torch.long)
        assigned_people[queries] = people
        agreements.append((assigned_people[output_queries] == output_people).to(reference))
        matched_count += people.numel()
    target_count = sum(target['labels'].numel() for target in targets)
    cost_values = torch.cat(matched_costs)
    agreement_values = torch.cat(agreements)
    zero = reference.new_zeros(())
    return {
        'input_match/cost': cost_values.mean() if cost_values.numel() else zero,
        'input_match/target_coverage': reference.new_tensor(matched_count / max(target_count, 1)),
        'input_match/output_disagreement': 1 - agreement_values.mean() if agreement_values.numel() else zero,
    }


class PoseFlowV42MatchedTargetPipeline(PoseFlowV42Pipeline):
    def _losses_for_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        model = self.model
        noisy_poses, timesteps, targets = build_diffusionregpose_targets(
            batch, model, num_proposals=model.num_proposals,
            num_body_points=model.transformer.decoder.num_body_points,
            noise_seed=self.diffusion_noise_seed,
        )
        costs = noisy_pose_matching_costs(noisy_poses, targets, scale=model.scale,
                                         matcher=self.criterion.matcher)
        matches = _assign_pose_costs(costs)
        outputs = self._forward_diffusion_batch(batch, noisy_poses, timesteps, targets)
        auxiliary_matches = [
            matches if index >= self.criterion.num_box_decoder_layers else None
            for index, _ in enumerate(outputs.get('aux_outputs', []))
        ]
        losses = self.criterion(outputs, targets, final_matches=matches,
                                auxiliary_matches=auxiliary_matches)
        output_matches = self.criterion.matcher(outputs, targets)
        losses.update(input_assignment_diagnostics(costs, matches, output_matches,
                                                   targets, losses['total']))
        return losses


class PoseFlowV42MatchedTargetPipelineConfig(PoseFlowV42PipelineConfig):
    pipeline_class: ClassVar[type[PoseFlowV42Pipeline]] = PoseFlowV42MatchedTargetPipeline
    target_assignment: Literal['noisy_input_hungarian'] = 'noisy_input_hungarian'

    def build(self, **kwargs):
        if self.matching_strategy != 'hungarian':
            raise ValueError('V42 matched targets require Hungarian matching')
        pipeline = super().build(**kwargs)
        pipeline.save_hyperparameters({'target_assignment': self.target_assignment})
        return pipeline
