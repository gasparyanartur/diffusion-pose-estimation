from functools import partial
from typing import ClassVar, Literal

import torch
from torch import Tensor

from .sampling_v42 import match_proposals_to_noisy_poses, sample_v42_with_proposal_pose_matching
from .training_v42_paired_proposal import (
    PoseFlowV42PairedProposalPipeline, PoseFlowV42PairedProposalPipelineConfig,
    build_proposal_paired_pose_inputs,
)


@torch.no_grad()
def build_noisy_proposal_pose_inputs(
    selected_logits: Tensor,
    reference_boxes: Tensor,
    timesteps: Tensor,
    targets: list[dict[str, Tensor]],
    *,
    diffusion,
    matcher,
    noise: Tensor | None = None,
    noise_seed: int | None = None,
) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
    noisy, source_matches = build_proposal_paired_pose_inputs(
        selected_logits, reference_boxes, timesteps, targets,
        diffusion=diffusion, matcher=matcher, noise=noise, noise_seed=noise_seed,
    )
    # B = images, N = person slots. permutation[B,N] maps proposal -> source.
    permutation = match_proposals_to_noisy_poses(
        reference_boxes, noisy, scale=diffusion.scale, matcher=matcher,
        selected_logits=selected_logits,
    )
    ordered = noisy.gather(1, permutation[..., None].expand_as(noisy))
    inverse = permutation.argsort(dim=1)
    matches = [(inverse[image, queries], people)
               for image, (queries, people) in enumerate(source_matches)]
    return ordered, matches


class PoseFlowV42NoisyProposalPipeline(PoseFlowV42PairedProposalPipeline):
    def _build_paired_pose_inputs(self, selected_logits, reference_boxes, timesteps, targets):
        return build_noisy_proposal_pose_inputs(
            selected_logits, reference_boxes, timesteps, targets,
            diffusion=self.model, matcher=self.criterion.matcher, noise_seed=self.diffusion_noise_seed,
        )

    def _evaluation_sampler_kwargs(self):
        return dict(
            sampler=partial(sample_v42_with_proposal_pose_matching, matcher=self.criterion.matcher),
            sampler_description='coordinate cosine DDIM eta=0; proposal/noisy-pose Hungarian every call; source-following training',
        )

    def _sample_validation_outputs(self, image, **sampling_kwargs):
        return sample_v42_with_proposal_pose_matching(
            self.model, image, matcher=self.criterion.matcher, **sampling_kwargs,
        )


class PoseFlowV42NoisyProposalPipelineConfig(PoseFlowV42PairedProposalPipelineConfig):
    pipeline_class: ClassVar[type[PoseFlowV42NoisyProposalPipeline]] = PoseFlowV42NoisyProposalPipeline
    target_assignment: Literal['noisy_proposal_source_fixed'] = 'noisy_proposal_source_fixed'
    inference_association: Literal['proposal_noisy_pose_hungarian'] = 'proposal_noisy_pose_hungarian'

    def build(self, **kwargs):
        pipeline = super().build(**kwargs)
        pipeline.save_hyperparameters({'inference_association': self.inference_association})
        return pipeline
