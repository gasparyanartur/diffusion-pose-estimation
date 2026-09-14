"""PoseFlow model identity and flow-sampling boundary."""

from __future__ import annotations

import torch
from torch import Tensor

from src.models.pipelines.pose_estimators.diffusionregpose.model import (
    DiffusionRegPoseModel,
)


class PoseFlowModel(DiffusionRegPoseModel):
    """ED-Pose architecture whose proposal state is sampled as endpoint flow."""

    def __init__(self, *args, flow_sampling_steps: int = 1, **kwargs) -> None:
        if flow_sampling_steps < 1:
            raise ValueError("flow_sampling_steps must be positive")
        super().__init__(*args, **kwargs)
        self.flow_sampling_steps = flow_sampling_steps

    def sample_keypoints(
        self,
        images: Tensor,
        *,
        padding_mask: Tensor | None = None,
        num_proposals: int | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        """Sample through PoseFlow's fixed-source Euler solver."""
        from .training import flow_sample_keypoints

        return flow_sample_keypoints(
            self,
            images,
            padding_mask=padding_mask,
            num_proposals=num_proposals,
            sampling_steps=self.flow_sampling_steps,
            generator=generator,
        )
