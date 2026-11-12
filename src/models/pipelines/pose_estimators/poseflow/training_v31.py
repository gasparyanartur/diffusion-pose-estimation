from pydantic import Field

from .training_v28 import PoseFlowV28PipelineConfig


class PoseFlowV31PipelineConfig(PoseFlowV28PipelineConfig):
    denoised_task_loss_weight: float = Field(default=.5, ge=0)
