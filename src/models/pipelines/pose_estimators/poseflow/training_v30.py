from typing import ClassVar

from pydantic import Field

from .training_v26 import PoseFlowV26Pipeline
from .training_v28 import PoseFlowV28PipelineConfig


class PoseFlowV30Pipeline(PoseFlowV26Pipeline):
    def _denoised_task_predictions(self, denoised):
        return self.model.readout(denoised['pred_latents'].detach())


class PoseFlowV30PipelineConfig(PoseFlowV28PipelineConfig):
    pipeline_class: ClassVar[type] = PoseFlowV30Pipeline
    denoised_task_loss_weight: float = Field(default=.5, ge=0)
