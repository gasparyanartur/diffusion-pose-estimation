from typing import ClassVar

from pydantic import Field, model_validator

from .model_v27 import PoseFlowV27Model
from .training_v26 import PoseFlowV26PipelineConfig


class PoseFlowV27PipelineConfig(PoseFlowV26PipelineConfig):
    image_source_feature_levels: ClassVar[tuple[int, ...]] = (0, 1, 2)
    num_feature_levels: int = 3
    denoised_task_loss_weight: float = Field(default=0., ge=0)

    @model_validator(mode='after')
    def validate_v27(self):
        if self.num_feature_levels != 3:
            raise ValueError('V27 requires three coarse-to-fine image feature levels')
        return self

    def _build_poseflow_model(self):
        fields = ('backbone_pretrained', 'timesteps', 'num_proposals', 'num_body_points',
                  'hidden_dim', 'num_heads', 'feedforward_dim', 'num_feature_levels',
                  'freeze_backbone_stem_and_layer1', 'freeze_backbone')
        return PoseFlowV27Model(**{name:getattr(self, name) for name in fields},
                               sampling_steps=self.flow_sampling_steps)
