from pydantic import Field

from .model_v28 import PoseFlowV28Model
from .training_v26 import PoseFlowV26PipelineConfig


class PoseFlowV28PipelineConfig(PoseFlowV26PipelineConfig):
    denoised_task_loss_weight: float = Field(default=0., ge=0)

    def _build_poseflow_model(self):
        fields = ('backbone_pretrained', 'timesteps', 'num_proposals', 'num_body_points',
                  'hidden_dim', 'num_heads', 'feedforward_dim', 'num_feature_levels',
                  'freeze_backbone_stem_and_layer1', 'freeze_backbone')
        return PoseFlowV28Model(**{name:getattr(self, name) for name in fields},
                               sampling_steps=self.flow_sampling_steps)
