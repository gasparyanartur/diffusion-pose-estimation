from pydantic import Field

from .model_v33 import PoseFlowV33Model
from .training_v32 import PoseFlowV32PipelineConfig


class PoseFlowV33PipelineConfig(PoseFlowV32PipelineConfig):
    pose_readout_layers: int = Field(default=3, ge=1)

    def _build_poseflow_model(self):
        names = ('backbone_pretrained', 'timesteps', 'num_proposals', 'num_body_points',
            'hidden_dim', 'num_heads', 'feedforward_dim', 'num_feature_levels',
            'freeze_backbone_stem_and_layer1', 'freeze_backbone')
        return PoseFlowV33Model(**{name:getattr(self, name) for name in names},
            sampling_steps=self.flow_sampling_steps, dino_grid_size=self.dino_grid_size,
            pose_readout_layers=self.pose_readout_layers)
