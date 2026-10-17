from typing import ClassVar

from .latent_v16 import V16StateCodec
from .model_v16 import PoseFlowV16Model
from .training_v15 import PoseFlowV15Pipeline, PoseFlowV15PipelineConfig


class PoseFlowV16Pipeline(PoseFlowV15Pipeline):
    latent_role_slices = V16StateCodec.latent_role_slices

    def _quick_architecture(self):
        return {**super()._quick_architecture(), 'state_tokens': self.model.num_state_tokens,
                'decoder_representation_tokens': self.model.num_state_tokens+1,
                'state_roles': ['confidence', 'center', 'size', 'joints'],
                'box_role_references': 'assembled 4D box for confidence, center and size',
                'latent_loss_groups': 'mean(confidence, center, size), joints; equal family weighting'}


class PoseFlowV16PipelineConfig(PoseFlowV15PipelineConfig):
    latent_model_class: ClassVar[type] = PoseFlowV16Model
    latent_pipeline_class: ClassVar[type] = PoseFlowV16Pipeline
