from typing import ClassVar

from torch import nn

from .model_v15 import PoseFlowV15Model
from .training_v15_rollout import PoseFlowV15RolloutPipeline, PoseFlowV15RolloutPipelineConfig


class PoseFlowQueryNormModel(PoseFlowV15Model):
    def __init__(self, image_model, *, sampling_steps=3):
        super().__init__(image_model, sampling_steps=sampling_steps)
        self.hypothesis_update.query_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
        self.initialization_report['hypothesis_query_normalization'] = 'non-affine LayerNorm, queries only'


class PoseFlowQueryNormPipeline(PoseFlowV15RolloutPipeline):
    def _quick_architecture(self):
        return {
            **super()._quick_architecture(),
            'training_variant': 'V15-rollout-QNorm',
            'hypothesis_query_normalization': 'non-affine LayerNorm, queries only',
        }


class PoseFlowQueryNormPipelineConfig(PoseFlowV15RolloutPipelineConfig):
    latent_model_class: ClassVar[type] = PoseFlowQueryNormModel
    latent_pipeline_class: ClassVar[type] = PoseFlowQueryNormPipeline
