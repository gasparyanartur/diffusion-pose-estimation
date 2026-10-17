from typing import ClassVar

from .model_v15 import PoseFlowV15Model
from .training_v14 import PoseFlowV14Pipeline, PoseFlowV14PipelineConfig


class PoseFlowV15Pipeline(PoseFlowV14Pipeline):
    def _quick_architecture(self):
        return {**super()._quick_architecture(), 'decoder_representation_tokens': self.model.num_body_points+2,
                'evidence_path': 'candidate-local residual before within-person self-attention',
                'hypothesis_update': 'normalized attention evidence, scalar retention interpolation',
                'memory_retention_initial': .9,
                'memory_retention': float(self.model.hypothesis_update.retention_logit.detach().sigmoid())}


class PoseFlowV15PipelineConfig(PoseFlowV14PipelineConfig):
    latent_model_class: ClassVar[type] = PoseFlowV15Model
    latent_pipeline_class: ClassVar[type] = PoseFlowV15Pipeline
