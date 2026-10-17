from typing import ClassVar

from .model_v17 import PoseFlowV17Model
from .training_v14 import PoseFlowV14Pipeline
from .training_v15_rollout import PoseFlowV15RolloutPipeline, PoseFlowV15RolloutPipelineConfig


class PoseFlowV17Pipeline(PoseFlowV15RolloutPipeline):
    def _quick_architecture(self):
        return {
            **PoseFlowV14Pipeline._quick_architecture(self),
            'training_variant': 'V17-tied-memory',
            'decoder_representation_tokens': self.model.num_body_points + 3,
            'hypothesis_update': '17-token candidate-local residual self-attention/FFN',
            'hypothesis_communication': 'shared self-attention/FFN across candidate hypotheses before local update',
            'shared_components': ['evidence_extractor', 'hypothesis_attention', 'latent_head'],
            'rollout_probability': self.v12_config.rollout_probability,
            'rollout_call_counts': self.v12_config.rollout_call_counts,
            'rollout_schedule': 'per-image descending distinct timestep indices; Gaussian t=1 start',
        }


class PoseFlowV17PipelineConfig(PoseFlowV15RolloutPipelineConfig):
    latent_model_class: ClassVar[type] = PoseFlowV17Model
    latent_pipeline_class: ClassVar[type] = PoseFlowV17Pipeline
