from typing import ClassVar

from .model_v18 import PoseFlowV18Model
from .training_v17 import PoseFlowV17Pipeline, PoseFlowV17PipelineConfig


class PoseFlowV18Pipeline(PoseFlowV17Pipeline):
    def _quick_architecture(self):
        return {
            **super()._quick_architecture(),
            'training_variant': 'V18-preimage-FFN',
            'pre_image_refinement': 'distinct per-iteration LayerNorm, residual FFN, LayerNorm; 16 representations',
        }


class PoseFlowV18PipelineConfig(PoseFlowV17PipelineConfig):
    latent_model_class: ClassVar[type] = PoseFlowV18Model
    latent_pipeline_class: ClassVar[type] = PoseFlowV18Pipeline
