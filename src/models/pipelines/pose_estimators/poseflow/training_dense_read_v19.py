from typing import ClassVar

from pydantic import model_validator

from .dense_read_v19 import PoseFlowV19DenseReadModel
from .training_v19 import PoseFlowV19Pipeline, PoseFlowV19PipelineConfig


class PoseFlowV19DenseReadPipeline(PoseFlowV19Pipeline):
    def _quick_architecture(self):
        return super()._quick_architecture() | {
            'training_variant': 'V19-dense-read',
            'shared_components': ['hypothesis_attention', 'dense_read'],
            'dense_read_calls_per_denoise': 1,
            'dense_read_time_strength': '.1+.9*(1-t)',
            'dense_read_gate_initialization': 0.,
            'evidence_ranking': False,
        }


class PoseFlowV19DenseReadPipelineConfig(PoseFlowV19PipelineConfig):
    latent_model_class: ClassVar[type] = PoseFlowV19DenseReadModel
    latent_pipeline_class: ClassVar[type] = PoseFlowV19DenseReadPipeline
    detach_decoder_geometry: bool = True

    @model_validator(mode='after')
    def _validate_dense_read_baseline(self):
        if not self.detach_decoder_geometry:
            raise ValueError('Dense hypothesis reads use the full geometry-detach V19 baseline')
        return self
