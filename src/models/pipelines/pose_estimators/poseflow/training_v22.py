from typing import ClassVar, Literal

from pydantic import Field

from .model_v22 import PoseFlowV22Model
from .training_v21 import PoseFlowV21Pipeline, PoseFlowV21PipelineConfig


class PoseFlowV22Pipeline(PoseFlowV21Pipeline):
    diagnostic_namespace = 'v22'


class PoseFlowV22PipelineConfig(PoseFlowV21PipelineConfig):
    model_class: ClassVar[type] = PoseFlowV22Model
    pipeline_class: ClassVar[type] = PoseFlowV22Pipeline
    new_state_prefixes: ClassVar[tuple[str, ...]] = (
        'dense_read.', 'time_bottleneck.', 'dense_query_conditioner.',
        'proposal_conditioner.', 'coordinate_conditioners.', 'conditioners.')
    hypothesis_update_order: Literal['h_then_g'] = 'h_then_g'
    time_embedding_scale: float = Field(default=999., gt=0, allow_inf_nan=False)
    coordinate_batch_layout: Literal['legacy', 'per_image'] = 'legacy'

    def _extra_model_arguments(self):
        return {'time_embedding_scale': self.time_embedding_scale,
                'coordinate_batch_layout': self.coordinate_batch_layout}
