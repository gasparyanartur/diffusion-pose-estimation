from typing import ClassVar, Literal

from .model_v36 import PoseFlowV36Model
from .training_v35 import PoseFlowV35Pipeline, PoseFlowV35PipelineConfig


class PoseFlowV36Pipeline(PoseFlowV35Pipeline):
    def _training_outputs(self, batch, targets, times, noise):
        existence = targets['positive'] if self.model.clean_existence_input else None
        input_key = 'input_boxes' if self.model.clean_box_input_format == 'normalized' else 'input_pixels'
        return self.model.training_outputs(batch['image'], targets[input_key],
            timesteps=times, noise=noise, padding_mask=batch.get('padding_mask'), existence=existence)


class PoseFlowV36PipelineConfig(PoseFlowV35PipelineConfig):
    pipeline_class: ClassVar[type] = PoseFlowV36Pipeline
    clean_existence_input: bool = False
    clean_box_input_format: Literal['pixels', 'normalized'] = 'pixels'
    denoised_task_loss_weight: float = 0.

    def _build_poseflow_model(self):
        names = ('backbone_pretrained', 'timesteps', 'num_proposals', 'num_body_points',
            'hidden_dim', 'num_heads', 'feedforward_dim', 'num_feature_levels',
            'freeze_backbone_stem_and_layer1', 'freeze_backbone', 'spatial_queries',
            'grid_rows', 'grid_columns', 'clean_existence_input', 'clean_box_input_format')
        return PoseFlowV36Model(**{name: getattr(self, name) for name in names},
                               sampling_steps=self.flow_sampling_steps)
