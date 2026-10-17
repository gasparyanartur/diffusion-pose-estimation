from typing import ClassVar, Literal

from pydantic import Field

from .losses_v14 import matched_v14_layer_losses
from .model_v14 import PoseFlowV14Model
from .training_v13 import PoseFlowV13Pipeline, PoseFlowV13PipelineConfig


class PoseFlowV14Pipeline(PoseFlowV13Pipeline):
    latent_role_slices = [('box', slice(0, 1)), ('joints', slice(1, None))]
    matched_loss = staticmethod(matched_v14_layer_losses)

    def _evaluate_quick_grid(self, *args):
        from .quick_eval_v13 import evaluate_quick_grid
        report = evaluate_quick_grid(*args, hypothesis_modes=('persistent', 'reset'))
        report['raw_box_quality_contract'] = {
            'targets': 'person_mask-eligible transformed targets; separate from full COCO AP denominator',
            'coverage': 'best raw candidate IoU per target at .5/.75',
            'duplicates': 'excess IoU>=.5 candidates sharing a best target / all raw candidates',
            'confidence': 'mean score for raw candidates with best target IoU>=.5 versus below .5',
        }
        return report

    def _quick_architecture(self):
        return {'hidden_dim': self.model.hidden_dim, 'state_tokens': self.model.num_body_points+1,
                'decoder_layers': len(self.model.layers), 'hypothesis_slots': self.model.num_hypotheses,
                'global_tokens': 'one per person; reset each denoising call',
                'shared_components': ['evidence_extractor', 'hypothesis_update', 'latent_head']}


class PoseFlowV14PipelineConfig(PoseFlowV13PipelineConfig):
    latent_model_class: ClassVar[type] = PoseFlowV14Model
    latent_pipeline_class: ClassVar[type] = PoseFlowV14Pipeline
    matching_strategy: Literal['hungarian', 'ota'] = 'ota'
    reconstruction_weight_start: float = Field(default=1., gt=0, allow_inf_nan=False)
    reconstruction_weight_end: float = Field(default=.01, gt=0, allow_inf_nan=False)
    reconstruction_decay_epochs: int = Field(default=40, ge=2)
    optimization_diagnostics_enabled: bool = True
