from typing import ClassVar, Literal

import torch
from pydantic import Field

from .latent_v35 import box_task_loss
from .latent_v37 import LatentMetricAlignment, assign_latent_targets
from .training_v35 import PoseFlowV35Pipeline, PoseFlowV35PipelineConfig


class PoseFlowV37Pipeline(PoseFlowV35Pipeline):
    def __init__(self, model, *, config, **kwargs):
        super().__init__(model, config=config, **kwargs)
        del self.matcher
        self.alignment = LatentMetricAlignment(position_scale=config.alignment_position_scale,
            size_scale=config.alignment_size_scale, weights=(config.alignment_position_weight,
                config.alignment_size_weight, config.alignment_existence_weight))

    def _losses_for_batch(self, batch, *, training=True, batch_idx=0):
        targets, times, noise = self._draw_training_inputs(batch, training=training, batch_idx=batch_idx)
        clean, recovered, _, _ = self._training_outputs(batch, targets, times, noise)
        assigned, latent_targets, metadata = assign_latent_targets(clean, recovered, targets,
            candidate_topk=self.v26_config.ota_candidate_topk)
        weights = self.v26_config.weights
        coefficients = (weights.cls, weights.bbox, weights.giou)
        clean_loss = box_task_loss(clean, targets, weights=coefficients)
        recovered_loss = box_task_loss(recovered, assigned, weights=coefficients)
        alignment, alignment_metrics = self.alignment(clean['pred_latents'], targets)
        difference = (recovered['pred_latents']-latent_targets).square()
        losses = {f'clean/{name}': value for name, value in clean_loss.items()}
        losses.update({f'denoised/{name}': value for name, value in recovered_loss.items()})
        losses.update(loss_latent=difference.mean(), loss_alignment=alignment)
        losses['total'] = (.5*clean_loss['total']+self.v26_config.denoised_task_loss_weight*recovered_loss['total']
            +self.v26_config.latent_loss_weight*losses['loss_latent']+self.v26_config.alignment_loss_weight*alignment)
        self._record_diagnostics(targets, clean, recovered, times, difference)
        self._diagnostics.update({f'latent_ota/{name}': value.detach() for name, value in metadata.items() if value.ndim == 0})
        self._diagnostics.update({f'alignment/{name}': value for name, value in alignment_metrics.items()})
        for name, mask in (('positive', assigned['positive']), ('negative', ~assigned['positive'])):
            scores = recovered['existence_logits'].detach().sigmoid()
            self._diagnostics[f'denoised/{name}_score'] = scores[mask].mean() if mask.any() else scores.new_zeros(())
            values = difference.detach()[mask]
            self._diagnostics[f'latent_mse_{name}'] = values.mean() if values.numel() else difference.new_zeros(())
        return losses


class PoseFlowV37PipelineConfig(PoseFlowV35PipelineConfig):
    pipeline_class: ClassVar[type] = PoseFlowV37Pipeline
    matching_strategy: Literal['latent_ota'] = 'latent_ota'
    spatial_queries: bool = True
    spatial_cost_weight: float = Field(default=0., ge=0, le=0)
    spatial_cost_schedule: Literal['constant'] = 'constant'
    alignment_loss_weight: float = Field(default=1., ge=0, allow_inf_nan=False)
    alignment_position_scale: float = Field(default=.25, gt=0, allow_inf_nan=False)
    alignment_size_scale: float = Field(default=.25, gt=0, allow_inf_nan=False)
    alignment_position_weight: float = Field(default=1., ge=0, allow_inf_nan=False)
    alignment_size_weight: float = Field(default=1., ge=0, allow_inf_nan=False)
    alignment_existence_weight: float = Field(default=4., ge=0, allow_inf_nan=False)
