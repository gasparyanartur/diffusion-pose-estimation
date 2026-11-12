from typing import ClassVar, Literal

import torch
from pydantic import Field

from ..diffusionregpose.training import DiffusionRegPoseOTAMatcher
from .latent_v26 import v26_task_loss
from .latent_v29 import build_v29_ota_targets
from .training_v26 import PoseFlowV26Pipeline
from .training_v28 import PoseFlowV28PipelineConfig


class PoseFlowV29Pipeline(PoseFlowV26Pipeline):
    def __init__(self, model, *, config, **kwargs):
        super().__init__(model, config=config, **kwargs)
        self.matcher = DiffusionRegPoseOTAMatcher(num_body_points=model.num_body_points,
            weights=config.weights.to_dataclass(), candidate_topk=config.ota_candidate_topk,
            center_radius=config.ota_center_radius)

    def _losses_for_batch(self, batch, *, training=True, batch_idx=0):
        targets, times, noise = self._draw_training_inputs(batch, training=training, batch_idx=batch_idx)
        clean, recovered, _, _ = self.model.training_outputs(batch['image'], targets['input_pixels'],
            timesteps=times, noise=noise, padding_mask=batch.get('padding_mask'))
        assigned, latent_targets, metadata = build_v29_ota_targets(clean, recovered, targets, self.matcher)
        weights = self.v26_config.weights
        coefficients = (weights.cls, weights.bbox, weights.giou, weights.keypoints, weights.oks)
        clean_loss = v26_task_loss(clean, targets, weights=coefficients)
        with torch.no_grad():
            recovered_loss = v26_task_loss(recovered, assigned, weights=coefficients)
        difference = (recovered['pred_latents']-latent_targets).square()
        latent_loss = difference.mean()
        losses = {f'clean/{name}':value for name, value in clean_loss.items()}
        losses.update({f'denoised/{name}':value for name, value in recovered_loss.items()})
        losses.update(loss_latent=latent_loss,
                      total=.5*clean_loss['total']+self.v26_config.latent_loss_weight*latent_loss)
        self._record_diagnostics(targets, clean, recovered, times, difference)
        self._diagnostics.update(clean_task=clean_loss['total'].detach(),
            denoised_task=recovered_loss['total'], latent_mse=latent_loss.detach())
        for name, value in metadata.items():
            if value.ndim == 0:
                self._diagnostics[f'ota/{name}'] = value.detach()
        scores = recovered['existence_logits'].detach().sigmoid()
        for name, mask in (('positive', assigned['positive']), ('negative', ~assigned['positive'])):
            self._diagnostics[f'denoised/{name}_score'] = scores[mask].mean() if mask.any() else scores.new_zeros(())
            self._diagnostics[f'ota/{name}_latent_mse'] = difference.detach()[mask].mean() if mask.any() else scores.new_zeros(())
        return losses


class PoseFlowV29PipelineConfig(PoseFlowV28PipelineConfig):
    pipeline_class: ClassVar[type] = PoseFlowV29Pipeline
    matching_strategy: Literal['ota'] = 'ota'
    denoised_task_loss_weight: Literal[0.] = 0.
    ota_candidate_topk: int = Field(default=5, ge=1)
    ota_center_radius: float = Field(default=2.5, gt=0)
