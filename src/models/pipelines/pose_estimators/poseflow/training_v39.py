from typing import ClassVar, Literal

import torch
from pydantic import Field

from .latent_fixed_source import build_fixed_source_targets, positive_latent_mse
from .latent_v35 import box_task_loss
from .training_v26 import PoseFlowV26Pipeline
from .training_v35 import PoseFlowV35Pipeline, PoseFlowV35PipelineConfig


@torch.no_grad()
def fixed_source_noise_metrics(output, targets, latent_error, times, num_timesteps):
    scores = output['existence_logits'].sigmoid()
    box_error = (output['pred_boxes']-targets['boxes']).abs().sum(-1)
    bins = times*3//num_timesteps
    metrics = {}
    for index, name in enumerate(('low', 'middle', 'high')):
        images = (bins == index)[:, None]
        positive = targets['positive'] & images
        fake = ~targets['positive'] & images
        if positive.any():
            count = int(positive.sum())
            for metric, value in [('bbox_l1', box_error), ('latent_mse', latent_error),
                                  ('positive_score', scores), ('positive_recall50', (scores >= .5).float())]:
                metrics[f'{metric}/{name}'] = (value[positive].mean(), count)
        if fake.any():
            count = int(fake.sum())
            metrics[f'fake_score/{name}'] = (scores[fake].mean(), count)
            metrics[f'fake_acceptance50/{name}'] = ((scores[fake] >= .5).float().mean(), count)
    return metrics


class PoseFlowV39Pipeline(PoseFlowV26Pipeline):
    def __init__(self, model, *, config, **kwargs):
        kwargs['val_ann_file'] = None
        super().__init__(model, config=config, **kwargs)
        self._diagnostic_counts = {}

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return batch

    def _draw_training_inputs(self, batch, *, training=True, batch_idx=0):
        generator = torch.Generator()
        if training:
            generator.set_state(self.training_rng_state.cpu())
        else:
            generator.manual_seed(self.v26_config.proposal_seed+batch_idx)
        targets = build_fixed_source_targets(batch, proposals=self.model.num_proposals, generator=generator)
        times = torch.randint(self.model.num_timesteps, (len(batch['image']),), generator=generator)
        noise = torch.randn(len(times), self.model.num_proposals, self.model.hidden_dim, generator=generator)
        if training:
            self.training_rng_state.copy_(generator.get_state().to(self.training_rng_state.device))
        return targets, times.to(batch['image'].device), noise.to(batch['image'])

    def _losses_for_batch(self, batch, *, training=True, batch_idx=0):
        targets, times, noise = self._draw_training_inputs(batch, training=training, batch_idx=batch_idx)
        clean, recovered, _, _ = self.model.training_outputs(batch['image'], targets['input_pixels'],
            timesteps=times, noise=noise, padding_mask=batch.get('padding_mask'),
            detach_corruption_source=self.v26_config.detach_corruption_source)
        weights = self.v26_config.weights
        coefficients = (weights.cls, weights.bbox, weights.giou)
        clean_loss = box_task_loss(clean, targets, weights=coefficients)
        recovered_weight = self.v26_config.denoised_task_loss_weight
        with torch.set_grad_enabled(torch.is_grad_enabled() and recovered_weight > 0):
            recovered_loss = box_task_loss(recovered, targets, weights=coefficients)
        latent_loss = positive_latent_mse(recovered['pred_latents'], clean['pred_latents'], targets['positive'])
        losses = {f'clean/{name}': value for name, value in clean_loss.items()}
        losses.update({f'denoised/{name}': value for name, value in recovered_loss.items()})
        losses['loss_latent'] = latent_loss
        losses['total'] = (.5*clean_loss['total']+recovered_weight*recovered_loss['total']
                           +self.v26_config.latent_loss_weight*latent_loss)
        self._record_fixed_source_diagnostics(targets, clean, recovered, times, latent_loss)
        return losses

    @torch.no_grad()
    def _record_fixed_source_diagnostics(self, targets, clean, recovered, times, latent_loss):
        positive = targets['positive']
        fake = ~positive
        self._diagnostics = dict(unique_people=targets['unique_people'].float().mean(),
            positive_proposals=positive.sum(-1).float().mean(),
            duplicate_proposals=positive.new_zeros((), dtype=torch.float), latent_mse=latent_loss.detach(),
            random_negative_fraction=targets['random_negative'].float().mean(), time_mean=times.float().mean())
        self._diagnostic_counts = {}
        if fake.any():
            overlap = targets['negative_gt_iou'][fake]
            self._diagnostics['negative_gt_iou'] = overlap.mean()
            self._diagnostics['negative_overlap50_fraction'] = (overlap >= .5).float().mean()
            self._diagnostic_counts.update(negative_gt_iou=int(fake.sum()), negative_overlap50_fraction=int(fake.sum()))
        for branch, output in [('clean', clean), ('denoised', recovered)]:
            scores = output['existence_logits'].sigmoid()
            for label, mask in [('positive', positive), ('fake', fake)]:
                if mask.any():
                    key = f'{branch}/{label}_score'
                    self._diagnostics[key] = scores[mask].mean()
                    self._diagnostic_counts[key] = int(mask.sum())
        error = (recovered['pred_latents']-clean['pred_latents']).square().mean(-1)
        for name, (value, count) in fixed_source_noise_metrics(recovered, targets, error, times,
                                                             self.model.num_timesteps).items():
            self._diagnostics[name] = value
            self._diagnostic_counts[name] = count

    def _log_v26(self, prefix, losses, batch_size):
        self._log_losses(prefix, losses, batch_size)
        for name, value in self._diagnostics.items():
            self.log(f'debug/fixed_source/{prefix}/{name}', value,
                     batch_size=self._diagnostic_counts.get(name, batch_size),
                     on_step=prefix == 'train', on_epoch=True)

    on_validation_epoch_end = PoseFlowV35Pipeline.on_validation_epoch_end


class PoseFlowV39PipelineConfig(PoseFlowV35PipelineConfig):
    pipeline_class: ClassVar[type] = PoseFlowV39Pipeline
    detach_corruption_source: bool = True
    matching_strategy: Literal['constructed'] = 'constructed'
    spatial_queries: Literal[False] = False
    spatial_cost_weight: float = Field(default=0., ge=0, le=0)
    spatial_cost_schedule: Literal['constant'] = 'constant'
