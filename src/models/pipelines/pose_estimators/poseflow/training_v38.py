from typing import ClassVar

import torch
from pydantic import Field, model_validator

from .latent_v35 import box_task_loss
from .latent_v37 import assign_latent_targets
from .model_v38 import PoseFlowV38Model
from .training_v35 import PoseFlowV35Pipeline
from .training_v37 import PoseFlowV37PipelineConfig


def prior_kl(mean, logvar):
    return .5*(mean.square()+logvar.exp()-1-logvar).mean()


@torch.no_grad()
def bbox_noise_metrics(boxes, assigned, times, num_timesteps):
    error = (boxes-assigned['boxes']).abs().sum(-1)
    bins = times*3//num_timesteps
    metrics = {}
    for index, name in enumerate(('low', 'middle', 'high')):
        positive = assigned['positive'] & (bins == index)[:, None]
        count = int(positive.sum())
        if count:
            metrics[name] = (error[positive].mean(), count)
    return metrics


class PoseFlowV38Pipeline(PoseFlowV35Pipeline):
    def __init__(self, model, *, config, **kwargs):
        super().__init__(model, config=config, **kwargs)
        del self.matcher
        generator = torch.Generator().manual_seed(config.posterior_seed)
        self.register_buffer('posterior_rng_state', generator.get_state())
        self._bbox_noise_metrics = {}

    def _draw_posterior_noise(self, reference, *, training=True, batch_idx=0):
        generator = torch.Generator()
        if training:
            generator.set_state(self.posterior_rng_state.cpu())
        else:
            generator.manual_seed(self.v26_config.posterior_seed+batch_idx)
        noise = torch.randn(reference.shape, generator=generator)
        if training:
            self.posterior_rng_state.copy_(generator.get_state().to(self.posterior_rng_state.device))
        return noise.to(reference)

    def _losses_for_batch(self, batch, *, training=True, batch_idx=0):
        targets, times, noise = self._draw_training_inputs(batch, training=training, batch_idx=batch_idx)
        posterior_noise = self._draw_posterior_noise(noise, training=training, batch_idx=batch_idx)
        clean, recovered, _, _ = self.model.training_outputs(batch['image'], targets['input_pixels'],
            timesteps=times, noise=noise, posterior_noise=posterior_noise, padding_mask=batch.get('padding_mask'))
        assigned, latent_targets, metadata = assign_latent_targets(clean, recovered, targets,
            candidate_topk=self.v26_config.ota_candidate_topk)
        weights = self.v26_config.weights
        coefficients = (weights.cls, weights.bbox, weights.giou)
        sampled_predictions = dict(pred_boxes=clean['sampled_pred_boxes'],
                                   existence_logits=clean['sampled_existence_logits'])
        clean_loss = box_task_loss(sampled_predictions, targets, weights=coefficients)
        recovered_loss = box_task_loss(recovered, assigned, weights=coefficients)
        difference = (recovered['pred_latents']-latent_targets).square()
        kl = prior_kl(clean['pred_latents'], clean['posterior_logvar'])
        losses = {f'clean/{name}': value for name, value in clean_loss.items()}
        losses.update({f'denoised/{name}': value for name, value in recovered_loss.items()})
        losses.update(loss_latent=difference.mean(), loss_kl=kl)
        losses['total'] = (.5*clean_loss['total']+self.v26_config.denoised_task_loss_weight*recovered_loss['total']
            +self.v26_config.latent_loss_weight*losses['loss_latent']+self.v26_config.kl_loss_weight*kl)
        self._record_diagnostics(targets, clean, recovered, times, difference)
        self._diagnostics.update({f'latent_ota/{name}': value.detach() for name, value in metadata.items() if value.ndim == 0})
        self._posterior_diagnostics(clean, targets)
        with torch.no_grad():
            mean_losses = box_task_loss(clean, targets, weights=coefficients)
            for name in ('loss_bbox', 'loss_giou', 'total'):
                self._diagnostics[f'clean_mean/{name}'] = mean_losses[name]
                self._diagnostics[f'clean_sampled/{name}'] = clean_loss[name].detach()
            self._diagnostics['kl/raw'] = kl.detach()
            self._diagnostics['kl/weighted'] = self.v26_config.kl_loss_weight*kl.detach()
            for name, mask in (('positive', assigned['positive']), ('negative', ~assigned['positive'])):
                scores = recovered['existence_logits'].detach().sigmoid()
                self._diagnostics[f'denoised/{name}_score'] = scores[mask].mean() if mask.any() else scores.new_zeros(())
                values = difference.detach()[mask]
                self._diagnostics[f'latent_mse_{name}'] = values.mean() if values.numel() else difference.new_zeros(())
        self._bbox_noise_metrics = bbox_noise_metrics(recovered['pred_boxes'], assigned, times, self.model.num_timesteps)
        return losses

    @torch.no_grad()
    def _posterior_diagnostics(self, clean, targets):
        mean = clean['pred_latents'].detach()
        logvar = clean['posterior_logvar'].detach()
        for name, mask in (('real', targets['positive']), ('fake', ~targets['positive'])):
            selected_mean, selected_logvar = mean[mask], logvar[mask]
            zero = mean.new_zeros(())
            values = dict(mean_code_variance=selected_mean.var(0, unbiased=False).mean() if mask.any() else zero,
                posterior_std=(selected_logvar*.5).exp().mean() if mask.any() else zero,
                kl_raw=prior_kl(selected_mean, selected_logvar) if mask.any() else zero,
                kl_weighted=self.v26_config.kl_loss_weight*prior_kl(selected_mean, selected_logvar) if mask.any() else zero,
                clamp_low=(selected_logvar <= self.model.posterior_logvar_min).float().mean() if mask.any() else zero,
                clamp_high=(selected_logvar >= self.model.posterior_logvar_max).float().mean() if mask.any() else zero)
            self._diagnostics.update({f'posterior/{name}/{key}': value for key, value in values.items()})

    def _log_v26(self, prefix, losses, batch_size):
        super()._log_v26(prefix, losses, batch_size)
        for name, (value, count) in self._bbox_noise_metrics.items():
            self.log(f'debug/v38/{prefix}/denoised_bbox_l1/{name}', value,
                     batch_size=count, on_step=prefix == 'train', on_epoch=True)


class PoseFlowV38PipelineConfig(PoseFlowV37PipelineConfig):
    pipeline_class: ClassVar[type] = PoseFlowV38Pipeline
    latent_loss_weight: float = Field(default=100., ge=0, allow_inf_nan=False)
    alignment_loss_weight: float = Field(default=0., ge=0, le=0, allow_inf_nan=False)
    kl_loss_weight: float = Field(default=.1, ge=0, allow_inf_nan=False)
    posterior_seed: int = 1042
    posterior_logvar_init: float = Field(default=-4., allow_inf_nan=False)
    posterior_logvar_min: float = Field(default=-10., allow_inf_nan=False)
    posterior_logvar_max: float = Field(default=2., allow_inf_nan=False)

    @model_validator(mode='after')
    def validate_posterior(self):
        if not self.posterior_logvar_min <= self.posterior_logvar_init <= self.posterior_logvar_max:
            raise ValueError('Posterior initialization must be inside the logvariance bounds')
        return self

    def _build_poseflow_model(self):
        names = ('backbone_pretrained', 'timesteps', 'num_proposals', 'num_body_points',
            'hidden_dim', 'num_heads', 'feedforward_dim', 'num_feature_levels',
            'freeze_backbone_stem_and_layer1', 'freeze_backbone', 'spatial_queries', 'grid_rows', 'grid_columns',
            'posterior_logvar_init', 'posterior_logvar_min', 'posterior_logvar_max')
        return PoseFlowV38Model(**{name: getattr(self, name) for name in names},
                               sampling_steps=self.flow_sampling_steps)
