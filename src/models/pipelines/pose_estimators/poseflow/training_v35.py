from typing import ClassVar, Literal

import torch
from pydantic import Field, model_validator

from .latent_v35 import BoxOTAMatcher, assign_box_targets, box_task_loss, build_v35_targets
from .model_v35 import PoseFlowV35Model
from .training_v26 import PoseFlowV26Pipeline, PoseFlowV26PipelineConfig


def late_linear_spatial_multipliers(timesteps, *, cutoff, num_timesteps):
    if not 0 < cutoff < num_timesteps:
        raise ValueError('Spatial cutoff must be inside the diffusion schedule')
    if timesteps.ndim != 1 or timesteps.is_floating_point():
        raise ValueError('Timesteps must be a one-dimensional integer tensor')
    if ((timesteps < 0) | (timesteps >= num_timesteps)).any():
        raise ValueError('Timestep outside the diffusion schedule')
    return (1-timesteps.float()/cutoff).clamp(0, 1)


def early_linear_spatial_multipliers(timesteps, *, cutoff, num_timesteps):
    return late_linear_spatial_multipliers(num_timesteps-1-timesteps,
        cutoff=cutoff, num_timesteps=num_timesteps)


class PoseFlowV35Pipeline(PoseFlowV26Pipeline):
    def __init__(self, model, *, config, **kwargs):
        kwargs['val_ann_file'] = None
        super().__init__(model, config=config, **kwargs)
        self.matcher = BoxOTAMatcher(weights=config.weights.to_dataclass(),
            anchors=getattr(model, 'query_anchors', None), spatial_cost_weight=config.spatial_cost_weight,
            candidate_topk=config.ota_candidate_topk, center_radius=config.ota_center_radius)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return batch

    def _draw_training_inputs(self, batch, *, training=True, batch_idx=0):
        generator = torch.Generator()
        if training:
            generator.set_state(self.training_rng_state.cpu())
        else:
            generator.manual_seed(self.v26_config.proposal_seed+batch_idx)
        targets = build_v35_targets(batch, proposals=self.model.num_proposals, generator=generator,
            anchors=getattr(self.model, 'query_anchors', None), perturb_pixels=self.v26_config.perturb_pixels)
        times = torch.randint(self.model.num_timesteps, (len(batch['image']),), generator=generator)
        noise = torch.randn(len(times), self.model.num_proposals, self.model.hidden_dim, generator=generator)
        if training:
            self.training_rng_state.copy_(generator.get_state().to(self.training_rng_state.device))
        return targets, times.to(batch['image'].device), noise.to(batch['image'])

    def _training_outputs(self, batch, targets, times, noise):
        return self.model.training_outputs(batch['image'], targets['input_pixels'],
            timesteps=times, noise=noise, padding_mask=batch.get('padding_mask'))

    def _losses_for_batch(self, batch, *, training=True, batch_idx=0):
        targets, times, noise = self._draw_training_inputs(batch, training=training, batch_idx=batch_idx)
        clean, recovered, _, _ = self._training_outputs(batch, targets, times, noise)
        multipliers = None
        if self.v26_config.spatial_cost_schedule == 'late_linear':
            multipliers = late_linear_spatial_multipliers(times,
                cutoff=self.v26_config.spatial_cost_ramp_cutoff, num_timesteps=self.model.num_timesteps)
        elif self.v26_config.spatial_cost_schedule == 'early_linear':
            multipliers = early_linear_spatial_multipliers(times,
                cutoff=self.v26_config.spatial_cost_ramp_cutoff, num_timesteps=self.model.num_timesteps)
        assigned, latent_targets, metadata = assign_box_targets(clean, recovered, targets, self.matcher,
            spatial_multipliers=multipliers)
        weights = self.v26_config.weights
        coefficients = (weights.cls, weights.bbox, weights.giou)
        clean_loss = box_task_loss(clean, targets, weights=coefficients)
        recovered_weight = self.v26_config.denoised_task_loss_weight
        with torch.set_grad_enabled(torch.is_grad_enabled() and recovered_weight > 0):
            recovered_loss = box_task_loss(recovered, assigned, weights=coefficients)
        difference = (recovered['pred_latents']-latent_targets).square()
        losses = {f'clean/{k}': v for k, v in clean_loss.items()}
        losses.update({f'denoised/{k}': v for k, v in recovered_loss.items()})
        losses['loss_latent'] = difference.mean()
        losses['total'] = (.5*clean_loss['total']+recovered_weight*recovered_loss['total']
                           +self.v26_config.latent_loss_weight*losses['loss_latent'])
        self._record_diagnostics(targets, clean, recovered, times, difference)
        self._diagnostics.update(clean_task=clean_loss['total'].detach(),
            denoised_task=recovered_loss['total'].detach(), latent_mse=losses['loss_latent'].detach(),
            zero_input_fraction=targets['zero_fallback'].float().mean(),
            perturbed_positive_fraction=targets['perturbed'].float().mean())
        for name, value in metadata.items():
            if value.ndim == 0:
                self._diagnostics[f'ota/{name}'] = value.detach()
        self._diagnostics.update({f'ota/{k}': v.detach() for k, v in self.matcher.diagnostics.items()})
        if multipliers is not None:
            for index, name in enumerate(('low', 'middle', 'high')):
                selected = times*3//self.model.num_timesteps == index
                if selected.any():
                    self._diagnostics[f'ota/spatial_weight/{name}'] = (
                        self.v26_config.spatial_cost_weight*multipliers[selected]).mean().detach()
        for name, mask in [('positive', assigned['positive']), ('negative', ~assigned['positive'])]:
            scores = recovered['existence_logits'].detach().sigmoid()
            self._diagnostics[f'denoised/{name}_score'] = scores[mask].mean() if mask.any() else scores.new_zeros(())
        return losses

    def on_validation_epoch_end(self):
        if not self.v26_config.quick_eval_enabled or self.trainer.sanity_checking or not self.trainer.is_global_zero:
            return
        from pathlib import Path
        from .quick_eval_v35 import run_v35_evaluation
        callback = self.trainer.checkpoint_callback
        root = Path(callback.dirpath).parent if callback else Path(self.trainer.default_root_dir)
        report = run_v35_evaluation(self.model, self.trainer.datamodule, root/'quick_test',
            epoch=self.current_epoch, global_step=self.global_step,
            count=self.v26_config.quick_eval_images, seed=self.v26_config.proposal_seed)
        for condition, row in report['conditions'].items():
            for name, value in row['metrics'].items():
                self.log(f'quick-test/{condition}/bbox/{name}', value, rank_zero_only=True)
            for name, value in row['diagnostics'].items():
                self.log(f'quick-test/{condition}/{name}', value, rank_zero_only=True)


class PoseFlowV35PipelineConfig(PoseFlowV26PipelineConfig):
    pipeline_class: ClassVar[type] = PoseFlowV35Pipeline
    matching_strategy: Literal['ota'] = 'ota'
    latent_loss_weight: float = Field(default=10., ge=0)
    ota_center_radius: float = Field(default=2.5, gt=0)
    spatial_queries: bool = False
    grid_rows: int = Field(default=5, ge=1)
    grid_columns: int = Field(default=10, ge=1)
    spatial_cost_weight: float = Field(default=0., ge=0, allow_inf_nan=False)
    spatial_cost_schedule: Literal['constant', 'late_linear', 'early_linear'] = 'constant'
    spatial_cost_ramp_cutoff: int = Field(default=500, gt=0)
    perturb_pixels: float = Field(default=5., ge=0, allow_inf_nan=False)

    @model_validator(mode='after')
    def validate_grid(self):
        if self.grid_rows*self.grid_columns != self.num_proposals:
            raise ValueError('Grid rows*columns must equal proposal budget')
        if not self.spatial_queries and self.spatial_cost_weight:
            raise ValueError('Spatial cost requires spatial queries')
        if self.spatial_cost_schedule != 'constant':
            if not self.spatial_queries or self.spatial_cost_ramp_cutoff >= self.timesteps:
                raise ValueError('Scheduled spatial cost requires spatial queries and a cutoff inside the schedule')
        return self

    def _build_poseflow_model(self):
        names = ('backbone_pretrained', 'timesteps', 'num_proposals', 'num_body_points',
            'hidden_dim', 'num_heads', 'feedforward_dim', 'num_feature_levels',
            'freeze_backbone_stem_and_layer1', 'freeze_backbone', 'spatial_queries', 'grid_rows', 'grid_columns')
        return PoseFlowV35Model(**{name: getattr(self, name) for name in names},
                               sampling_steps=self.flow_sampling_steps)
