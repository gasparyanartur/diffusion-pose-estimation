"""Image-conditioned fixed-slot latent diffusion training for V12."""

from contextlib import nullcontext
import logging
from typing import ClassVar, Literal

import torch
from torch import nn
from pydantic import model_validator

from ..diffusionregpose.model import DiffusionRegPoseModel
from ..diffusionregpose.training import (
    DiffusionRegPosePipeline, DiffusionRegPosePipelineConfig,
    _postprocess_diffusionregpose_ap, update_diffusionregpose_ap_metric,
    update_diffusionregpose_bbox_ap_metric,
)
from .training import PoseFlowPipelineConfig
from .latent_v12 import build_v12_targets, prediction_loss, latent_loss, corrupt_latents
from .model_v12 import PoseFlowV12Model
from src.training.optim import build_adamw_with_schedule


class PoseFlowV12Pipeline(DiffusionRegPosePipeline):
    latent_role_slices = [('existence', slice(0, 1)), ('center', slice(1, 2)),
                          ('size', slice(2, 3)), ('joints', slice(3, None))]

    def __init__(self, model, *, config, **kwargs):
        super().__init__(model, nn.Identity(), **kwargs)
        self.v12_config = config
        self._v12_diagnostics = {}

    def forward(self, images, noisy_latents, timesteps, padding_mask=None, **kwargs):
        return self.model(images, noisy_latents, timesteps, padding_mask, **kwargs)

    def _losses_for_batch(self, batch):
        image = batch['image']
        generator = None
        if self.diffusion_noise_seed is not None:
            generator = torch.Generator(device=image.device).manual_seed(self.diffusion_noise_seed)
        targets = build_v12_targets(batch, groups=self.model.num_proposals, generator=generator)
        clean = self.model.codec.encode(targets)
        times = torch.randint(self.v12_config.timesteps, (image.shape[0],), device=image.device,
                               generator=generator).to(image.dtype) / (self.v12_config.timesteps - 1)
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        corrupted = corrupt_latents(clean.detach(), noise, times)
        context = nullcontext() if self.deterministic_seed is None else torch.random.fork_rng(
            devices=[image.get_device()] if image.is_cuda else [],
        )
        with context:
            if self.deterministic_seed is not None:
                torch.manual_seed(self.deterministic_seed)
            output = self.model(image, corrupted, times, batch.get('padding_mask'), collect_diagnostics=True)
        weights = self.v12_config.weights.to_dataclass()
        losses = self._denoising_losses(output, targets, clean, weights)
        reconstruction = prediction_loss(self.model.codec.decode(clean), targets, reconstruct=True, weights=weights)
        losses['loss_reconstruction'] = reconstruction['total']
        losses['total'] = losses['loss_latent'] + losses['loss_prediction'] + losses['loss_reconstruction']
        self._v12_diagnostics = {
            'real_people': targets['real'].sum().detach().float() / image.shape[0],
            'overflow_people': targets['overflow'].sum().detach().float(),
            'diffusion_time_mean': times.detach().mean(),
        }
        for name, subset in self.latent_role_slices:
            values = clean[:, :, subset].detach().float()
            self._v12_diagnostics[f'clean/{name}_rms'] = values.square().mean().sqrt()
            self._v12_diagnostics[f'clean/{name}_variance'] = values.var(unbiased=False)
        for row in output['latent_diagnostics']:
            self._v12_diagnostics.update({f'layer{row["layer"]}/{name}': value for name, value in row.items()
                                          if name != 'layer'})
        return losses

    def _denoising_losses(self, output, targets, clean, weights):
        stages = output['aux_outputs'] + [output]
        weights = self.v12_config.weights.to_dataclass()
        stage_losses = [prediction_loss(stage, targets, weights=weights) for stage in stages]
        losses = {name: torch.stack([stage[name] for stage in stage_losses]).mean()
                  for name in stage_losses[0] if name != 'total'}
        losses['loss_latent'] = torch.stack([
            latent_loss(stage['pred_latents'], clean.detach()) for stage in stages
        ]).mean()
        losses['loss_prediction'] = torch.stack([stage['total'] for stage in stage_losses]).mean()
        for index, values in enumerate(stage_losses):
            losses.update({f'{name}_layer{index}': value for name, value in values.items() if name != 'total'})
        return losses

    def _log_latent_diagnostics(self, prefix, batch_size):
        for name, value in self._v12_diagnostics.items():
            self.log(f'debug/latent/{prefix}/{name}', value, batch_size=batch_size,
                      on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        losses = self._losses_for_batch(batch)
        self._log_losses('train', losses, batch['image'].shape[0])
        if batch_idx == 0:
            self._log_latent_diagnostics('train_first_batch', batch['image'].shape[0])
        return losses['total']

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        with torch.random.fork_rng(devices=[image.get_device()] if image.is_cuda else []):
            losses = self._losses_for_batch(batch)
            self._log_losses('val', losses, image.shape[0])
            if batch_idx == 0:
                self._log_latent_diagnostics('val_first_batch', image.shape[0])
            if self._ap_epoch_active():
                generator = None if self.deterministic_seed is None else torch.Generator(
                    device=image.device).manual_seed(self.deterministic_seed)
                sampled = self.model.sample_keypoints(image, padding_mask=batch.get('padding_mask'),
                    generator=generator, collect_step_diagnostics=batch_idx == 0)
                results = _postprocess_diffusionregpose_ap(sampled, batch, num_body_points=self.model.num_body_points)
                for result in results:
                    result['labels'] = torch.ones_like(result['labels'])
                update_diffusionregpose_ap_metric(self._val_ap, sampled, batch,
                    num_body_points=self.model.num_body_points, results=results)
                update_diffusionregpose_bbox_ap_metric(self._val_bbox_ap, results, batch)
                if batch_idx == 0:
                    for step, row in enumerate(sampled['latent_step_diagnostics']):
                        state = row['latent_state'].float()
                        self.log(f'debug/latent/sampling/step{step}/rms', state.square().mean().sqrt(),
                                  batch_size=image.shape[0], on_step=False, on_epoch=True)
                        self.log(f'debug/latent/sampling/step{step}/update_rms', row['update_rms'],
                                  batch_size=image.shape[0], on_step=False, on_epoch=True)
                        for layer in row['layers']:
                            for name, value in layer.items():
                                if name != 'layer':
                                    self.log(f'debug/latent/sampling/step{step}/layer{layer["layer"]}/{name}',
                                              value, batch_size=image.shape[0], on_step=False, on_epoch=True)
                        self._log_sampling_quality(batch, row['predictions'], step)

    def _log_sampling_quality(self, batch, stages, step):
        from src.eval.metrics import CROWDPOSE_SIGMAS
        from src.eval.poseflow_diagnostics import stage_pose_oks
        from src.eval.poseflow_training_diagnostics import _image_targets

        sigmas = batch['image'].new_tensor(CROWDPOSE_SIGMAS[:self.model.num_body_points])
        for layer, stage in enumerate(stages):
            oks_sum = sigmas.new_zeros(())
            covered = sigmas.new_zeros(())
            count = 0
            for index in range(batch['image'].shape[0]):
                target = _image_targets(batch, index)
                output = {name: value[index:index+1] for name, value in stage.items()}
                oks = stage_pose_oks(output, target, sigmas)
                count += oks.shape[1]
                if oks.numel():
                    best = oks.amax(0)
                    oks_sum += best.sum()
                    covered += (best >= .75).sum()
            prefix = f'debug/latent/sampling/step{step}/layer{layer}'
            for name, value in [('oracle_oks', oks_sum / max(count,1)),
                                ('coverage75', covered / max(count,1))]:
                self.log(f'{prefix}/{name}', value, batch_size=batch['image'].shape[0],
                          on_step=False, on_epoch=True)

    def configure_optimizers(self):
        config = self.v12_config
        return build_adamw_with_schedule(self, lr=self.lr, lr_backbone=self.lr_backbone,
            weight_decay=self.weight_decay, lr_T_max=self.max_epochs,
            lr_warmup_epochs=self.lr_warmup_epochs, lr_min_frac=self.lr_min_frac,
            scheduler_type=config.scheduler_type, split_weight_decay=self.split_weight_decay,
            lr_cycle_restart_epoch=config.lr_cycle_restart_epoch,
            lr_cycle_first_min=config.lr_cycle_first_min,
            lr_cycle_restart_peak=config.lr_cycle_restart_peak,
            lr_cycle_final_min=config.lr_cycle_final_min)


class PoseFlowV12PipelineConfig(PoseFlowPipelineConfig):
    latent_model_class: ClassVar[type] = PoseFlowV12Model
    latent_pipeline_class: ClassVar[type] = PoseFlowV12Pipeline
    num_queries: int = 50
    num_proposals: int = 50
    num_group: int = 50
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_box_decoder_layers: int = 1
    dn_number: int = 0
    matching_strategy: Literal['fixed'] = 'fixed'
    flow_sampling_steps: int = 3
    flow_state_conditioning: bool = False
    lr: float = 1e-4
    lr_backbone: float | None = 5e-5
    scheduler_type: Literal['two_cycle_cosine'] = 'two_cycle_cosine'
    lr_cycle_restart_epoch: int = 40
    lr_cycle_first_min: float = 1e-5
    lr_cycle_restart_peak: float = 3e-5
    lr_cycle_final_min: float = 2e-6

    @model_validator(mode='after')
    def _validate_v12(self):
        if not (self.num_queries == self.num_proposals == self.num_group > 0):
            raise ValueError('V12 requires one fixed number of input/output groups')
        if self.dn_number or self.flow_renewal or self.flow_ensemble or self.flow_state_conditioning or self.pose_codebook_size:
            raise ValueError('V12 has no DN, renewal, ensemble, coordinate conditioning or codebook')
        if self.timesteps < 2 or not 1 <= self.num_body_points <= 14:
            raise ValueError('V12 requires at least2 timesteps and1-14 joints')
        if self.freeze_backbone_and_transformer:
            raise ValueError('V12 trains its new latent decoder')
        if self.lr_cycle_restart_epoch < 1 or min(self.lr_cycle_first_min,
                self.lr_cycle_restart_peak, self.lr_cycle_final_min) < 0:
            raise ValueError('V12 scheduler settings must be nonnegative with a positive restart epoch')
        return self

    def _build_poseflow_model(self):
        names = ('backbone_pretrained', 'timesteps', 'sampling_timesteps', 'num_queries',
            'num_proposals', 'num_classes', 'num_feature_levels', 'num_encoder_layers',
            'num_decoder_layers', 'hidden_dim', 'feedforward_dim', 'num_heads', 'num_body_points',
            'num_box_decoder_layers', 'num_group', 'dn_number', 'freeze_backbone_stem_and_layer1',
            'freeze_backbone', 'freeze_backbone_and_transformer')
        image_model = DiffusionRegPoseModel(**{name: getattr(self, name) for name in names})
        return self.latent_model_class(image_model, sampling_steps=self.flow_sampling_steps)

    def build(self, **kwargs):
        initialized = DiffusionRegPosePipelineConfig.build(self, **kwargs)
        model = self.latent_model_class(initialized.model, sampling_steps=self.flow_sampling_steps)
        model.initialization_report['source_checkpoint'] = (
            self.pretrained_checkpoint if kwargs.get('load_pretrained_checkpoint', True) else None
        )
        logging.info('%s initialization: %s', type(model).__name__, model.initialization_report)
        config = self.model_copy(deep=True)
        return self.latent_pipeline_class(model, config=config,
            lr=self.lr, lr_backbone=self.lr_backbone, weight_decay=self.weight_decay,
            max_epochs=kwargs.get('max_epochs'), lr_warmup_epochs=self.lr_warmup_epochs,
            lr_min_frac=self.lr_min_frac, scheduler_type=self.scheduler_type,
            lr_decay_milestones=self.lr_decay_milestones, split_weight_decay=self.split_weight_decay,
            val_ap_every_n_epochs=self.val_ap_every_n_epochs, val_ann_file=kwargs.get('val_ann_file'),
            diffusion_noise_seed=self.diffusion_noise_seed, deterministic_seed=self.deterministic_seed)
