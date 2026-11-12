import hashlib
import json
from pathlib import Path
from typing import ClassVar, Literal

import torch
from torch import nn
from pydantic import Field, model_validator

from ..diffusionregpose.training import DiffusionRegPosePipeline
from .training import PoseFlowPipelineConfig
from .latent_v26 import build_v26_targets, v26_task_loss
from .model_v26 import PoseFlowV26Model


def initialize_v26_image_weights(model, checkpoint_path, *, feature_level_indices=None):
    if feature_level_indices is not None and tuple(feature_level_indices) != tuple(range(model.num_feature_levels)):
        raise ValueError('Image level subsets must be an ordered prefix matching the model levels')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    source = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    source = {name.removeprefix('module.'): value for name, value in source.items()}
    state = model.state_dict()
    copied, missing = {}, []
    for name, target in state.items():
        if not name.startswith(('backbone.', 'input_proj.', 'level_embed')):
            continue
        source_name = 'transformer.level_embed' if name == 'level_embed' else name
        value = source.get(source_name)
        if name == 'level_embed' and value is not None and feature_level_indices is not None:
            value = value[:len(feature_level_indices)]
        if value is None or value.shape != target.shape:
            missing.append(name)
        else:
            copied[name] = value
    if missing:
        raise ValueError(f'V26 image initialization missing/incompatible tensors: {missing}')
    model.load_state_dict(dict(state, **copied), strict=True)
    digest = hashlib.sha256()
    for name, value in sorted(copied.items()):
        digest.update(name.encode()); digest.update(value.contiguous().numpy().tobytes())
    report = dict(source_checkpoint=str(checkpoint_path), copied_image_tensors=len(copied),
                image_tensor_coverage=1., image_sha256=digest.hexdigest(),
                fresh_tensors=len(state)-len(copied), decoder_transfer=False)
    if feature_level_indices is not None:
        report['source_feature_levels'] = list(feature_level_indices)
    return report


class PoseFlowV26Pipeline(DiffusionRegPosePipeline):
    def __init__(self, model, *, config, **kwargs):
        super().__init__(model, nn.Identity(), **kwargs)
        self.v26_config = config
        generator = torch.Generator().manual_seed(config.proposal_seed)
        self.register_buffer('training_rng_state', generator.get_state())
        self._diagnostics = {}

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if isinstance(batch.get('image'), list):
            batch['v26_original_points'] = batch['gt_keypoints'].clone()
            batch['v26_original_labeled'] = batch['gt_labeled'].clone()
            batch['v26_original_people'] = batch['person_mask'].clone()
        return batch

    def forward(self, images, noisy_latents, timesteps, padding_mask=None):
        return self.model(images, noisy_latents, timesteps, padding_mask)

    def _draw_training_inputs(self, batch, *, training=True, batch_idx=0):
        generator = torch.Generator()
        if training:
            generator.set_state(self.training_rng_state.cpu())
        else:
            generator.manual_seed(self.v26_config.proposal_seed+batch_idx)
        targets = build_v26_targets(batch, proposals=self.model.num_proposals,
            generator=generator, fallback_noise_pixels=self.v26_config.fallback_noise_pixels)
        times = torch.randint(self.model.num_timesteps, (len(batch['image']),), generator=generator)
        noise = torch.randn(len(times), self.model.num_proposals, self.model.hidden_dim, generator=generator)
        if training:
            self.training_rng_state.copy_(generator.get_state().to(self.training_rng_state.device))
        device = batch['image'].device
        return targets, times.to(device), noise.to(device=device, dtype=batch['image'].dtype)

    def _denoised_task_predictions(self, denoised):
        return denoised

    def _latent_auxiliary_losses(self, batch, targets, clean, denoised):
        return {}

    def _losses_for_batch(self, batch, *, training=True, batch_idx=0):
        targets, times, noise = self._draw_training_inputs(batch, training=training, batch_idx=batch_idx)
        clean, denoised, _, _ = self.model.training_outputs(batch['image'], targets['input_pixels'],
            timesteps=times, noise=noise, padding_mask=batch.get('padding_mask'))
        weights = self.v26_config.weights
        coefficients = (weights.cls, weights.bbox, weights.giou, weights.keypoints, weights.oks)
        clean_loss = v26_task_loss(clean, targets, weights=coefficients)
        denoised_task_weight = self.v26_config.denoised_task_loss_weight
        with torch.set_grad_enabled(torch.is_grad_enabled() and denoised_task_weight > 0):
            denoised_loss = v26_task_loss(self._denoised_task_predictions(denoised),
                                         targets, weights=coefficients)
        difference = (denoised['pred_latents']-clean['pred_latents'].detach()).square()
        latent_loss = difference.mean()
        losses = {f'clean/{name}': value for name, value in clean_loss.items()}
        losses.update({f'denoised/{name}': value for name, value in denoised_loss.items()})
        losses['loss_latent'] = latent_loss
        losses['total'] = (.5*clean_loss['total']+denoised_task_weight*denoised_loss['total']
                           +self.v26_config.latent_loss_weight*latent_loss)
        self._record_diagnostics(targets, clean, denoised, times, difference)
        self._diagnostics.update(clean_task=clean_loss['total'].detach(),
            denoised_task=denoised_loss['total'].detach(), latent_mse=latent_loss.detach())
        auxiliary = self._latent_auxiliary_losses(batch, targets, clean, denoised)
        if auxiliary:
            losses['total'] = losses['total']+auxiliary['loss_dino']
            losses.update(auxiliary)
        return losses

    def _record_diagnostics(self, targets, clean, denoised, times, difference):
        positive = targets['positive']
        clean_latents = clean['pred_latents'].detach()
        self._noise_range_counts = torch.bincount((times*3//self.model.num_timesteps).cpu(), minlength=3).tolist()
        self._diagnostics = dict(
            unique_people=targets['unique_people'].float().mean(),
            positive_proposals=positive.sum(-1).float().mean(),
            duplicate_proposals=(positive.sum(-1)-targets['unique_people']).float().mean(),
            synthetic_fraction=targets['synthetic'].float().mean(),
            clean_latent_variance=clean_latents.var(dim=(0, 1), unbiased=False).mean(),
            clean_latent_rms=clean_latents.square().mean().sqrt(),
            time_mean=times.float().mean())
        for name, output in (('clean', clean), ('denoised', denoised)):
            scores = output['existence_logits'].detach().sigmoid()
            for kind, mask in (('positive', positive), ('negative', ~positive)):
                self._diagnostics[f'{name}/{kind}_score'] = scores[mask].mean() if mask.any() else scores.new_zeros(())
        for index, name in enumerate(('low', 'middle', 'high')):
            selected = (times*3//self.model.num_timesteps) == index
            if selected.any():
                self._diagnostics[f'latent_mse/{name}'] = difference.detach()[selected].mean()

    def _log_v26(self, prefix, losses, batch_size):
        self._log_losses(prefix, losses, batch_size)
        for name, value in self._diagnostics.items():
            metric_batch_size = batch_size
            if name.startswith(('latent_mse/', 'ota/spatial_weight/')):
                # Weight noise-range means by their actual image count.
                index = ('low', 'middle', 'high').index(name.rsplit('/', 1)[1])
                metric_batch_size = self._noise_range_counts[index]
            self.log(f'debug/v26/{prefix}/{name}', value.detach(), batch_size=metric_batch_size,
                     on_step=prefix == 'train', on_epoch=True)

    def training_step(self, batch, batch_idx):
        losses = self._losses_for_batch(batch, training=True, batch_idx=batch_idx)
        self._log_v26('train', losses, len(batch['image']))
        return losses['total']

    def validation_step(self, batch, batch_idx):
        losses = self._losses_for_batch(batch, training=False, batch_idx=batch_idx)
        self._log_v26('val', losses, len(batch['image']))

    def test_step(self, batch, batch_idx):
        losses = self._losses_for_batch(batch, training=False, batch_idx=batch_idx)
        self._log_v26('test', losses, len(batch['image']))

    def on_validation_epoch_end(self):
        if not self.v26_config.quick_eval_enabled or self.trainer.sanity_checking or not self.trainer.is_global_zero:
            return
        from .quick_eval_v26 import run_v26_evaluation
        callback = self.trainer.checkpoint_callback
        root = Path(callback.dirpath).parent if callback is not None else Path(self.trainer.default_root_dir)
        report = run_v26_evaluation(self.model, self.trainer.datamodule, root/'quick_test',
            epoch=self.current_epoch, global_step=self.global_step,
            count=self.v26_config.quick_eval_images, seed=self.v26_config.proposal_seed)
        for condition, row in report['conditions'].items():
            for name, value in row['metrics'].items():
                self.log(f'quick-test/{condition}/{name}', value, rank_zero_only=True)
            for name, value in row['diagnostics'].items():
                self.log(f'quick-test/{condition}/{name}', float(value), rank_zero_only=True)


class PoseFlowV26PipelineConfig(PoseFlowPipelineConfig):
    pipeline_class: ClassVar[type] = PoseFlowV26Pipeline
    training_protocol: ClassVar[str] = 'joint_clean_latent_diffusion'
    image_source_feature_levels: ClassVar[tuple[int, ...] | None] = None
    num_queries: int = 50
    num_proposals: int = 50
    num_group: int = 50
    num_encoder_layers: int = 0
    num_decoder_layers: int = 3
    num_box_decoder_layers: int = 0
    dn_number: int = 0
    matching_strategy: Literal['constructed'] = 'constructed'
    flow_sampling_steps: int = 4
    pretrained_keep_encoder_layers: tuple[int, ...] | None = None
    pretrained_keep_decoder_layers: tuple[int, ...] | None = None
    latent_loss_weight: float = Field(default=1., ge=0)
    denoised_task_loss_weight: float = Field(default=.5, ge=0)
    fallback_noise_pixels: float = Field(default=5., ge=0)
    proposal_seed: int = 42
    quick_eval_enabled: bool = True
    quick_eval_images: int = Field(default=32, ge=1)

    @model_validator(mode='after')
    def validate_v26(self):
        if not self.num_queries == self.num_proposals == self.num_group > 0:
            raise ValueError('V26 requires a fixed equal proposal budget')
        if self.num_encoder_layers or self.num_decoder_layers != 3 or self.num_box_decoder_layers:
            raise ValueError('V26 uses projected features and two three-layer latent stacks')
        if self.dn_number or self.flow_renewal or self.flow_ensemble or self.pose_codebook_size:
            raise ValueError('V26 excludes DN, renewal, ensemble and codebooks')
        if self.pretrained_keep_encoder_layers or self.pretrained_keep_decoder_layers or self.freeze_backbone_and_transformer:
            raise ValueError('V26 imports image-extraction tensors only and trains its latent stacks')
        return self

    def _build_poseflow_model(self):
        names = ('backbone_pretrained', 'timesteps', 'num_proposals', 'num_body_points',
            'hidden_dim', 'num_heads', 'feedforward_dim', 'num_feature_levels',
            'freeze_backbone_stem_and_layer1', 'freeze_backbone')
        return PoseFlowV26Model(**{name:getattr(self, name) for name in names},
                               sampling_steps=self.flow_sampling_steps)

    def build(self, **kwargs):
        model = self._build_poseflow_model()
        if self.pretrained_checkpoint and kwargs.get('load_pretrained_checkpoint', True):
            model.initialization_report = initialize_v26_image_weights(model, self.pretrained_checkpoint,
                feature_level_indices=self.image_source_feature_levels)
        fields = ('lr', 'lr_backbone', 'weight_decay', 'lr_warmup_epochs', 'lr_min_frac',
            'scheduler_type', 'lr_decay_milestones', 'split_weight_decay',
            'val_ap_every_n_epochs', 'diffusion_noise_seed', 'deterministic_seed')
        return self.pipeline_class(model, config=self,
            **{name:getattr(self, name) for name in fields},
            max_epochs=kwargs.get('max_epochs'), val_ann_file=kwargs.get('val_ann_file'))
