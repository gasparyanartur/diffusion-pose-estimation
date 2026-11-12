import hashlib
import random
from pathlib import Path
from typing import Literal

import numpy as np
import pytorch_lightning as pl
import torch
from pydantic import Field, model_validator

from src.configs.base_config import BaseConfig
from src.models.pipelines.pose_autoencoder.model import geometry_loss
from src.models.pipelines.pose_vae.fidelity import FidelityPoseVAE
from src.training.optim import build_adamw_with_schedule
from ..diffusionregpose.training import (
    DiffusionRegPoseCriterion, DiffusionRegPoseMatcher, DiffusionRegPosePipelineConfig,
)
from .codec_v43 import CODEC_CHECKPOINT, CODEC_SHA256, FrozenPoseCodec, load_frozen_codec
from .latent_v43 import assign_latent_sources, balanced_latent_loss, geometry_targets, reference_statistics, slot_focal_loss
from .model_v44 import PoseFlowV44Model


NATIVE_MODEL_FIELDS = ('timesteps', 'sampling_timesteps', 'num_queries', 'num_proposals', 'num_classes',
    'num_feature_levels', 'num_encoder_layers', 'num_decoder_layers', 'hidden_dim', 'feedforward_dim',
    'num_heads', 'num_body_points', 'num_box_decoder_layers', 'num_group', 'dn_number',
    'dn_box_noise_scale', 'dn_label_noise_ratio', 'freeze_backbone_stem_and_layer1',
    'freeze_backbone', 'freeze_backbone_and_transformer')


class PoseFlowV44PipelineConfig(BaseConfig):
    prediction_type: Literal['x0', 'v'] = 'x0'
    native: dict = Field(default_factory=dict)
    codec_checkpoint: str = CODEC_CHECKPOINT
    codec_sha256: str = CODEC_SHA256
    codec_width: int = 256
    latent_dim: int = 32
    latent_statistics_file: str | None = None
    pretrained_checkpoint: str | None = 'external/DiffusionRegPose/pretrain_models/edpose_r50_crowdpose.pth'
    latent_loss_weight: float = 10.
    geometry_loss_weight: float = .5
    seed: int = 42
    quick_eval_enabled: bool = True
    quick_eval_images: int = 32
    quick_eval_steps: tuple[int, ...] = (1, 4, 8)
    small_noise_levels: tuple[float, ...] = (.01, .025, .05, .1)
    lr: float = 2e-4
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4

    @model_validator(mode='after')
    def validate_contract(self):
        native = self.native_config()
        if native.matching_strategy != 'hungarian' or native.num_proposals != native.num_group:
            raise ValueError('V44 requires Hungarian matching and groups == selected proposals')
        if not 1 <= native.num_box_decoder_layers < native.num_decoder_layers:
            raise ValueError('V44 needs box and latent decoder stages')
        if not self.quick_eval_steps or any(not 1 <= value <= native.timesteps for value in self.quick_eval_steps):
            raise ValueError('Invalid evaluation steps')
        if not self.small_noise_levels or any(not np.isfinite(value) or value <= 0 for value in self.small_noise_levels):
            raise ValueError('Small-noise levels must be finite and positive')
        if self.latent_loss_weight < 0 or self.geometry_loss_weight < 0:
            raise ValueError('Loss weights must be nonnegative')
        return self

    def native_config(self):
        defaults = dict(backbone_pretrained=False, num_encoder_layers=3, num_decoder_layers=3,
            num_box_decoder_layers=1, num_queries=200, num_proposals=50, num_group=50, dn_number=50,
            pretrained_keep_encoder_layers=(2, 4, 6), pretrained_keep_decoder_layers=(2, 4, 6),
            pretrained_minimum_critical_coverage=.99, sampling_timesteps=4,
            freeze_backbone_stem_and_layer1=True)
        return DiffusionRegPosePipelineConfig(**dict(defaults, **self.native,
            pretrained_checkpoint=self.pretrained_checkpoint))

    def create_model(self, codec=None, native_state=None):
        native = self.native_config()
        if codec is None:
            with torch.random.fork_rng(devices=[]):
                codec = FrozenPoseCodec(FidelityPoseVAE(width=self.codec_width, latent_dim=self.latent_dim))
        return PoseFlowV44Model(codec=codec, native_state=native_state, backbone_pretrained=False,
            prediction_type=self.prediction_type,
            separate_expansion_pose_head=native.pretrained_keep_decoder_layers is not None,
            **{name: getattr(native, name) for name in NATIVE_MODEL_FIELDS})

    def create_pipeline(self, *, model, max_epochs):
        return PoseFlowV44Pipeline(config=self, model=model, max_epochs=max_epochs)

    def build(self, *, max_epochs=80, load_pretrained_checkpoint=True, **kwargs):
        provenance, native_state, initialization = {}, None, {}
        codec = None
        if load_pretrained_checkpoint:
            codec = load_frozen_codec(self.codec_checkpoint, expected_sha256=self.codec_sha256,
                width=self.codec_width, latent_dim=self.latent_dim)
            if not self.latent_statistics_file:
                raise ValueError('V44 requires TRAIN posterior-mean statistics')
            path = Path(self.latent_statistics_file)
            stats = torch.load(path, map_location='cpu', weights_only=False)
            if (stats.get('split') != 'train' or stats.get('posterior_mode') != 'mean'
                    or not stats.get('complete') or stats.get('codec_sha256') != self.codec_sha256):
                raise ValueError('V44 statistics must be complete TRAIN posterior means for this codec')
            if hashlib.sha256(Path(stats['annotation_path']).read_bytes()).hexdigest() != stats['annotation_sha256']:
                raise ValueError('Statistics annotation hash mismatch')
            codec.set_statistics(stats['mean'], stats['std'])
            provenance = {key: value for key, value in stats.items() if key not in ('mean', 'std')}
            provenance['statistics_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            if self.pretrained_checkpoint:
                native = self.native_config().build(max_epochs=max_epochs)
                native_state = native.model.state_dict()
                initialization = dict(source=self.pretrained_checkpoint, transfer='native strict load before replacing expansion',
                    encoder_source_layers=[2, 4, 6], decoder_source_layers=[2, 4, 6],
                    native_tensors_before_replacing_expansion=len(native_state),
                    critical_minimum_coverage=self.native_config().pretrained_minimum_critical_coverage)
        model = self.create_model(codec=codec, native_state=native_state)
        model.initialization_report = initialization
        module = self.create_pipeline(model=model, max_epochs=max_epochs)
        module.codec_provenance = provenance
        return module


class PoseFlowV44Pipeline(pl.LightningModule):
    def __init__(self, *, config, model=None, max_epochs=80):
        super().__init__()
        self.config = config if isinstance(config, PoseFlowV44PipelineConfig) else PoseFlowV44PipelineConfig(**config)
        self.max_epochs = max_epochs
        self.save_hyperparameters(dict(config=self.config.model_dump(), max_epochs=max_epochs))
        self.model = self.config.create_model() if model is None else model
        if self.model.prediction_type != self.config.prediction_type:
            raise ValueError('V44 model and configured prediction types differ')
        native = self.config.native_config()
        weights = native.weights.to_dataclass()
        self.frontend_criterion = DiffusionRegPoseCriterion(num_body_points=14,
            num_box_decoder_layers=native.num_box_decoder_layers, weights=weights,
            dn_label_coef=native.dn_label_coef, dn_bbox_coef=native.dn_bbox_coef,
            interm_loss_coef=native.interm_loss_coef, no_interm_box_loss=native.no_interm_box_loss)
        self.slot_matcher = DiffusionRegPoseMatcher(num_body_points=14, weights=weights)
        self.register_buffer('training_rng_state', torch.Generator().manual_seed(self.config.seed).get_state())
        self.codec_provenance = {}

    def _frontend_loss(self, context, targets):
        criterion = self.frontend_criterion
        rows = context['frontend_outputs']
        loss = rows[-1]['pred_logits'].sum() * 0
        for row in rows:
            loss = loss + criterion._loss_for_outputs(row, targets, include_keypoints=False)['total']
        encoder = criterion._loss_for_outputs(context['encoder_output'], targets, include_keypoints=False)
        loss = loss + criterion.interm_loss_coef * (
            criterion.weights.cls * encoder['loss_ce'] + (0 if criterion.no_interm_box_loss else
            criterion.weights.bbox * encoder['loss_bbox'] + criterion.weights.giou * encoder['loss_giou']))
        if 'dn_output' in context:
            dn = criterion._denoising_losses(context['dn_output'])
            loss = loss + criterion.dn_label_coef * dn['dn_loss_ce'] + criterion.dn_bbox_coef * (
                criterion.weights.bbox * dn['dn_loss_bbox'] + criterion.weights.giou * dn['dn_loss_giou'])
        return loss

    def _reconstruction_loss(self, output, geometry, sources):
        positive = sources['positive']
        if not positive.any():
            return output['pred_latents'].sum() * 0
        rows = sources['target_indices'][positive]
        return geometry_loss({name: output[name][positive] for name in ('keypoints_px', 'boxes_cxcywh_px')},
            {name: value[rows] for name, value in geometry.items()})['loss']

    def _diffusion_inputs(self, clean, generator, context, image_wh, *, training):
        times = torch.randint(self.model.num_timesteps, (len(clean),), generator=generator).to(clean.device)
        noise = torch.randn(clean.shape, generator=generator).to(clean)
        return self.model.q_sample(clean, times, noise), times, noise, {}

    def _losses_for_batch(self, batch, *, training=True, batch_idx=0):
        geometry, targets = geometry_targets(batch)
        if any(len(target['labels']) > self.model.num_proposals for target in targets):
            raise ValueError('Valid person count exceeds proposal capacity')
        context = self.model.prepare_context(batch['image'], batch.get('padding_mask'), targets=targets)
        matches = self.slot_matcher(dict(pred_boxes=context['boxes'], pred_logits=context['pred_logits']),
            targets, include_keypoints=False)
        generator = torch.Generator()
        if training:
            generator.set_state(self.training_rng_state.cpu())
        else:
            generator.manual_seed(self.config.seed + batch_idx)
        sources = assign_latent_sources(self.model.codec, geometry, targets, matches,
            slots=self.model.num_proposals, generator=generator, posterior_mode='mean')
        image_wh = batch['img_hw'].flip(-1).to(sources['latents'])
        noisy, times, noise, state_debug = self._diffusion_inputs(sources['latents'], generator,
            context, image_wh, training=training)
        output = self.model.denoise(noisy, times, context, image_wh)
        velocity_target = self.model.velocity_target(sources['latents'], noise, times) if self.config.prediction_type == 'v' else None
        if training:
            self.training_rng_state.copy_(generator.get_state().to(self.training_rng_state.device))
        frontend = self._frontend_loss(context, targets)
        focal = slot_focal_loss(context['existence_logits'], sources['positive'])
        losses = dict(frontend=frontend, loss_slot_focal=focal, total=frontend + 2 * focal)
        for index, prediction in enumerate([*output['aux_outputs'], output]):
            prefix = '' if index == len(output['aux_outputs']) else f'aux{index}/'
            latent = balanced_latent_loss(prediction['pred_latents'], sources['latents'], sources['positive'])
            regression = latent
            if velocity_target is not None:
                regression = balanced_latent_loss(prediction['pred_velocity'], velocity_target, sources['positive'])
                losses.update({f'{prefix}velocity/{name}': value for name, value in regression.items()})
            geometry_loss_value = self._reconstruction_loss(prediction, geometry, sources)
            losses.update({f'{prefix}latent/{name}': value for name, value in latent.items()})
            losses[f'{prefix}geometry'] = geometry_loss_value
            losses['total'] = losses['total'] + self.config.latent_loss_weight * regression['total'] + (
                self.config.geometry_loss_weight * geometry_loss_value)
        losses.update({'debug/timestep_mean': times.float().mean(), 'debug/timestep_min': times.min().float(),
            'debug/timestep_max': times.max().float(), 'debug/positive_slots': sources['positive'].sum().float(),
            'debug/predicted_latent_rms': output['pred_latents'].detach().square().mean().sqrt()})
        losses.update({f'debug/{name}': value for name, value in state_debug.items()})
        candidates = dict(pred_boxes=context['frontend_outputs'][-1]['pred_boxes'])
        losses.update({f'debug/{name}': value for name, value in reference_statistics(
            candidates, dict(boxes=context['boxes']), targets, matches).items()})
        return losses

    def training_step(self, batch, batch_idx):
        losses = self._losses_for_batch(batch, batch_idx=batch_idx)
        if not torch.isfinite(losses['total']):
            raise FloatingPointError('Nonfinite V44 training loss')
        self.log_dict({f'train/{name}': value for name, value in losses.items()},
            on_step=True, on_epoch=True, batch_size=len(batch['image']))
        return losses['total']

    def validation_step(self, batch, batch_idx):
        losses = self._losses_for_batch(batch, training=False, batch_idx=batch_idx)
        self.log_dict({f'val/{name}': value for name, value in losses.items()},
            on_step=False, on_epoch=True, batch_size=len(batch['image']))
        return losses['total']

    def on_before_optimizer_step(self, optimizer):
        norms = [parameter.grad.detach().norm() for parameter in self.parameters() if parameter.grad is not None]
        norm = torch.stack(norms).norm()
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite V44 gradients')
        self.log('train/gradient_norm', norm)

    def on_validation_epoch_end(self):
        if not self.config.quick_eval_enabled or self.trainer.sanity_checking:
            return
        from .quick_eval_v44 import run_v44_evaluation
        checkpoint = self.trainer.checkpoint_callback
        root = Path(checkpoint.dirpath).parent if checkpoint is not None else Path(self.trainer.default_root_dir)
        report = run_v44_evaluation(self.model, self.trainer.datamodule,
            root / 'quick_test', epoch=self.current_epoch,
            count=self.config.quick_eval_images, steps=self.config.quick_eval_steps, seed=self.config.seed,
            levels=self.config.small_noise_levels)
        scores = []
        for condition, row in report['conditions'].items():
            for name, value in row['metrics'].items():
                self.log(f'quick-test/{condition}/{name}', value)
            scores.append(min(row['metrics']['pose_AP'], row['metrics']['bbox_AP']))
        for name, value in report['small_noise'].items():
            self.log(f'quick-test/{name}', value)
        self.log('val/codec_selection_AP', max(scores))

    def configure_optimizers(self):
        return build_adamw_with_schedule(self, lr=self.config.lr, lr_backbone=self.config.lr_backbone,
            weight_decay=self.config.weight_decay, lr_T_max=self.max_epochs, lr_warmup_epochs=0,
            scheduler_type='multistep', lr_decay_milestones=(31, 61), split_weight_decay=False)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['v44_prediction_type'] = self.config.prediction_type
        checkpoint['v44_contract'] = dict(version=1, posterior_mode='mean', state='latent', fixed_assignment=True)
        checkpoint['v44_codec_provenance'] = self.codec_provenance
        checkpoint['v44_global_rng'] = dict(python=random.getstate(), numpy=np.random.get_state(),
            torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('v44_prediction_type', 'x0') != self.config.prediction_type:
            raise ValueError('Checkpoint prediction type disagrees with configured V44 head')
        if checkpoint.get('v44_contract') != dict(version=1, posterior_mode='mean', state='latent', fixed_assignment=True):
            raise ValueError('Checkpoint does not satisfy the V44 latent/assignment contract')
        self.codec_provenance = checkpoint['v44_codec_provenance']
        rng = checkpoint['v44_global_rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'].cpu())
        if rng['cuda'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in rng['cuda']])
