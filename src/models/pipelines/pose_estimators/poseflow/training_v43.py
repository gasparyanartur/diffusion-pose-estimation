from pathlib import Path

import hashlib
import math
import random
from typing import Literal
import numpy as np
import pytorch_lightning as pl
import torch
from pydantic import Field, model_validator

from src.configs.base_config import BaseConfig
from src.models.pipelines.pose_autoencoder.model import geometry_loss
from src.models.pipelines.pose_vae.fidelity import FidelityPoseVAE
from src.training.optim import build_adamw_with_schedule
from ..diffusionregpose.training import DiffusionRegPoseMatcher, DiffusionRegPoseLossWeightsConfig
from .codec_v43 import CODEC_CHECKPOINT, CODEC_SHA256, FrozenPoseCodec, load_frozen_codec
from .latent_v40 import proposal_detection_loss
from .latent_v43 import (
    assign_latent_sources, balanced_latent_loss, geometry_targets, proposal_targets,
    reference_statistics, slot_focal_loss,
)
from .model_v43 import PoseFlowV43Model
from .noise_v43 import noise_coefficients, velocity_target
from .training_v40 import initialize_v40_weights


def validate_statistics_data(provenance, data_config):
    annotation = Path(data_config.root)/data_config.train_ann
    if hashlib.sha256(annotation.read_bytes()).hexdigest() != provenance['annotation_sha256']:
        raise ValueError('Current training annotation differs from latent-statistics annotation')
    fields = ('external_train_transforms', 'train_random_flip', 'train_random_crop', 'train_resize_scales',
              'train_max_size', 'train_crop_resize_scales', 'train_crop_size')
    if any(getattr(data_config, name) != provenance['data_config'][name] for name in fields):
        raise ValueError('Current training augmentation differs from latent-statistics augmentation')


class PoseFlowV43Pipeline(pl.LightningModule):
    def __init__(self, *, config, max_epochs=80, model=None):
        super().__init__()
        self.config = config if isinstance(config, PoseFlowV43PipelineConfig) else PoseFlowV43PipelineConfig(**config)
        self.max_epochs = max_epochs
        self.model = self.config.create_model() if model is None else model
        if self.model.prediction_type != self.config.prediction_type:
            raise ValueError('Model prediction type disagrees with pipeline configuration')
        self.save_hyperparameters(dict(config=self.config.model_dump(), max_epochs=max_epochs))
        self.matcher = DiffusionRegPoseMatcher(weights=self.config.weights.to_dataclass())
        generator = torch.Generator().manual_seed(self.config.seed)
        self.register_buffer('training_rng_state', generator.get_state())
        self.register_buffer('posterior_rng_state', torch.Generator().manual_seed(self.config.seed+1).get_state())
        self.codec_provenance = {}

    def _generators(self, training, batch_idx):
        diffusion, posterior = torch.Generator(), torch.Generator()
        if training:
            diffusion.set_state(self.training_rng_state.cpu())
            posterior.set_state(self.posterior_rng_state.cpu())
        else:
            diffusion.manual_seed(self.config.seed+batch_idx)
            posterior.manual_seed(self.config.seed+1+batch_idx)
        return diffusion, posterior

    def _losses_for_batch(self, batch, *, training=True, batch_idx=0):
        geometry, targets = geometry_targets(batch)
        if any(len(target['labels']) > self.model.num_proposals for target in targets):
            raise ValueError('Valid person count exceeds proposal capacity')
        context = self.model.encode_context(batch['image'], batch.get('padding_mask'))
        candidates = self.model.propose(context)
        references = self.model.select_references(candidates)
        candidate_matches = self.matcher(candidates, targets, include_keypoints=False)
        selected_matches = self.matcher(dict(pred_boxes=references['boxes'], pred_logits=references['pred_logits']),
                                        targets, include_keypoints=False)
        diffusion, posterior = self._generators(training, batch_idx)
        sources = assign_latent_sources(self.model.codec, geometry, targets, selected_matches,
            slots=self.model.num_proposals, generator=posterior, posterior_mode=self.config.posterior_mode)
        if self.config.training_task == 'image_alignment':
            output = self.model.predict_image_context(context['image'], references=references,
                image_wh=batch['img_hw'].flip(-1).to(batch['image']))
            times = torch.zeros(len(targets), device=batch['image'].device, dtype=torch.long)
            sigma = sources['latents'].new_zeros(len(targets))
            noise = torch.zeros_like(sources['latents'])
        elif self.config.noise_policy == 'small_noise':
            levels = sources['latents'].new_tensor(self.config.small_noise_levels)
            indices = torch.randint(len(levels), (len(targets),), generator=diffusion).to(levels.device)
            sigma = levels[indices]
            noise = torch.randn(sources['latents'].shape, generator=diffusion).to(sources['latents'])
            noisy = sources['latents']+sigma[:, None, None]*noise
            output = self.model.denoise_at_sigma(noisy, sigma, context['image'], references=references,
                image_wh=batch['img_hw'].flip(-1).to(noisy))
            times = self.model._sigma_time(sigma)
        else:
            times = torch.randint(self.model.num_timesteps, (len(targets),), generator=diffusion).to(batch['image'].device)
            noise = torch.randn(sources['latents'].shape, generator=diffusion).to(sources['latents'])
            noisy = self.model.q_sample(sources['latents'], times, noise)
            output = self.model.denoise(noisy, times, context['image'], references=references,
                                        image_wh=batch['img_hw'].flip(-1).to(noisy))
            sigma = self.model.sqrt_recipm1_alphas_cumprod[times].to(noisy)
        if training:
            self.training_rng_state.copy_(diffusion.get_state().to(self.training_rng_state.device))
            self.posterior_rng_state.copy_(posterior.get_state().to(self.posterior_rng_state.device))
        proposal = proposal_detection_loss(candidates, proposal_targets(candidates, targets, candidate_matches),
            weights=(self.config.weights.cls, self.config.weights.bbox, self.config.weights.giou))
        latent = balanced_latent_loss(output['pred_latents'], sources['latents'], sources['positive'])
        regression = latent
        if self.config.prediction_type == 'v':
            regression = balanced_latent_loss(output['pred_velocity'],
                velocity_target(sources['latents'], noise, sigma), sources['positive'])
        focal = slot_focal_loss(references['existence_logits'], sources['positive'])
        positive = sources['positive']
        if positive.any():
            rows = sources['target_indices'][positive]
            reconstruction = geometry_loss(
                {name: output[name][positive] for name in ('keypoints_px', 'boxes_cxcywh_px')},
                {name: value[rows] for name, value in geometry.items()},
                center_min_side_px=self.config.center_min_side_px)
        else:
            reconstruction = {name: output['pred_latents'].sum()*0
                              for name in ('loss', 'pose', 'box_center', 'box_size', 'giou')}
        losses = {f'proposal/{name}': value for name, value in proposal.items()}
        losses.update({f'latent/{name}': value for name, value in latent.items()})
        if self.config.prediction_type == 'v':
            losses.update({f'velocity/{name}': value for name, value in regression.items()})
        losses.update({f'geometry/{"total" if name == "loss" else name}': value for name, value in reconstruction.items()})
        losses.update(loss_slot_focal=focal, loss_latent=regression['total'],
            total=proposal['total']+2*focal+self.config.latent_loss_weight*regression['total']
                  +self.config.geometry_loss_weight*reconstruction['loss'])
        errors = (output['pred_latents'].detach()-sources['latents']).square().mean(-1)
        losses.update({'debug/timestep_mean': times.float().mean(), 'debug/timestep_min': times.min().float(),
            'debug/timestep_max': times.max().float(), 'debug/positive_slots': positive.sum().float(),
            'debug/predicted_latent_rms': output['pred_latents'].detach().square().mean().sqrt(),
            'debug/target_latent_rms': sources['latents'].square().mean().sqrt()})
        if self.config.noise_policy == 'small_noise':
            losses.update({'debug/noise_sigma_mean': sigma.mean(),
                           'debug/noise_sigma_min': sigma.min(), 'debug/noise_sigma_max': sigma.max()})
        losses.update({f'debug/{name}': value for name, value in
            reference_statistics(candidates, references, targets, selected_matches).items()})
        for name, mask in (('positive', positive), ('background', ~positive)):
            if mask.any():
                losses[f'debug/{name}_confidence'] = references['existence_logits'].detach()[mask].sigmoid().mean()
        if positive.any():
            for name, values in (('target', sources['latents'][positive]), ('predicted', output['pred_latents'].detach()[positive])):
                means, stds = values.mean(0), values.std(0, unbiased=False)
                for dimension in range(self.model.latent_dim):
                    losses[f'debug/dimension_{dimension:02d}/{name}_mean'] = means[dimension]
                    losses[f'debug/dimension_{dimension:02d}/{name}_std'] = stds[dimension]
        for index, label in enumerate(('low', 'middle', 'high')):
            selected = positive & (times[:, None]*3//self.model.num_timesteps == index)
            if selected.any():
                losses[f'debug/latent_positive_mse_{label}'] = errors[selected].mean()
        return losses

    def training_step(self, batch, batch_idx):
        losses = self._losses_for_batch(batch, batch_idx=batch_idx)
        if not torch.isfinite(losses['total']):
            raise FloatingPointError('Nonfinite V43 training loss')
        self.log_dict({f'train/{name}': value for name, value in losses.items() if '/dimension_' not in name},
                      on_step=True, on_epoch=True, batch_size=len(batch['image']))
        self.log_dict({f'train/{name}': value for name, value in losses.items() if '/dimension_' in name},
                      on_step=False, on_epoch=True, batch_size=len(batch['image']))
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
            raise FloatingPointError('Nonfinite V43 gradients')
        self.log('train/gradient_norm', norm)

    def on_validation_epoch_end(self):
        if not self.config.quick_eval_enabled or self.trainer.sanity_checking or not self.trainer.is_global_zero:
            return
        if self.config.training_task == 'image_alignment':
            from .alignment_v43 import run_alignment_evaluation
            root = Path(self.trainer.default_root_dir)/'alignment_validation'
            report = run_alignment_evaluation(self.model, self.trainer.datamodule, root,
                epoch=self.current_epoch, count=self.config.quick_eval_images, matcher=self.matcher)
            for condition, result in report['conditions'].items():
                for name, value in {**result['metrics'], **result['paired']}.items():
                    self.log(f'alignment-val/{condition}/{name}', value, rank_zero_only=True)
            return
        from .quick_eval_v43 import run_v43_evaluation
        root = Path(self.trainer.default_root_dir)/'quick_test'
        report = run_v43_evaluation(self.model, self.trainer.datamodule, root,
            epoch=self.current_epoch, global_step=self.global_step,
            count=self.config.quick_eval_images, steps=self.config.quick_eval_steps, seed=self.config.seed)
        for condition, result in report['conditions'].items():
            for name, value in result['metrics'].items():
                self.log(f'quick-test/{condition}/{name}', value, rank_zero_only=True)

    def configure_optimizers(self):
        config = self.config
        return build_adamw_with_schedule(self, lr=config.lr, lr_backbone=config.lr_backbone,
            weight_decay=config.weight_decay, lr_T_max=self.max_epochs, lr_warmup_epochs=0,
            scheduler_type=config.scheduler_type, lr_decay_milestones=(31, 61), split_weight_decay=False)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['v43_training_task'] = self.config.training_task
        checkpoint['v43_prediction_type'] = self.config.prediction_type
        checkpoint['v43_codec_provenance'] = self.codec_provenance
        checkpoint['v43_global_rng'] = dict(python=random.getstate(), numpy=np.random.get_state(),
            torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('v43_training_task', 'denoising') != self.config.training_task:
            raise ValueError('Checkpoint training task disagrees with configured task')
        if checkpoint.get('v43_prediction_type', 'x0') != self.config.prediction_type:
            raise ValueError('Checkpoint prediction type disagrees with configured head semantics')
        self.codec_provenance = checkpoint.get('v43_codec_provenance', {})
        state = checkpoint.get('v43_global_rng')
        if state is not None:
            random.setstate(state['python'])
            np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch'].cpu())
            if state['cuda'] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


class PoseFlowV43PipelineConfig(BaseConfig):
    training_task: Literal['denoising', 'image_alignment'] = 'denoising'
    scheduler_type: Literal['multistep', 'cosine'] = 'multistep'
    prediction_type: Literal['x0', 'v'] = 'x0'
    noise_policy: Literal['uniform', 'small_noise'] = 'uniform'
    small_noise_levels: tuple[float, ...] = (.01, .025, .05)
    posterior_mode: Literal['sampled', 'mean'] = 'sampled'
    center_min_side_px: float = Field(default=0., ge=0)
    codec_checkpoint: str = CODEC_CHECKPOINT
    codec_sha256: str = CODEC_SHA256
    latent_statistics_file: str | None = None
    codec_width: int = 256
    latent_dim: int = 32
    hidden_dim: int = 256
    num_heads: int = 8
    feedforward_dim: int = 2048
    num_proposals: int = Field(default=50, ge=1)
    reference_candidates: int = 200
    reference_encoder_source_layers: tuple[int, ...] = (2, 4, 6)
    num_feature_levels: int = 4
    timesteps: int = 1000
    flow_sampling_steps: int = 4
    backbone_pretrained: bool = False
    freeze_backbone_stem_and_layer1: bool = True
    freeze_backbone: bool = False
    pretrained_checkpoint: str | None = 'external/DiffusionRegPose/pretrain_models/edpose_r50_crowdpose.pth'
    latent_loss_weight: float = Field(default=10., ge=0)
    geometry_loss_weight: float = Field(default=.5, ge=0)
    lr: float = 2e-4
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    quick_eval_enabled: bool = True
    quick_eval_images: int = 32
    quick_eval_steps: tuple[int, ...] = (1, 4, 8)
    weights: DiffusionRegPoseLossWeightsConfig = Field(default_factory=DiffusionRegPoseLossWeightsConfig)

    @model_validator(mode='after')
    def validate_contract(self):
        if self.training_task == 'image_alignment' and (self.prediction_type != 'x0' or self.posterior_mode != 'mean'):
            raise ValueError('Image alignment requires absolute x0 prediction and fixed posterior means')
        layers = self.reference_encoder_source_layers
        if (not layers or len(set(layers)) != len(layers) or any(layer < 1 or layer > 6 for layer in layers)
                or self.reference_candidates < self.num_proposals or self.num_feature_levels != 4
                or not 1 <= self.flow_sampling_steps <= self.timesteps
                or any(not 1 <= steps <= self.timesteps for steps in self.quick_eval_steps)):
            raise ValueError('Invalid V43 proposal, source-layer or timestep configuration')
        if not self.small_noise_levels or any(not math.isfinite(value) or value <= 0 for value in self.small_noise_levels):
            raise ValueError('Small-noise training requires finite positive sigma levels')
        return self

    def create_model(self, codec=None):
        if codec is None:
            with torch.random.fork_rng(devices=[]):
                codec = FrozenPoseCodec(FidelityPoseVAE(width=self.codec_width, latent_dim=self.latent_dim))
        names = ('backbone_pretrained', 'timesteps', 'num_proposals', 'hidden_dim', 'num_heads',
                 'feedforward_dim', 'num_feature_levels', 'reference_candidates',
                 'freeze_backbone_stem_and_layer1', 'freeze_backbone', 'prediction_type')
        return PoseFlowV43Model(codec=codec, **{name: getattr(self, name) for name in names},
            sampling_steps=self.flow_sampling_steps, reference_encoder_layers=len(self.reference_encoder_source_layers))

    def build(self, *, max_epochs=80, load_pretrained_checkpoint=True, **kwargs):
        model = self.create_model()
        provenance = {}
        if load_pretrained_checkpoint:
            model.codec = load_frozen_codec(self.codec_checkpoint, expected_sha256=self.codec_sha256,
                                            width=self.codec_width, latent_dim=self.latent_dim)
            if not self.latent_statistics_file:
                raise ValueError('V43 requires audited trainval latent statistics')
            path = Path(self.latent_statistics_file)
            statistics = torch.load(path, map_location='cpu', weights_only=False)
            if not statistics.get('complete_trainval') or statistics.get('codec_sha256') != self.codec_sha256:
                raise ValueError('Latent statistics require the complete trainval pass and matching codec')
            annotation = Path(statistics['annotation_path'])
            if hashlib.sha256(annotation.read_bytes()).hexdigest() != statistics['annotation_sha256']:
                raise ValueError('Latent statistics annotation hash mismatch')
            model.codec.set_statistics(statistics['mean'], statistics['std'])
            provenance = {key: value for key, value in statistics.items() if key not in ('mean', 'std')}
            provenance['statistics_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            if self.pretrained_checkpoint:
                model.initialization_report = initialize_v40_weights(model, self.pretrained_checkpoint,
                    encoder_source_layers=self.reference_encoder_source_layers)
                model.initialize_slot_confidence()
        module = PoseFlowV43Pipeline(model=model, config=self, max_epochs=max_epochs)
        module.codec_provenance = provenance
        return module
