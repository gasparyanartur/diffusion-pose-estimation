from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
import torch
from pydantic import Field

from src.configs.base_config import BaseConfig
from src.data.pose_geometry import flip_geometry, random_tensor
from .evaluation import CodecAPEvaluator, clean_gate, export_reconstructions, perturb_latents
from .model import PoseBoxAutoencoder, geometry_loss


GEOMETRY_FIELDS = ('keypoints_px', 'boxes_cxcywh_px', 'joint_mask', 'image_wh')
TARGET_FIELDS = (*GEOMETRY_FIELDS, 'annotation_id', 'image_id')


def encode_batch(model, batch):
    return model.encode(**{name:batch[name] for name in GEOMETRY_FIELDS})


def concatenate_batches(batches):
    return {name:torch.cat([batch[name] for batch in batches]) for name in batches[0]}


class PoseAutoencoderPipeline(pl.LightningModule):
    def __init__(self, width=256, latent_dim=32, phase='clean', lr=1e-3, min_lr=1e-5,
                 weight_decay=1e-4, seed=42, max_epochs=100, gate_ap=97.5,
                 val_ann=None, val_annotation_ids=None, artifact_directory=None, export_every_n_epochs=10):
        super().__init__()
        if phase not in ('clean', 'robust'):
            raise ValueError('phase must be clean or robust')
        self.save_hyperparameters()
        self.model = PoseBoxAutoencoder(width, latent_dim)
        self.register_buffer('latent_mean', torch.zeros(latent_dim))
        self.register_buffer('latent_std', torch.ones(latent_dim))
        self.register_buffer('latent_calibrated', torch.tensor(False))
        self.training_generator = torch.Generator().manual_seed(seed)
        self.evaluator = None
        self.latest_metrics = {}
        if phase == 'robust':
            self.model.freeze_encoder()

    def train(self, mode=True):
        super().train(mode)
        if self.hparams.phase == 'robust':
            self.model.freeze_encoder()
        return self

    def set_latent_statistics(self, mean, std):
        if mean.shape != self.latent_mean.shape or std.shape != self.latent_std.shape:
            raise ValueError('Latent statistics have incorrect dimensions')
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std >= 0).all():
            raise ValueError('Latent statistics must be finite with nonnegative standard deviations')
        self.latent_mean.copy_(mean.to(self.latent_mean))
        self.latent_std.copy_(std.to(self.latent_std))
        self.latent_calibrated.fill_(True)

    def training_objective(self, batch):
        if self.hparams.phase == 'robust' and not self.latent_calibrated:
            raise ValueError('Robust training requires fixed training-only latent statistics')
        selected = random_tensor((len(batch['image_wh']),), batch['image_wh'], self.training_generator) < .5
        target = flip_geometry(batch, selected)
        latent = encode_batch(self.model, target)
        clean_prediction = self.model.decode(latent, target['image_wh'])
        clean = geometry_loss(clean_prediction, target)
        if self.hparams.phase == 'clean':
            return clean['loss'], clean
        fraction = .05*random_tensor((len(latent), 1), latent, self.training_generator)
        noise = random_tensor(latent.shape, latent, self.training_generator, normal=True)
        noisy_prediction = self.model.decode(latent+fraction*self.latent_std*noise, target['image_wh'])
        noisy = geometry_loss(noisy_prediction, target)
        return .5*(clean['loss']+noisy['loss']), dict(clean_loss=clean['loss'], noisy_loss=noisy['loss'])

    def training_step(self, batch, batch_idx):
        loss, metrics = self.training_objective(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite geometry autoencoder objective')
        self.log_dict({f'train/{name}':value for name,value in dict(metrics, total=loss).items()},
                      on_step=True, on_epoch=True, batch_size=len(batch['image_wh']))
        return loss

    def on_before_optimizer_step(self, optimizer):
        norm = torch.stack([parameter.grad.detach().norm() for parameter in self.parameters()
                            if parameter.grad is not None]).norm()
        if not torch.isfinite(norm):
            raise FloatingPointError('Non-finite geometry autoencoder gradients')
        self.log('train/gradient_norm', norm, on_step=True, on_epoch=False)

    def on_validation_epoch_start(self):
        self.predictions, self.targets, self.latents = [], [], []

    def validation_step(self, batch, batch_idx):
        latent = encode_batch(self.model, batch)
        prediction = self.model.decode(latent, batch['image_wh'])
        self.predictions.append({name:value.detach().cpu() for name,value in prediction.items()})
        self.targets.append({name:batch[name].detach().cpu() for name in TARGET_FIELDS})
        self.latents.append(latent.detach().cpu())

    def on_validation_epoch_end(self):
        if self.evaluator is None:
            self.evaluator = CodecAPEvaluator(self.hparams.val_ann, self.hparams.val_annotation_ids)
        prediction, target = concatenate_batches(self.predictions), concatenate_batches(self.targets)
        clean = self.evaluator.evaluate(prediction, target)
        metrics = {f'clean/{name}':value for name,value in clean.items()}
        metrics.update({f'clean/{name}':float(value) for name,value in geometry_loss(prediction, target).items()})
        score = min(clean['pose_AP'], clean['bbox_AP'])
        if self.hparams.phase == 'robust':
            latent = torch.cat(self.latents).to(self.device)
            perturbed = perturb_latents(latent, self.latent_std, .05, seed=1042)
            noisy_prediction = {name:value.cpu() for name,value in self.model.decode(
                perturbed, target['image_wh'].to(self.device)).items()}
            noisy = self.evaluator.evaluate(noisy_prediction, target)
            metrics.update({f'noise05/{name}':value for name,value in noisy.items()})
            score = min(noisy['pose_AP'], noisy['bbox_AP']) if clean_gate(clean, self.hparams.gate_ap) else -1e6+score
        metrics['selection_score'] = score
        self.latest_metrics = metrics
        self.log_dict({f'val/{name}':value for name,value in metrics.items()}, on_step=False, on_epoch=True)
        if self.hparams.artifact_directory and self.trainer.is_global_zero:
            directory = Path(self.hparams.artifact_directory)
            directory.mkdir(parents=True, exist_ok=True)
            (directory/f'epoch{self.current_epoch:03d}.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
            if (self.current_epoch == 0 or (self.current_epoch+1) % self.hparams.export_every_n_epochs == 0
                    or self.current_epoch+1 == self.hparams.max_epochs):
                export_reconstructions(directory/f'epoch{self.current_epoch:03d}.png', prediction, target)
                torch.save(dict(prediction=prediction, target=target), directory/f'epoch{self.current_epoch:03d}.pt')
        self.predictions.clear()
        self.targets.clear()
        self.latents.clear()

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs,
                                                               eta_min=self.hparams.min_lr)
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='epoch'))

    def on_save_checkpoint(self, checkpoint):
        checkpoint['pose_autoencoder_rng'] = self.training_generator.get_state()

    def on_load_checkpoint(self, checkpoint):
        self.training_generator.set_state(checkpoint['pose_autoencoder_rng'].cpu())


class PoseAutoencoderConfig(BaseConfig):
    width: int = Field(default=256, ge=1)
    latent_dim: int = Field(default=32, ge=1)
    phase: Literal['clean', 'robust'] = 'clean'
    lr: float = Field(default=1e-3, gt=0)
    min_lr: float = Field(default=1e-5, ge=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    seed: int = 42
    gate_ap: float = Field(default=97.5, ge=0, le=100)
    export_every_n_epochs: int = Field(default=10, ge=1)

    def build(self, **kwargs):
        return PoseAutoencoderPipeline(**self.model_dump(), **kwargs)
