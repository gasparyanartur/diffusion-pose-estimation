from __future__ import annotations

import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import nn

from src.data.pose_geometry import flip_geometry, normalize_geometry, random_tensor
from src.models.pipelines.pose_autoencoder.evaluation import CodecAPEvaluator, export_reconstructions
from src.models.pipelines.pose_autoencoder.model import geometry_loss
from src.models.pipelines.pose_autoencoder.training import GEOMETRY_FIELDS, TARGET_FIELDS, concatenate_batches
from .model import PoseBoxVAE, prior_kl


class FidelityPoseVAE(PoseBoxVAE):
    def encode(self, keypoints_px, boxes_cxcywh_px, joint_mask, image_wh):
        relative, placement = normalize_geometry(keypoints_px, boxes_cxcywh_px, joint_mask, image_wh)
        features = self.encoder(torch.cat((relative.flatten(-2), placement, joint_mask.to(relative.dtype)), -1))
        return self.posterior_mean(features), self.posterior_logvar(features).clamp(-30., 10.)

    def initialize_from_autoencoder(self, autoencoder, latent_std):
        if latent_std.shape != (self.latent_dim,) or not torch.isfinite(latent_std).all() or (latent_std < 0).any():
            raise ValueError('Expected finite nonnegative training latent standard deviations')
        for name in ('encoder', 'decoder', 'box_head', 'joint_head'):
            getattr(self, name).load_state_dict(getattr(autoencoder, name).state_dict(), strict=True)
        self.posterior_mean.load_state_dict(autoencoder.latent_head.state_dict(), strict=True)
        nn.init.zeros_(self.posterior_logvar.weight)
        with torch.no_grad():
            self.posterior_logvar.bias.copy_(2*(.05*latent_std).clamp_min(1e-6).log())


def mask_joints(batch, maximum_probability, generator, *, fixed_probability=False):
    if not 0 <= maximum_probability <= 1:
        raise ValueError('Mask probability must lie in [0,1]')
    reference, labels = batch['image_wh'], batch['joint_mask']
    probability = random_tensor((len(labels), 1), reference, generator)*maximum_probability
    if fixed_probability:
        probability.fill_(maximum_probability)
    observed = labels & (random_tensor(labels.shape, reference, generator) >= probability)
    priority = random_tensor(labels.shape, reference, generator).masked_fill(~labels, -1)
    fallback = torch.nn.functional.one_hot(priority.argmax(-1), labels.shape[-1]).bool() & labels
    observed |= fallback & ~observed.any(-1, keepdim=True)
    return dict(batch, joint_mask=observed)


def calibrated_kl_weight(reconstruction, kl, target_fraction):
    if reconstruction <= 0 or kl <= 0 or not 0 <= target_fraction <= 1:
        raise ValueError('KL calibration requires positive finite losses and a fraction in [0,1]')
    values = torch.tensor([reconstruction, kl, target_fraction])
    if not torch.isfinite(values).all():
        raise ValueError('KL calibration values must be finite')
    return target_fraction*reconstruction/kl


def posterior_statistics(mean, logvar):
    posterior_std = (.5*logvar).exp()
    mean_std = mean.std(0, unbiased=False)
    return dict(kl=float(prior_kl(mean, logvar)), posterior_std=float(posterior_std.mean()),
                posterior_to_mean_std=float((posterior_std.mean(0)/mean_std.clamp_min(1e-8)).mean()),
                active_dimensions=int((mean_std.square() > .01).sum()),
                mean_norm=float(mean.norm(dim=-1).mean()), mean_variance=float(mean_std.square().mean()),
                logvar_clamp_fraction=float(((logvar <= -30) | (logvar >= 10)).float().mean()))


class FidelityVAEPipeline(pl.LightningModule):
    def __init__(self, width=256, latent_dim=32, mode='sampled', kl_weight=0., mask_max=0.,
                 lr=1e-4, min_lr=1e-6, weight_decay=1e-4, seed=42, max_epochs=100,
                 kl_warmup_epochs=10, val_ann=None, val_annotation_ids=None, artifact_directory=None):
        super().__init__()
        if mode not in ('deterministic', 'sampled') or not 0 <= mask_max <= 1 or kl_weight < 0:
            raise ValueError('Invalid fidelity VAE mode, mask probability or KL weight')
        if mode == 'deterministic' and kl_weight:
            raise ValueError('The deterministic control must have zero KL')
        self.save_hyperparameters()
        self.model = FidelityPoseVAE(width, latent_dim)
        if mode == 'deterministic':
            self.model.posterior_logvar.requires_grad_(False)
        self.training_generator = torch.Generator().manual_seed(seed)
        self.mask_generator = torch.Generator().manual_seed(seed+1)
        self.posterior_generator = torch.Generator().manual_seed(seed+2)
        self.register_buffer('initial_latent_std', torch.ones(latent_dim))
        self.evaluator = None
        self.latest_metrics = {}

    def training_objective(self, batch, *, epoch=None):
        epoch = self.current_epoch if epoch is None else epoch
        selected = random_tensor((len(batch['image_wh']),), batch['image_wh'], self.training_generator) < .5
        target = flip_geometry(batch, selected)
        observed = mask_joints(target, self.hparams.mask_max, self.mask_generator)
        mean, logvar = self.model.encode(**{name: observed[name] for name in GEOMETRY_FIELDS})
        latent = mean
        if self.hparams.mode == 'sampled':
            noise = random_tensor(mean.shape, mean, self.posterior_generator, normal=True)
            latent = mean+(.5*logvar).exp()*noise
        reconstruction = geometry_loss(self.model.decode(latent, target['image_wh']), target)
        kl = prior_kl(mean, logvar)
        coefficient = self.hparams.kl_weight*min(epoch/max(1, self.hparams.kl_warmup_epochs), 1.)
        loss = reconstruction['loss']+coefficient*kl
        metrics = dict(reconstruction, reconstruction=reconstruction['loss'], kl=kl,
                       weighted_kl=coefficient*kl, kl_coefficient=mean.new_tensor(coefficient))
        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self.training_objective(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite fidelity VAE objective')
        self.log_dict({f'train/{name}': value for name, value in metrics.items()},
                      on_step=True, on_epoch=True, batch_size=len(batch['image_wh']))
        return loss

    def on_before_optimizer_step(self, optimizer):
        norm = torch.stack([p.grad.detach().norm() for p in self.parameters() if p.grad is not None]).norm()
        if not torch.isfinite(norm):
            raise FloatingPointError('Non-finite fidelity VAE gradients')
        self.log('train/gradient_norm', norm)

    def on_validation_epoch_start(self):
        self.targets, self.means, self.logvars = [], [], []

    def validation_step(self, batch, batch_idx):
        mean, logvar = self.model.encode(**{name: batch[name] for name in GEOMETRY_FIELDS})
        self.means.append(mean.detach().cpu())
        self.logvars.append(logvar.detach().cpu())
        self.targets.append({name: batch[name].detach().cpu() for name in TARGET_FIELDS})

    def on_validation_epoch_end(self):
        target = concatenate_batches(self.targets)
        mean, logvar = torch.cat(self.means), torch.cat(self.logvars)
        if self.evaluator is None:
            self.evaluator = CodecAPEvaluator(self.hparams.val_ann, self.hparams.val_annotation_ids)
        image_wh = target['image_wh'].to(self.device)
        mean_prediction = {name: value.cpu() for name, value in self.model.decode(mean.to(self.device), image_wh).items()}
        sample = mean
        if self.hparams.mode == 'sampled':
            noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(1042))
            sample = mean+(.5*logvar).exp()*noise
        sample_prediction = {name: value.cpu() for name, value in self.model.decode(sample.to(self.device), image_wh).items()}
        metrics = {f'mean/{name}': value for name, value in self.evaluator.evaluate(mean_prediction, target).items()}
        sample_metrics = (self.evaluator.evaluate(sample_prediction, target) if self.hparams.mode == 'sampled'
                          else {name.removeprefix('mean/'): value for name, value in metrics.items()})
        metrics.update({f'sample/{name}': value for name, value in sample_metrics.items()})
        metrics.update({f'posterior/{name}': value for name, value in posterior_statistics(mean, logvar).items()})
        metrics['selection_score'] = min(metrics[f'{variant}/{kind}_AP']
                                         for variant in ('mean', 'sample') for kind in ('pose', 'bbox'))
        self.latest_metrics = metrics
        self.log_dict({f'val/{name}': value for name, value in metrics.items()})
        if self.hparams.artifact_directory:
            directory = Path(self.hparams.artifact_directory)
            directory.mkdir(parents=True, exist_ok=True)
            (directory/f'epoch{self.current_epoch:03d}.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
            export_reconstructions(directory/f'epoch{self.current_epoch:03d}.png', sample_prediction, target,
                                   title='Ground truth: gray | Posterior reconstruction: blue | Labeled joints only')
        self.targets.clear()
        self.means.clear()
        self.logvars.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs,
                                                               eta_min=self.hparams.min_lr)
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='epoch'))

    def on_save_checkpoint(self, checkpoint):
        checkpoint['fidelity_rng'] = {name: getattr(self, name).get_state() for name in
                                     ('training_generator', 'mask_generator', 'posterior_generator')}

    def on_load_checkpoint(self, checkpoint):
        for name, state in checkpoint['fidelity_rng'].items():
            getattr(self, name).set_state(state.cpu())
