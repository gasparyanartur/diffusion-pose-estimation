from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
from pydantic import Field
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from src.configs.base_config import BaseConfig
from src.data.pose_geometry import PoseGeometryDataset, corrupt_geometry, flip_geometry, random_tensor
from .evaluation import export_examples, geometry_metrics
from .model import PoseBoxVAE, prior_kl, reconstruction_loss


GEOMETRY_FIELDS = ('keypoints_px', 'boxes_cxcywh_px', 'joint_mask', 'image_wh')


def encode_batch(model: PoseBoxVAE, batch: dict[str, Tensor]):
    return model.encode(**{name: batch[name] for name in GEOMETRY_FIELDS})


class PoseVAEPipeline(pl.LightningModule):
    def __init__(self, width=256, latent_dim=32, lr=1e-3, min_lr=1e-5, weight_decay=1e-4,
                 kl_weight=1e-4, kl_warmup_epochs=10, seed=42, max_epochs=50,
                 artifact_directory=None, export_every_n_epochs=10):
        super().__init__()
        self.save_hyperparameters()
        self.model = PoseBoxVAE(width, latent_dim)
        self.training_generator = torch.Generator().manual_seed(seed)
        self.validation_generator = torch.Generator()
        self.latest_metrics = {}

    def kl_coefficient(self, epoch):
        return self.hparams.kl_weight * min(epoch / self.hparams.kl_warmup_epochs, 1.)

    def training_objective(self, batch: dict[str, Tensor], *, epoch: int):
        selected = random_tensor((len(batch['image_wh']),), batch['image_wh'], self.training_generator) < .5
        target = flip_geometry(batch, selected)
        corrupted = corrupt_geometry(target, generator=self.training_generator)
        mean, logvar = encode_batch(self.model, corrupted)
        noise = random_tensor(mean.shape, mean, self.training_generator, normal=True)
        prediction = self.model.decode(mean + (.5*logvar).exp()*noise, target['image_wh'])
        losses = reconstruction_loss(prediction, target)
        kl = prior_kl(mean, logvar)
        coefficient = self.kl_coefficient(epoch)
        loss = losses['reconstruction'] + coefficient*kl
        return loss, dict(losses, kl=kl, weighted_kl=coefficient*kl, kl_coefficient=loss.new_tensor(coefficient))

    def training_step(self, batch, batch_idx):
        loss, metrics = self.training_objective(batch, epoch=self.current_epoch)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite pose VAE training objective')
        self.log_dict({f'train/{name}': value for name, value in metrics.items()},
                      on_step=True, on_epoch=True, batch_size=len(batch['image_wh']))
        return loss

    def on_before_optimizer_step(self, optimizer):
        norm = torch.stack([parameter.grad.detach().norm() for parameter in self.parameters()
                            if parameter.grad is not None]).norm()
        if not torch.isfinite(norm):
            raise FloatingPointError('Non-finite pose VAE gradients')
        self.log('train/gradient_norm', norm, on_step=True, on_epoch=False)

    def on_validation_epoch_start(self):
        self.validation_generator.manual_seed(self.hparams.seed + 1000)
        self.metric_totals = {}
        self.posterior_means = []
        self.examples = []

    def validation_outputs(self, batch):
        corrupted = corrupt_geometry(batch, generator=self.validation_generator)
        clean_mean, clean_logvar = encode_batch(self.model, batch)
        noisy_mean, noisy_logvar = encode_batch(self.model, corrupted)
        clean_noise = random_tensor(clean_mean.shape, clean_mean, self.validation_generator, normal=True)
        noisy_noise = random_tensor(noisy_mean.shape, noisy_mean, self.validation_generator, normal=True)
        predictions = {
            'clean_mean': self.model.decode(clean_mean, batch['image_wh']),
            'clean_sample': self.model.decode(clean_mean + (.5*clean_logvar).exp()*clean_noise, batch['image_wh']),
            'noisy_mean': self.model.decode(noisy_mean, batch['image_wh']),
            'noisy_sample': self.model.decode(noisy_mean + (.5*noisy_logvar).exp()*noisy_noise, batch['image_wh']),
        }
        return dict(predictions=predictions, corrupted=corrupted,
                    mean=clean_mean, logvar=clean_logvar)

    def _accumulate(self, name, numerator, denominator):
        pair = torch.stack((torch.as_tensor(numerator).detach().double().cpu(),
                            torch.as_tensor(denominator).detach().double().cpu()))
        self.metric_totals[name] = self.metric_totals.get(name, torch.zeros(2, dtype=torch.float64)) + pair

    def validation_step(self, batch, batch_idx):
        output = self.validation_outputs(batch)
        count = len(batch['image_wh'])
        for variant, prediction in output['predictions'].items():
            observed = output['corrupted']['joint_mask'] if variant.startswith('noisy') else batch['joint_mask']
            for name, (numerator, denominator) in geometry_metrics(prediction, batch, observed).items():
                self._accumulate(f'{variant}/{name}', numerator, denominator)
            for name, value in reconstruction_loss(prediction, batch).items():
                self._accumulate(f'{variant}/{name}', value*count, count)
        mean, logvar = output['mean'], output['logvar']
        self._accumulate('kl', prior_kl(mean, logvar)*count, count)
        self._accumulate('posterior_std', (.5*logvar).exp().sum(), logvar.numel())
        self._accumulate('logvar_clamp_fraction', ((logvar <= -10) | (logvar >= 2)).sum(), logvar.numel())
        self.posterior_means.append(mean.detach().cpu())
        # Spread qualitative records over validation batches rather than showing only one image.
        for index in range(min(2, count)):
            if len(self.examples) >= 16:
                break
            self.examples.append(dict(
                target={name: batch[name][index].detach().cpu() for name in GEOMETRY_FIELDS[:3]},
                input={name: output['corrupted'][name][index].detach().cpu() for name in GEOMETRY_FIELDS[:3]},
                prediction={name: value[index].detach().cpu()
                            for name, value in output['predictions']['noisy_mean'].items()},
                image_wh=batch['image_wh'][index].detach().cpu(),
                annotation_id=int(batch.get('annotation_id', torch.arange(count))[index])))

    def _finish_evaluation(self, prefix):
        metrics = {name: float(pair[0]/pair[1]) for name, pair in self.metric_totals.items() if pair[1] > 0}
        means = torch.cat(self.posterior_means)
        variance = means.var(0, unbiased=False)
        metrics.update(active_latent_dimensions=float((variance > .01).sum()),
                       mean_latent_variance=float(variance.mean()),
                       mean_latent_norm=float(means.norm(dim=-1).mean()))
        metrics['reconstruction'] = metrics['noisy_sample/reconstruction']
        self.latest_metrics = {f'{prefix}/{name}': value for name, value in metrics.items()}
        self.log_dict(self.latest_metrics, on_epoch=True, on_step=False)
        should_export = (prefix == 'test' or self.current_epoch == 0
                         or (self.current_epoch+1) % self.hparams.export_every_n_epochs == 0
                         or self.current_epoch+1 == self.hparams.max_epochs)
        if self.hparams.artifact_directory and should_export and self.trainer.is_global_zero:
            generator = torch.Generator().manual_seed(self.hparams.seed + 2000)
            image_wh = torch.stack([example['image_wh'] for example in self.examples]).to(self.device)
            noise = random_tensor((len(image_wh), self.model.latent_dim), image_wh, generator, normal=True)
            prior = {name: value.detach().cpu() for name, value in self.model.decode(noise, image_wh).items()}
            directory = Path(self.hparams.artifact_directory) / prefix / f'epoch{self.current_epoch:03d}'
            export_examples(directory, self.examples, prior, self.latest_metrics)

    def on_validation_epoch_end(self):
        self._finish_evaluation('val')

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self._finish_evaluation('test')

    def on_save_checkpoint(self, checkpoint):
        checkpoint['pose_vae_training_rng'] = self.training_generator.get_state()

    def on_load_checkpoint(self, checkpoint):
        self.training_generator.set_state(checkpoint['pose_vae_training_rng'].cpu())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs,
                                                               eta_min=self.hparams.min_lr)
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='epoch'))


class PoseVAEConfig(BaseConfig):
    width: int = Field(default=256, ge=1)
    latent_dim: int = Field(default=32, ge=1)
    lr: float = Field(default=1e-3, gt=0)
    min_lr: float = Field(default=1e-5, ge=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    kl_weight: float = Field(default=1e-4, ge=0)
    kl_warmup_epochs: int = Field(default=10, ge=1)
    seed: int = 42
    export_every_n_epochs: int = Field(default=10, ge=1)

    def build(self, *, max_epochs=50, artifact_directory=None):
        return PoseVAEPipeline(**self.model_dump(), max_epochs=max_epochs, artifact_directory=artifact_directory)


class PoseGeometryDataModule(pl.LightningDataModule):
    def __init__(self, train_ann, val_ann, test_ann, batch_size=512, num_workers=0, subset_frac=1.):
        super().__init__()
        if not 0 < subset_frac <= 1:
            raise ValueError('subset_frac must be in (0,1]')
        self.paths = dict(train=Path(train_ann), val=Path(val_ann), test=Path(test_ann))
        self.batch_size, self.num_workers, self.subset_frac = batch_size, num_workers, subset_frac
        self.datasets = {}

    def setup(self, stage=None):
        splits = ('train', 'val') if stage == 'fit' else ('test',) if stage == 'test' else ('val',)
        for split in splits:
            if split not in self.datasets:
                self.datasets[split] = PoseGeometryDataset(self.paths[split])
        if {'train', 'val'} <= self.datasets.keys():
            if self.datasets['train'].image_ids & self.datasets['val'].image_ids:
                raise ValueError('Training and validation image IDs overlap')
        if {'train', 'test'} <= self.datasets.keys():
            if self.datasets['train'].image_ids & self.datasets['test'].image_ids:
                raise ValueError('Training and test image IDs overlap')

    def _loader(self, split):
        dataset = self.datasets[split]
        if self.subset_frac < 1:
            dataset = Subset(dataset, range(max(1, int(len(dataset)*self.subset_frac))))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=split == 'train',
                          num_workers=self.num_workers, pin_memory=torch.cuda.is_available())

    def train_dataloader(self):
        return self._loader('train')

    def val_dataloader(self):
        return self._loader('val')

    def test_dataloader(self):
        return self._loader('test')
