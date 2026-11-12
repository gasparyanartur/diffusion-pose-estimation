from typing import ClassVar, Literal
import random
import json
from pathlib import Path

import numpy as np
import torch
from pydantic import Field, model_validator

from ..diffusionregpose.training import DiffusionRegPosePipelineConfig
from .model_v45_task_pose import EMATaskPoseEncoder, PoseFlowV45TaskPoseModel
from .training_v42_image_confidence import PoseFlowV42ImageConfidencePipelineConfig
from .training_v42_paired_proposal import PoseFlowV42PairedProposalPipeline
from .training_v44 import NATIVE_MODEL_FIELDS


def pose_geometry_loss(predicted_xy, target_keypoints, areas, sigmas, weights):
    points = len(sigmas)
    if not len(predicted_xy):
        return predicted_xy.sum() * 0
    target_xy = target_keypoints[..., :2 * points]
    valid = target_keypoints[..., 2 * points:]
    l1 = ((predicted_xy - target_xy).abs() * valid.repeat_interleave(2, -1)).sum() / len(predicted_xy)
    error = (predicted_xy.reshape(-1, points, 2) - target_xy.reshape(-1, points, 2)).square().sum(-1)
    variance = (sigmas.to(error) * 2).square()
    oks = torch.exp(-error / (areas[:, None] * variance[None] * 2))
    oks_loss = (1 - (oks * valid).sum(-1) / (valid.sum(-1) + 1e-6)).mean()
    return weights.keypoints * l1 + weights.oks * oks_loss


class PoseFlowV45TaskPosePipeline(PoseFlowV42PairedProposalPipeline):
    def __init__(self, model, criterion, **kwargs):
        super().__init__(model, criterion, **kwargs)
        self.teacher = EMATaskPoseEncoder(model.pose_codec)
        self.register_buffer('ema_updates', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update_ema(self):
        self.teacher.update_from(self.model.pose_codec, self.v42_config.ema_decay)
        self.ema_updates.add_(1)

    def _losses_for_batch(self, batch):
        if (batch['gt_labeled'].bool().any(-1).sum(-1) > self.model.num_proposals).any():
            raise ValueError('Valid person count exceeds selected proposal capacity')
        return super()._losses_for_batch(batch)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.update_ema()

    def _additional_pose_losses(self, outputs, targets, timesteps, losses):
        predicted = outputs['pred_latents']
        points = self.criterion.num_body_points
        complete_xy, complete_predictions, complete_targets, complete_areas, complete_times = [], [], [], [], []
        for b, (queries, people) in enumerate(outputs['pose_target_matches']):
            rows = targets[b]['keypoints'][people]
            eligible = rows[:, 2 * points:].bool().all(-1)
            complete_targets.append(rows[eligible])
            complete_xy.append(rows[eligible, :2 * points])
            complete_areas.append(targets[b]['area'][people][eligible])
            complete_predictions.append(predicted[b, queries][eligible])
            complete_times.append(timesteps[b].expand(int(eligible.sum())))
        xy, target_rows, areas = torch.cat(complete_xy), torch.cat(complete_targets), torch.cat(complete_areas)
        estimates, times = torch.cat(complete_predictions), torch.cat(complete_times)
        count = len(xy)
        clean_loss = latent_loss = predicted.sum() * 0
        diagnostics = {}
        if count:
            online = self.model.pose_codec.encode(xy)
            teacher = self.teacher(xy)
            online_xy, teacher_xy = self.model.pose_codec.decode(online), self.model.pose_codec.decode(teacher)
            arguments = (target_rows, areas, self.criterion.matcher.sigmas, self.criterion.weights)
            online_loss = pose_geometry_loss(online_xy, *arguments)
            teacher_loss = pose_geometry_loss(teacher_xy, *arguments)
            clean_loss = .5 * (online_loss + teacher_loss)
            latent_loss = (estimates - teacher).square().mean()
            with torch.no_grad():
                diagnostics.update({
                    'task_pose/online_geometry': online_loss.detach(), 'task_pose/teacher_geometry': teacher_loss.detach(),
                    'task_pose/online_variance': online.var(0, unbiased=False).mean(),
                    'task_pose/teacher_variance': teacher.var(0, unbiased=False).mean(),
                    'task_pose/teacher_lag': (online - teacher).square().mean(),
                    'task_pose/online_rms': online.square().mean().sqrt(), 'task_pose/teacher_rms': teacher.square().mean().sqrt(),
                })
                for label, prediction in (('online', online_xy), ('teacher', teacher_xy)):
                    distances = (prediction - xy).reshape(-1, points, 2).norm(dim=-1)
                    oks = torch.exp(-distances.square() / (areas[:, None] * (2 * self.criterion.matcher.sigmas.to(xy)).square()[None] * 2)).mean(-1)
                    diagnostics[f'task_pose/{label}_oks'] = oks.mean()
                    diagnostics[f'task_pose/{label}_oks_p05'] = torch.quantile(oks, .05)
                    diagnostics[f'task_pose/{label}_error_p95'] = torch.quantile(distances.flatten(), .95)
                    for joint in range(points):
                        diagnostics[f'task_pose/{label}_joint_{joint}_error'] = distances[:, joint].mean()
                for label, low, high in (('low', 0, 333), ('mid', 333, 667), ('high', 667, 1000)):
                    selected = (times >= low) & (times < high)
                    if selected.any():
                        diagnostics[f'task_pose/{label}_latent_mse'] = (estimates[selected] - teacher[selected]).square().mean()
        baseline = losses['total']
        losses['total'] = baseline + self.v42_config.clean_pose_weight * clean_loss + self.v42_config.jepa_weight * latent_loss
        matched_count = sum(len(people) for _, people in outputs['pose_target_matches'])
        losses.update({'task_pose/baseline': baseline.detach(), 'task_pose/clean': clean_loss,
            'task_pose/jepa': latent_loss, 'task_pose/complete_count': predicted.new_tensor(count),
            'task_pose/complete_fraction': predicted.new_tensor(count / max(matched_count, 1)),
            'task_pose/ema_updates': self.ema_updates.float().clone(), **diagnostics})
        return losses

    def _log_losses(self, prefix, losses, batch_size):
        super()._log_losses(prefix, losses, batch_size)
        if prefix == 'train':
            for name in ('task_pose/clean', 'task_pose/jepa', 'task_pose/complete_count'):
                self.log(f'train/{name}_step', losses[name].detach(), batch_size=batch_size, on_step=True, on_epoch=False)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['v45_task_pose'] = self.contract()
        checkpoint['v45_task_pose_rng'] = dict(python=random.getstate(), numpy=np.random.get_state(),
            torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('v45_task_pose') != self.contract():
            raise ValueError('Checkpoint does not match the V45 task-pose contract')
        super().on_load_checkpoint(checkpoint)
        state = checkpoint['v45_task_pose_rng']
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'].cpu())
        if state['cuda'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])

    def contract(self):
        return dict(version='45-task-pose', coordinates='absolute_image_normalized', state='coordinates',
            latent_dim=self.model.latent_dim, fixed_context=True, jepa_weight=self.v42_config.jepa_weight,
            clean_pose_weight=self.v42_config.clean_pose_weight, ema_decay=self.v42_config.ema_decay)

    def _evaluation_sampler_kwargs(self):
        return dict(sampler_description=self.model.sampler_kind)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if self.trainer.sanity_checking or not self.v42_config.quick_eval_enabled or self.current_epoch % 5:
            return
        from .diagnostics_v45_task_pose import evaluate_task_pose_diagnostics
        checkpoint = self.trainer.checkpoint_callback
        root = Path(checkpoint.dirpath).parent if checkpoint is not None else Path(self.trainer.default_root_dir)
        directory = root / 'quick_test' / f'epoch{self.current_epoch:03d}'
        report = json.loads((directory / 'metrics.json').read_text())
        diagnostics = evaluate_task_pose_diagnostics(self, self.trainer.datamodule, report['image_ids'], directory)
        metrics = {'task-pose-diagnostic/complete_people': diagnostics['complete_people']}
        for name in ('online_clean', 'teacher_clean'):
            for key, value in diagnostics.get(name, {}).items():
                metrics[f'task-pose-diagnostic/{name}/{key}'] = value
        for name, values in diagnostics.get('state_use', {}).items():
            for key, value in values.items():
                metrics[f'task-pose-diagnostic/state/{name}/{key}'] = value
        for level, values in diagnostics['perturbations'].items():
            for key, value in values.items():
                metrics[f'task-pose-diagnostic/perturb_{level}/{key}'] = value
        for logger in self.trainer.loggers:
            logger.log_metrics(metrics, step=self.global_step)
            if hasattr(logger.experiment, 'flush'):
                logger.experiment.flush()


class PoseFlowV45TaskPosePipelineConfig(PoseFlowV42ImageConfidencePipelineConfig):
    pipeline_class: ClassVar[type] = PoseFlowV45TaskPosePipeline
    pose_readout_mode: Literal['absolute'] = 'absolute'
    latent_dim: Literal[32] = 32
    jepa_weight: float = Field(default=0., ge=0., allow_inf_nan=False)
    clean_pose_weight: float = Field(default=.5, ge=0., allow_inf_nan=False)
    ema_decay: float = Field(default=.999, ge=0., lt=1., allow_inf_nan=False)

    @model_validator(mode='after')
    def validate_task_pose_contract(self):
        if self.matching_strategy != 'hungarian':
            raise ValueError('V45 task-pose requires fixed Hungarian proposal pairing')
        return self

    def build(self, **kwargs):
        values = {name: getattr(self, name) for name in DiffusionRegPosePipelineConfig.model_fields}
        native = DiffusionRegPosePipelineConfig(**values).build(**kwargs)
        model = PoseFlowV45TaskPoseModel(
            **{name: getattr(self, name) for name in NATIVE_MODEL_FIELDS}, backbone_pretrained=False,
            separate_expansion_pose_head=self.pretrained_keep_decoder_layers is not None,
            native_state=native.model.state_dict(), latent_dim=self.latent_dim)
        model.initialization_report = dict(source=self.pretrained_checkpoint if kwargs.get('load_pretrained_checkpoint', True) else None,
            transfer='strict native load before task-pose replacement; fresh codec/residual/adapter',
            coordinate_convention='absolute_image_normalized', cached_context=True)
        arguments = ('lr', 'lr_backbone', 'weight_decay', 'max_epochs', 'lr_warmup_epochs', 'lr_min_frac',
            'scheduler_type', 'lr_decay_milestones', 'split_weight_decay', 'val_ap_every_n_epochs',
            'diffusion_noise_seed', 'deterministic_seed')
        pipeline = self.pipeline_class(model, native.criterion,
            **{name: getattr(native, name) for name in arguments}, val_ann_file=kwargs.get('val_ann_file'))
        pipeline.v42_config = self
        pipeline.save_hyperparameters(dict(training_timestep_policy='uniform', target_assignment='proposal_hungarian_fixed',
            confidence_mode='image_only', pose_readout_mode='absolute', v45_task_pose=pipeline.contract()))
        return pipeline
