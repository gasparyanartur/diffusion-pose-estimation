from typing import ClassVar
import random

import numpy as np
import torch
from pydantic import Field

from .model_v42_jepa import EMAPoseEncoder
from .training_v42_image_confidence import PoseFlowV42ImageConfidencePipelineConfig
from .training_v42_paired_proposal import PoseFlowV42PairedProposalPipeline


class PoseFlowV42JEPAPipeline(PoseFlowV42PairedProposalPipeline):
    def __init__(self, model, criterion, **kwargs):
        super().__init__(model, criterion, **kwargs)
        self.teacher = EMAPoseEncoder(model)
        self.model.return_pose_references = True
        self.register_buffer('ema_updates', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update_ema(self):
        self.teacher.update_from(self.model, self.v42_config.ema_decay)
        self.ema_updates.add_(1)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.update_ema()

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['v42_jepa_rng'] = dict(
            python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        state = checkpoint['v42_jepa_rng']
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'].cpu())
        if state['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])

    def _log_losses(self, prefix, losses, batch_size):
        super()._log_losses(prefix, losses, batch_size)
        if prefix == 'train':
            for name in ('jepa/loss', 'jepa/clean_geometry', 'jepa/teacher_variance', 'jepa/target_lag'):
                if name in losses:
                    self.log(f'train/{name}_step', losses[name].detach(),
                             batch_size=batch_size, on_step=True, on_epoch=False)

    def _additional_pose_losses(self, outputs, targets, timesteps, losses):
        matches = outputs['pose_target_matches']
        features = outputs['pose_features']  # <B, N, D>
        points = self.criterion.num_body_points
        clean_xy = torch.cat([target['keypoints'][people, :2 * points]
                              for target, (_, people) in zip(targets, matches)])
        count = len(clean_xy)
        zero = features.sum() * 0
        diagnostics = {'jepa/positive_count': features.new_tensor(count),
                       'jepa/ema_updates': self.ema_updates.float().clone()}
        latent_loss = clean_geometry = zero
        if count:
            online = self.model.transformer.decoder.norm(self.model.pose_encoder(clean_xy))
            target_codes = self.teacher(clean_xy)
            predicted = torch.cat([features[b, queries] for b, (queries, _) in enumerate(matches)])
            references = torch.cat([outputs['pose_reference_boxes'][b, queries]
                                    for b, (queries, _) in enumerate(matches)])
            clean_prediction = self.model.read_pose(online[None], references[None],
                layer_index=len(self.model.pose_readers) - 1)[0]
            # Reuse native masks/OKS/normalizer with fixed matches; only geometry terms enter the loss.
            clean_keypoints = outputs['pred_keypoints'].new_zeros(outputs['pred_keypoints'].shape)
            offset = 0
            for b, (queries, _) in enumerate(matches):
                clean_keypoints[b, queries, :2 * points] = clean_prediction[offset:offset + len(queries)]
                offset += len(queries)
            clean_outputs = {**outputs, 'pred_keypoints': clean_keypoints}
            clean_losses = self.criterion._loss_for_outputs(clean_outputs, targets, matches=matches)
            clean_geometry = (self.criterion.weights.keypoints * clean_losses['loss_keypoints']
                              + self.criterion.weights.oks * clean_losses['loss_oks'])
            per_person = (predicted - target_codes).square().mean(-1)
            latent_loss = per_person.mean()
            with torch.no_grad():
                diagnostics.update({
                    'jepa/online_rms': online.square().mean().sqrt(),
                    'jepa/teacher_rms': target_codes.square().mean().sqrt(),
                    'jepa/teacher_variance': target_codes.var(dim=0, unbiased=False).mean(),
                    'jepa/online_variance': online.var(dim=0, unbiased=False).mean(),
                    'jepa/target_lag': (online - target_codes).square().mean(),
                })
                person_times = torch.cat([timesteps[b].expand(len(queries))
                                          for b, (queries, _) in enumerate(matches)])
                validity = torch.cat([target['keypoints'][people, 2 * points:]
                                      for target, (_, people) in zip(targets, matches)])
                noisy_xy = torch.cat([outputs['pred_keypoints'][b, queries, :2 * points]
                                      for b, (queries, _) in enumerate(matches)])
                for name, start, stop in (('low', 0, 333), ('mid', 333, 667), ('high', 667, 1000)):
                    selected = (person_times >= start) & (person_times < stop)
                    if selected.any():
                        valid = validity[selected].repeat_interleave(2, dim=-1)
                        for label, xy in (('clean', clean_prediction), ('denoised', noisy_xy)):
                            diagnostics[f'jepa/{name}/{label}_coordinate_mae'] = (
                                ((xy[selected] - clean_xy[selected]).abs() * valid).sum() / valid.sum().clamp_min(1))
                        diagnostics[f'jepa/{name}/latent_mse'] = per_person[selected].mean()
        losses['jepa/baseline_total'] = losses['total'].detach()
        losses['total'] = (losses['total'] + self.v42_config.jepa_weight * latent_loss
                           + self.v42_config.clean_pose_weight * clean_geometry)
        losses.update({'jepa/loss': latent_loss, 'jepa/clean_geometry': clean_geometry,
                       'jepa/weighted_loss': self.v42_config.jepa_weight * latent_loss,
                       'jepa/weighted_clean_geometry': self.v42_config.clean_pose_weight * clean_geometry,
                       **diagnostics})
        return losses


class PoseFlowV42JEPAPipelineConfig(PoseFlowV42ImageConfidencePipelineConfig):
    pipeline_class: ClassVar[type] = PoseFlowV42JEPAPipeline
    jepa_weight: float = Field(default=1., ge=0., allow_inf_nan=False)
    clean_pose_weight: float = Field(default=.5, ge=0., allow_inf_nan=False)
    ema_decay: float = Field(default=.999, ge=0., lt=1., allow_inf_nan=False)

    def build(self, **kwargs):
        pipeline = super().build(**kwargs)
        pipeline.save_hyperparameters({'jepa_weight': self.jepa_weight,
                                      'clean_pose_weight': self.clean_pose_weight,
                                      'ema_decay': self.ema_decay,
                                      'representation_target': 'ema_pose_coordinates'})
        return pipeline
