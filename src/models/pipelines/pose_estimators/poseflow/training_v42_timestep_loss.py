import random
from typing import ClassVar, Literal

import numpy as np
import torch

from ..diffusionregpose.training import DiffusionRegPoseCriterion, build_diffusionregpose_targets
from .training_v42 import PoseFlowV42Pipeline, PoseFlowV42PipelineConfig


def geometry_factors(timesteps, total_timesteps):
    if total_timesteps < 2 or timesteps.ndim != 1:
        raise ValueError('Expected a timestep vector and at least two diffusion steps')
    if timesteps.dtype != torch.long:
        raise TypeError('Timesteps must be long integers')
    if ((timesteps < 0) | (timesteps >= total_timesteps)).any():
        raise ValueError('Timesteps outside the diffusion schedule')
    progress = timesteps.float() / (total_timesteps-1)
    return .2+1.6*progress, 1.8-1.6*progress


class TimestepLossCriterion(DiffusionRegPoseCriterion):
    def __init__(self, *, total_timesteps=1000, **kwargs):
        super().__init__(**kwargs)
        self.total_timesteps = total_timesteps
        self._active_geometry_weights = None
        self._assignment_cache = None

    def _loss_for_outputs(self, outputs, targets, *, include_keypoints=True,
                          include_classification_diagnostics=False, matches=None):
        if matches is None and self._assignment_cache is not None:
            identity = id(outputs)
            if identity not in self._assignment_cache:
                self._assignment_cache[identity] = (
                    self.matcher(outputs,targets) if 'pred_keypoints' in outputs
                    else self.matcher(outputs,targets,include_keypoints=False))
            matches = self._assignment_cache[identity]
        values = super()._loss_for_outputs(outputs,targets,include_keypoints=include_keypoints,
            include_classification_diagnostics=include_classification_diagnostics,matches=matches)
        if self._active_geometry_weights is None:
            return values
        box_weights,pose_weights = self._active_geometry_weights
        normalizer = self._loss_normalizer(matches,targets,outputs['pred_logits'])
        corrections = {key: values[key]*0 for key in ('loss_bbox','loss_giou','loss_keypoints','loss_oks')}
        for index,target in enumerate(targets):
            row = {key: outputs[key][index:index+1] for key in
                   ('pred_logits','pred_boxes','pred_keypoints') if key in outputs}
            local_matches = [matches[index]]
            local = super()._loss_for_outputs(row,[target],include_keypoints=include_keypoints,matches=local_matches)
            local_normalizer = self._loss_normalizer(local_matches,[target],row['pred_logits'])
            for key in corrections:
                factor = box_weights[index] if key in ('loss_bbox','loss_giou') else pose_weights[index]
                corrections[key] = corrections[key]+local[key]*local_normalizer*(factor-1)/normalizer
        for key,correction in corrections.items():
            values[key] = values[key]+correction
        values['total'] = self._weighted_clean_loss(values)
        return values

    def _denoising_losses(self, outputs):
        values = super()._denoising_losses(outputs)
        if self._active_geometry_weights is None:
            return values
        box_weights,_ = self._active_geometry_weights
        for index,factor in enumerate(box_weights):
            row = {key: outputs[key][index:index+1] for key in
                   ('dn_class_pred','dn_class_input','dn_bbox_pred','dn_bbox_input')}
            local = super()._denoising_losses(row)
            # Native DN reduction divides by queries, not by people or batch size.
            for key in ('dn_loss_bbox','dn_loss_giou'):
                values[key] = values[key]+local[key]*(factor-1)
        return values

    def forward(self, outputs, targets, *, timesteps=None, geometry_weights=None, **kwargs):
        if self._assignment_cache is not None:
            raise RuntimeError('Timestep criterion cannot be called recursively')
        reference = outputs['pred_logits']
        if geometry_weights is None:
            if timesteps is None or timesteps.shape != (len(targets),) or timesteps.device != reference.device:
                raise ValueError('Expected per-image timesteps on the output device')
            geometry_weights = geometry_factors(timesteps,self.total_timesteps)
        factors = tuple(value.to(reference) for value in geometry_weights)
        if len(factors) != 2 or any(value.shape != (len(targets),) or not torch.isfinite(value).all()
                                   or (value < 0).any() for value in factors):
            raise ValueError('Expected finite nonnegative per-image box/pose weights')
        self._assignment_cache = {}
        try:
            native = super().forward(outputs,targets,**kwargs)
            if all(torch.equal(value,torch.ones_like(value)) for value in factors):
                losses = native
            else:
                self._active_geometry_weights = factors
                losses = super().forward(outputs,targets,**kwargs)
            losses['schedule/native_total'] = native['total'].detach()
            losses['schedule/box_factor'] = factors[0].mean()
            losses['schedule/pose_factor'] = factors[1].mean()
            losses['schedule/weighted_total'] = losses['total'].detach()
            if timesteps is not None:
                bins = (timesteps*3//self.total_timesteps).clamp_max(2)
                for index in range(3):
                    selected = bins == index
                    if selected.any():
                        losses[f'schedule/box_factor_bin{index}'] = factors[0][selected].mean()
                        losses[f'schedule/pose_factor_bin{index}'] = factors[1][selected].mean()
            return losses
        finally:
            self._active_geometry_weights = None
            self._assignment_cache = None


class TimestepLossPipeline(PoseFlowV42Pipeline):
    def _log_losses(self, prefix, losses, batch_size):
        super()._log_losses(prefix,losses,batch_size)
        if prefix == 'train':
            for key in ('schedule/native_total','schedule/weighted_total','schedule/box_factor','schedule/pose_factor'):
                self.log(f'train/{key}_step',losses[key].detach(),batch_size=batch_size,on_step=True,on_epoch=False)

    def _losses_for_batch(self, batch):
        keypoints = batch['gt_keypoints']
        generator = None
        if self.diffusion_noise_seed is not None:
            generator = torch.Generator(device=keypoints.device).manual_seed(self.diffusion_noise_seed)
        timesteps = torch.randint(self.model.num_timesteps,(len(keypoints),),device=keypoints.device,generator=generator)
        noisy,timesteps,targets = build_diffusionregpose_targets(batch,self.model,
            num_proposals=self.model.num_proposals,num_body_points=self.model.transformer.decoder.num_body_points,
            noise_seed=self.diffusion_noise_seed,timesteps=timesteps)
        outputs = self._forward_diffusion_batch(batch,noisy,timesteps,targets)
        losses = self.criterion(outputs,targets,timesteps=timesteps)
        losses.update({'diffusion/timestep_mean':timesteps.float().mean(),
            'diffusion/timestep_min':timesteps.min().float(),'diffusion/timestep_max':timesteps.max().float()})
        return losses

    def contract(self):
        return dict(base='v42_box_center_uniform_t',high_t_box=.9,low_t_box=.1,
            scale=2.,interpolation='linear_t_over_T_minus_1',classification='unchanged',
            geometry='all_final_aux_encoder_DN_box_and_all_pose_terms',matching='native_costs_unchanged')

    def on_save_checkpoint(self, checkpoint):
        checkpoint['v42_timestep_loss'] = self.contract()
        checkpoint['v42_timestep_loss_rng'] = dict(python=random.getstate(),numpy=np.random.get_state(),
            torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('v42_timestep_loss') != self.contract():
            raise ValueError('Checkpoint does not match timestep-loss contract')
        state = checkpoint['v42_timestep_loss_rng']
        random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
        if state['cuda'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


class TimestepLossConfig(PoseFlowV42PipelineConfig):
    pipeline_class: ClassVar[type] = TimestepLossPipeline
    training_timestep_policy: Literal['uniform'] = 'uniform'
    pose_readout_mode: Literal['box_center_residual'] = 'box_center_residual'
    matching_strategy: Literal['hungarian'] = 'hungarian'
    timestep_loss_policy: Literal['box_to_pose_linear_90_10_scale2'] = 'box_to_pose_linear_90_10_scale2'

    def build(self, **kwargs):
        pipeline = super().build(**kwargs)
        native = pipeline.criterion
        pipeline.criterion = TimestepLossCriterion(total_timesteps=pipeline.model.num_timesteps,
            num_classes=native.num_classes,matcher=native.matcher,num_body_points=native.num_body_points,
            num_box_decoder_layers=native.num_box_decoder_layers,dn_label_coef=native.dn_label_coef,
            dn_bbox_coef=native.dn_bbox_coef,interm_loss_coef=native.interm_loss_coef,
            no_interm_box_loss=native.no_interm_box_loss,weights=native.weights)
        pipeline.save_hyperparameters({'v42_timestep_loss':pipeline.contract()})
        return pipeline
