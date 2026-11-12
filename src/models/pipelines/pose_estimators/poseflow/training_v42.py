from pathlib import Path
from typing import ClassVar, Literal

import torch
from pydantic import Field, model_validator

from ..diffusionregpose.training import (
    DiffusionRegPosePipeline, DiffusionRegPosePipelineConfig, build_diffusionregpose_targets,
)
from .model_v42 import PoseFlowV42Model


class PoseFlowV42Pipeline(DiffusionRegPosePipeline):
    def _losses_for_batch(self, batch):
        if self.v42_config.training_timestep_policy == 'timestep_zero':
            return super()._losses_for_batch(batch)
        keypoints = batch['gt_keypoints']
        generator = None
        if self.diffusion_noise_seed is not None:
            generator = torch.Generator(device=keypoints.device).manual_seed(self.diffusion_noise_seed)
        timesteps = torch.randint(self.model.num_timesteps, (keypoints.shape[0],),
            device=keypoints.device, generator=generator)
        noisy, timesteps, targets = build_diffusionregpose_targets(
            batch, self.model, num_proposals=self.model.num_proposals,
            num_body_points=self.model.transformer.decoder.num_body_points,
            noise_seed=self.diffusion_noise_seed, timesteps=timesteps,
        )
        outputs = self._forward_diffusion_batch(batch, noisy, timesteps, targets)
        losses = self.criterion(outputs, targets)
        losses.update({
            'diffusion/timestep_mean': timesteps.float().mean(),
            'diffusion/timestep_min': timesteps.min().float(),
            'diffusion/timestep_max': timesteps.max().float(),
        })
        return losses

    def _evaluation_sampler_kwargs(self):
        return {}

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if self.trainer.sanity_checking or not self.v42_config.quick_eval_enabled:
            return
        from .quick_eval_v42 import run_v42_evaluation
        checkpoint = self.trainer.checkpoint_callback
        root = Path(checkpoint.dirpath).parent if checkpoint is not None else Path(self.trainer.default_root_dir)
        report = run_v42_evaluation(self.model, self.trainer.datamodule, self.v42_config,
            root / 'quick_test',
            epoch=self.current_epoch, global_step=self.global_step,
            **self._evaluation_sampler_kwargs())
        metrics = {}
        for condition, row in report['conditions'].items():
            for name, value in row['metrics'].items():
                metrics[f'quick-test/{condition}/{name}'] = value
        for logger in self.trainer.loggers:
            logger.log_metrics(metrics, step=self.global_step)
            if hasattr(logger.experiment, 'flush'):
                logger.experiment.flush()


class PoseFlowV42PipelineConfig(DiffusionRegPosePipelineConfig):
    pipeline_class: ClassVar[type[PoseFlowV42Pipeline]] = PoseFlowV42Pipeline
    pose_readout_mode: Literal['absolute', 'box_center_residual'] = 'absolute'
    confidence_mode: Literal['pose_conditioned', 'image_only'] = 'pose_conditioned'
    training_timestep_policy: Literal['timestep_zero', 'uniform'] = 'timestep_zero'
    quick_eval_enabled: bool = True
    quick_eval_images: int = Field(default=32, ge=1)
    quick_eval_seed: int = 42
    quick_eval_steps: tuple[int, ...] = (1, 4, 8)

    @model_validator(mode='after')
    def validate_steps(self):
        if not self.quick_eval_steps or any(not 1 <= steps <= self.timesteps for steps in self.quick_eval_steps):
            raise ValueError('Quick evaluation steps must lie in the diffusion schedule')
        return self

    def build(self, **kwargs):
        native = super().build(**kwargs)
        fields = ('timesteps', 'sampling_timesteps', 'num_queries', 'num_proposals',
            'num_classes', 'num_feature_levels', 'num_encoder_layers', 'num_decoder_layers',
            'hidden_dim', 'feedforward_dim', 'num_heads', 'num_body_points',
            'num_box_decoder_layers', 'num_group', 'dn_number', 'dn_box_noise_scale',
            'dn_label_noise_ratio', 'freeze_backbone_stem_and_layer1', 'freeze_backbone',
            'freeze_backbone_and_transformer')
        model = PoseFlowV42Model(
            **{name: getattr(self, name) for name in fields}, backbone_pretrained=False,
            separate_expansion_pose_head=self.pretrained_keep_decoder_layers is not None,
            native_state=native.model.state_dict(),
            pose_readout_mode=self.pose_readout_mode,
            confidence_mode=self.confidence_mode,
        )
        model.initialization_report = {
            'source': self.pretrained_checkpoint if kwargs.get('load_pretrained_checkpoint', True) else None,
            'transfer': 'strict native load before replacing expansion; attention value/output projections retained',
            'fresh': ['pose_encoder', 'pose_readers', 'person_attention sampling_offsets/attention_weights'],
            'removed': ['joint query embeddings', 'joint coordinate refinement heads'],
            'pose_readout_mode': self.pose_readout_mode,
            'confidence_mode': self.confidence_mode,
            'image_confidence_initialization': 'cloned box-stage classifier' if self.confidence_mode == 'image_only' else None,
        }
        arguments = ('lr', 'lr_backbone', 'weight_decay', 'max_epochs', 'lr_warmup_epochs',
            'lr_min_frac', 'scheduler_type', 'lr_decay_milestones', 'split_weight_decay',
            'val_ap_every_n_epochs', 'diffusion_noise_seed', 'deterministic_seed')
        pipeline = self.pipeline_class(model, native.criterion,
            **{name: getattr(native, name) for name in arguments}, val_ann_file=kwargs.get('val_ann_file'))
        pipeline.v42_config = self
        pipeline.save_hyperparameters({
            'pose_readout_mode': self.pose_readout_mode,
            'confidence_mode': self.confidence_mode,
            'training_timestep_policy': self.training_timestep_policy,
        })
        return pipeline
