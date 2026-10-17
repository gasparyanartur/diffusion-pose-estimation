"""Matched clean-latent set prediction with bounded epoch evaluation."""

import json
from pathlib import Path
from typing import ClassVar, Literal

import torch
from pydantic import Field, model_validator

from ..diffusionregpose.training import DiffusionRegPoseMatcher, DiffusionRegPoseOTAMatcher
from .training_v12 import PoseFlowV12Pipeline, PoseFlowV12PipelineConfig
from .model_v13 import PoseFlowV13Model
from .losses_v13 import matched_layer_losses
from .optimization_v13 import gradient_norms, reconstruction_weight


class PoseFlowV13Pipeline(PoseFlowV12Pipeline):
    matched_loss = staticmethod(matched_layer_losses)

    def _evaluate_quick_grid(self, *args):
        from .quick_eval_v13 import evaluate_quick_grid
        return evaluate_quick_grid(*args)

    def _quick_architecture(self):
        return {'hidden_dim': self.model.hidden_dim, 'state_tokens': self.model.num_body_points+3,
                'decoder_layers': len(self.model.layers)}

    def __init__(self, model, *, config, **kwargs):
        super().__init__(model, config=config, **kwargs)
        matcher_arguments = {'num_body_points': model.num_body_points,
                             'weights': config.weights.to_dataclass()}
        self.matcher = (DiffusionRegPoseOTAMatcher(**matcher_arguments, candidate_topk=config.ota_candidate_topk)
                        if config.matching_strategy == 'ota' else DiffusionRegPoseMatcher(**matcher_arguments))
        self._quick_eval_epoch = None
        self._quick_eval_dataset = None
        self._gradient_logged_epoch = None

    def _losses_for_batch(self, batch):
        losses = super()._losses_for_batch(batch)
        config = self.v12_config
        weight = reconstruction_weight(config.reconstruction_weight_start,
            config.reconstruction_weight_end, config.reconstruction_decay_epochs, self.current_epoch)
        losses['reconstruction_weight'] = losses['loss_reconstruction'].new_tensor(weight)
        losses['loss_reconstruction_weighted'] = weight * losses['loss_reconstruction']
        losses['total'] = (losses['loss_latent_mean'] + losses['loss_prediction']
                           + losses['loss_reconstruction_weighted'])
        return losses

    def training_step(self, batch, batch_idx):
        losses = self._losses_for_batch(batch)
        self._log_losses('train', losses, batch['image'].shape[0])
        if batch_idx == 0:
            self._log_latent_diagnostics('train_first_batch', batch['image'].shape[0])
            if self.v12_config.optimization_diagnostics_enabled and self.trainer.is_global_zero:
                self._log_objective_gradients(losses)
        return losses['total']

    def _log_optimization_metrics(self, metrics):
        values = {name: float(value.detach()) if torch.is_tensor(value) else float(value)
                  for name, value in metrics.items()}
        for logger in self.loggers:
            logger.log_metrics(values, step=self.global_step)

    def _log_objective_gradients(self, losses):
        named = [(name, parameter) for name, parameter in self.named_parameters() if parameter.requires_grad]
        parameters = [parameter for _, parameter in named]
        metrics = {}
        for label, key in [('latent', 'loss_latent_mean'), ('prediction', 'loss_prediction'),
                           ('reconstruction_raw', 'loss_reconstruction')]:
            gradients = torch.autograd.grad(losses[key], parameters, retain_graph=True, allow_unused=True)
            for component, norm in gradient_norms(named, gradients).items():
                metrics[f'debug/optimization/objective/{label}/{component}'] = norm
                if label == 'reconstruction_raw':
                    metrics[f'debug/optimization/objective/reconstruction_weighted/{component}'] = (
                        norm * losses['reconstruction_weight'])
            del gradients
        metrics['debug/optimization/reconstruction_weight'] = losses['reconstruction_weight']
        self._log_optimization_metrics(metrics)

    def on_before_optimizer_step(self, optimizer):
        if not self.v12_config.optimization_diagnostics_enabled or not self.trainer.is_global_zero:
            return
        if self._gradient_logged_epoch == self.current_epoch and self.global_step % 100:
            return
        named = list(self.named_parameters())
        norms = gradient_norms(named, [parameter.grad for _, parameter in named])
        if not norms:
            return
        metrics = {f'debug/optimization/preclip/{component}': norm for component, norm in norms.items()}
        cap = self.trainer.gradient_clip_val
        factor = (cap / (norms['global'] + 1e-6)).clamp(max=1) if cap else norms['global'].new_tensor(1.)
        metrics['debug/optimization/clip_factor'] = factor
        metrics['debug/optimization/expected_postclip_norm'] = norms['global'] * factor
        for index, group in enumerate(optimizer.param_groups):
            metrics[f'debug/optimization/lr/group{index}'] = group['lr']
        self._log_optimization_metrics(metrics)
        self._gradient_logged_epoch = self.current_epoch

    def _denoising_losses(self, output, targets, clean, weights):
        stages = output['aux_outputs'] + [output]
        stage_losses = [self.matched_loss(stage, targets, clean, self.model.codec,
                                            self.matcher, weights) for stage in stages]
        losses = {}
        for name in stage_losses[0]:
            if name == 'total':
                continue
            mean = torch.stack([stage[name] for stage in stage_losses]).mean()
            losses[f'{name}_mean'] = mean
            losses[name] = stage_losses[-1][name]
            for index, stage in enumerate(stage_losses):
                losses[f'{name}_{index}'] = stage[name]
        losses['loss_prediction'] = torch.stack([stage['total'] for stage in stage_losses]).mean()
        return losses


    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        trainer = self.trainer
        config = self.v12_config
        if (not config.quick_eval_enabled or trainer.sanity_checking
                or not trainer.is_global_zero or self._quick_eval_epoch == self.current_epoch):
            return
        from .quick_eval_v13 import select_test_images
        from src.data.datasets import PoseDataset

        data = trainer.datamodule
        root = Path(trainer.checkpoint_callback.dirpath).parent/'quick_test'
        if self._quick_eval_dataset is None:
            self._quick_eval_dataset = PoseDataset.from_path(
                data.test_ann_path,img_root=data.img_dir,num_keypoints=data.num_keypoints)
        candidates = [int(sample['image_id']) for sample in self._quick_eval_dataset.samples]
        image_ids = select_test_images(candidates,root/'subset.json',
            count=config.quick_eval_images,seed=config.quick_eval_seed,annotations=data.test_ann_path)
        report = self._evaluate_quick_grid(self.model,self._quick_test_batches(data,image_ids),config,
            root/f'epoch{self.current_epoch:03d}',data.test_ann_path,image_ids)
        report.update(epoch=self.current_epoch,global_step=self.global_step,
            classification_normalization='focal sum / max(positive assignments,1)',
            matching_strategy=config.matching_strategy,
            ota_candidate_topk=config.ota_candidate_topk if config.matching_strategy == 'ota' else None,
            groups=self.model.num_proposals,resize_scale=data.eval_resize_scale,
            resize_max_size=data.eval_max_size,model=type(self.model).__name__.removesuffix('Model'),
            architecture=self._quick_architecture(),
            weights_stage='post-training epoch, before next epoch updates')
        report['reconstruction_weight'] = reconstruction_weight(config.reconstruction_weight_start,
            config.reconstruction_weight_end, config.reconstruction_decay_epochs, self.current_epoch)
        report['gradient_clip_norm'] = trainer.gradient_clip_val
        (root/f'epoch{self.current_epoch:03d}'/'metrics.json').write_text(
            json.dumps(report,indent=2,allow_nan=False))
        from scripts.utils.quick_test_logging import comparison_tag

        for condition,row in report['conditions'].items():
            for name,value in row['metrics'].items():
                if value >= 0:
                    self.log(comparison_tag(f'quick-test/{condition}/{name}'),value,
                             on_step=False,on_epoch=True,rank_zero_only=True)
        self._quick_eval_epoch = self.current_epoch

    def _quick_test_batches(self, data, image_ids):
        from src.data.datasets import SetPoseDataset, set_pose_collate

        samples = self._quick_eval_dataset.samples
        indices = {int(sample['image_id']):index for index,sample in enumerate(samples)}
        dataset = SetPoseDataset(self._quick_eval_dataset)
        for image_id in image_ids:
            batch = set_pose_collate([dataset[indices[image_id]]])
            batch = {key:([image.to(self.device) for image in value] if key == 'image'
                          else value.to(self.device) if torch.is_tensor(value) else value)
                     for key,value in batch.items()}
            yield data.on_after_batch_transfer(batch,0)


class PoseFlowV13PipelineConfig(PoseFlowV12PipelineConfig):
    latent_model_class: ClassVar[type] = PoseFlowV13Model
    latent_pipeline_class: ClassVar[type] = PoseFlowV13Pipeline
    matching_strategy: Literal['hungarian', 'ota'] = 'hungarian'
    quick_eval_enabled: bool = True
    quick_eval_images: int = Field(default=32, ge=1)
    quick_eval_seed: int = 42
    quick_eval_noise_fractions: list[float] = Field(default_factory=lambda: [.25,.5,1.])
    quick_eval_steps: list[int] = Field(default_factory=lambda: [1,2,3])
    reconstruction_weight_start: float = Field(default=1., gt=0, allow_inf_nan=False)
    reconstruction_weight_end: float = Field(default=1., gt=0, allow_inf_nan=False)
    reconstruction_decay_epochs: int = Field(default=40, ge=2)
    optimization_diagnostics_enabled: bool = False

    @model_validator(mode='after')
    def _validate_quick_eval(self):
        if not self.quick_eval_noise_fractions or any(not 0 < p <= 1 for p in self.quick_eval_noise_fractions):
            raise ValueError('Quick evaluation noise variance fractions must be in (0,1]')
        if len(set(self.quick_eval_noise_fractions)) != len(self.quick_eval_noise_fractions):
            raise ValueError('Quick evaluation noise fractions must be unique')
        if not self.quick_eval_steps or any(step < 1 for step in self.quick_eval_steps):
            raise ValueError('Quick evaluation requires positive sampling steps')
        if len(set(self.quick_eval_steps)) != len(self.quick_eval_steps):
            raise ValueError('Quick evaluation steps must be unique')
        return self
