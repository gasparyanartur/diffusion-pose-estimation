from contextlib import nullcontext
import json
import logging
from pathlib import Path
from typing import ClassVar, Literal

import torch
from pydantic import Field, model_validator

from .coordinate_v19 import matched_coordinate_loss
from .latent_v12 import build_v12_targets, corrupt_latents, ddim_update
from .model_v14 import tensor_rms
from .model_v19 import PoseFlowV19Model
from .optimization_v13 import gradient_norms
from .training_v12 import PoseFlowV12Pipeline, PoseFlowV12PipelineConfig
from .training_v13 import PoseFlowV13Pipeline
from .training_v15_rollout import sample_rollout_schedule


class PoseFlowV19Pipeline(PoseFlowV13Pipeline):
    def _denoising_losses(self, output, targets):
        weights = self.v12_config.weights.to_dataclass()
        stages = [self._stage_losses(stage, targets, weights)
                  for stage in output['aux_outputs']+[output]]
        losses = {}
        for name in stages[0]:
            if name == 'total':
                continue
            losses[name] = stages[-1][name]
            losses[name+'_mean'] = torch.stack([stage[name] for stage in stages]).mean()
            for index, stage in enumerate(stages):
                losses[f'{name}_{index}'] = stage[name]
        losses['loss_prediction'] = torch.stack([stage['total'] for stage in stages]).mean()
        losses['total'] = losses['loss_prediction']
        return losses

    def _stage_losses(self, stage, targets, weights):
        return matched_coordinate_loss(stage, targets, self.matcher, weights)

    def _call_loss_diagnostics(self, losses):
        return {}

    def _losses_for_batch(self, batch):
        image, config = batch['image'], self.v12_config
        generator = (None if self.diffusion_noise_seed is None else
                     torch.Generator(device=image.device).manual_seed(self.diffusion_noise_seed))
        targets = build_v12_targets(batch, groups=self.model.num_proposals, generator=generator)
        clean = self.model.codec.encode(targets)
        rollout = (self.training and config.rollout_probability > 0
                   and torch.rand((), device=image.device, generator=generator) < config.rollout_probability)
        if rollout:
            choice = torch.randint(len(config.rollout_call_counts), (), device=image.device, generator=generator).item()
            calls = config.rollout_call_counts[choice]
            schedule = sample_rollout_schedule(batch_size=image.shape[0], calls=calls,
                timesteps=config.timesteps, device=image.device, dtype=image.dtype, generator=generator)
        else:
            times = torch.randint(config.timesteps, (image.shape[0],), device=image.device,
                                  generator=generator).to(image.dtype)/(config.timesteps-1)
            schedule = torch.stack((times, torch.zeros_like(times)), 1)
            calls = 1
        self._rollout_call_count = calls
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        state = noise if rollout else corrupt_latents(clean, noise, schedule[:, 0])
        context = nullcontext() if self.deterministic_seed is None else torch.random.fork_rng(
            devices=[image.get_device()] if image.is_cuda else [])
        with context:
            if self.deterministic_seed is not None:
                torch.manual_seed(self.deterministic_seed)
            return self._trajectory_losses(batch, targets, state, schedule)

    def _trajectory_losses(self, batch, targets, state, schedule):
        encoded = self.model.encode_image(batch['image'], batch.get('padding_mask'))
        memory, losses = None, []
        diagnostics = {'diffusion_time_mean': schedule[:, :-1].mean(),
                       'rollout/calls': state.new_tensor(schedule.shape[1]-1)}
        for index in range(schedule.shape[1]-1):
            incoming = self.model.hypothesis_slots.expand(state.shape[0], -1, -1) if memory is None else memory
            output = self.model.denoise(state, schedule[:, index], encoded,
                hypothesis_memory=memory, collect_diagnostics=True)
            values = self._denoising_losses(output, targets)
            losses.append(values)
            if index + 1 < schedule.shape[1] - 1:
                updated = ddim_update(state, output['pred_coordinates'], schedule[:, index], schedule[:, index+1])
            else:
                updated = output['pred_coordinates']
            memory = output['hypothesis_memory']
            prefix = f'rollout/call{index}'
            diagnostics.update({f'{prefix}/time_mean': schedule[:, index].mean(),
                f'{prefix}/time_min': schedule[:, index].min(), f'{prefix}/time_max': schedule[:, index].max(),
                f'{prefix}/objective': values['total'].detach(),
                f'{prefix}/coordinate_update_rms': tensor_rms(updated-state),
                f'{prefix}/incoming_memory_rms': tensor_rms(incoming),
                f'{prefix}/outgoing_memory_rms': tensor_rms(memory)})
            diagnostics.update({f'{prefix}/{name}': value.detach()
                                for name, value in self._call_loss_diagnostics(values).items()})
            for row in output['latent_diagnostics']:
                diagnostics.update({f'{prefix}/layer{row["layer"]}/{name}':value
                                    for name,value in row.items() if name != 'layer'})
            state = updated
        self._v12_diagnostics = diagnostics
        averaged = {name:torch.stack([loss[name] for loss in losses]).mean() for name in losses[0]}
        auxiliary = self._trajectory_auxiliary_losses(output, targets, batch)
        if auxiliary:
            averaged.update(auxiliary)
            averaged['total'] = averaged['total']+auxiliary['loss_auxiliary']
        return averaged

    def _trajectory_auxiliary_losses(self, output, targets, batch):
        return {}

    def _log_latent_diagnostics(self, prefix, batch_size):
        for name, value in self._v12_diagnostics.items():
            self.log(f'debug/coordinates/{prefix}/{name}', value, batch_size=batch_size,
                     on_step=False, on_epoch=True)

    def _log_objective_gradients(self, losses):
        named = [(name, parameter) for name, parameter in self.named_parameters() if parameter.requires_grad]
        gradients = torch.autograd.grad(losses['loss_prediction'], [p for _,p in named],
                                        retain_graph=True, allow_unused=True)
        self._log_optimization_metrics({f'debug/optimization/objective/prediction/{key}':value
                                       for key,value in gradient_norms(named, gradients).items()})

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        for calls in [1, *self.v12_config.rollout_call_counts]:
            self.log(f'debug/rollout/call_count_{calls}_fraction', float(self._rollout_call_count == calls),
                     batch_size=batch['image'].shape[0], on_step=False, on_epoch=True)
        self.log('debug/rollout/mean_calls', float(self._rollout_call_count),
                 batch_size=batch['image'].shape[0], on_step=False, on_epoch=True)
        self.log('debug/rollout/branch_fraction', float(self._rollout_call_count > 1),
                 batch_size=batch['image'].shape[0], on_step=False, on_epoch=True)
        if (self._rollout_call_count > 1 and self.trainer.is_global_zero
                and getattr(self, '_rollout_logged_epoch', None) != self.current_epoch):
            self._log_optimization_metrics({f'debug/coordinates/train_first_rollout_batch/{name}':value
                                           for name,value in self._v12_diagnostics.items()})
            self._rollout_logged_epoch = self.current_epoch
        return loss

    def _evaluate_quick_grid(self, *args):
        from .quick_eval_v13 import evaluate_quick_grid
        report = evaluate_quick_grid(*args, hypothesis_modes=('persistent', 'reset'))
        report['sampler'] = 'coordinate-logit DDIM eta0'
        return report

    def _quick_architecture(self):
        variant = 'V19-detached-geometry' if self.model.detach_decoder_geometry else 'V19-coordinates'
        if self.model.detach_sampling_references:
            variant = 'V19-sampling-detach'
        return {'training_variant': variant, 'hidden_dim': self.model.hidden_dim,
                'detach_decoder_geometry': self.model.detach_decoder_geometry,
                'detach_sampling_references': self.model.detach_sampling_references,
                'coordinate_values': self.model.codec.width, 'state_space': 'coordinate logits',
                'decoder_representation_tokens': self.model.num_body_points+2,
                'global_tokens': 0, 'learned_codec': False, 'reconstruction_loss': False,
                'shared_components': ['hypothesis_attention'],
                'rollout_probability': self.v12_config.rollout_probability,
                'rollout_call_counts': self.v12_config.rollout_call_counts}

    def on_validation_epoch_end(self):
        PoseFlowV12Pipeline.on_validation_epoch_end(self)
        trainer, config = self.trainer, self.v12_config
        if (not config.quick_eval_enabled or trainer.sanity_checking or not trainer.is_global_zero
                or self._quick_eval_epoch == self.current_epoch):
            return
        from src.data.datasets import PoseDataset
        from .quick_eval_v13 import select_test_images
        from scripts.utils.quick_test_logging import comparison_tag

        data = trainer.datamodule
        root = Path(trainer.checkpoint_callback.dirpath).parent/'quick_test'
        if self._quick_eval_dataset is None:
            self._quick_eval_dataset = PoseDataset.from_path(data.test_ann_path,
                img_root=data.img_dir, num_keypoints=data.num_keypoints)
        candidates = [int(sample['image_id']) for sample in self._quick_eval_dataset.samples]
        ids = select_test_images(candidates, root/'subset.json', count=config.quick_eval_images,
                                seed=config.quick_eval_seed, annotations=data.test_ann_path)
        directory = root/f'epoch{self.current_epoch:03d}'
        report = self._evaluate_quick_grid(self.model, self._quick_test_batches(data, ids),
                                         config, directory, data.test_ann_path, ids)
        report.update(epoch=self.current_epoch, global_step=self.global_step, architecture=self._quick_architecture(),
            matching_strategy=config.matching_strategy, ota_candidate_topk=config.ota_candidate_topk,
            classification_normalization='focal sum / max(positive assignments,1)',
            model=type(self.model).__name__, resize_scale=data.eval_resize_scale,
            resize_max_size=data.eval_max_size, gradient_clip_norm=trainer.gradient_clip_val,
            weights_stage='post-training epoch, before next epoch updates')
        (directory/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        for condition, row in report['conditions'].items():
            for name, value in row['metrics'].items():
                if value >= 0:
                    self.log(comparison_tag(f'quick-test/{condition}/{name}'), value,
                             on_step=False, on_epoch=True, rank_zero_only=True)
        self._quick_eval_epoch = self.current_epoch


class PoseFlowV19PipelineConfig(PoseFlowV12PipelineConfig):
    latent_model_class: ClassVar[type] = PoseFlowV19Model
    latent_pipeline_class: ClassVar[type] = PoseFlowV19Pipeline
    matching_strategy: Literal['ota', 'hungarian'] = 'ota'
    quick_eval_enabled: bool = True
    quick_eval_images: int = Field(default=32, ge=1)
    quick_eval_seed: int = 42
    quick_eval_noise_fractions: list[float] = Field(default_factory=lambda:[.25,.5,1.])
    quick_eval_steps: list[int] = Field(default_factory=lambda:[1,2,3])
    optimization_diagnostics_enabled: bool = True
    rollout_probability: float = Field(default=.5, ge=0, le=1)
    rollout_call_counts: list[int] = Field(default_factory=lambda:[2,3])
    detach_decoder_geometry: bool = False
    detach_sampling_references: bool = False

    def _build_poseflow_model(self):
        model = super()._build_poseflow_model()
        model.detach_decoder_geometry = self.detach_decoder_geometry
        model.detach_sampling_references = self.detach_sampling_references
        return model

    def build(self, **kwargs):
        pipeline = super().build(**kwargs)
        pipeline.model.detach_decoder_geometry = self.detach_decoder_geometry
        pipeline.model.detach_sampling_references = self.detach_sampling_references
        if self.pretrained_checkpoint and kwargs.get('load_pretrained_checkpoint', True):
            from .initialization_v19 import checkpoint_transfer_coverage
            report = checkpoint_transfer_coverage(pipeline.model, self)
            pipeline.model.initialization_report.update(report)
            logging.info('V19 verified checkpoint state coverage %.3f%%; parameters %.3f%%',
                         100*report['verified_checkpoint_fraction'],
                         100*report['verified_checkpoint_parameter_fraction'])
        return pipeline

    @model_validator(mode='after')
    def _validate_coordinate_rollouts(self):
        if self.detach_decoder_geometry and self.detach_sampling_references:
            raise ValueError('Full geometry and sampling-reference detachment are mutually exclusive')
        if (not self.rollout_call_counts or len(set(self.rollout_call_counts)) != len(self.rollout_call_counts)
                or min(self.rollout_call_counts) < 2 or self.timesteps < max(self.rollout_call_counts)+1):
            raise ValueError('V19 needs distinct rollout call counts and sufficient timestep indices')
        if (not self.quick_eval_noise_fractions or any(not 0 < f <= 1 for f in self.quick_eval_noise_fractions)
                or not self.quick_eval_steps or min(self.quick_eval_steps) < 1):
            raise ValueError('Invalid V19 quick evaluation grid')
        return self
