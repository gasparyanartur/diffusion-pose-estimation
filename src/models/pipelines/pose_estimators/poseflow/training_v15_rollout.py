from contextlib import nullcontext
from typing import ClassVar

import torch
from pydantic import Field, model_validator

from .latent_v12 import build_v12_targets, ddim_update, prediction_loss
from .optimization_v13 import reconstruction_weight
from .training_v15 import PoseFlowV15Pipeline, PoseFlowV15PipelineConfig


def sample_rollout_schedule(*, batch_size: int, calls: int, timesteps: int,
                            device: torch.device, dtype: torch.dtype,
                            generator: torch.Generator | None = None) -> torch.Tensor:
    if batch_size < 1 or calls < 2 or timesteps < calls+1:
        raise ValueError('Rollout needs distinct interior timesteps and at least two calls')
    interior = torch.stack([
        (torch.randperm(timesteps-2, device=device, generator=generator)[:calls-1]+1)
        .sort(descending=True).values for _ in range(batch_size)
    ]).to(dtype)/(timesteps-1)
    return torch.cat((torch.ones(batch_size,1,device=device,dtype=dtype), interior,
                      torch.zeros(batch_size,1,device=device,dtype=dtype)), dim=1)


def detached_rms(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().square().mean().sqrt()


class PoseFlowV15RolloutPipeline(PoseFlowV15Pipeline):
    def _losses_for_batch(self, batch):
        config = self.v12_config
        self._rollout_call_count = 1
        if not self.training or config.rollout_probability == 0:
            return super()._losses_for_batch(batch)
        image = batch['image']
        generator = (None if self.diffusion_noise_seed is None else
                     torch.Generator(device=image.device).manual_seed(self.diffusion_noise_seed))
        if config.rollout_probability < 1 and torch.rand((), device=image.device,
                generator=generator).item() >= config.rollout_probability:
            return super()._losses_for_batch(batch)
        choice = torch.randint(len(config.rollout_call_counts), (), device=image.device, generator=generator).item()
        calls = config.rollout_call_counts[choice]
        self._rollout_call_count = calls
        targets = build_v12_targets(batch, groups=self.model.num_proposals, generator=generator)
        clean = self.model.codec.encode(targets)
        schedule = sample_rollout_schedule(batch_size=image.shape[0], calls=calls,
            timesteps=config.timesteps, device=image.device, dtype=image.dtype, generator=generator)
        state = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        context = nullcontext() if self.deterministic_seed is None else torch.random.fork_rng(
            devices=[image.get_device()] if image.is_cuda else [])
        with context:
            if self.deterministic_seed is not None:
                torch.manual_seed(self.deterministic_seed)
            losses, diagnostics = self._rollout_losses(batch, targets, clean, state, schedule)
        self._v12_diagnostics = diagnostics
        return losses

    def _rollout_losses(self, batch, targets, clean, state, schedule):
        weights = self.v12_config.weights.to_dataclass()
        encoded = self.model.encode_image(batch['image'], batch.get('padding_mask'))
        memory, call_losses = None, []
        diagnostics = {'real_people': targets['real'].sum().detach().float()/state.shape[0],
                       'overflow_people': targets['overflow'].sum().detach().float(),
                       'diffusion_time_mean': schedule[:,:-1].mean(),
                       'rollout/calls': state.new_tensor(schedule.shape[1]-1)}
        for role, subset in self.latent_role_slices:
            values = clean[:,:,subset].detach().float()
            diagnostics[f'clean/{role}_rms'] = detached_rms(values)
            diagnostics[f'clean/{role}_variance'] = values.var(unbiased=False)
        for index in range(schedule.shape[1]-1):
            times, next_times = schedule[:,index], schedule[:,index+1]
            incoming_memory = (self.model.hypothesis_slots.expand(state.shape[0],-1,-1)
                               if memory is None else memory)
            output = self.model.denoise(state, times, encoded, hypothesis_memory=memory,
                                        collect_diagnostics=True)
            values = self._denoising_losses(output, targets, clean, weights)
            call_losses.append(values)
            next_state = ddim_update(state, output['pred_latents'], times, next_times)
            memory = output['hypothesis_memory']
            prefix = f'rollout/call{index}'
            diagnostics.update({f'{prefix}/time_mean': times.mean(),
                f'{prefix}/time_min': times.min(), f'{prefix}/time_max': times.max(),
                f'{prefix}/next_time_mean': next_times.mean(),
                f'{prefix}/incoming_memory_rms': detached_rms(incoming_memory),
                f'{prefix}/outgoing_memory_rms': detached_rms(memory),
                f'{prefix}/incoming_memory_slot_variance': incoming_memory.detach().float().var(1,unbiased=False).mean(),
                f'{prefix}/outgoing_memory_slot_variance': memory.detach().float().var(1,unbiased=False).mean(),
                f'{prefix}/latent_rms': detached_rms(state),
                f'{prefix}/latent_update_rms': detached_rms(next_state-state),
                f'{prefix}/objective': (values['loss_latent_mean']+values['loss_prediction']).detach()})
            for name in ['loss_latent_mean','loss_prediction','loss_latent_box_mean','loss_latent_joints_mean']:
                diagnostics[f'{prefix}/{name}'] = values[name].detach()
            for row in output['latent_diagnostics']:
                for name, value in row.items():
                    if name != 'layer':
                        diagnostics[f'{prefix}/layer{row["layer"]}/{name}'] = value
            state = next_state
        losses = {name: torch.stack([values[name] for values in call_losses]).mean()
                  for name in call_losses[0]}
        reconstruction = prediction_loss(self.model.codec.decode(clean), targets,
                                       reconstruct=True, weights=weights)['total']
        config = self.v12_config
        weight = reconstruction_weight(config.reconstruction_weight_start,
            config.reconstruction_weight_end, config.reconstruction_decay_epochs, self.current_epoch)
        losses['loss_reconstruction'] = reconstruction
        losses['reconstruction_weight'] = reconstruction.new_tensor(weight)
        losses['loss_reconstruction_weighted'] = weight*reconstruction
        losses['total'] = losses['loss_latent_mean']+losses['loss_prediction']+losses['loss_reconstruction_weighted']
        return losses, diagnostics

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch,batch_idx)
        for calls in [1,*self.v12_config.rollout_call_counts]:
            self.log(f'debug/rollout/call_count_{calls}_fraction',
                float(self._rollout_call_count == calls), batch_size=batch['image'].shape[0],
                on_step=False, on_epoch=True)
        self.log('debug/rollout/branch_fraction', float(self._rollout_call_count > 1),
            batch_size=batch['image'].shape[0], on_step=False, on_epoch=True)
        self.log('debug/rollout/mean_calls', float(self._rollout_call_count),
            batch_size=batch['image'].shape[0], on_step=False, on_epoch=True)
        if (self._rollout_call_count > 1 and self.trainer.is_global_zero
                and getattr(self,'_rollout_logged_epoch',None) != self.current_epoch):
            self._log_optimization_metrics({f'debug/rollout/train_first_rollout_batch/{name.removeprefix("rollout/")}':value
                for name,value in self._v12_diagnostics.items() if name.startswith('rollout/')})
            self._rollout_logged_epoch = self.current_epoch
        return loss

    def _quick_architecture(self):
        return {**super()._quick_architecture(), 'training_variant':'V15-rollout',
                'rollout_probability':self.v12_config.rollout_probability,
                'rollout_call_counts':self.v12_config.rollout_call_counts,
                'rollout_schedule':'per-image descending distinct timestep indices; Gaussian t=1 start'}


class PoseFlowV15RolloutPipelineConfig(PoseFlowV15PipelineConfig):
    latent_pipeline_class: ClassVar[type] = PoseFlowV15RolloutPipeline
    rollout_probability: float = Field(default=.5, ge=0, le=1, allow_inf_nan=False)
    rollout_call_counts: list[int] = Field(default_factory=lambda:[2,3])

    @model_validator(mode='after')
    def _validate_rollout(self):
        counts = self.rollout_call_counts
        if not counts or len(set(counts)) != len(counts) or min(counts) < 2:
            raise ValueError('Rollout call counts must be distinct integers of at least two')
        if self.timesteps < max(counts)+1:
            raise ValueError('Rollout needs enough distinct interior timestep indices')
        return self
