from contextlib import nullcontext
from typing import ClassVar

import torch
from pydantic import Field

from .quick_eval_v13 import evaluation_context
from .training import build_poseflow_targets
from .training_v22 import PoseFlowV22Pipeline, PoseFlowV22PipelineConfig


def sample_two_step_schedule(batch_size, generator):
    times = torch.rand(batch_size, 2, generator=generator).sort(-1).values
    use_second = bool(torch.rand((), generator=generator) < .5)
    return times[:, 0], times[:, 1], use_second


class PoseFlowV23Pipeline(PoseFlowV22Pipeline):
    diagnostic_namespace = 'v23'

    def __init__(self, *args, rollout_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        generator = torch.Generator().manual_seed(rollout_seed)
        self.register_buffer('two_step_rng_state', generator.get_state())
        self._two_step_diagnostics = {}

    def _sample_schedule(self, batch_size, device, dtype):
        generator = torch.Generator().set_state(self.two_step_rng_state.cpu())
        first, second, use_second = sample_two_step_schedule(batch_size, generator)
        self.two_step_rng_state.copy_(generator.get_state().to(self.two_step_rng_state.device))
        return first.to(device=device, dtype=dtype), second.to(device=device, dtype=dtype), use_second

    def _losses_for_batch(self, batch, *, collect_codebook_attention=False):
        if not self.training:
            return super()._losses_for_batch(batch, collect_codebook_attention=collect_codebook_attention)
        image = batch['image']
        first, second, use_second = self._sample_schedule(image.shape[0], image.device, image.dtype)
        decoder_state, source, _, targets = build_poseflow_targets(batch, self.model,
            num_proposals=self.model.num_proposals,
            num_body_points=self.model.transformer.decoder.num_body_points,
            padding_policy=self.flow_padding_policy, noise_seed=self.diffusion_noise_seed, flow_times=first)
        state = (decoder_state*2-1)*self.model.scale
        memory = None
        rng_context = nullcontext() if self.deterministic_seed is None else torch.random.fork_rng(
            devices=[image.device] if image.is_cuda else [])
        with rng_context:
            if self.deterministic_seed is not None:
                torch.manual_seed(self.deterministic_seed)
            if use_second:
                with evaluation_context(self.model):
                    warmup = self.model(image, state, first, padding_mask=batch.get('padding_mask'))
                    endpoint = (warmup['pred_keypoints'][..., :source.shape[-1]]*2-1)*self.model.scale
                    state = (state+(second-first)[:, None, None]*(endpoint-source)).detach()
                    memory = warmup['hypothesis_memory'].detach()
            supervised_time = second if use_second else first
            output = self.model(image, state, supervised_time, targets, batch.get('padding_mask'),
                                hypothesis_memory=memory)
        self._latest_codebook_diagnostics = self.model.last_codebook_diagnostics()
        self._two_step_diagnostics = {
            'second_step_fraction': float(use_second), 'supervised_time': supervised_time.detach().mean(),
            'first_time': first.detach().mean(), 'second_time': second.detach().mean()}
        return self.criterion(output, targets)

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        for name, value in self._two_step_diagnostics.items():
            self.log(f'debug/v23/{name}', value, batch_size=batch['image'].shape[0],
                     on_step=True, on_epoch=True)
        return loss


class PoseFlowV23PipelineConfig(PoseFlowV22PipelineConfig):
    training_protocol: ClassVar[str] = 'random_first_or_second_detached_warmup'
    pipeline_class: ClassVar[type] = PoseFlowV23Pipeline
    rollout_seed: int = Field(default=42, ge=0)

    def _extra_pipeline_arguments(self):
        return {'rollout_seed': self.rollout_seed}
