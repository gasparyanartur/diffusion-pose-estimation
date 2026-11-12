from copy import deepcopy
from typing import Literal

from pydantic import Field, model_validator
import torch

from .quick_eval_v13 import evaluation_context
from .training_v44 import PoseFlowV44Pipeline, PoseFlowV44PipelineConfig


class PoseFlowV44RolloutConfig(PoseFlowV44PipelineConfig):
    state_policy: Literal['forward', 'rollout'] = 'forward'
    trajectory_probability: float = Field(default=.5, ge=0., le=1.)

    @model_validator(mode='after')
    def validate_state_policy(self):
        if self.prediction_type != 'x0':
            raise ValueError('Detached-rollout comparison requires x0 prediction')
        if self.native_config().timesteps != 1000:
            raise ValueError('The matched comparison uses the fixed1000-step K4 schedule')
        return self


class PoseFlowV44RolloutPipeline(PoseFlowV44Pipeline):
    def __init__(self, *, config, model=None, max_epochs=2):
        config = config if isinstance(config, PoseFlowV44RolloutConfig) else PoseFlowV44RolloutConfig(**config)
        super().__init__(config=config, model=model, max_epochs=max_epochs)

    def _diffusion_inputs(self, clean, generator, context, image_wh, *, training):
        forward, times, noise, _ = super()._diffusion_inputs(clean, generator, context, image_wh, training=training)
        if not training:
            return forward, times, noise, {}
        debug = {name: clean.new_tensor(0.) for name in ('trajectory_batch', 'rollout_batch', 'warmup_steps')}
        if torch.rand((), generator=generator).item() >= self.config.trajectory_probability:
            return forward, times, noise, debug
        schedule = (999, 749, 499, 249)
        selected = int(torch.randint(1, len(schedule), (), generator=generator))
        times = torch.full((len(clean),), schedule[selected], dtype=torch.long, device=clean.device)
        debug['trajectory_batch'] = clean.new_tensor(1.)
        if self.config.state_policy == 'forward':
            return self.model.q_sample(clean, times, noise), times, noise, debug
        state = noise.detach()
        with evaluation_context(self.model):
            for time, next_time in zip(schedule[:selected], schedule[1:selected+1]):
                timestep = torch.full_like(times, time)
                prediction = self.model.denoise(state, timestep, context, image_wh)['pred_latents']
                epsilon = self.model.predict_noise_from_start(state, timestep, prediction)
                state = self.model.ddim_step(prediction, epsilon, time, next_time, noise=torch.zeros_like(state))
        debug.update(rollout_batch=clean.new_tensor(1.), warmup_steps=clean.new_tensor(float(selected)))
        return state.detach(), times, noise, debug

    def training_state_policy(self):
        return dict(state_policy=self.config.state_policy, trajectory_probability=self.config.trajectory_probability,
            schedule=[999, 749, 499, 249], warmup='eval_no_grad', version=1)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['v44_training_state_policy'] = self.training_state_policy()

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('v44_training_state_policy') != self.training_state_policy():
            raise ValueError('Checkpoint training-state policy disagrees with configured comparison arm')
        super().on_load_checkpoint(checkpoint)


def warmstart_rollout(record, *, state_policy, max_epochs=2):
    if record.get('v44_prediction_type', 'x0') != 'x0':
        raise ValueError('Detached-rollout warmstart requires an x0 source')
    expected = dict(version=1, posterior_mode='mean', state='latent', fixed_assignment=True)
    if record.get('v44_contract') != expected or record.get('v44_training_state_policy'):
        raise ValueError('Detached-rollout warmstart requires the original V44 contract/source')
    config = PoseFlowV44RolloutConfig(**dict(record['hyper_parameters']['config'],
        state_policy=state_policy, trajectory_probability=.5, lr=2e-5, lr_backbone=1e-5))
    module = PoseFlowV44RolloutPipeline(config=config, max_epochs=max_epochs)
    module.load_state_dict(record['state_dict'], strict=True)
    module.codec_provenance = deepcopy(record['v44_codec_provenance'])
    return module
