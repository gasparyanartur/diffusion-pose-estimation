"""V11 center/size flow, box-only coarse supervision, and token refinement."""

from contextlib import nullcontext

import torch
from pydantic import model_validator

from .model_v11 import PoseFlowV11Model
from .training import (
    PoseFlowV5Pipeline, PoseFlowV5PipelineConfig,
    build_poseflow_v5_targets, flow_sample_poseflow_v5_states,
)


def build_poseflow_v11_targets(batch, model, **kwargs):
    return build_poseflow_v5_targets(batch, model, box_format='center_size', **kwargs)


def flow_sample_poseflow_v11_states(model, images, **kwargs):
    return flow_sample_poseflow_v5_states(
        model, images, box_format='center_size',
        forward_kwargs={'apply_joint_conditioning': model.sampling_joint_conditioning},
        **kwargs,
    )


class PoseFlowV11Pipeline(PoseFlowV5Pipeline):
    def _losses_for_batch(self, batch, *, collect_codebook_attention=False):
        del collect_codebook_attention
        model = self.model
        person_state, _, times, targets = build_poseflow_v11_targets(
            batch, model, num_queries=model.num_queries,
            num_body_points=model.transformer.decoder.num_body_points,
            padding_policy=self.flow_padding_policy, noise_seed=self.diffusion_noise_seed,
        )
        rng_context = nullcontext() if self.deterministic_seed is None else torch.random.fork_rng(
            devices=[batch['image'].device] if batch['image'].is_cuda else [],
        )
        with rng_context:
            if self.deterministic_seed is not None:
                torch.manual_seed(self.deterministic_seed)
            output = model(
                batch['image'], person_state[..., 4:], times, targets,
                batch.get('padding_mask'), person_state=person_state, collect_diagnostics=True,
            )
        self._latest_v4_diagnostics = model.last_v4_diagnostics()
        return self.criterion(output, targets)

    def on_train_epoch_start(self):
        for group in self.trainer.optimizers[0].param_groups:
            self.log(f"debug/lr/{group['schedule_group']}", group['lr'],
                     on_step=False, on_epoch=True, sync_dist=True)


class PoseFlowV11PipelineConfig(PoseFlowV5PipelineConfig):
    corner_order_loss_weight: float = 0.0
    center_loss_factor: float = 0.0
    flow_sampling_steps: int = 3
    sampling_joint_conditioning: bool = False

    @model_validator(mode='after')
    def _validate_v11_configuration(self):
        if self.corner_order_loss_weight or self.center_loss_factor:
            raise ValueError('V11 has no corner or auxiliary center losses')
        if self.decoder_loss_mode != 'all':
            raise ValueError('V11 supervises all refinement layers')
        if self.flow_padding_policy != 'source_prior':
            raise ValueError('V11 requires positive-size source_prior padding')
        if self.freeze_backbone_and_transformer:
            raise ValueError('V11 retrains the structured decoder; use freeze_backbone to freeze only ResNet')
        return self

    def _build_poseflow_model(self):
        return PoseFlowV11Model(
            **self._poseflow_model_kwargs(),
            sampling_joint_conditioning=self.sampling_joint_conditioning,
        )

    def _pipeline_class(self):
        return PoseFlowV11Pipeline

    def _additional_initialization_missing_keys(self, model):
        prefixes = ('role_embeddings.', 'joint_conditioning.', 'dn_box_conditioning.',
                    'point_head.', 'size_head.', 'existence_head.')
        return {
            name for name in model.state_dict()
            if name.startswith(prefixes) or '.exist_attn.' in name or '.exist_norm.' in name
        }
