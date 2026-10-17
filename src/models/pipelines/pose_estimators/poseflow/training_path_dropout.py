from typing import ClassVar

import torch
from pydantic import Field

from .path_dropout import PoseFlowPathDropoutModel
from .training_v15_rollout import PoseFlowV15RolloutPipeline, PoseFlowV15RolloutPipelineConfig


class PoseFlowPathDropoutPipeline(PoseFlowV15RolloutPipeline):
    def _losses_for_batch(self, batch):
        probabilities = (self.v12_config.joint_image_path_dropout_probability,
                         self.v12_config.joint_self_attention_path_dropout_probability)
        if not self.training or not any(0 < value < 1 for value in probabilities):
            return super()._losses_for_batch(batch)
        device = batch['image'].device
        # Attention uses a fixed forked RNG; message masks get fresh batch-local randomness.
        seed = torch.randint(2**31, (), device=device).item()
        generator = torch.Generator(device=device).manual_seed(seed)
        self.model.set_path_dropout_generator(generator)
        try:
            return super()._losses_for_batch(batch)
        finally:
            self.model.set_path_dropout_generator(None)

    def _quick_architecture(self):
        return {**super()._quick_architecture(), 'training_variant': 'V15-rollout-path-dropout',
                'joint_image_path_dropout_probability': self.v12_config.joint_image_path_dropout_probability,
                'joint_self_attention_path_dropout_probability': self.v12_config.joint_self_attention_path_dropout_probability,
                'path_dropout': 'independent joint message masks per layer/call; survival scaling; eval disabled'}


class PoseFlowPathDropoutPipelineConfig(PoseFlowV15RolloutPipelineConfig):
    latent_model_class: ClassVar[type] = PoseFlowPathDropoutModel
    latent_pipeline_class: ClassVar[type] = PoseFlowPathDropoutPipeline
    joint_image_path_dropout_probability: float = Field(default=.25, ge=0, le=1, allow_inf_nan=False)
    joint_self_attention_path_dropout_probability: float = Field(default=.25, ge=0, le=1, allow_inf_nan=False)

    def _configure_model_dropout(self, model):
        model.configure_path_dropout(self.joint_image_path_dropout_probability,
                                     self.joint_self_attention_path_dropout_probability)
        return model

    def _build_poseflow_model(self):
        return self._configure_model_dropout(super()._build_poseflow_model())

    def build(self, **kwargs):
        pipeline = super().build(**kwargs)
        self._configure_model_dropout(pipeline.model)
        return pipeline
