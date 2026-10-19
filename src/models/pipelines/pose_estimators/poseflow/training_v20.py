from typing import ClassVar

import torch
from pydantic import Field, model_validator

from .guidance_v20 import projected_pose_loss, score_distillation_loss
from .losses_v13 import real_person_targets
from .model_v20 import PoseFlowV20Model
from .training_dense_read_v19 import PoseFlowV19DenseReadPipeline, PoseFlowV19DenseReadPipelineConfig


class PoseFlowV20Pipeline(PoseFlowV19DenseReadPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        generator = torch.Generator().manual_seed(self.v12_config.pose3d_noise_seed)
        self.register_buffer('guidance_rng_state', generator.get_state())
        self._teacher = None

    def _guidance_noise(self, poses):
        generator = torch.Generator().set_state(self.guidance_rng_state.cpu())
        config = self.v12_config
        times = torch.rand(poses.shape[0], generator=generator)
        times = config.pose3d_noise_min+(config.pose3d_noise_max-config.pose3d_noise_min)*times
        noise = torch.randn(poses.shape, generator=generator)
        self.guidance_rng_state.copy_(generator.get_state().to(self.guidance_rng_state.device))
        return times.to(poses.device), noise.to(poses.device)

    def _get_teacher(self):
        if self._teacher is None:
            from .gfpose.teacher import FrozenGFPose
            self._teacher = FrozenGFPose(self.v12_config.gfpose_checkpoint,
                                        self.v12_config.gfpose_sha256)
        return self._teacher

    def _matched_pose_targets(self, output, targets):
        matches = self.matcher(output, real_person_targets(targets))
        poses = self.model.read_pose3d(output['hypothesis_memory'])
        selected, joints, labeled = [], [], []
        for index, (candidates, people) in enumerate(matches):
            slots = targets['real'][index].nonzero(as_tuple=False).flatten()[people]
            selected.append(poses[index, candidates])
            joints.append(targets['joints'][index, slots])
            labeled.append(targets['labeled'][index, slots])
        return torch.cat(selected), torch.cat(joints), torch.cat(labeled)

    def _trajectory_auxiliary_losses(self, output, targets, batch):
        config = self.v12_config
        if not self.training or not (config.pose3d_prior_weight or config.pose3d_projection_weight):
            return {}
        poses, joints, labeled = self._matched_pose_targets(output, targets)
        zero = poses.sum()*0
        projection, prior, metrics = zero, zero, {}
        if poses.shape[0]:
            if config.pose3d_projection_weight:
                projection = projected_pose_loss(poses, joints, labeled)
            if config.pose3d_prior_weight:
                times, noise = self._guidance_noise(poses)
                prior, metrics = score_distillation_loss(poses, self._get_teacher(), times, noise)
            metrics['pose_rms'] = poses.detach().float().square().mean().sqrt()
        weighted = config.pose3d_projection_weight*projection+config.pose3d_prior_weight*prior
        losses = {'loss_pose3d_projection': projection, 'loss_pose3d_prior': prior,
                  'loss_pose3d_weighted': weighted, 'loss_auxiliary': weighted}
        metrics.update({name: value.detach() for name, value in losses.items()})
        metrics['positive_assignments'] = poses.new_tensor(poses.shape[0])
        self._v12_diagnostics.update({f'pose3d/{name}': value for name, value in metrics.items()})
        return losses

    def _quick_architecture(self):
        return super()._quick_architecture() | {
            'training_variant': 'V20-3D-guidance', 'pose3d_common_joints': 12,
            'pose3d_teacher_frame': 'GFPose image-plane XY plus scaled depth',
            'pose3d_guidance_stage': 'final layer of final training call',
            'pose3d_prior_weight': self.v12_config.pose3d_prior_weight,
            'pose3d_projection_weight': self.v12_config.pose3d_projection_weight,
            'gfpose_sha256': self.v12_config.gfpose_sha256,
            'teacher_at_inference': False}


class PoseFlowV20PipelineConfig(PoseFlowV19DenseReadPipelineConfig):
    latent_model_class: ClassVar[type] = PoseFlowV20Model
    latent_pipeline_class: ClassVar[type] = PoseFlowV20Pipeline
    pose3d_projection_weight: float = Field(default=1., ge=0, allow_inf_nan=False)
    pose3d_prior_weight: float = Field(default=1., ge=0, allow_inf_nan=False)
    pose3d_noise_min: float = Field(default=.1, gt=0, le=1, allow_inf_nan=False)
    pose3d_noise_max: float = Field(default=.6, gt=0, le=1, allow_inf_nan=False)
    pose3d_noise_seed: int = Field(default=42, ge=0)
    gfpose_checkpoint: str = 'checkpoints/gfpose/best_model.pth'
    gfpose_sha256: str = '742a37757f3ccc0c96989462576112d4dcac0e3eeb83d071e8012df06726cbc6'

    @model_validator(mode='after')
    def _validate_guidance(self):
        if self.pose3d_noise_min >= self.pose3d_noise_max:
            raise ValueError('Teacher noise minimum must be below maximum')
        if len(self.gfpose_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.gfpose_sha256):
            raise ValueError('GFPose requires a lowercase SHA256 checkpoint digest')
        return self
