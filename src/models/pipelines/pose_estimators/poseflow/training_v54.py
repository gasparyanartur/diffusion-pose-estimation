from copy import deepcopy

from pydantic import Field
import torch

from .prefix_augmentation_v54 import augment_prefix_rows, bank_digest
from .training_v53 import PoseFlowV53Pipeline, PoseFlowV53PipelineConfig


class PoseFlowV54PipelineConfig(PoseFlowV53PipelineConfig):
    residual_bank_capacity: int = Field(default=2048, ge=1)
    prefix_clean_probability: float = Field(default=.25, ge=0., le=1.)

    def build(self, *, max_epochs=80, **kwargs):
        original = super().build(max_epochs=max_epochs, **kwargs)
        return PoseFlowV54Pipeline(config=self, model=original.model, max_epochs=max_epochs)


class PoseFlowV54Pipeline(PoseFlowV53Pipeline):
    def __init__(self, *, config, model=None, max_epochs=80):
        config = config if isinstance(config, PoseFlowV54PipelineConfig) else PoseFlowV54PipelineConfig(**config)
        super().__init__(config=config, model=model, max_epochs=max_epochs)
        self.model.architecture = 'V54'
        self.register_buffer('prefix_residuals', torch.zeros(config.residual_bank_capacity, 32))
        self.register_buffer('prefix_joint_mask', torch.zeros(config.residual_bank_capacity, 14, dtype=torch.bool))
        self.register_buffer('prefix_bank_count', torch.tensor(0, dtype=torch.long))
        self.prefix_bank_provenance = None

    def set_residual_bank(self, residuals, joint_mask, provenance):
        if (self.prefix_bank_count or residuals.ndim != 2 or not 0 < len(residuals) <= self.config.residual_bank_capacity
                or residuals.shape[-1] != 32 or joint_mask.shape != (len(residuals), 14)
                or joint_mask.dtype != torch.bool or not torch.isfinite(residuals).all() or provenance.get('split') != 'train'):
            raise ValueError('Invalid or already frozen V54 TRAIN residual bank')
        if provenance.get('residual_sha256', bank_digest(residuals, joint_mask)) != bank_digest(residuals, joint_mask):
            raise ValueError('Corrupted V54 input residual bank')
        self.prefix_residuals[:len(residuals)].copy_(residuals.to(self.prefix_residuals))
        self.prefix_joint_mask[:len(residuals)].copy_(joint_mask.to(self.prefix_joint_mask))
        self.prefix_bank_count.fill_(len(residuals))
        self.prefix_bank_provenance = dict(deepcopy(provenance), count=len(residuals),
            sha256=bank_digest(self.prefix_residuals, self.prefix_joint_mask))

    def matched_rows(self, batch, context, *, generator=None):
        rows, sources, geometry, targets = super().matched_rows(batch, context, generator=generator)
        if self.training:
            if not self.prefix_bank_count:
                raise RuntimeError('V54 training requires frozen TRAIN residual bank')
            count = int(self.prefix_bank_count)
            rows, statistics = augment_prefix_rows(rows, sources, self.prefix_residuals[:count],
                self.prefix_joint_mask[:count], batch['img_hw'].flip(-1),
                clean_probability=self.config.prefix_clean_probability, generator=generator)
            rows['augmentation_statistics'] = statistics
        return rows, sources, geometry, targets

    def _losses_for_batch(self, batch, *, training=True, batch_idx=0):
        if training != self.training:
            raise ValueError('V54 training/evaluation mode must match augmentation policy')
        return super()._losses_for_batch(batch, training=training, batch_idx=batch_idx)

    def checkpoint_contract(self):
        contract = super().checkpoint_contract()
        contract.update(architecture='V54', state='fixed_completed_coordinates;TRAIN_only_empirical_prefix_augmentation;no_hidden_recurrence',
            noise=dict(space='image_normalized_geometry_before_shared_coordinate_encoder',
                distribution='frozen_TRAIN_V53_paired_full_person_residual_vectors',
                parameterization='box_relative_center;log_WH;box_relative_joint_displacement',
                clean_probability=self.config.prefix_clean_probability, scale=[.5, 1.],
                bounds=dict(center_box_units=.15, log_WH_ratio=1.25, joint_box_units=.25, nearest_earlier_center_fraction=.25),
                lifetime='once_per_person_per_forward;shared_across_teacher_rows',
                masks='preserve_source_visibility;bank_unknown_joints_zero_local_error',
                bank_capacity=self.config.residual_bank_capacity, inference='none'))
        return contract

    def on_save_checkpoint(self, checkpoint):
        if not self.prefix_bank_count:
            raise ValueError('Cannot save V54 without frozen residual bank')
        super().on_save_checkpoint(checkpoint)
        for key in ('contract', 'initialization', 'calibration', 'global_rng'):
            checkpoint[f'v54_{key}'] = checkpoint.pop(f'v53_{key}')
        checkpoint['v54_prefix_bank'] = deepcopy(self.prefix_bank_provenance)

    def on_load_checkpoint(self, checkpoint):
        state, provenance = checkpoint['state_dict'], checkpoint.get('v54_prefix_bank')
        count = int(state.get('prefix_bank_count', 0))
        residuals, masks = state.get('prefix_residuals'), state.get('prefix_joint_mask')
        if (not provenance or provenance.get('split') != 'train' or not 0 < count <= self.config.residual_bank_capacity
                or count != provenance.get('count') or residuals is None or masks is None
                or residuals.shape != (self.config.residual_bank_capacity, 32)
                or masks.shape != (self.config.residual_bank_capacity, 14) or masks.dtype != torch.bool
                or not torch.isfinite(residuals).all() or bank_digest(residuals, masks) != provenance.get('sha256')):
            raise ValueError('Invalid or corrupted V54 residual bank')
        translated = dict(checkpoint)
        for key in ('contract', 'initialization', 'calibration', 'global_rng'):
            translated[f'v53_{key}'] = checkpoint.get(f'v54_{key}')
        super().on_load_checkpoint(translated)
        self.prefix_bank_provenance = deepcopy(provenance)
