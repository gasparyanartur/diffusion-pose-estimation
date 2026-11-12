from typing import ClassVar

import torch
from pydantic import Field, model_validator

from .dino_box_teacher import BoxFeatureTargets, FrozenDinoBoxTeacher
from .model_v32 import PoseFlowV32Model
from .training_v25 import matched_grid_loss
from .training_v26 import PoseFlowV26Pipeline
from .training_v31 import PoseFlowV31PipelineConfig


def dino_person_correspondence(targets):
    boxes, matches = [], []
    for index, count in enumerate(targets['unique_people'].tolist()):
        boxes.append(targets['boxes'][index, :count])
        queries = targets['positive'][index].nonzero(as_tuple=False).flatten()
        originals = targets['target_indices'][index, :count].contiguous()
        people = torch.searchsorted(originals, targets['target_indices'][index, queries])
        matches.append((queries, people))
    return boxes, matches


class PoseFlowV32Pipeline(PoseFlowV26Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dino_teacher = None

    def _get_dino_teacher(self):
        if self._dino_teacher is None:
            config = self.v26_config
            self._dino_teacher = FrozenDinoBoxTeacher(config.dino_teacher_model,
                config.dino_teacher_revision, config.dino_teacher_sha256,
                grid_size=config.dino_grid_size)
        return self._dino_teacher

    def _latent_auxiliary_losses(self, batch, targets, clean, denoised, *, denoised_matches=None):
        weight = self.v26_config.dino_loss_weight
        if not weight:
            return {}
        boxes, matches = dino_person_correspondence(targets)
        predictions = [self.model.dino_head(output['pred_latents']) for output in (clean, denoised)]
        if any(len(box) for box in boxes):
            reference = self._get_dino_teacher().targets(batch['image'], batch['img_hw'], boxes)
        else:
            reference = [BoxFeatureTargets(
                predictions[0].new_zeros((0, *predictions[0].shape[-2:])),
                torch.zeros(0, dtype=torch.bool, device=predictions[0].device)) for _ in boxes]
        recovered_matches = matches if denoised_matches is None else denoised_matches
        results = [matched_grid_loss(predictions[0], reference, matches),
                   matched_grid_loss(predictions[1], reference, recovered_matches)]
        losses = {}
        for branch, result in zip(('clean', 'denoised'), results):
            losses[f'loss_dino_{branch}'] = result['loss_h_dino']
            for name, value in result.items():
                metric = 'loss' if name == 'loss_h_dino' else name.removeprefix('h_dino_')
                self._diagnostics[f'dino/{branch}/{metric}'] = value.detach()
        losses['loss_dino'] = weight*.5*(losses['loss_dino_clean']+losses['loss_dino_denoised'])
        self._diagnostics['dino/weighted_loss'] = losses['loss_dino'].detach()
        return losses


class PoseFlowV32PipelineConfig(PoseFlowV31PipelineConfig):
    pipeline_class: ClassVar[type] = PoseFlowV32Pipeline
    dino_loss_weight: float = Field(default=1., ge=0, allow_inf_nan=False)
    dino_grid_size: int = Field(default=7, ge=1)
    dino_teacher_model: str = 'facebook/dinov3-vits16-pretrain-lvd1689m'
    dino_teacher_revision: str = '114c1379950215c8b35dfcd4e90a5c251dde0d32'
    dino_teacher_sha256: str = '4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d'

    @model_validator(mode='after')
    def validate_teacher_identity(self):
        for value, length in ((self.dino_teacher_revision, 40), (self.dino_teacher_sha256, 64)):
            if len(value) != length or any(character not in '0123456789abcdef' for character in value):
                raise ValueError('DINO teacher revision and SHA256 must be pinned lowercase hex digests')
        return self

    def _build_poseflow_model(self):
        names = ('backbone_pretrained', 'timesteps', 'num_proposals', 'num_body_points',
            'hidden_dim', 'num_heads', 'feedforward_dim', 'num_feature_levels',
            'freeze_backbone_stem_and_layer1', 'freeze_backbone')
        return PoseFlowV32Model(**{name:getattr(self, name) for name in names},
            sampling_steps=self.flow_sampling_steps, dino_grid_size=self.dino_grid_size)
