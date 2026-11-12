from contextlib import nullcontext
from typing import ClassVar

import torch
from torch import Tensor
from torch.nn import functional as F
from pydantic import Field, model_validator

from .dino_box_teacher import BoxFeatureTargets, FrozenDinoBoxTeacher
from .model_v25 import PoseFlowV25Model
from .training import build_poseflow_targets
from .training_v24 import PoseFlowV24Pipeline, PoseFlowV24PipelineConfig


def matched_grid_loss(
    predictions: Tensor, targets: list[BoxFeatureTargets],
    matches: list[tuple[Tensor, Tensor]],
) -> dict[str, Tensor]:
    if predictions.ndim != 4 or len(targets) != predictions.shape[0] or len(matches) != len(targets):
        raise ValueError('DINO predictions, targets and matches must have the same batch size')
    selected, expected, wrong = [], [], []
    wrong_predictions, correct_for_wrong = [], []
    skipped = 0
    for image_index, ((queries, people), target) in enumerate(zip(matches, targets)):
        if target.features.shape[1:] != predictions.shape[-2:]:
            raise ValueError('DINO target grid dimensions must match the predictions')
        valid = target.valid[people]
        skipped += int((~valid).sum())
        queries, people = queries[valid], people[valid]
        selected.append(predictions[image_index, queries])
        expected.append(target.features.detach()[people])
        available = target.valid.nonzero(as_tuple=False).flatten()
        if len(available) > 1 and len(people):
            position = torch.searchsorted(available, people)
            donors = available[(position + 1) % len(available)]
            wrong_predictions.append(predictions[image_index, queries])
            wrong.append(target.features.detach()[donors])
            correct_for_wrong.append(target.features.detach()[people])
    zero = predictions.float().sum() * 0
    count = sum(len(value) for value in selected)
    normalizer = predictions.new_tensor(float(count), dtype=torch.float32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(normalizer)
        normalizer = normalizer.clamp_min(1) / torch.distributed.get_world_size()
    else:
        normalizer = normalizer.clamp_min(1)
    if count:
        prediction = torch.cat(selected).float()
        reference = torch.cat(expected).float()
        cosine = F.cosine_similarity(prediction, reference, dim=-1, eps=1e-6).mean(-1)
        loss = (1 - cosine).sum() / normalizer
        similarity = cosine.mean().detach()
        variance = prediction.detach().var(dim=0, unbiased=False).mean()
    else:
        loss, similarity, variance = zero, zero.detach(), zero.detach()
    gap = zero.detach()
    if wrong:
        inputs = torch.cat(wrong_predictions).float()
        wrong_cosine = F.cosine_similarity(inputs, torch.cat(wrong).float(), dim=-1, eps=1e-6).mean()
        correct_cosine = F.cosine_similarity(inputs, torch.cat(correct_for_wrong).float(),
                                             dim=-1, eps=1e-6).mean()
        gap = (correct_cosine - wrong_cosine).detach()
    return {
        'loss_h_dino': loss,
        'h_dino_positive_assignments': predictions.new_tensor(float(count)),
        'h_dino_skipped_assignments': predictions.new_tensor(float(skipped)),
        'h_dino_cosine': similarity,
        'h_dino_prediction_variance': variance,
        'h_dino_correct_minus_wrong_cosine': gap,
    }


class PoseFlowV25Pipeline(PoseFlowV24Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dino_teacher = None

    def _get_dino_teacher(self):
        if self._dino_teacher is None:
            config = self.quick_eval_config
            self._dino_teacher = FrozenDinoBoxTeacher(
                config.dino_teacher_model, config.dino_teacher_revision,
                config.dino_teacher_sha256, grid_size=config.h_dino_grid_size)
        return self._dino_teacher

    def _losses_for_batch(self, batch, *, collect_codebook_attention=False):
        config = self.quick_eval_config
        if not config.h_dino_weight:
            return super()._losses_for_batch(batch,
                collect_codebook_attention=collect_codebook_attention)
        state, _, times, targets = build_poseflow_targets(
            batch, self.model, num_proposals=self.model.num_proposals,
            num_body_points=self.model.transformer.decoder.num_body_points,
            padding_policy=self.flow_padding_policy, noise_seed=self.diffusion_noise_seed)
        devices = [batch['image'].device] if batch['image'].is_cuda else []
        context = (nullcontext() if self.deterministic_seed is None
                   else torch.random.fork_rng(devices=devices))
        with context:
            if self.deterministic_seed is not None:
                torch.manual_seed(self.deterministic_seed)
            output = self.model(batch['image'], (state * 2 - 1) * self.model.scale,
                                times, targets, batch.get('padding_mask'))
        with torch.no_grad():
            matches = self.criterion.matcher(output, targets)
        losses = self.criterion(output, targets, final_matches=matches)
        predictions = self.model.h_decoder(output['h_final'])
        boxes = [target['boxes'] for target in targets]
        if any(len(queries) for queries, _ in matches):
            target_grids = self._get_dino_teacher().targets(batch['image'], batch['img_hw'], boxes)
        else:
            target_grids = [BoxFeatureTargets(
                predictions.new_zeros((len(boxes_i), *predictions.shape[-2:])),
                torch.zeros(len(boxes_i), device=predictions.device, dtype=torch.bool))
                for boxes_i in boxes]
        auxiliary = matched_grid_loss(predictions, target_grids, matches)
        auxiliary['loss_h_dino_weighted'] = config.h_dino_weight * auxiliary['loss_h_dino']
        losses['total'] = losses['total'] + auxiliary['loss_h_dino_weighted']
        return losses | auxiliary


class PoseFlowV25PipelineConfig(PoseFlowV24PipelineConfig):
    training_protocol: ClassVar[str] = 'matched_single_call_h_dino_grid'
    model_class: ClassVar[type] = PoseFlowV25Model
    pipeline_class: ClassVar[type] = PoseFlowV25Pipeline
    new_model_prefixes: ClassVar[tuple[str, ...]] = ('h_branch.', 'h_decoder.')
    h_enabled: bool = True
    h_dino_grid_size: int = Field(default=7, ge=1)
    h_dino_weight: float = Field(default=1., ge=0, allow_inf_nan=False)
    dino_teacher_model: str = 'facebook/dinov3-vits16-pretrain-lvd1689m'
    dino_teacher_revision: str = '114c1379950215c8b35dfcd4e90a5c251dde0d32'
    dino_teacher_sha256: str = '4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d'

    @model_validator(mode='after')
    def validate_dino_supervision(self):
        if not self.h_enabled:
            raise ValueError('V25 DINO supervision requires h')
        for value, length in ((self.dino_teacher_revision, 40), (self.dino_teacher_sha256, 64)):
            if len(value) != length or any(character not in '0123456789abcdef' for character in value):
                raise ValueError('DINO teacher revision and SHA256 must be pinned lowercase hex digests')
        return self

    def _extra_model_kwargs(self):
        return {'h_grid_size': self.h_dino_grid_size}
