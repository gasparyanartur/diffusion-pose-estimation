import math
from typing import ClassVar

import torch
from pydantic import Field, model_validator

from .coordinate_v19 import matched_coordinate_loss
from .evidence_rank_v19 import PoseFlowV19EvidenceRankModel, evidence_ranking_loss
from .losses_v13 import real_person_targets
from .optimization_v13 import gradient_norms
from .training_v19 import PoseFlowV19Pipeline, PoseFlowV19PipelineConfig


class PoseFlowV19EvidenceRankPipeline(PoseFlowV19Pipeline):
    def _losses_for_batch(self, batch):
        if not self.training or self.v12_config.evidence_rank_weight == 0:
            return super()._losses_for_batch(batch)
        # Draw before the inherited fixed attention RNG fork, so corruption varies by batch.
        seed = torch.randint(2**31, (), device=batch['image'].device).item()
        return super()._losses_for_batch(batch | {'_evidence_noise_seed': seed})

    def _trajectory_losses(self, batch, targets, state, schedule):
        seed = batch.get('_evidence_noise_seed')
        if seed is None:
            return super()._trajectory_losses(batch, targets, state, schedule)
        image = batch['image']
        with torch.random.fork_rng(devices=[image.get_device()] if image.is_cuda else []):
            torch.manual_seed(seed)
            return super()._trajectory_losses(batch, targets, state, schedule)

    def _denoising_losses(self, output, targets):
        losses = super()._denoising_losses(output, targets)
        if 'evidence_rank_scores' not in output:
            return losses
        # Stage losses are already independently assigned and equally averaged by the parent.
        losses['loss_evidence_rank'] = losses['loss_evidence_rank_mean']
        losses['loss_evidence_rank_weighted'] = self.v12_config.evidence_rank_weight*losses['loss_evidence_rank']
        losses['total'] = losses['loss_prediction']+losses['loss_evidence_rank_weighted']
        return losses

    def _stage_losses(self, stage, targets, weights):
        matches = self.matcher(stage, real_person_targets(targets))
        losses = matched_coordinate_loss(stage, targets, self.matcher, weights, matches=matches)
        if 'evidence_rank_scores' in stage:
            positive = torch.zeros_like(stage['pred_logits'][..., 0], dtype=torch.bool)
            for index, (candidates, _) in enumerate(matches):
                positive[index, candidates] = True
            ranking, metrics = evidence_ranking_loss(stage['evidence_rank_scores'], positive,
                self.v12_config.evidence_noise_levels, margin=self.v12_config.evidence_rank_margin)
            denominator = positive.sum().clamp_min(1)
            metrics.update({name: (value*positive).sum()/denominator
                            for name, value in stage['evidence_rank_statistics'].items()})
            losses['loss_evidence_rank'] = ranking
            losses.update({f'evidence_rank/{name}': value for name, value in metrics.items()})
        return losses

    def _call_loss_diagnostics(self, losses):
        metrics = {name: value for name, value in losses.items() if name.startswith('loss_evidence_rank')}
        for index in range(len(self.model.layers)):
            for name in losses:
                if name.startswith('evidence_rank/') and name.endswith(f'_{index}'):
                    metric = name.removeprefix('evidence_rank/').removesuffix(f'_{index}')
                    metrics[f'evidence_rank/layer{index}/{metric}'] = losses[name]
        return metrics

    def _log_objective_gradients(self, losses):
        if 'loss_evidence_rank' not in losses:
            return super()._log_objective_gradients(losses)
        named = [(name, parameter) for name, parameter in self.named_parameters() if parameter.requires_grad]
        parameters = [parameter for _, parameter in named]
        pose = torch.autograd.grad(losses['loss_prediction'], parameters, retain_graph=True, allow_unused=True)
        ranking = torch.autograd.grad(losses['loss_evidence_rank_weighted'], parameters,
                                      retain_graph=True, allow_unused=True)
        metrics = {}
        for objective, gradients in [('prediction', pose), ('evidence_rank_weighted', ranking)]:
            metrics.update({f'debug/optimization/objective/{objective}/{name}': value
                            for name, value in gradient_norms(named, gradients).items()})
        shared = [(left.detach().float(), right.detach().float())
                  for left, right in zip(pose, ranking) if left is not None and right is not None]
        if shared:
            dot = sum((left*right).sum() for left, right in shared)
            pose_norm = sum(left.square().sum() for left, _ in shared).sqrt()
            rank_norm = sum(right.square().sum() for _, right in shared).sqrt()
            prefix = 'debug/optimization/evidence_rank_vs_prediction'
            metrics[f'{prefix}/shared_gradient_cosine'] = dot/(pose_norm*rank_norm).clamp_min(1e-12)
            metrics[f'{prefix}/weighted_shared_norm_ratio'] = rank_norm/pose_norm.clamp_min(1e-12)
        self._log_optimization_metrics(metrics)

    def _quick_architecture(self):
        return super()._quick_architecture() | {
            'training_variant': 'V19-evidence-ranking',
            'shared_components': ['hypothesis_attention', 'evidence_confidence'],
            'evidence_noise_levels': self.v12_config.evidence_noise_levels,
            'evidence_rank_margin': self.v12_config.evidence_rank_margin,
            'evidence_rank_weight': self.v12_config.evidence_rank_weight,
            'confidence_semantics': 'ordinal evidence-corruption ranking, OTA positives only',
            'inference_confidence_reranking': False}


class PoseFlowV19EvidenceRankPipelineConfig(PoseFlowV19PipelineConfig):
    latent_model_class: ClassVar[type] = PoseFlowV19EvidenceRankModel
    latent_pipeline_class: ClassVar[type] = PoseFlowV19EvidenceRankPipeline
    detach_decoder_geometry: bool = True
    evidence_noise_levels: list[float] = Field(default_factory=lambda: [0., .25, .5])
    evidence_rank_margin: float = Field(default=1., gt=0, allow_inf_nan=False)
    evidence_rank_weight: float = Field(default=.1, ge=0, allow_inf_nan=False)

    @model_validator(mode='after')
    def _validate_evidence_ranking(self):
        levels = self.evidence_noise_levels
        if (len(levels) != 3 or not all(math.isfinite(level) for level in levels)
                or not levels[0] == 0 < levels[1] < levels[2]):
            raise ValueError('Evidence ranking needs three finite strictly increasing levels starting at zero')
        if not self.detach_decoder_geometry:
            raise ValueError('Evidence ranking uses the full geometry-detach baseline')
        return self

    def _configure_evidence_ranking(self, model):
        model.evidence_noise_levels = list(self.evidence_noise_levels)
        model.evidence_rank_enabled = self.evidence_rank_weight > 0
        return model

    def _build_poseflow_model(self):
        return self._configure_evidence_ranking(super()._build_poseflow_model())

    def build(self, **kwargs):
        pipeline = super().build(**kwargs)
        self._configure_evidence_ranking(pipeline.model)
        return pipeline
