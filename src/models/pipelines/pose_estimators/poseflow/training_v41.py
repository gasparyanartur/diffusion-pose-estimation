import random
from typing import ClassVar

import numpy as np
import torch
from pydantic import Field
from torchvision.ops import box_iou

from ..diffusionregpose.evaluation import box_cxcywh_to_xyxy
from .coordinates_v41 import (
    assign_coordinate_targets, assign_detection_targets, build_coordinate_targets, matched_embedding_loss,
)
from .latent_v35 import box_task_loss
from .latent_v40 import proposal_detection_loss
from .latent_v40_sigreg import latent_distribution_metrics, sigreg_penalty
from .model_v41 import PoseFlowV41Model
from .training_v40 import PoseFlowV40Pipeline, PoseFlowV40PipelineConfig


class PoseFlowV41Pipeline(PoseFlowV40Pipeline):
    def __init__(self, model, *, config, **kwargs):
        super().__init__(model, config=config, **kwargs)
        self.register_buffer('sigreg_rng_state', torch.Generator().manual_seed(config.sigreg_seed).get_state())

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['v41_rng_state'] = dict(python=random.getstate(), numpy=np.random.get_state(),
            torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        state = checkpoint['v41_rng_state']
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'].cpu())
        if state['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])

    def _draw_training_inputs(self, batch, *, training=True, batch_idx=0):
        generator = torch.Generator()
        if training:
            generator.set_state(self.training_rng_state.cpu())
        else:
            generator.manual_seed(self.v26_config.proposal_seed+batch_idx)
        targets = build_coordinate_targets(batch, slots=self.model.num_proposals,
            scale=self.model.scale, generator=generator)
        times = torch.randint(self.model.num_timesteps, (len(batch['image']),), generator=generator)
        noise = torch.randn(len(times), self.model.num_proposals, 4, generator=generator)
        if training:
            self.training_rng_state.copy_(generator.get_state().to(self.training_rng_state.device))
        return targets, times.to(batch['image'].device), noise.to(batch['image'])

    def _regularization(self, clean, targets, *, training, batch_idx):
        codes = clean['pred_embeddings'][targets['positive']]
        config = self.v26_config
        enabled = config.sigreg_weight > 0 and len(codes) >= 2
        self._diagnostics.update({'sigreg/count': codes.new_tensor(len(codes)),
                                 'sigreg/skipped': codes.new_tensor(not enabled)})
        if not enabled:
            raw = codes.float().sum()*0
        else:
            generator = torch.Generator()
            if training:
                generator.set_state(self.sigreg_rng_state.cpu())
            else:
                generator.manual_seed(config.sigreg_seed+batch_idx)
            directions = torch.randn(self.model.hidden_dim, config.sigreg_projections, generator=generator)
            directions /= directions.norm(dim=0)
            raw = sigreg_penalty(codes, directions.to(codes.device),
                knots=config.sigreg_knots, t_max=config.sigreg_t_max)
            if training:
                self.sigreg_rng_state.copy_(generator.get_state().to(self.sigreg_rng_state.device))
        self._diagnostics.update({'sigreg/raw': raw.detach(), 'sigreg/weighted': config.sigreg_weight*raw.detach()})
        return raw

    def _losses_for_batch(self, batch, *, training=True, batch_idx=0):
        targets, times, noise = self._draw_training_inputs(batch, training=training, batch_idx=batch_idx)
        context = self.model.encode_context(batch['image'], batch.get('padding_mask'))
        candidates = self.model.propose(context)
        references = self.model.select_references(candidates)
        proposal_targets, proposal_indices = assign_detection_targets(candidates, targets, self.matcher)
        clean, predicted, noisy, _ = self.model.training_outputs(batch['image'], targets['boxes'],
            source_coordinates=targets['source_coordinates'], timesteps=times, noise=noise,
            references=references, context=context)
        assigned, embeddings, _ = assign_coordinate_targets(predicted, clean['pred_embeddings'], targets, self.matcher)
        weights = self.v26_config.weights
        coefficients = (weights.cls, weights.bbox, weights.giou)
        branches = dict(proposal=proposal_detection_loss(candidates, proposal_targets, weights=coefficients),
            clean=box_task_loss(clean, targets, weights=coefficients),
            denoised=box_task_loss(predicted, assigned, weights=coefficients))
        losses = {f'{branch}/{name}': value for branch, values in branches.items() for name, value in values.items()}
        losses['loss_jepa'] = matched_embedding_loss(predicted['pred_embeddings'], embeddings, assigned['positive'])
        self._record_diagnostics(targets, assigned, clean, predicted, embeddings, candidates,
                                references, proposal_indices, times, noisy)
        losses['regularization/sigreg'] = self._regularization(clean, targets, training=training, batch_idx=batch_idx)
        losses['regularization/total'] = self.v26_config.sigreg_weight*losses['regularization/sigreg']
        losses['total'] = (branches['proposal']['total']+.5*branches['clean']['total']
            +.5*branches['denoised']['total']+self.v26_config.jepa_loss_weight*losses['loss_jepa']
            +losses['regularization/total'])
        self._diagnostics.update({f'{name}_task': values['total'].detach() for name, values in branches.items()})
        self._diagnostics['jepa_mse'] = losses['loss_jepa'].detach()
        return losses

    @torch.no_grad()
    def _record_diagnostics(self, targets, assigned, clean, predicted, embeddings, candidates,
                            references, proposal_indices, times, noisy):
        self._diagnostics = dict(unique_people=targets['unique_people'].float().mean(),
            coordinate_rms=noisy.square().mean().sqrt(),
            coordinate_clamp_fraction=(noisy.abs() > self.model.scale).float().mean(),
            time_mean=times.float().mean(), jepa_target_detached=noisy.new_zeros(()))
        self._diagnostic_counts = {}
        for name, output, mask in [('clean', clean, targets['positive']), ('predicted', predicted, assigned['positive'])]:
            for key, value in latent_distribution_metrics(output['pred_embeddings'][mask]).items():
                self._diagnostics[f'embeddings/{name}/{key}'] = value
        error = (predicted['pred_embeddings']-embeddings).square().mean(-1)
        for index, name in enumerate(('low', 'middle', 'high')):
            mask = (times*3//self.model.num_timesteps == index)[:, None].expand_as(assigned['positive'])
            positive, background = mask & assigned['positive'], mask & ~assigned['positive']
            for key, values, selected in [('jepa', error, positive),
                ('bbox_l1', (predicted['pred_boxes']-assigned['boxes']).abs().mean(-1), positive),
                ('positive_score', predicted['existence_logits'].sigmoid(), positive),
                ('background_score', predicted['existence_logits'].sigmoid(), background)]:
                if selected.any():
                    self._diagnostics[f'{key}/{name}'] = values[selected].mean()
                    self._diagnostic_counts[f'{key}/{name}'] = int(selected.sum())
        positive = assigned['positive']
        if positive.any():
            self._diagnostics['matched_iou'] = box_iou(box_cxcywh_to_xyxy(predicted['pred_boxes'][positive]),
                box_cxcywh_to_xyxy(assigned['boxes'][positive])).diag().mean()
        for name, scores, mask in [
            ('denoised', predicted['existence_logits'].sigmoid(), positive),
            ('proposal', candidates['existence_logits'].sigmoid(), proposal_indices >= 0),
        ]:
            for label, selected in [('positive', mask), ('background', ~mask)]:
                if selected.any():
                    self._diagnostics[f'{name}/{label}_score'] = scores[selected].mean()
        coverage, selected_coverage, matched_ranks = [], [], []
        for image, truth in enumerate(targets['targets']):
            if not len(truth['boxes']):
                continue
            iou = box_iou(box_cxcywh_to_xyxy(candidates['pred_boxes'][image]), box_cxcywh_to_xyxy(truth['boxes']))
            coverage.append((iou.amax(0) >= .5).float())
            selected_coverage.append((iou[references['indices'][image]].amax(0) >= .5).float())
            ranking = candidates['pred_logits'][image].amax(-1).argsort(descending=True, stable=True)
            matched_ranks.append((ranking.argsort()[proposal_indices[image] >= 0]+1).float())
        for name, values in [('coverage_iou50', coverage), ('top50_coverage_iou50', selected_coverage),
                             ('matched_rank', matched_ranks)]:
            if values:
                self._diagnostics[f'proposal/{name}'] = torch.cat(values).mean()

    def _log_v26(self, prefix, losses, batch_size):
        self._log_losses(prefix, losses, batch_size)
        for name, value in self._diagnostics.items():
            self.log(f'debug/v41/{prefix}/{name}', value.detach(),
                batch_size=self._diagnostic_counts.get(name, batch_size), on_step=prefix == 'train', on_epoch=True)


class PoseFlowV41PipelineConfig(PoseFlowV40PipelineConfig):
    pipeline_class: ClassVar[type] = PoseFlowV41Pipeline
    model_class: ClassVar[type] = PoseFlowV41Model
    training_protocol: ClassVar[str] = 'coordinate_box_jepa'
    coordinate_scale: float = Field(default=6., gt=0, allow_inf_nan=False)
    jepa_loss_weight: float = Field(default=10., ge=0, allow_inf_nan=False)
    sigreg_weight: float = Field(default=.05, ge=0, allow_inf_nan=False)
    sigreg_projections: int = Field(default=256, ge=1)
    sigreg_knots: int = Field(default=17, ge=2)
    sigreg_t_max: float = Field(default=3., gt=0, allow_inf_nan=False)
    sigreg_seed: int = Field(default=42, ge=0)

    def _build_poseflow_model(self):
        model = super()._build_poseflow_model()
        model.scale = self.coordinate_scale
        return model
