import hashlib
from typing import ClassVar, Literal

import torch
from pydantic import Field, model_validator
from torchvision.ops import box_iou

from ..diffusionregpose.evaluation import box_cxcywh_to_xyxy
from ..diffusionregpose.training import DiffusionRegPoseMatcher
from .latent_fixed_source import positive_latent_mse
from .latent_v35 import box_task_loss
from .latent_v40 import proposal_detection_loss, select_training_references
from .model_v40 import PoseFlowV40Model
from .training_v26 import PoseFlowV26Pipeline
from .training_v35 import PoseFlowV35Pipeline, PoseFlowV35PipelineConfig
from .training_v39 import fixed_source_noise_metrics


def initialize_v40_weights(model, checkpoint_path, *, encoder_source_layers):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    source = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    source = {name.removeprefix('module.'): value for name, value in source.items()}
    copied, missing = {}, []
    for name, target in model.state_dict().items():
        source_name = model.pretrained_source_name(name, encoder_source_layers)
        if source_name is None:
            continue
        value = source.get(source_name)
        if value is None or value.shape != target.shape:
            missing.append((name, source_name))
        else:
            copied[name] = value
    if missing:
        raise ValueError(f'V40 initialization missing/incompatible tensors: {missing}')
    model.load_state_dict(dict(model.state_dict(), **copied), strict=True)
    digest = hashlib.sha256()
    for name, value in sorted(copied.items()):
        digest.update(name.encode())
        digest.update(value.contiguous().numpy().tobytes())
    return dict(source_checkpoint=str(checkpoint_path), copied_tensors=len(copied), tensor_coverage=1.,
        encoder_source_layers=list(encoder_source_layers), pretrained_sha256=digest.hexdigest(),
        fresh_tensors=len(model.state_dict())-len(copied), decoder_transfer=False,
        source_mapping={name:model.pretrained_source_name(name, encoder_source_layers) for name in copied})


class PoseFlowV40Pipeline(PoseFlowV26Pipeline):
    def __init__(self, model, *, config, **kwargs):
        kwargs['val_ann_file'] = None
        super().__init__(model, config=config, **kwargs)
        self.matcher = DiffusionRegPoseMatcher(weights=config.weights.to_dataclass())
        self._diagnostic_counts = {}

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return batch

    def _draw_training_inputs(self, batch, *, training=True, batch_idx=0):
        generator = torch.Generator()
        if training:
            generator.set_state(self.training_rng_state.cpu())
        else:
            generator.manual_seed(self.v26_config.proposal_seed+batch_idx)
        times = torch.randint(self.model.num_timesteps, (len(batch['image']),), generator=generator)
        noise = torch.randn(len(times), self.model.num_proposals, self.model.hidden_dim, generator=generator)
        if training:
            self.training_rng_state.copy_(generator.get_state().to(self.training_rng_state.device))
        return times.to(batch['image'].device), noise.to(batch['image'])

    def _losses_for_batch(self, batch, *, training=True, batch_idx=0):
        times, noise = self._draw_training_inputs(batch, training=training, batch_idx=batch_idx)
        context = self.model.encode_context(batch['image'], batch.get('padding_mask'))
        candidates = self.model.propose(context)
        references, targets, proposal_targets = select_training_references(candidates, batch,
            self.matcher, slots=self.model.num_proposals)
        latent_target = self._clean_latent_target(batch, targets)
        clean, recovered, _, _ = self.model.training_outputs(batch['image'], targets['input_pixels'],
            timesteps=times, noise=noise, references=references, context=context, latent_target=latent_target,
            **self._training_output_options())
        latent_target = clean['pred_latents'].detach() if latent_target is None else latent_target
        weights = self.v26_config.weights
        coefficients = (weights.cls, weights.bbox, weights.giou)
        proposal_loss = proposal_detection_loss(candidates, proposal_targets, weights=coefficients)
        clean_loss = box_task_loss(clean, targets, weights=coefficients)
        recovered_loss = box_task_loss(recovered, targets, weights=coefficients)
        latent_loss = self._latent_reconstruction_loss(clean, recovered, latent_target, targets)
        losses = {f'{prefix}/{name}': value for prefix, branch in
                  (('proposal', proposal_loss), ('clean', clean_loss), ('denoised', recovered_loss))
                  for name, value in branch.items()}
        losses['loss_latent'] = latent_loss
        losses['total'] = (proposal_loss['total']+.5*clean_loss['total']
            +self.v26_config.denoised_task_loss_weight*recovered_loss['total']
            +self.v26_config.latent_loss_weight*latent_loss)
        self._record_reference_diagnostics(candidates, references, targets, proposal_targets,
                                           recovered, clean, times, losses, latent_target=latent_target)
        self._record_target_diagnostics(clean, latent_target, targets)
        regularization = self._clean_regularization(clean, targets, training=training, batch_idx=batch_idx)
        if regularization is not None:
            losses.update({f'regularization/{name}': value for name, value in regularization.items()})
            losses['total'] = losses['total']+regularization['total']
        return losses

    def _training_output_options(self):
        return {}

    def _latent_reconstruction_loss(self, clean, recovered, latent_target, targets):
        return positive_latent_mse(recovered['pred_latents'], latent_target, targets['positive'])

    def _clean_regularization(self, clean, targets, *, training, batch_idx):
        return None

    def _clean_latent_target(self, batch, targets):
        return None

    def _record_target_diagnostics(self, clean, latent_target, targets):
        pass

    @torch.no_grad()
    def _record_reference_diagnostics(self, candidates, references, targets, proposal_targets,
                                      recovered, clean, times, losses, latent_target=None):
        positive = targets['positive']
        self._diagnostic_counts = {}
        self._diagnostics = dict(unique_people=targets['unique_people'].float().mean(),
            positive_proposals=positive.sum(-1).float().mean(),
            clean_task=losses['clean/total'].detach(), denoised_task=losses['denoised/total'].detach(),
            proposal_task=losses['proposal/total'].detach(), latent_mse=losses['loss_latent'].detach())
        coverage, top50_coverage, matched_iou, ranks = [], [], [], []
        for image in range(len(positive)):
            truth = targets['boxes'][image, positive[image]]
            if not len(truth):
                continue
            predicted = candidates['pred_boxes'][image]
            ranking = candidates['pred_logits'][image].amax(-1).argsort(descending=True, stable=True)
            iou = box_iou(box_cxcywh_to_xyxy(predicted), box_cxcywh_to_xyxy(truth))
            coverage.append((iou.amax(0) >= .5).float())
            top50_coverage.append((iou[ranking[:self.model.num_proposals]].amax(0) >= .5).float())
            assigned = proposal_targets['positive'][image]
            assigned_boxes = proposal_targets['boxes'][image, assigned]
            matched_iou.append(box_iou(box_cxcywh_to_xyxy(predicted[assigned]),
                box_cxcywh_to_xyxy(assigned_boxes)).diag())
            inverse_rank = ranking.argsort()+1
            ranks.append(inverse_rank[assigned].float())
        for name, values in (('proposal/coverage_iou50', coverage),
                             ('proposal/top50_coverage_iou50', top50_coverage),
                             ('proposal/matched_iou', matched_iou), ('proposal/matched_rank', ranks)):
            if values:
                joined = torch.cat(values)
                self._diagnostics[name] = joined.mean()
                self._diagnostic_counts[name] = len(joined)
        latent_target = clean['pred_latents'] if latent_target is None else latent_target
        error = (recovered['pred_latents']-latent_target).square().mean(-1)
        for name, (value, count) in fixed_source_noise_metrics(recovered, targets, error, times,
                                                             self.model.num_timesteps).items():
            self._diagnostics[name] = value
            self._diagnostic_counts[name] = count
        for label, mask in (('positive', proposal_targets['positive']), ('background', ~proposal_targets['positive'])):
            if mask.any():
                name = f'proposal/{label}_score'
                self._diagnostics[name] = candidates['existence_logits'].sigmoid()[mask].mean()
                self._diagnostic_counts[name] = int(mask.sum())

    def _log_v26(self, prefix, losses, batch_size):
        self._log_losses(prefix, losses, batch_size)
        for name, value in self._diagnostics.items():
            self.log(f'debug/v40/{prefix}/{name}', value,
                batch_size=self._diagnostic_counts.get(name, batch_size),
                on_step=prefix == 'train', on_epoch=True)

    on_validation_epoch_end = PoseFlowV35Pipeline.on_validation_epoch_end


class PoseFlowV40PipelineConfig(PoseFlowV35PipelineConfig):
    pipeline_class: ClassVar[type] = PoseFlowV40Pipeline
    model_class: ClassVar[type] = PoseFlowV40Model
    matching_strategy: Literal['hungarian'] = 'hungarian'
    spatial_queries: Literal[False] = False
    spatial_cost_weight: float = Field(default=0., ge=0, le=0)
    spatial_cost_schedule: Literal['constant'] = 'constant'
    reference_candidates: int = Field(default=200, ge=1)
    reference_encoder_source_layers: tuple[int, ...] = (2, 4, 6)

    @model_validator(mode='after')
    def validate_references(self):
        layers = self.reference_encoder_source_layers
        if not layers or len(set(layers)) != len(layers) or any(index < 1 or index > 6 for index in layers):
            raise ValueError('Reference encoder source layers must be unique indices in1..6')
        if self.reference_candidates < self.num_proposals:
            raise ValueError('Reference candidate budget must cover latent slots')
        return self

    def _build_poseflow_model(self):
        names = ('backbone_pretrained', 'timesteps', 'num_proposals', 'num_body_points',
            'hidden_dim', 'num_heads', 'feedforward_dim', 'num_feature_levels',
            'freeze_backbone_stem_and_layer1', 'freeze_backbone', 'grid_rows', 'grid_columns')
        return self.model_class(**{name:getattr(self, name) for name in names},
            reference_candidates=self.reference_candidates,
            reference_encoder_layers=len(self.reference_encoder_source_layers), sampling_steps=self.flow_sampling_steps)

    def build(self, **kwargs):
        model = self._build_poseflow_model()
        if self.pretrained_checkpoint and kwargs.get('load_pretrained_checkpoint', True):
            model.initialization_report = initialize_v40_weights(model, self.pretrained_checkpoint,
                encoder_source_layers=self.reference_encoder_source_layers)
        fields = ('lr', 'lr_backbone', 'weight_decay', 'lr_warmup_epochs', 'lr_min_frac',
            'scheduler_type', 'lr_decay_milestones', 'split_weight_decay',
            'val_ap_every_n_epochs', 'diffusion_noise_seed', 'deterministic_seed')
        return self.pipeline_class(model, config=self, **{name:getattr(self, name) for name in fields},
            max_epochs=kwargs.get('max_epochs'), val_ann_file=kwargs.get('val_ann_file'))
