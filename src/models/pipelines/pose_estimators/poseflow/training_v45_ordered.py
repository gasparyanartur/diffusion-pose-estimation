from typing import ClassVar, Literal

import torch
from pydantic import Field

from ..diffusionregpose.training import DiffusionRegPoseMatcher, DiffusionRegPosePipelineConfig
from .model_v45_ordered import OrderedTaskPoseModel
from .training_v44 import NATIVE_MODEL_FIELDS
from .training_v45_task_pose import PoseFlowV45TaskPosePipeline, PoseFlowV45TaskPosePipelineConfig


def ordered_people(target):
    boxes = target['boxes'].detach().cpu()
    points = target['keypoints'].detach().cpu()
    joints = points.shape[-1] // 3
    xy = points[:, :2*joints].reshape(-1, joints, 2)
    labeled = points[:, 2*joints:].bool()
    centroids = xy.masked_fill(~labeled[..., None], 0.).sum(1) / labeled.sum(1).clamp_min(1)[:, None]
    if not torch.isfinite(boxes).all() or not torch.isfinite(centroids).all():
        raise ValueError('Canonical order requires finite boxes and labeled joints')
    def key(index):
        cx, cy, width, height = boxes[index].tolist()
        column = min(511, max(0, int(cx*512)))
        return (column, cy, *centroids[index].tolist(), -width*height, index)
    return torch.tensor(sorted(range(len(boxes)), key=key), device=target['boxes'].device, dtype=torch.long)


class OrderedRankTargets(DiffusionRegPoseMatcher):
    @torch.no_grad()
    def forward(self, outputs, targets, *, include_keypoints=True):
        result = []
        for logits, target in zip(outputs['pred_logits'], targets):
            count = len(target['labels'])
            if count > len(logits):
                raise ValueError('Valid person count exceeds ordered slot capacity')
            result.append((torch.arange(count, device=logits.device), ordered_people(target)))
        return result


class OrderedTaskPosePipeline(PoseFlowV45TaskPosePipeline):
    def contract(self):
        return dict(super().contract(), assignment='ordered_rank_512', slots=self.model.num_proposals,
            order='column_cy_labeled_centroid_xy_negative_box_area_stable',
            context='learned_rank_queries_native_box_stage_no_candidate_selection_no_DN')


class OrderedTaskPoseConfig(PoseFlowV45TaskPosePipelineConfig):
    pipeline_class: ClassVar[type] = OrderedTaskPosePipeline
    matching_strategy: Literal['ordered_rank'] = 'ordered_rank'
    target_assignment: Literal['ordered_rank_fixed'] = 'ordered_rank_fixed'
    jepa_weight: float = Field(default=1., ge=0., allow_inf_nan=False)

    def validate_task_pose_contract(self):
        return self

    def build(self, **kwargs):
        values = {name: getattr(self, name) for name in DiffusionRegPosePipelineConfig.model_fields}
        # Native construction loads detector weights; no native matcher is ever executed.
        values['matching_strategy'] = 'hungarian'
        native = DiffusionRegPosePipelineConfig(**values).build(**kwargs)
        model = OrderedTaskPoseModel(**{name: getattr(self, name) for name in NATIVE_MODEL_FIELDS},
            backbone_pretrained=False, separate_expansion_pose_head=self.pretrained_keep_decoder_layers is not None,
            native_state=native.model.state_dict(), latent_dim=self.latent_dim)
        native.criterion.matcher = OrderedRankTargets(num_body_points=self.num_body_points, weights=native.criterion.weights)
        arguments = ('lr', 'lr_backbone', 'weight_decay', 'max_epochs', 'lr_warmup_epochs', 'lr_min_frac',
            'scheduler_type', 'lr_decay_milestones', 'split_weight_decay', 'val_ap_every_n_epochs',
            'diffusion_noise_seed', 'deterministic_seed')
        pipeline = self.pipeline_class(model, native.criterion,
            **{name: getattr(native, name) for name in arguments}, val_ann_file=kwargs.get('val_ann_file'))
        pipeline.v42_config = self
        model.initialization_report = dict(source=self.pretrained_checkpoint,
            transfer='native ED-Pose weights; fresh ordered queries and task codec; no trained warmstart')
        pipeline.save_hyperparameters(dict(training_timestep_policy='uniform', target_assignment='ordered_rank_fixed',
            confidence_mode='image_only', pose_readout_mode='absolute', v45_task_pose=pipeline.contract()))
        return pipeline
