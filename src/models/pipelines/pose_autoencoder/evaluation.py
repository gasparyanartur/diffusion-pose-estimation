from __future__ import annotations

import builtins
import contextlib
import io
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .model import aligned_box_metrics, joint_exponents


NOISE_LEVELS = (0., .01, .025, .05, .1)
EVALUATION_SEEDS = (1042, 2042, 3042)


def clean_gate(metrics, threshold=97.5):
    return metrics['pose_AP'] >= threshold and metrics['bbox_AP'] >= threshold


def robust_gain(control, candidate):
    return all(np.mean([row[name] for row in candidate]) > np.mean([row[name] for row in control])
               for name in ('pose_AP', 'bbox_AP'))


def perturb_latents(latents: Tensor, std: Tensor, fraction: float, *, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(latents.shape, generator=generator, dtype=latents.dtype).to(latents.device)
    return latents + fraction*std.to(latents)*noise


def paired_metrics(prediction: dict[str, Tensor], target: dict[str, Tensor]) -> dict[str, float]:
    mask, boxes = target['joint_mask'], target['boxes_cxcywh_px']
    exponent = joint_exponents(prediction, target)
    oks = ((-exponent).exp()*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    iou, _ = aligned_box_metrics(prediction['boxes_cxcywh_px'], boxes)
    distances = (prediction['keypoints_px']-target['keypoints_px']).norm(dim=-1)
    errors = dict(joint_error=(distances/boxes[:, 2:].amax(-1)[:, None])[mask],
                  joint_error_px=distances[mask],
                  box_center_error=((prediction['boxes_cxcywh_px'][:, :2]-boxes[:, :2])/boxes[:, 2:]).norm(dim=-1),
                  box_logsize_error=(prediction['boxes_cxcywh_px'][:, 2:]/boxes[:, 2:]).log().abs().mean(-1))
    metrics = dict(paired_OKS_mean=float(oks.mean()), paired_OKS_p05=float(oks.quantile(.05)),
                   paired_IoU_mean=float(iou.mean()), paired_IoU_p05=float(iou.quantile(.05)))
    for name, values in errors.items():
        metrics.update({f'{name}_median': float(values.quantile(.5)), f'{name}_p95': float(values.quantile(.95))})
    return metrics


@contextlib.contextmanager
def native_compatibility():
    missing_unicode, missing_float = not hasattr(builtins, 'unicode'), not hasattr(np, 'float')
    if missing_unicode:
        builtins.unicode = str
    if missing_float:
        np.float = float
    try:
        yield
    finally:
        if missing_unicode:
            del builtins.unicode
        if missing_float:
            del np.float


class CodecAPEvaluator:
    """Known-person codec AP: constant scores, annotation order, no pose NMS."""

    def __init__(self, annotation_path: str | Path, annotation_ids=None):
        from crowdposetools.coco import COCO
        self.annotation_path = str(annotation_path)
        with contextlib.redirect_stdout(io.StringIO()):
            self.ground = COCO(self.annotation_path)
        selected = set(annotation_ids) if annotation_ids is not None else None
        annotations = [record for record in self.ground.dataset['annotations']
                       if not record.get('iscrowd', 0) and record.get('num_keypoints', 0) > 0
                       and (selected is None or record['id'] in selected)]
        self.annotation_ids = [record['id'] for record in annotations]
        if not annotations or (selected is not None and selected != set(self.annotation_ids)):
            raise ValueError('Codec evaluation requires nonempty valid annotation IDs')
        for record in annotations:
            record['area'] = record['bbox'][2]*record['bbox'][3]
        self.ground.dataset['annotations'] = annotations
        with contextlib.redirect_stdout(io.StringIO()):
            self.ground.createIndex()

    def evaluate(self, prediction: dict[str, Tensor], target: dict[str, Tensor]) -> dict[str, float]:
        from crowdposetools.cocoeval import COCOeval
        if target['annotation_id'].tolist() != self.annotation_ids:
            raise ValueError('Codec annotations must cover the evaluator in deterministic annotation order')
        points = prediction['keypoints_px'].detach().cpu()
        boxes = prediction['boxes_cxcywh_px'].detach().cpu()
        if not torch.isfinite(points).all() or not torch.isfinite(boxes).all() or not (boxes[:, 2:] > 0).all():
            raise ValueError('Codec predictions must be finite with positive box sizes')
        keypoints = torch.cat((points, torch.ones_like(points[..., :1])), -1).flatten(1).tolist()
        xywh = torch.cat((boxes[:, :2]-boxes[:, 2:]/2, boxes[:, 2:]), -1).tolist()
        image_ids = target['image_id'].tolist()
        metrics = {}
        for kind, prefix in (('keypoints', 'pose'), ('bbox', 'bbox')):
            records = [dict(image_id=int(image), category_id=1, score=1., keypoints=pose, bbox=box)
                       for image, pose, box in zip(image_ids, keypoints, xywh)]
            with native_compatibility(), contextlib.redirect_stdout(io.StringIO()):
                detected = self.ground.loadRes(records)
                evaluator = COCOeval(self.ground, detected, kind)
                evaluator.evaluate()
                evaluator.accumulate()
            # loadRes appends the record list to anno_file; avoid retaining every epoch's predictions.
            self.ground.anno_file = [self.annotation_path]
            precision = evaluator.eval['precision'][:, :, :, 0, -1]
            metrics[f'{prefix}_AP'] = float(precision[precision >= 0].mean()*100)
            for threshold in (.75, .95):
                index = np.flatnonzero(np.isclose(evaluator.params.iouThrs, threshold))[0]
                values = precision[index]
                metrics[f'{prefix}_AP{int(threshold*100)}'] = float(values[values >= 0].mean()*100)
        cpu_target = {name: value.detach().cpu() for name, value in target.items()}
        metrics.update(paired_metrics(dict(keypoints_px=points, boxes_cxcywh_px=boxes), cpu_target))
        return metrics


def export_reconstructions(path: Path, prediction, target, *, title=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from src.data.skeleton import CROWDPOSE_EDGE_LIST

    indices = torch.linspace(0, len(target['image_wh'])-1, min(16, len(target['image_wh']))).long()
    figure, axes = plt.subplots((len(indices)+3)//4, 4, figsize=(12, 3*((len(indices)+3)//4)), squeeze=False)
    for axis, index in zip(axes.flat, indices):
        mask = target['joint_mask'][index]
        for geometry, color in ((target, '0.65'), (prediction, '#176ab4')):
            points, box = geometry['keypoints_px'][index], geometry['boxes_cxcywh_px'][index]
            for first, second in CROWDPOSE_EDGE_LIST:
                if mask[first] and mask[second]:
                    axis.plot(points[[first, second], 0], points[[first, second], 1], color=color, lw=1)
            axis.scatter(points[mask, 0], points[mask, 1], s=7, color=color)
            axis.add_patch(Rectangle((box[0]-box[2]/2, box[1]-box[3]/2), box[2], box[3],
                                     fill=False, edgecolor=color))
        width, height = target['image_wh'][index].tolist()
        axis.set_xlim(-.1*width, 1.1*width)
        axis.set_ylim(1.1*height, -.1*height)
        axis.set_aspect('equal')
        axis.set_title(str(int(target['annotation_id'][index])), fontsize=9)
    for axis in axes.flat:
        axis.axis('off')
    figure.suptitle(title or 'Ground truth: gray | Deterministic reconstruction: blue | Labeled joints only')
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
