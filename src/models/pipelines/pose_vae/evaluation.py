from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor

from src.data.skeleton import CROWDPOSE_EDGE_LIST


def box_iou(prediction: Tensor, target: Tensor) -> Tensor:
    pred_low, pred_high = prediction[:, :2] - prediction[:, 2:] / 2, prediction[:, :2] + prediction[:, 2:] / 2
    true_low, true_high = target[:, :2] - target[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2
    overlap = (torch.minimum(pred_high, true_high) - torch.maximum(pred_low, true_low)).clamp_min(0).prod(-1)
    union = prediction[:, 2:].prod(-1) + target[:, 2:].prod(-1) - overlap
    return overlap / union.clamp_min(1e-8)


def geometry_metrics(prediction: dict[str, Tensor], target: dict[str, Tensor],
                     observed: Tensor) -> dict[str, tuple[Tensor, Tensor]]:
    boxes, mask = target['boxes_cxcywh_px'], target['joint_mask']
    scale = boxes[:, 2:].amax(-1)
    distances_px = torch.linalg.vector_norm(prediction['keypoints_px'] - target['keypoints_px'], dim=-1)
    distances = torch.where(mask, distances_px / scale[:, None], 0.)
    metrics = {}

    def add_masked(name, values, valid):
        metrics[name] = (torch.where(valid, values, 0.).sum(), valid.sum())

    add_masked('joint_error', distances, mask)
    add_masked('joint_error_px', distances_px, mask)
    add_masked('masked_joint_error', distances, mask & ~observed)
    add_masked('observed_joint_error', distances, mask & observed)
    metrics['bbox_iou'] = (box_iou(prediction['boxes_cxcywh_px'], boxes).sum(), boxes.new_tensor(len(boxes)))
    relative_scale = scale / target['image_wh'].amax(-1)
    for name, people in (('small', relative_scale < .1), ('medium', (relative_scale >= .1) & (relative_scale < .3)),
                         ('large', relative_scale >= .3)):
        add_masked(f'size_{name}_joint_error', distances, mask & people[:, None])
    for joint in range(14):
        add_masked(f'joint_{joint}_error', distances[:, joint], mask[:, joint])
    return metrics


def export_examples(directory: Path, examples: list[dict], prior: dict[str, Tensor],
                    metrics: dict[str, float]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(dict(examples=examples, prior=prior), directory / 'geometry.pt')
    (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
    render_grid(directory / 'reconstruction.png', examples)
    prior_examples = [dict(prediction={name: value[index] for name, value in prior.items()},
                           image_wh=example['image_wh']) for index, example in enumerate(examples)]
    render_grid(directory / 'prior_samples.png', prior_examples)


def render_grid(path: Path, examples: list[dict]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows = (len(examples) + 3) // 4
    figure, axes = plt.subplots(rows, 4, figsize=(12, 3*rows), squeeze=False)
    for axis, example in zip(axes.flat, examples):
        for name, color in (('target', '0.65'), ('input', '#d68a10'), ('prediction', '#176ab4')):
            if name not in example:
                continue
            geometry = example[name]
            points = geometry['keypoints_px']
            mask = geometry.get('joint_mask', torch.ones(14, dtype=torch.bool))
            for first, second in CROWDPOSE_EDGE_LIST:
                if mask[first] and mask[second]:
                    axis.plot(points[[first, second], 0], points[[first, second], 1], color=color, lw=1)
            axis.scatter(points[mask, 0], points[mask, 1], s=7, color=color)
            box = geometry['boxes_cxcywh_px']
            axis.add_patch(Rectangle((box[0]-box[2]/2, box[1]-box[3]/2), box[2], box[3],
                                     fill=False, edgecolor=color, lw=1))
        width, height = example['image_wh'].tolist()
        axis.set_xlim(-.1*width, 1.1*width)
        axis.set_ylim(1.1*height, -.1*height)
        axis.set_aspect('equal')
        axis.set_title(str(example.get('annotation_id', 'prior')), fontsize=9)
    for axis in axes.flat:
        axis.axis('off')
    figure.suptitle('Clean: gray | Corrupted: orange | Posterior mean: blue' if 'target' in examples[0]
                   else 'Fixed N(0,I) latent draws; image dimensions are metadata')
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
