from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


CROWDPOSE_FLIP = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13)


def normalize_geometry(
    keypoints_px: Tensor, boxes_cxcywh_px: Tensor, joint_mask: Tensor, image_wh: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return box-centered isotropic joint offsets and image-relative box placement."""
    if (keypoints_px.shape[-2:] != (14, 2)
            or boxes_cxcywh_px.shape != (*keypoints_px.shape[:-2], 4)
            or joint_mask.shape != keypoints_px.shape[:-1]
            or image_wh.shape != (*keypoints_px.shape[:-2], 2)):
        raise ValueError('Expected matching [...,14,2] joints, [...,4] boxes and [...,2] image sizes')
    if not (torch.isfinite(boxes_cxcywh_px).all() and torch.isfinite(image_wh).all()
            and (boxes_cxcywh_px[..., 2:] > 0).all() and (image_wh > 0).all()
            and torch.isfinite(keypoints_px[joint_mask.bool()]).all()):
        raise ValueError('Geometry must be finite where observed, with positive box/image sizes')
    scale = boxes_cxcywh_px[..., 2:].amax(-1, keepdim=True)
    relative = (keypoints_px - boxes_cxcywh_px[..., None, :2]) / scale[..., None]
    relative = torch.where(joint_mask[..., None].bool(), relative, 0.)
    placement = torch.cat((boxes_cxcywh_px[..., :2] / image_wh,
                           (boxes_cxcywh_px[..., 2:] / image_wh).log()), -1)
    return relative, placement


def restore_geometry(relative: Tensor, placement: Tensor, image_wh: Tensor) -> dict[str, Tensor]:
    center = placement[..., :2] * image_wh
    size = placement[..., 2:].exp() * image_wh
    boxes = torch.cat((center, size), -1)
    keypoints = center[..., None, :] + relative * size.amax(-1, keepdim=True)[..., None]
    return dict(keypoints_px=keypoints, boxes_cxcywh_px=boxes)


def flip_geometry(batch: dict[str, Tensor], selected: Tensor) -> dict[str, Tensor]:
    points = batch['keypoints_px'][:, CROWDPOSE_FLIP].clone()
    points[..., 0] = batch['image_wh'][:, None, 0] - points[..., 0]
    boxes = batch['boxes_cxcywh_px'].clone()
    boxes[:, 0] = batch['image_wh'][:, 0] - boxes[:, 0]
    return dict(batch,
                keypoints_px=torch.where(selected[:, None, None], points, batch['keypoints_px']),
                boxes_cxcywh_px=torch.where(selected[:, None], boxes, batch['boxes_cxcywh_px']),
                joint_mask=torch.where(selected[:, None], batch['joint_mask'][:, CROWDPOSE_FLIP],
                                       batch['joint_mask']))


def random_tensor(shape, reference: Tensor, generator: torch.Generator, *, normal=False) -> Tensor:
    draw = torch.randn if normal else torch.rand
    return draw(shape, generator=generator, dtype=reference.dtype).to(reference.device)


def corrupt_geometry(batch: dict[str, Tensor], *, generator: torch.Generator) -> dict[str, Tensor]:
    boxes, points, mask = (batch[name] for name in ('boxes_cxcywh_px', 'keypoints_px', 'joint_mask'))
    count = len(boxes)
    selected = random_tensor((count,), boxes, generator) >= .25
    scale = boxes[:, 2:].amax(-1, keepdim=True)
    joint_std = random_tensor((count, 1, 1), boxes, generator) * .05 * scale[:, None]
    center_std = random_tensor((count, 1), boxes, generator) * .05 * scale
    noisy_points = points + joint_std * random_tensor(points.shape, points, generator, normal=True)
    center = boxes[:, :2] + center_std * random_tensor((count, 2), boxes, generator, normal=True)
    size = boxes[:, 2:] * (.1 * random_tensor((count, 2), boxes, generator, normal=True)).exp()
    drop_probability = random_tensor((count, 1), boxes, generator) * .3
    observed = mask & (random_tensor(mask.shape, boxes, generator) >= drop_probability)
    # Retain a random labeled joint if dropout removed every observation.
    priority = random_tensor(mask.shape, boxes, generator).masked_fill(~mask, -1)
    fallback = torch.nn.functional.one_hot(priority.argmax(-1), 14).bool() & mask
    observed = observed | (fallback & ~observed.any(-1, keepdim=True))
    return dict(batch,
                keypoints_px=torch.where(selected[:, None, None], noisy_points, points),
                boxes_cxcywh_px=torch.where(selected[:, None], torch.cat((center, size), -1), boxes),
                joint_mask=torch.where(selected[:, None], observed, mask))


class PoseGeometryDataset(Dataset):
    """One labeled person per record; only COCO annotation JSON is opened."""

    def __init__(self, annotation_path: str | Path):
        data = json.loads(Path(annotation_path).read_text())
        images = {image['id']: image for image in data['images']}
        self.records = []
        self.image_ids = set()
        for annotation in data['annotations']:
            if annotation.get('iscrowd', 0) or annotation.get('num_keypoints', 0) <= 0:
                continue
            image = images[annotation['image_id']]
            joints = torch.tensor(annotation['keypoints'], dtype=torch.float32).reshape(14, 3)
            box = torch.tensor(annotation['bbox'], dtype=torch.float32)
            box[:2] += box[2:] / 2
            record = dict(keypoints_px=joints[:, :2], joint_mask=joints[:, 2] > 0,
                          boxes_cxcywh_px=box,
                          image_wh=torch.tensor([image['width'], image['height']], dtype=torch.float32),
                          annotation_id=torch.tensor(annotation['id']),
                          image_id=torch.tensor(image['id']))
            if not record['joint_mask'].any():
                raise ValueError(f"Annotation {annotation['id']} claims keypoints but has no labels")
            normalize_geometry(**{name: record[name] for name in (
                'keypoints_px', 'boxes_cxcywh_px', 'joint_mask', 'image_wh')})
            self.records.append(record)
            self.image_ids.add(image['id'])
        if not self.records:
            raise ValueError(f'No usable people in {annotation_path}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]
