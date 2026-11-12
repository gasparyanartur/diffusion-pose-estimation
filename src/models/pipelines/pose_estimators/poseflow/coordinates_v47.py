import torch

from .rows_v46 import build_training_rows, canonical_order


def geometry_coordinates(geometry):
    boxes, sizes = geometry['boxes_cxcywh_px'], geometry['image_wh']
    points = torch.where(geometry['joint_mask'][..., None], geometry['keypoints_px'], boxes[..., None, :2])
    return torch.cat(((points/sizes[..., None, :]).flatten(-2), boxes/torch.cat((sizes, sizes), -1)), -1)


def coordinate_geometry(coordinates, image_wh):
    return dict(keypoints_px=coordinates[..., :28].unflatten(-1, (14, 2))*image_wh[..., None, :],
                boxes_cxcywh_px=coordinates[..., 28:]*torch.cat((image_wh, image_wh), -1))


@torch.no_grad()
def ordered_coordinates(geometry, locations, *, batch_size, capacity=50):
    counts = torch.bincount(locations[:, 0], minlength=batch_size)
    if (counts > capacity).any():
        raise ValueError('Valid person count exceeds sequence capacity')
    clean = geometry_coordinates(geometry)
    coordinates = clean.new_zeros(batch_size, capacity, 32)
    target_indices = torch.full((batch_size, capacity), -1, device=clean.device, dtype=torch.long)
    coordinate_mask = torch.zeros_like(coordinates, dtype=torch.bool)
    for image in range(batch_size):
        indices = canonical_order(geometry, (locations[:, 0] == image).nonzero().flatten())
        coordinates[image, :len(indices)] = clean[indices]
        target_indices[image, :len(indices)] = indices
        coordinate_mask[image, :len(indices), :28] = geometry['joint_mask'][indices].repeat_interleave(2, -1)
        coordinate_mask[image, :len(indices), 28:] = True
    return dict(coordinates=coordinates, counts=counts, target_indices=target_indices, coordinate_mask=coordinate_mask)


def training_rows(clean, counts, *, rows=64, scale=6., generator=None):
    result = build_training_rows(scale*(2*clean-1), counts, rows=rows, generator=generator)
    result['noisy_coordinates'] = result.pop('noisy_latents')
    return result
