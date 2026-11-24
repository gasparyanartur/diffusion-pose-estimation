import hashlib
import math

import torch
from torch.nn import functional as F


def bank_digest(residuals, joint_mask):
    checksum = hashlib.sha256()
    for value in (residuals, joint_mask):
        checksum.update(value.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def geometric_residuals(prediction, target, joint_mask):
    target_size = target[..., 30:].clamp_min(1e-6)
    predicted_size = prediction[..., 30:].clamp_min(1e-6)
    translation = (prediction[..., 28:30]-target[..., 28:30])/target_size
    log_scale = (predicted_size/target_size).log()
    predicted_local = (prediction[..., :28].unflatten(-1, (14, 2))-prediction[..., None, 28:30])/predicted_size[..., None, :]
    target_local = (target[..., :28].unflatten(-1, (14, 2))-target[..., None, 28:30])/target_size[..., None, :]
    joint_error = torch.where(joint_mask[..., None], predicted_local-target_local, 0.)
    return torch.cat((joint_error.flatten(-2), translation, log_scale), -1)


def apply_residuals(clean, residuals, joint_mask):
    size = clean[..., 30:].clamp_min(1e-6)
    center = clean[..., 28:30]+size*residuals[..., 28:30]
    new_size = size*residuals[..., 30:].exp()
    local = (clean[..., :28].unflatten(-1, (14, 2))-clean[..., None, 28:30])/size[..., None, :]
    points = center[..., None, :]+new_size[..., None, :]*(local+residuals[..., :28].unflatten(-1, (14, 2)))
    points = torch.where(joint_mask[..., None], points, center[..., None, :])
    return torch.cat((points.flatten(-2), center, new_size), -1)


def bounded_residuals(residuals):
    joints = residuals[..., :28].unflatten(-1, (14, 2))
    joints = joints*(.25/joints.norm(dim=-1, keepdim=True).clamp_min(.25)).clamp_max(1.)
    translation = residuals[..., 28:30]
    translation = translation*(.15/translation.norm(dim=-1, keepdim=True).clamp_min(.15)).clamp_max(1.)
    return torch.cat((joints.flatten(-2), translation, residuals[..., 30:].clamp(-math.log(1.25), math.log(1.25))), -1)


def limit_center_displacement(residuals, clean, counts, image_wh):
    capacity = clean.shape[1]
    valid = torch.arange(capacity, device=clean.device)[None] < counts[:, None]
    centers = clean[..., 28:30]*image_wh[:, None]
    distances = torch.cdist(centers, centers)
    excluded = torch.ones(capacity, capacity, device=clean.device, dtype=torch.bool).triu()[None] | ~valid[:, None]
    nearest = distances.masked_fill(excluded, float('inf')).amin(-1)
    movement = (residuals[..., 28:30]*clean[..., 30:]*image_wh[:, None]).norm(dim=-1)
    ratio = (.25*nearest/movement.clamp_min(1e-12)).clamp_max(1.)
    result = residuals.clone()
    result[..., 28:30] *= ratio[..., None]
    return result


@torch.no_grad()
def augment_prefix_rows(rows, sources, residuals, bank_joint_mask, image_wh, *, clean_probability=.25, generator=None):
    if (residuals.ndim != 2 or residuals.shape[-1] != 32 or not len(residuals)
            or bank_joint_mask.shape != (len(residuals), 14) or bank_joint_mask.dtype != torch.bool
            or not torch.isfinite(residuals).all() or not 0 <= clean_probability <= 1):
        raise ValueError('Invalid V54 empirical residual bank or clean probability')
    clean = sources['coordinates']  # <B,C,32>; one draw per person, shared across R teacher rows.
    shape, device = clean.shape[:2], clean.device
    random_device = device if generator is None else generator.device
    indices = torch.randint(len(residuals), shape, generator=generator, device=random_device).to(device)
    selected = torch.rand(shape, generator=generator, device=random_device).to(device) >= clean_probability
    selected &= torch.arange(shape[1], device=device)[None] < sources['counts'][:, None]
    scale = .5+.5*torch.rand(shape, generator=generator, device=random_device).to(device)
    sample = residuals[indices].to(clean)*scale[..., None]
    observed = bank_joint_mask[indices].to(device)
    sample[..., :28] *= observed.repeat_interleave(2, -1)
    bounded = limit_center_displacement(bounded_residuals(sample), clean, sources['counts'], image_wh)
    target_mask = sources['coordinate_mask'][..., :28].unflatten(-1, (14, 2)).all(-1)
    corrupted = apply_residuals(clean, bounded, target_mask)
    corrupted = corrupted.clamp(0., 1.)
    corrupted[..., 30:] = corrupted[..., 30:].clamp_min(1e-6)
    corrupted = torch.where(selected[..., None], corrupted, clean)
    queries = rows['token_types'].shape[-1]
    padded = F.pad(corrupted, (0, 0, 0, max(0, queries-clean.shape[1])))[:, :queries]
    result = dict(rows, prefix_coordinates=torch.where((rows['token_types'] == 1)[..., None], padded[:, None], 0.))
    people = sources['counts'].sum().clamp_min(1)
    clipped = ((sample-bounded).abs().amax(-1) > 1e-6) & selected
    applied = (corrupted-clean).square().mean(-1)*selected
    statistics = dict(prefix_corruption_fraction=selected.sum()/people,
                      prefix_corruption_clipped_fraction=clipped.sum()/selected.sum().clamp_min(1),
                      prefix_corruption_coordinate_mse=applied.sum()/selected.sum().clamp_min(1))
    return result, statistics
