import torch
from torch import Tensor
from torch.nn import functional as F

from .rows_v46 import PAD, PERSON, PREDICT, draw_noise


def token_age_sigmas(ages: Tensor, *, dtype=torch.float32) -> Tensor:
    """Gaussian coefficients: pure noise at birth, then .1, .05, .025, ..."""
    if ages.is_floating_point() or (ages < 0).any():
        raise ValueError('Token ages must be nonnegative integers')
    values = ages.to(dtype=dtype)
    return torch.where(ages == 0, 1., .1*torch.pow(2., 1-values))


def _validate_inputs(clean_coordinates, counts, coordinate_mask, target_indices, row_count):
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 2:
        raise ValueError('At least two rows are required for birth and EOS/refinement')
    if (clean_coordinates.ndim != 3 or clean_coordinates.shape[-1] != 32
            or not clean_coordinates.is_floating_point() or not clean_coordinates.shape[0]):
        raise ValueError('Coordinates must be a nonempty floating batch with shape [B,C,32]')
    batch_size, capacity, _ = clean_coordinates.shape
    if (counts.shape != (batch_size,) or counts.dtype != torch.long
            or counts.device != clean_coordinates.device or (counts < 0).any()):
        raise ValueError('Person counts must be nonnegative int64 values on the coordinates device')
    if (counts > capacity).any() or (counts >= row_count).any():
        raise ValueError('Person counts exceed input capacity or leave no EOS/refinement row')
    if coordinate_mask is not None and (
            coordinate_mask.shape != clean_coordinates.shape or coordinate_mask.dtype != torch.bool
            or coordinate_mask.device != clean_coordinates.device):
        raise ValueError('Coordinate mask must be boolean and match coordinates shape/device')
    if target_indices is not None and (
            target_indices.shape != (batch_size, capacity) or target_indices.dtype != torch.long
            or target_indices.device != clean_coordinates.device):
        raise ValueError('Target indices must be int64 with shape [B,C] on the coordinates device')
    valid = torch.arange(capacity, device=counts.device)[None] < counts[:, None]
    if (~torch.isfinite(clean_coordinates).all(-1) & valid).any():
        raise ValueError('Valid coordinates must be finite; fill missing input joints before corruption')


def _shuffle_sources(clean_coordinates, counts, coordinate_mask, target_indices, generator):
    batch_size, capacity, dimension = clean_coordinates.shape
    device = clean_coordinates.device
    order = torch.arange(capacity, device=device).repeat(batch_size, 1)
    random_device = device if generator is None else generator.device
    for image, count in enumerate(counts.tolist()):
        order[image, :count] = torch.randperm(count, device=random_device, generator=generator).to(device)
    valid = torch.arange(capacity, device=device)[None] < counts[:, None]
    coordinates = clean_coordinates.gather(1, order[..., None].expand(-1, -1, dimension))
    if coordinate_mask is None:
        coordinate_mask = torch.ones_like(clean_coordinates, dtype=torch.bool)
    masks = coordinate_mask.gather(1, order[..., None].expand(-1, -1, dimension))
    if target_indices is None:
        target_indices = torch.arange(capacity, device=device).expand(batch_size, -1)
    identities = target_indices.gather(1, order)
    return dict(
        coordinates=torch.where(valid[..., None], coordinates, 0.),
        coordinate_mask=masks & valid[..., None],
        target_indices=torch.where(valid, identities, -1),
        source_indices=torch.where(valid, order, -1),
        counts=counts.clone(),
    )


def build_training_rows(
    clean_coordinates: Tensor,
    counts: Tensor,
    *,
    coordinate_mask: Tensor | None = None,
    target_indices: Tensor | None = None,
    row_count: int = 50,
    generator: torch.Generator | None = None,
) -> dict:
    """Build independent teacher-forced rows; clean targets stay in `sources`.

    Noise is applied in caller-supplied coordinate units without rescaling.
    Missing input joints must already be filled; their supervision mask is kept.
    PREDICT slots always contain pure noise, including EOS rows. Only earlier
    people reach zero input noise in the final row, which is reserved for cleanup.
    """
    _validate_inputs(clean_coordinates, counts, coordinate_mask, target_indices, row_count)
    sources = _shuffle_sources(clean_coordinates, counts, coordinate_mask, target_indices, generator)
    # B = images, R = rows, Q = token positions (R), D = coordinate channels (32).
    batch_size, capacity, dimension = clean_coordinates.shape
    positions = torch.arange(row_count, device=clean_coordinates.device)
    row_indices = positions[None, :, None]  # <1,R,1>
    ranks = positions[None, None, :]  # <1,1,Q>
    prefix = torch.minimum(positions[None], counts[:, None])  # <B,R>
    is_person = ranks < prefix[..., None]
    is_predict = ranks == prefix[..., None]
    token_types = torch.where(is_person, PERSON, torch.where(is_predict, PREDICT, PAD))
    padding_mask = token_types == PAD
    birth_mask = is_predict & (row_indices < counts[:, None, None])
    ages = (row_indices-ranks).clamp_min(0)
    age_sigmas = token_age_sigmas(ages, dtype=clean_coordinates.dtype)
    age_sigmas = torch.where(row_indices == row_count-1, 0., age_sigmas)
    noise_sigmas = torch.where(is_person, age_sigmas, torch.where(is_predict, 1., 0.))

    noise = draw_noise((batch_size, row_count, row_count, dimension), clean_coordinates, generator)
    noise = noise.masked_fill(padding_mask[..., None], 0.)
    padded_sources = F.pad(sources['coordinates'][:, :row_count], (0, 0, 0, max(0, row_count-capacity)))
    sigma = noise_sigmas[..., None]
    noisy_coordinates = (1-sigma.square()).sqrt()*padded_sources[:, None]+sigma*noise
    noisy_coordinates = noisy_coordinates.masked_fill(padding_mask[..., None], 0.)
    return dict(
        sources=sources,
        noisy_coordinates=noisy_coordinates,
        noise=noise,
        noise_sigmas=noise_sigmas,
        token_types=token_types,
        padding_mask=padding_mask,
        predict_indices=prefix,
        eos_targets=positions[None] >= counts[:, None],
        supervised=is_person | birth_mask,
        birth_mask=birth_mask,
        refinement_mask=is_person,
    )
