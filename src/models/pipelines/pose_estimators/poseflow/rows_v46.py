import math

import torch


PAD, PERSON, PREDICT = 0, 1, 2


def canonical_order(geometry, indices, *, columns=512):
    boxes = geometry['boxes_cxcywh_px'][indices]
    widths = geometry['image_wh'][indices, 0]
    column = (columns*boxes[:, 0]/widths).floor().clamp(0, columns-1)
    points = torch.where(geometry['joint_mask'][indices, :, None],
                         geometry['keypoints_px'][indices], 0.).flatten(1)
    keys = torch.cat((column[:, None], boxes[:, [1, 0, 2, 3]], points), -1).detach().cpu().tolist()
    order = sorted(range(len(keys)), key=keys.__getitem__)
    return indices[torch.tensor(order, device=indices.device, dtype=torch.long)]


@torch.no_grad()
def ordered_sources(codec, geometry, locations, *, batch_size, capacity=50):
    counts = torch.bincount(locations[:, 0], minlength=batch_size)
    if (counts > capacity).any():
        raise ValueError('Valid person count exceeds sequence capacity')
    mean, _ = codec.encode_posterior(geometry)
    codes = codec.standardize(mean)
    latents = codes.new_zeros(batch_size, capacity, codec.latent_dim)
    target_indices = torch.full((batch_size, capacity), -1, dtype=torch.long, device=codes.device)
    for image in range(batch_size):
        indices = canonical_order(geometry, (locations[:, 0] == image).nonzero().flatten())
        latents[image, :len(indices)] = codes[indices]
        target_indices[image, :len(indices)] = indices
    return dict(latents=latents, counts=counts, target_indices=target_indices)


def row_alpha_bar(row_times):
    times = row_times.double()
    value = ((times+.008)/1.008*(math.pi/2)).cos().square()/math.cos(.008/1.008*(math.pi/2))**2
    value = torch.where(times == 1, 0., torch.where(times == 0, 1., value))
    return value.clamp(0, 1).to(row_times)


def draw_noise(shape, reference, generator=None):
    device = reference.device if generator is None else generator.device
    return torch.randn(shape, device=device, dtype=reference.dtype, generator=generator).to(reference.device)


def build_training_rows(clean, counts, *, rows=64, generator=None):
    # B images, R independent rows, Q padded token positions, D latent channels.
    batch, capacity, dimension = clean.shape
    if counts.shape != (batch,) or (counts < 0).any() or (counts > capacity).any():
        raise ValueError('Person counts exceed sequence capacity')
    if rows <= capacity:
        raise ValueError('Training rows must include an EOS row beyond capacity')
    positions = torch.arange(rows, device=clean.device)
    row_indices = positions[None, :, None]
    ranks = positions[None, None, :]
    prefix = torch.minimum(positions[None], counts[:, None])
    is_person = ranks < prefix[..., None]
    is_predict = ranks == prefix[..., None]
    types = torch.where(is_person, PERSON, torch.where(is_predict, PREDICT, PAD))
    times = torch.linspace(1., 0., rows, device=clean.device, dtype=clean.dtype)
    alpha = row_alpha_bar(times)[None, :, None, None]
    noise = draw_noise((batch, rows, rows, dimension), clean, generator)
    padded_clean = torch.nn.functional.pad(clean, (0, 0, 0, rows-capacity))
    noisy = alpha.sqrt()*padded_clean[:, None]+(1-alpha).sqrt()*noise
    noisy = torch.where(is_person[..., None], noisy, 0.)
    supervised = is_person | (is_predict & (row_indices < counts[:, None, None]))
    birth = is_predict & supervised
    return dict(noisy_latents=noisy, token_types=types, padding_mask=types == PAD,
                row_times=times, eos_targets=positions[None] >= counts[:, None],
                predict_indices=prefix, supervised=supervised, birth_mask=birth,
                refinement_mask=is_person, noise=noise)
