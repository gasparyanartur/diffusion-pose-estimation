import torch

from .rows_v46 import PAD, PERSON, PREDICT


def training_rows(clean, counts):
    # B images, C person capacity, R=max(P)+1 independent rows/token positions.
    batch, capacity, dimension = clean.shape
    if (dimension != 32 or counts.shape != (batch,) or counts.dtype != torch.long
            or (counts < 0).any() or (counts > capacity).any()):
        raise ValueError('Invalid coordinates/counts or person capacity exceeded')
    rows = int(counts.max())+1
    positions = torch.arange(rows, device=clean.device)
    row_valid = positions[None] <= counts[:, None]
    prefix = torch.where(row_valid, positions[None], 0)
    person = positions[None, None] < prefix[..., None]
    predict = positions[None, None] == prefix[..., None]
    types = torch.where(person, PERSON, torch.where(predict, PREDICT, PAD))
    padded = torch.nn.functional.pad(clean, (0, 0, 0, max(0, rows-capacity)))[:, :rows]
    coordinates = torch.where(person[..., None], padded[:, None], 0.)
    return dict(prefix_coordinates=coordinates, token_types=types, padding_mask=types == PAD,
                row_valid=row_valid, predict_indices=prefix,
                birth_rows=positions[None] < counts[:, None],
                eos_targets=positions[None] == counts[:, None])
