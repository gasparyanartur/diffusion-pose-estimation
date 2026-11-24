import torch
from torch.nn import functional as F

from .codec_v43 import geometry_from_batch
from .coordinates_v47 import ordered_coordinates
from .rows_token_age import _shuffle_sources, _validate_inputs
from .rows_v48 import training_rows as clean_prefix_rows


def training_rows(sources, *, generator=None):
    _validate_inputs(sources['coordinates'], sources['counts'], sources['coordinate_mask'],
                     sources['target_indices'], sources['coordinates'].shape[1]+1)
    shuffled = _shuffle_sources(sources['coordinates'], sources['counts'], sources['coordinate_mask'],
                               sources['target_indices'], generator)
    return clean_prefix_rows(shuffled['coordinates'], shuffled['counts']), shuffled


def matched_training_rows(model, matcher, batch, context, *, generator=None):
    from .latent_v43 import geometry_targets
    geometry, targets = geometry_targets(batch)
    _, locations = geometry_from_batch(batch)
    sources = ordered_coordinates(geometry, locations, batch_size=len(batch['image']), capacity=model.capacity)
    rows, sources = training_rows(sources, generator=generator)
    matches = matcher(dict(pred_boxes=context['boxes'], pred_logits=context['pred_logits']), targets, include_keypoints=False)
    mapping = torch.full((len(geometry['image_wh']),), -1, dtype=torch.long, device=batch['image'].device)
    for target, (candidates, people) in zip(targets, matches):
        if len(people) != len(target['labels']) or len(candidates.unique()) != len(candidates):
            raise ValueError('V53 selected proposals must cover every GT person once')
        mapping[target['geometry_indices'][people]] = candidates
    ids = torch.full_like(sources['target_indices'], -1)
    valid = sources['target_indices'] >= 0
    ids[valid] = mapping[sources['target_indices'][valid]]
    queries = rows['token_types'].shape[-1]
    supplied = F.pad(ids, (0, max(0, queries-ids.shape[-1])))[:, :queries]
    rows['proposal_indices'] = supplied[:, None].expand_as(rows['token_types']).clone()
    active = (rows['token_types'] == 1) | ((rows['token_types'] == 2) & rows['birth_rows'][..., None])
    rows['proposal_indices'].masked_fill_(~active, -1)
    rows['selection_targets'] = rows['proposal_indices'].gather(-1, rows['predict_indices'][..., None]).squeeze(-1)
    sources['proposal_indices'] = ids
    return rows, sources, geometry, targets
