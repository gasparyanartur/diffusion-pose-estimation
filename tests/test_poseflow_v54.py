from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.prefix_augmentation_v54 import (
    geometric_residuals, apply_residuals, augment_prefix_rows, bank_digest,
)
from src.models.pipelines.pose_estimators.poseflow.rows_v53 import training_rows


def people():
    coordinates = torch.full((1, 6, 32), .5)
    coordinates[..., 30:] = .2
    coordinates[0, :3, 28] = torch.tensor([.2, .5, .8])
    coordinates[..., :28] = coordinates[..., 28:30].repeat_interleave(14, dim=-1).reshape(1, 6, 2, 14).transpose(-1, -2).flatten(-2)
    mask = torch.ones_like(coordinates, dtype=torch.bool)
    mask[0, 1, :2] = False
    return dict(coordinates=coordinates, coordinate_mask=mask, counts=torch.tensor([3]),
                target_indices=torch.arange(6)[None])


def test_geometric_residual_roundtrip_keeps_correlated_pose_and_missing_masks():
    clean = people()['coordinates'][0, :2]
    prediction = clean.clone()
    prediction[..., 28:30] += torch.tensor([.01, -.02])
    prediction[..., 30:] *= torch.tensor([1.1, .9])
    local = (clean[..., :28].unflatten(-1, (14, 2))-clean[..., None, 28:30])/clean[..., None, 30:]
    prediction[..., :28] = (prediction[..., None, 28:30]+(local+.02)*prediction[..., None, 30:]).flatten(-2)
    mask = torch.ones(2, 14, dtype=torch.bool)
    residual = geometric_residuals(prediction, clean, mask)
    reconstructed = apply_residuals(clean, residual, mask)
    torch.testing.assert_close(reconstructed, prediction)
    mask[:, 0] = False
    residual = geometric_residuals(prediction, clean, mask)
    assert not residual[:, :2].any()
    assert bank_digest(residual, mask) != bank_digest(residual+1, mask)


def test_noise_is_coherent_across_rows_without_target_or_identity_mutation():
    rows, source = training_rows(people(), generator=torch.Generator().manual_seed(4))
    rows['proposal_indices'] = torch.arange(4).expand(1, 4, 4).clone()
    original = deepcopy((rows, source))
    residual = torch.full((5, 32), .1)
    mask = torch.ones(5, 14, dtype=torch.bool)
    first, stats = augment_prefix_rows(rows, source, residual, mask, torch.tensor([[240., 120.]]),
        clean_probability=0., generator=torch.Generator().manual_seed(14))
    second, _ = augment_prefix_rows(rows, source, residual, mask, torch.tensor([[240., 120.]]),
        clean_probability=0., generator=torch.Generator().manual_seed(14))
    assert torch.equal(first['prefix_coordinates'], second['prefix_coordinates'])
    assert stats['prefix_corruption_fraction'] == 1
    for rank in range(3):
        introduced = first['prefix_coordinates'][0, rank+1, rank]
        for row in range(rank+1, 4):
            assert torch.equal(introduced, first['prefix_coordinates'][0, row, rank])
    assert not first['prefix_coordinates'][first['token_types'] != 1].any()
    assert not torch.equal(first['prefix_coordinates'], rows['prefix_coordinates'])
    for key in rows:
        assert torch.equal(rows[key], original[0][key])
        if key != 'prefix_coordinates':
            assert torch.equal(first[key], rows[key])
    for key in source:
        assert torch.equal(source[key], original[1][key])
    clean, _ = augment_prefix_rows(rows, source, residual, mask, torch.tensor([[240., 120.]]),
        clean_probability=1., generator=torch.Generator().manual_seed(14))
    assert torch.equal(clean['prefix_coordinates'], rows['prefix_coordinates'])


def test_bounds_missing_joints_coincident_centers_empty_and_invalid_bank():
    source = people()
    source['coordinates'][0, 1, 28:30] = source['coordinates'][0, 0, 28:30]
    rows, source = training_rows(source, generator=torch.Generator().manual_seed(2))
    residual, mask = torch.full((2, 32), 100.), torch.ones(2, 14, dtype=torch.bool)
    noisy, stats = augment_prefix_rows(rows, source, residual, mask, torch.tensor([[240., 120.]]),
        clean_probability=0., generator=torch.Generator().manual_seed(0))
    coordinates = noisy['prefix_coordinates'][0, -1, :3]
    clean = source['coordinates'][0, :3]
    assert torch.isfinite(coordinates).all()
    assert ((coordinates[:, 30:]/clean[:, 30:]).log().abs() <= torch.log(torch.tensor(1.25))+1e-6).all()
    displacement = ((coordinates[:, 28:30]-clean[:, 28:30])/clean[:, 30:]).norm(dim=-1)
    assert (displacement <= .15+1e-6).all()
    assert (displacement == 0).sum() >= 1
    absent = ~source['coordinate_mask'][0, :3, :28].unflatten(-1, (14, 2)).all(-1)
    joints = coordinates[:, :28].unflatten(-1, (14, 2))
    torch.testing.assert_close(joints[absent], coordinates[:, None, 28:30].expand_as(joints)[absent])
    empty_source = people()
    empty_source['counts'].zero_()
    empty, shuffled = training_rows(empty_source)
    result, _ = augment_prefix_rows(empty, shuffled, residual, mask, torch.tensor([[240., 120.]]))
    assert torch.equal(empty['prefix_coordinates'], result['prefix_coordinates'])
    with pytest.raises(ValueError, match='bank'):
        augment_prefix_rows(rows, source, residual[:0], mask[:0], torch.tensor([[240., 120.]]))


def test_future_ground_truth_centers_cannot_change_an_earlier_prefix():
    rows, sources = training_rows(people(), generator=torch.Generator().manual_seed(54))
    residuals, mask = torch.full((2, 32), .1), torch.ones(2, 14, dtype=torch.bool)
    sizes = torch.tensor([[240., 120.]])
    original, _ = augment_prefix_rows(rows, sources, residuals, mask, sizes,
        clean_probability=0., generator=torch.Generator().manual_seed(4))
    changed = deepcopy(sources)
    changed['coordinates'][0, 2, 28:30] = changed['coordinates'][0, 0, 28:30]+.00001
    later, _ = augment_prefix_rows(rows, changed, residuals, mask, sizes,
        clean_probability=0., generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(original['prefix_coordinates'][0, 1], later['prefix_coordinates'][0, 1], atol=0, rtol=0)
