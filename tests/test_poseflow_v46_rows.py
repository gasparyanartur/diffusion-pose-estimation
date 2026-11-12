import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.rows_v46 import (
    PERSON, PREDICT, build_training_rows, canonical_order, row_alpha_bar,
)


def test_columns_preserve_collisions_and_resolve_ties_with_geometry():
    boxes = torch.tensor([[50.05, 20., 10., 10.], [50., 30., 10., 10.],
                          [50., 10., 10., 10.], [100., 0., 10., 10.],
                          [0., 90., 10., 10.]])
    geometry = dict(boxes_cxcywh_px=boxes, image_wh=torch.tensor([[100., 100.]]).repeat(5, 1),
                    keypoints_px=torch.zeros(5, 14, 2), joint_mask=torch.ones(5, 14, dtype=torch.bool))
    assert canonical_order(geometry, torch.arange(5)).tolist() == [4, 2, 0, 1, 3]
    geometry['boxes_cxcywh_px'][:, 0] = 100 - boxes[:, 0]
    assert canonical_order(geometry, torch.arange(5)).tolist()[0] == 3


@pytest.mark.parametrize('counts', [[0], [1], [3, 1, 0], [50]])
def test_prefix_growth_prediction_and_refinement_tail(counts):
    clean = torch.randn(len(counts), 50, 8)
    result = build_training_rows(clean, torch.tensor(counts), generator=torch.Generator().manual_seed(1))
    assert result['noisy_latents'].shape == (len(counts), 64, 64, 8)
    for image, people in enumerate(counts):
        for row in range(64):
            prefix = min(row, people)
            assert (result['token_types'][image, row] == PERSON).sum() == prefix
            assert result['token_types'][image, row, prefix] == PREDICT
            assert result['padding_mask'][image, row].sum() == 63-prefix
            assert bool(result['eos_targets'][image, row]) == (row >= people)
            assert result['noisy_latents'][image, row, prefix:].count_nonzero() == 0


def test_schedule_endpoints_noise_independence_and_no_future_leakage():
    clean = torch.randn(1, 50, 8)
    counts = torch.tensor([3])
    result = build_training_rows(clean, counts, generator=torch.Generator().manual_seed(8))
    assert row_alpha_bar(result['row_times'])[[0, -1]].tolist() == [0., 1.]
    torch.testing.assert_close(result['noisy_latents'][0, -1, :3], clean[0, :3], rtol=0, atol=0)
    assert not torch.equal(result['noise'][0, 1, 0], result['noise'][0, 2, 0])
    changed = clean.clone()
    changed[:, 2:] += 1000
    other = build_training_rows(changed, counts, generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(result['noisy_latents'][:, :3], other['noisy_latents'][:, :3], rtol=0, atol=0)


def test_capacity_and_row_budget_rejection():
    with pytest.raises(ValueError, match='capacity'):
        build_training_rows(torch.zeros(1, 50, 8), torch.tensor([51]))
    with pytest.raises(ValueError, match='rows'):
        build_training_rows(torch.zeros(1, 50, 8), torch.tensor([50]), rows=50)
