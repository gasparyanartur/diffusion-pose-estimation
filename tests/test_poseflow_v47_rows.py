import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.codec_v43 import geometry_from_batch
from src.models.pipelines.pose_estimators.poseflow.coordinates_v47 import (
    coordinate_geometry, geometry_coordinates, ordered_coordinates, training_rows,
)
from tests.test_poseflow_v43_training import batch


def test_coordinate_roundtrip_mask_fill_and_rectangular_images():
    sample = batch((2, 1))
    sample['img_hw'] = torch.tensor([[80., 160.], [200., 100.]])
    sample['gt_labeled'] = sample['gt_labeled'].clone()
    sample['gt_labeled'][0, 0, 3] = False
    sample['gt_keypoints'] = sample['gt_keypoints'].clone()
    sample['gt_keypoints'][0, 0, 3] = float('nan')
    geometry, locations = geometry_from_batch(sample)
    coordinates = geometry_coordinates(geometry)
    assert torch.isfinite(coordinates).all()
    decoded = coordinate_geometry(coordinates, geometry['image_wh'])
    torch.testing.assert_close(decoded['boxes_cxcywh_px'], geometry['boxes_cxcywh_px'])
    mask = geometry['joint_mask']
    torch.testing.assert_close(decoded['keypoints_px'][mask], geometry['keypoints_px'][mask])
    torch.testing.assert_close(decoded['keypoints_px'][0, 3], geometry['boxes_cxcywh_px'][0, :2])
    sources = ordered_coordinates(geometry, locations, batch_size=2)
    assert sources['counts'].tolist() == [2, 1]
    assert sources['target_indices'][0, :2].tolist() == [0, 1]
    assert sources['target_indices'][1, 0] == 2


@pytest.mark.parametrize('counts', [[0], [1], [3, 1, 0], [50]])
def test_coordinate_rows_have_exact_prefix_noise_and_masking(counts):
    clean = torch.rand(len(counts), 50, 32)
    rows = training_rows(clean, torch.tensor(counts), generator=torch.Generator().manual_seed(3))
    assert rows['noisy_coordinates'].shape == (len(counts), 64, 64, 32)
    for image, people in enumerate(counts):
        for row in range(64):
            prefix = min(row, people)
            assert rows['predict_indices'][image, row] == prefix
            assert rows['noisy_coordinates'][image, row, prefix:].count_nonzero() == 0
            assert bool(rows['eos_targets'][image, row]) == (row >= people)
        torch.testing.assert_close(rows['noisy_coordinates'][image, -1, :people],
                                   6*(2*clean[image, :people]-1), rtol=0, atol=0)
    assert not torch.equal(rows['noise'][0, 1], rows['noise'][0, 2])
    changed = clean.clone()
    changed[:, 2:] += 1000
    other = training_rows(changed, torch.tensor(counts), generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(rows['noisy_coordinates'][:, :3], other['noisy_coordinates'][:, :3], rtol=0, atol=0)


def test_capacity_rejection_and_empty_ordered_coordinates():
    geometry, locations = geometry_from_batch(batch((0,)))
    result = ordered_coordinates(geometry, locations, batch_size=1)
    assert result['coordinates'].shape == (1, 50, 32)
    assert result['counts'].tolist() == [0]
    with pytest.raises(ValueError, match='capacity'):
        training_rows(torch.zeros(1, 50, 32), torch.tensor([51]))
