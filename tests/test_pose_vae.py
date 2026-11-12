import json
from pathlib import Path

import pytest
import torch

from src.data.pose_geometry import (
    PoseGeometryDataset, corrupt_geometry, flip_geometry,
    normalize_geometry, restore_geometry,
)
from src.models.pipelines.pose_vae.model import PoseBoxVAE, reconstruction_loss, prior_kl


def geometry(count=3):
    boxes = torch.tensor([[200., 120., 100., 160.]]).repeat(count, 1)
    points = torch.arange(28).reshape(1, 14, 2).float().repeat(count, 1, 1) * 8
    mask = torch.ones(count, 14, dtype=torch.bool)
    mask[:, 2] = False
    return dict(keypoints_px=points, boxes_cxcywh_px=boxes, joint_mask=mask,
                image_wh=torch.tensor([[640., 360.]]).repeat(count, 1))


def test_normalization_roundtrip_non_square_outside_box_and_mask():
    batch = geometry()
    relative, placement = normalize_geometry(**batch)
    expected = (batch['keypoints_px'] - batch['boxes_cxcywh_px'][:, None, :2]) / 160
    torch.testing.assert_close(relative[batch['joint_mask']], expected[batch['joint_mask']])
    assert relative[:, 2].count_nonzero() == 0
    restored = restore_geometry(relative, placement, batch['image_wh'])
    torch.testing.assert_close(restored['keypoints_px'][batch['joint_mask']],
                               batch['keypoints_px'][batch['joint_mask']])
    torch.testing.assert_close(restored['boxes_cxcywh_px'], batch['boxes_cxcywh_px'])
    noisy = dict(batch, boxes_cxcywh_px=batch['boxes_cxcywh_px'] + torch.tensor([20., 0., 0., 0.]))
    noisy_relative, _ = normalize_geometry(**noisy)
    torch.testing.assert_close(noisy_relative[batch['joint_mask']],
                               (expected - torch.tensor([.125, 0.]))[batch['joint_mask']])


def test_flip_is_involution_and_swaps_masks():
    batch = geometry()
    batch['joint_mask'][:, 0] = False
    flipped = flip_geometry(batch, torch.ones(3, dtype=torch.bool))
    assert not flipped['joint_mask'][:, 1].any()
    torch.testing.assert_close(flipped['keypoints_px'][:, 1, 0], 640-batch['keypoints_px'][:, 0, 0])
    torch.testing.assert_close(flipped['boxes_cxcywh_px'][:, 0], 440*torch.ones(3))
    restored = flip_geometry(flipped, torch.ones(3, dtype=torch.bool))
    for name in batch:
        torch.testing.assert_close(restored[name], batch[name])


def test_corruption_keeps_clean_targets_and_at_least_one_joint():
    batch = geometry(1000)
    original = {key: value.clone() for key, value in batch.items()}
    corrupted = corrupt_geometry(batch, generator=torch.Generator().manual_seed(42))
    for key in batch:
        torch.testing.assert_close(batch[key], original[key])
    assert corrupted['joint_mask'].any(-1).all()
    assert not (corrupted['joint_mask'] & ~batch['joint_mask']).any()
    unchanged = (corrupted['boxes_cxcywh_px'] == batch['boxes_cxcywh_px']).all(-1)
    assert .20 < unchanged.float().mean() < .30
    assert (batch['joint_mask'] & ~corrupted['joint_mask']).any()
    assert (corrupted['boxes_cxcywh_px'][:, 2:] > 0).all()
    single = geometry(100)
    single['joint_mask'][:] = False
    single['joint_mask'][:, 7] = True
    assert corrupt_geometry(single, generator=torch.Generator().manual_seed(3))['joint_mask'][:, 7].all()


def test_geometry_dataset_never_loads_images_and_preserves_occluded_labels(tmp_path):
    annotation = dict(id=1, image_id=7, bbox=[10, 20, 30, 40], iscrowd=0,
                      num_keypoints=2, keypoints=[15, 25, 1, 40, 60, 2]+[0, 0, 0]*12)
    path = tmp_path / 'annotations.json'
    path.write_text(json.dumps(dict(images=[dict(id=7, width=800, height=400,
                                               file_name='does-not-exist.jpg')],
                                    annotations=[annotation, dict(annotation, id=2, iscrowd=1)])))
    dataset = PoseGeometryDataset(path)
    assert len(dataset) == 1 and dataset.image_ids == {7}
    sample = dataset[0]
    torch.testing.assert_close(sample['boxes_cxcywh_px'], torch.tensor([25., 40., 30., 40.]))
    assert sample['joint_mask'].sum() == 2
    assert tuple(sample['image_wh']) == (800, 400)


def test_codec_batch_one_bounds_gradients_and_mask_invariance():
    model = PoseBoxVAE()
    assert sum(parameter.numel() for parameter in model.parameters()) < 650000
    batch = geometry(1)
    mean, logvar = model.encode(**batch)
    assert mean.shape == logvar.shape == (1, 32)
    changed = dict(batch, keypoints_px=batch['keypoints_px'].clone())
    changed['keypoints_px'][:, 2] = float('nan')
    torch.testing.assert_close(model.encode(**changed)[0], mean)
    prediction = model.decode(mean + (.5*logvar).exp(), batch['image_wh'])
    assert prediction['keypoints_px'].shape == (1, 14, 2)
    assert (prediction['boxes_cxcywh_px'][:, 2:] > 0).all()
    loss = reconstruction_loss(prediction, batch)['reconstruction'] + 1e-4*prior_kl(mean, logvar)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert model.posterior_mean.weight.grad.abs().sum() > 0
    assert model.posterior_logvar.weight.grad.abs().sum() > 0


def test_reconstruction_scores_hidden_labels_and_ignores_missing_labels():
    batch = geometry(1)
    prediction = {name: value.clone() for name, value in batch.items()}
    assert reconstruction_loss(prediction, batch)['reconstruction'] == 0
    prediction['keypoints_px'][:, 2] = float('nan')
    assert reconstruction_loss(prediction, batch)['reconstruction'] == 0
    prediction['keypoints_px'][:, 0] += 160
    assert reconstruction_loss(prediction, batch)['pose'] > 0
    torch.testing.assert_close(prior_kl(torch.ones(2, 32), torch.zeros(2, 32)), torch.tensor(16.))


@pytest.mark.parametrize('field', ['image_wh', 'boxes_cxcywh_px'])
def test_invalid_geometry_is_rejected(field):
    batch = geometry(1)
    batch[field][..., -1] = 0
    with pytest.raises(ValueError):
        normalize_geometry(**batch)
