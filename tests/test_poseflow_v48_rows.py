import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.rows_v46 import PAD, PERSON, PREDICT
from src.models.pipelines.pose_estimators.poseflow.rows_v48 import training_rows
from src.models.pipelines.pose_estimators.poseflow.losses_v48 import row_losses
from src.models.pipelines.pose_estimators.poseflow.codec_v43 import geometry_from_batch
from src.models.pipelines.pose_estimators.poseflow.coordinates_v47 import ordered_coordinates
from tests.test_poseflow_v43_training import batch


def test_clean_rows_have_only_births_and_one_eos_with_safe_batch_padding():
    clean = torch.rand(3, 6, 32)
    rows = training_rows(clean, torch.tensor([3, 1, 0]))
    assert rows['prefix_coordinates'].shape == (3, 4, 4, 32)
    assert rows['row_valid'].sum(-1).tolist() == [4, 2, 1]
    assert rows['birth_rows'].sum(-1).tolist() == [3, 1, 0]
    assert rows['eos_targets'].sum(-1).tolist() == [1, 1, 1]
    assert (rows['token_types'] == PREDICT).sum(-1).eq(1).all()
    for image, count in enumerate([3, 1, 0]):
        for row in range(count+1):
            assert rows['token_types'][image, row, :row].eq(PERSON).all()
            assert rows['token_types'][image, row, row] == PREDICT
            assert rows['token_types'][image, row, row+1:].eq(PAD).all()
            torch.testing.assert_close(rows['prefix_coordinates'][image, row, :row], clean[image, :row], atol=0, rtol=0)
    assert rows['prefix_coordinates'][rows['token_types'] != PERSON].eq(0).all()
    assert torch.equal(rows['padding_mask'], rows['token_types'] == PAD)
    changed = clean.clone()
    changed[:, 2:] += 100
    other = training_rows(changed, torch.tensor([3, 1, 0]))
    torch.testing.assert_close(rows['prefix_coordinates'][:, :3], other['prefix_coordinates'][:, :3], atol=0, rtol=0)


def test_empty_and_full_capacity_rows_and_invalid_counts():
    assert training_rows(torch.zeros(1, 50, 32), torch.tensor([0]))['prefix_coordinates'].shape == (1, 1, 1, 32)
    full = training_rows(torch.zeros(1, 50, 32), torch.tensor([50]))
    assert full['row_valid'].sum() == 51 and full['birth_rows'].sum() == 50
    with pytest.raises(ValueError, match='capacity'):
        training_rows(torch.zeros(1, 50, 32), torch.tensor([51]))


@pytest.mark.parametrize('counts', [(2, 1), (0,)])
def test_birth_only_objective_masks_prefix_eos_missing_joints_and_padded_rows(counts):
    sample = batch(counts)
    geometry, locations = geometry_from_batch(sample)
    sources = ordered_coordinates(geometry, locations, batch_size=len(counts), capacity=6)
    rows = training_rows(sources['coordinates'], sources['counts'])
    shape = rows['prefix_coordinates'].shape
    prediction = torch.full((2, *shape), 100., requires_grad=True)
    logits = torch.zeros(2, *shape[:-1], requires_grad=True)
    output = dict(pred_coordinates=prediction, eos_logits=logits)
    loss = row_losses(output, rows, sources, geometry, sample['img_hw'].flip(-1))
    torch.testing.assert_close(loss['total'], 10*loss['coordinate_birth']+.5*loss['geometry_birth']+loss['eos'])
    torch.testing.assert_close(loss['eos'], torch.log(torch.tensor(2.)))
    assert not any('refine' in key for key in loss)
    loss['total'].backward()
    grad = prediction.grad.abs().sum(-1)
    expected = (rows['token_types'] == PREDICT) & rows['birth_rows'][..., None]
    assert torch.equal(grad > 0, expected[None].expand_as(grad))
    if not sum(counts):
        assert loss['coordinate_birth'] == loss['geometry_birth'] == 0


def test_birth_mean_is_equal_per_person_and_eos_balanced_per_image():
    sample = batch((2, 1))
    sample['gt_labeled'] = sample['gt_labeled'].clone()
    sample['gt_labeled'][0, 0, :7] = False
    geometry, locations = geometry_from_batch(sample)
    sources = ordered_coordinates(geometry, locations, batch_size=2, capacity=6)
    rows = training_rows(sources['coordinates'], sources['counts'])
    shape = rows['prefix_coordinates'].shape
    pred = torch.zeros(2, *shape)
    for image, count in enumerate((2, 1)):
        for rank in range(count):
            pred[:, image, rank, rank] = sources['coordinates'][image, rank]+rank+1
            missing = ~sources['coordinate_mask'][image, rank]
            pred[:, image, rank, rank, missing] += 1000
    output = dict(pred_coordinates=pred, eos_logits=torch.zeros(2, *shape[:-1]))
    loss = row_losses(output, rows, sources, geometry, sample['img_hw'].flip(-1))
    torch.testing.assert_close(loss['coordinate_birth'], torch.tensor(2.))  # (1+4+1)/3 people
    changed = pred.clone()
    changed[:, 1, 2] = 9000  # Entirely inactive batch row.
    other = row_losses(dict(output, pred_coordinates=changed), rows, sources, geometry, sample['img_hw'].flip(-1))
    torch.testing.assert_close(loss['total'], other['total'], atol=0, rtol=0)


def test_eos_balances_person_group_and_single_stop_and_ignores_padding():
    from src.models.pipelines.pose_estimators.poseflow.losses_v48 import balanced_eos_loss
    rows = training_rows(torch.zeros(2, 6, 32), torch.tensor([3, 0]))
    logits = torch.full((2, 2, 4, 4), 100.)
    logits[:, 0, :3] = -2.
    logits[:, 0, 3] = -1.
    logits[:, 1, 0] = 2.
    negative_error = torch.nn.functional.softplus(torch.tensor(-2.))
    stop_error = torch.nn.functional.softplus(torch.tensor(1.))
    expected = ((negative_error+stop_error)/2+negative_error)/2
    torch.testing.assert_close(balanced_eos_loss(logits, rows), expected)
