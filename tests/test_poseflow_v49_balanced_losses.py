import pytest
import torch

from src.models.pipelines.pose_autoencoder.model import geometry_loss_per_person
from src.models.pipelines.pose_estimators.poseflow.codec_v43 import geometry_from_batch
from src.models.pipelines.pose_estimators.poseflow.coordinates_v47 import coordinate_geometry
from src.models.pipelines.pose_estimators.poseflow.losses_v49_balanced import (
    age_balanced_average, boundary_eos_loss, robust_geometry_errors, robust_square, row_losses)
from tests.test_poseflow_v43_training import batch
from tests.test_poseflow_v49_training import pipeline


def test_robust_squared_residual_retains_local_arithmetic_and_bounds_large_residual_derivative():
    residual = torch.tensor([0., .5, 1., 10., -100.], requires_grad=True)
    loss = robust_square(residual.square())
    torch.testing.assert_close(loss, torch.tensor([0., .25, 1., 19., 199.]))
    loss.sum().backward()
    torch.testing.assert_close(residual.grad, torch.tensor([0., 1., 2., 2., -2.]))


def test_robust_geometry_matches_original_for_small_errors_and_masks_missing_joints():
    sample = batch((2,))
    target, _ = geometry_from_batch(sample)
    prediction = {name:value.clone() for name,value in target.items() if name != 'joint_mask'}
    prediction['keypoints_px'] += .001
    target['joint_mask'][0, 0] = False
    prediction['keypoints_px'][0, 0] = 1e6
    actual = robust_geometry_errors(prediction, target)
    original = geometry_loss_per_person(prediction, target)
    for name in actual:
        torch.testing.assert_close(actual[name], original[name])


def test_age_buckets_average_people_then_present_buckets_and_use_row_age_at_terminal_sigma_zero():
    errors = torch.zeros(2, 1, 7, 7)
    mask = torch.zeros(1, 7, 7, dtype=torch.bool)
    # Person0 has four old-tail rows; person4 only has age1/age2.
    for rank in (0, 4):
        for row in range(rank+1, 7):
            age = row-rank
            errors[:, 0, row, rank] = 1 if age == 1 else 3 if age == 2 else 8
            mask[0, row, rank] = True
    torch.testing.assert_close(age_balanced_average(errors, mask), torch.tensor(3.))
    torch.testing.assert_close(age_balanced_average(errors, mask, weights=(.5,.75,.75)), torch.tensor(103/48))
    assert age_balanced_average(errors, torch.zeros_like(mask)) == 0


def test_first_eos_has_fixed_weight_independent_of_tail_length_and_empty_images_work():
    def evaluate(rows):
        logits = torch.zeros(2, 1, rows, 1, requires_grad=True)
        labels = torch.ones(1, rows, dtype=torch.bool)
        labels[:, :2] = False
        result = boundary_eos_loss(logits, dict(eos_targets=labels,predict_indices=torch.zeros(1,rows,dtype=torch.long)))
        result['eos'].backward()
        return logits.grad, result
    short, losses = evaluate(5)
    long, _ = evaluate(50)
    torch.testing.assert_close(short[:, :, 2], long[:, :, 2])
    torch.testing.assert_close(losses['eos'], torch.tensor(2.).log())
    empty = boundary_eos_loss(torch.zeros(2,2,7,1), dict(eos_targets=torch.ones(2,7,dtype=torch.bool),
        predict_indices=torch.zeros(2,7,dtype=torch.long)))
    torch.testing.assert_close(empty['eos'], torch.tensor(2.).log())


@pytest.mark.parametrize('counts', [(2,1),(0,)])
def test_balanced_loss_masks_targets_supervises_both_layers_and_applies_four_scales(counts):
    module = pipeline()
    sample = batch(counts)
    geometry, locations = geometry_from_batch(sample)
    rows, sources = module.training_rows(sample, geometry, locations, generator=torch.Generator().manual_seed(42))
    target = torch.nn.functional.pad(sources['coordinates'], (0,0,0,1))
    prediction = target[None,:,None].expand(2,len(counts),7,7,32).clone()
    prediction[..., :30] += .02
    prediction.requires_grad_()
    output = dict(pred_coordinates=prediction,eos_logits=torch.zeros(2,len(counts),7,7,requires_grad=True))
    scales = torch.tensor([2.,3.,4.,5.])
    losses = row_losses(output,rows,sources,geometry,sample['img_hw'].flip(-1),scales=scales)
    torch.testing.assert_close(losses['total'], losses['pose_birth_contribution']+losses['box_birth_contribution']+
        losses['pose_refine_contribution']+losses['box_refine_contribution']+losses['eos'])
    torch.testing.assert_close(losses['pose_birth_contribution'], .5*.25*losses['pose_birth']/scales[0])
    torch.testing.assert_close(losses['box_birth_contribution'], .5*.75*losses['box_birth']/scales[1])
    losses['total'].backward()
    assert torch.isfinite(prediction.grad).all()
    assert output['eos_logits'].grad.abs().sum() > 0
    if sum(counts):
        assert (prediction.grad.abs().sum((1,2,3,4)) > 0).all()
        missing = ~torch.nn.functional.pad(sources['coordinate_mask'],(0,0,0,1))
        assert prediction.grad.masked_select(missing[None,:,None]).eq(0).all()
    else:
        assert losses['pose_birth'] == losses['pose_refine'] == 0
        assert losses['box_birth'] == losses['box_refine'] == 0


def test_invalid_scales_rejected():
    with pytest.raises(ValueError,match='scales'):
        row_losses({}, {}, {}, {}, torch.ones(1,2), scales=torch.tensor([1.,0.,1.,1.]))
