from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from src.models.pipelines.pose_estimators.diffusionregpose.training import DiffusionRegPoseCriterion
from src.models.pipelines.pose_estimators.poseflow.training_v42_timestep_loss import (
    TimestepLossConfig, TimestepLossCriterion, geometry_factors,
)
from tests.test_poseflow_v42 import config
from tests.test_poseflow_v42_uniform_t import repeated_batch


def example():
    torch.manual_seed(12)
    def row():
        return dict(pred_logits=torch.randn(2,3,2,requires_grad=True),
            pred_boxes=(torch.rand(2,3,4)*.3+.3).requires_grad_(),
            pred_keypoints=torch.rand(2,3,6,requires_grad=True))
    outputs = row()
    outputs.update(aux_outputs=[row(),row()], interm_outputs=row(),
        dn_class_pred=torch.randn(2,2,2,requires_grad=True),
        dn_bbox_pred=(torch.rand(2,2,4)*.3+.3).requires_grad_(),
        dn_class_input=torch.ones(2,2,dtype=torch.long),
        dn_bbox_input=torch.full((2,2,4),.4))
    outputs['dn_aux_outputs'] = [{key:value[:,:2] for key,value in row().items()} for _ in range(2)]
    targets = [dict(labels=torch.ones(n,dtype=torch.long),boxes=torch.full((n,4),.4),
        keypoints=torch.tensor([[.2,.3,.4,.5,1.,1.]]).repeat(n,1),area=torch.full((n,),.16)) for n in (1,2)]
    return outputs, targets


def test_schedule_direction_midpoint_and_average_strength():
    box,pose = geometry_factors(torch.tensor([0,500,1000]),1001)
    torch.testing.assert_close(box,torch.tensor([.2,1.,1.8]))
    torch.testing.assert_close(pose,torch.tensor([1.8,1.,.2]))
    box,pose = geometry_factors(torch.arange(1000),1000)
    assert box.mean().item() == pytest.approx(1.)
    assert pose.mean().item() == pytest.approx(1.)


def test_unit_weights_exact_native_objective_gradients_and_matching():
    outputs,targets = example()
    criterion = TimestepLossCriterion(num_body_points=2,num_box_decoder_layers=1)
    native = DiffusionRegPoseCriterion(num_body_points=2,num_box_decoder_layers=1)
    expected = native(outputs,targets)
    with patch.object(criterion.matcher,'forward',wraps=criterion.matcher.forward) as matcher:
        actual = criterion(outputs,targets,geometry_weights=(torch.ones(2),torch.ones(2)))
    assert matcher.call_count == 4
    assert torch.equal(actual['total'],expected['total'])
    variables = [outputs[k] for k in ('pred_boxes','pred_logits','pred_keypoints','dn_bbox_pred')]
    for a,b in zip(torch.autograd.grad(actual['total'],variables,retain_graph=True),
                   torch.autograd.grad(expected['total'],variables)):
        assert torch.equal(a,b)


def slice_image(outputs,index):
    return {key: value[index:index+1] if torch.is_tensor(value) else
            [slice_image(row,index) for row in value] if isinstance(value,list) else
            slice_image(value,index) for key,value in outputs.items()}


def test_mixed_timestep_native_oracle_all_aux_encoder_and_dn_branches():
    outputs,targets = example()
    criterion = TimestepLossCriterion(num_body_points=2,num_box_decoder_layers=1)
    original = criterion.weights
    with patch.object(criterion.matcher,'forward',wraps=criterion.matcher.forward) as matcher:
        actual = criterion(outputs,targets,timesteps=torch.tensor([999,0]))
    assert matcher.call_count == 4, 'Changing losses must not add rematching'
    class GlobalNormalized(DiffusionRegPoseCriterion):
        def _loss_normalizer(self,matches,targets,reference):
            return reference.new_tensor(3.)
    expected = outputs['pred_boxes'].sum()*0
    for index,(box,pose) in enumerate(((1.8,.2),(.2,1.8))):
        weights = replace(original,bbox=original.bbox*box,giou=original.giou*box,
            keypoints=original.keypoints*pose,oks=original.oks*pose)
        oracle = GlobalNormalized(num_body_points=2,num_box_decoder_layers=1,weights=weights)
        expected = expected+oracle(slice_image(outputs,index),[targets[index]])['total']
    torch.testing.assert_close(actual['total'],expected)
    for key in ('loss_ce','loss_ce_interm','loss_ce_0','dn_loss_ce','dn_loss_ce_1'):
        assert torch.equal(actual[key],DiffusionRegPoseCriterion(num_body_points=2,num_box_decoder_layers=1)(outputs,targets)[key])
    assert criterion.weights is original and criterion.matcher.weights is original
    a,b = torch.autograd.grad(actual['total'],outputs['pred_boxes'],retain_graph=True)[0],torch.autograd.grad(expected,outputs['pred_boxes'])[0]
    torch.testing.assert_close(a,b)


@pytest.mark.parametrize('empty',[False,True])
def test_pipeline_exact_initialization_rng_actual_times_and_restore(empty):
    baseline_config = config(training_timestep_policy='uniform',pose_readout_mode='box_center_residual')
    torch.manual_seed(42)
    baseline = baseline_config.build(max_epochs=80)
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    arm = TimestepLossConfig(**baseline_config.model_dump()).build(max_epochs=80)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(value,arm.state_dict()[key]) for key,value in baseline.state_dict().items())
    value = repeated_batch(2);value['image'] = torch.randn(2,3,64,80)
    if empty:value['gt_labeled'].zero_()
    observed=[]
    arm.model.register_forward_pre_hook(lambda model,args:observed.append(args[2].clone()))
    with patch.object(arm.criterion,'forward',wraps=arm.criterion.forward) as loss:
        losses = arm._losses_for_batch(value)
    assert torch.equal(loss.call_args.kwargs['timesteps'],observed[0])
    assert torch.isfinite(losses['total']);losses['total'].backward()
    record={};arm.on_save_checkpoint(record);arm.on_load_checkpoint(record)
    record['v42_timestep_loss']['scale']=1.
    with pytest.raises(ValueError):arm.on_load_checkpoint(record)


@pytest.mark.parametrize('times',[torch.tensor([-1]),torch.tensor([1000]),torch.tensor([0.]),torch.tensor([[0]])])
def test_invalid_timesteps(times):
    with pytest.raises((ValueError,TypeError)):geometry_factors(times,1000)
