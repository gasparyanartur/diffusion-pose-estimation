from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.balance_v51 import GradientBalance, clip_families
from src.models.pipelines.pose_estimators.poseflow.training_v51 import PoseFlowV51Pipeline, PoseFlowV51PipelineConfig
from tests.test_poseflow_v51 import instance
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v43_training import batch


def pipeline():
    return PoseFlowV51Pipeline(config=PoseFlowV51PipelineConfig(native=arguments(),rows=7,capacity=6,
        codec_width=32,latent_dim=8,pretrained_checkpoint=None,quick_eval_enabled=False,calibration_batches=2),model=instance())


def test_balance_caps_missing_tasks_and_roundtrip():
    balance = GradientBalance(4)
    balance.update(torch.tensor([[10.,1.,2.,3.],[20.,2.,4.,6.],[.0001,.001,.0001,.0001],[0.,0.,0.,0.]]))
    assert balance.weights.min() >= .1 and balance.weights.max() <= 10
    assert balance.weights[2] > balance.weights[0]
    assert balance.weights[3] == 1
    restored = GradientBalance(4)
    restored.load_state_dict(balance.state_dict(),strict=True)
    torch.testing.assert_close(balance.weights,restored.weights,rtol=0,atol=0)


def test_disjoint_clipping_combined_bound_and_duplicate_rejection():
    parameters = [torch.nn.Parameter(torch.ones(2)) for _ in range(3)]
    for index,parameter in enumerate(parameters): parameter.grad = torch.full_like(parameter,100.**index)
    report = clip_families({str(index):[parameter] for index,parameter in enumerate(parameters)})
    assert sum(float(parameter.grad.square().sum()) for parameter in parameters)**.5 <= .100001
    assert all(report[f'{index}/preclip'] > report[f'{index}/postclip'] for index in range(3))
    with pytest.raises(ValueError): clip_families({'a':parameters,'b':parameters})


def test_empty_checkpoint_and_geometry_gradients(tmp_path):
    module = pipeline().train()
    from src.models.pipelines.pose_estimators.poseflow.calibration_v51 import calibrate
    calibrate(module,iter([batch((2,1)),batch((1,))]))
    assert module.calibration_ready
    losses = module._losses_for_batch(batch((0,)))
    assert torch.isfinite(losses['total'])
    losses = module._losses_for_batch(batch((2,1)))
    losses['total'].backward()
    assert module.model.person_codec.encoder[0].weight.grad.abs().sum() > 0
    assert all(head.weight.grad.abs().sum() > 0 for head in module.model.residual_heads)
    record = dict(state_dict=module.state_dict(),hyper_parameters=module.hparams)
    module.on_save_checkpoint(record)
    restored=pipeline()
    restored.on_load_checkpoint(record)
    restored.load_state_dict(record['state_dict'],strict=True)
    torch.testing.assert_close(module.balance.weights,restored.balance.weights,rtol=0,atol=0)
    broken=deepcopy(record);broken['v49_contract']['architecture']='V50'
    with pytest.raises(ValueError): restored.on_load_checkpoint(broken)


def test_v51_checkpointed_gradient_parity():
    module=pipeline().train()
    norms=[[[1.,1.,1.,1.]]*4]*2
    module.set_gradient_calibration(norms,dict(source='unit'))
    direct=deepcopy(module)
    direct.model.activation_checkpointing=False
    sample=batch((2,1))
    for item in (module,direct):
        torch.manual_seed(942)
        item._losses_for_batch(sample)['total'].backward()
    for name,parameter in module.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad,dict(direct.named_parameters())[name].grad,atol=4e-5,rtol=4e-4)


def test_calibration_provenance_allows_device_roundoff_and_rejects_changes():
    module=pipeline()
    module.set_gradient_calibration([[[1.,2.,3.,4.]]*4]*2,dict(source='unit'))
    record=dict(state_dict=module.state_dict(),hyper_parameters=module.hparams)
    module.on_save_checkpoint(record)
    record['v49_balanced_calibration']['initial_weights'][0] += 1e-7
    pipeline().on_load_checkpoint(record)
    record['v49_balanced_calibration']['initial_weights'][0] += .01
    with pytest.raises(ValueError): pipeline().on_load_checkpoint(record)


def test_launch_requires_fresh_noise_learning_gate(tmp_path):
    import json
    from scripts.training.train_poseflow_v51 import validate_verification
    proof=tmp_path/'verification.json'
    proof.write_text(json.dumps(dict(passed=True,native_preflight=dict(passed=True),evaluator_smoke=dict(images=2),learning=dict(passed=False))))
    with pytest.raises(ValueError,match='verification'):
        validate_verification(proof,None,'unused')
