from copy import deepcopy

import pytest
import pytorch_lightning as pl
import torch

from src.models.pipelines.pose_estimators.poseflow.calibration_v49_balanced import calibrate, fixed_scales
from src.models.pipelines.pose_estimators.poseflow.training_v49_balanced import (
    PoseFlowV49BalancedPipeline, PoseFlowV49BalancedPipelineConfig)
from tests.test_poseflow_v43_training import batch
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v49 import model


def pipeline():
    config = PoseFlowV49BalancedPipelineConfig(native=arguments(), rows=7, capacity=6, codec_width=32,
        latent_dim=8, pretrained_checkpoint=None, quick_eval_enabled=False, calibration_batches=2)
    return PoseFlowV49BalancedPipeline(config=config, model=model())


def ready_pipeline():
    module = pipeline()
    module.set_calibration([[2.,4.,6.,8.],[4.,6.,8.,10.]], dict(source='unit'))
    return module


def test_calibration_uses_midpoint_median_floor_and_rejects_bad_or_inactive_tasks():
    torch.testing.assert_close(fixed_scales([[1.,2.,3.,1e-10],[3.,4.,5.,1e-10]]), torch.tensor([2.,3.,4.,1e-8]))
    for bad in ([[float('nan')]*4], [[0.]*4], [[-1.]*4], [[1.,2.]]):
        with pytest.raises(ValueError):
            fixed_scales(bad)


def test_calibration_has_no_model_update_preserves_rng_and_is_frozen(monkeypatch):
    module = pipeline().train()
    before = deepcopy(module.model.state_dict())
    samples = [batch((2,1)),batch((1,))]
    rng = torch.get_rng_state().clone()
    calls = []
    original = module.model.prepare_context
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(module.model, 'prepare_context', counted)
    report = calibrate(module, iter(samples))
    assert len(calls) == 2 and bool(module.calibration_ready)
    assert report['task_names'] == ['pose_birth','box_birth','pose_refine','box_refine']
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value,module.model.state_dict()[name]) for name,value in before.items())
    assert not module.gradient_scales.requires_grad
    assert all(parameter.grad is None for parameter in module.parameters())
    with pytest.raises(RuntimeError,match='already'):
        calibrate(module, iter([]))


def test_pipeline_requires_calibration_and_uses_frozen_scales():
    module = pipeline().train()
    with pytest.raises(RuntimeError,match='calibrat'):
        module._losses_for_batch(batch((1,)))
    module = ready_pipeline().train()
    before = module.gradient_scales.clone()
    losses = module._losses_for_batch(batch((2,1)))
    losses['total'].backward()
    assert torch.equal(before,module.gradient_scales)
    assert all(head.weight.grad.abs().sum() > 0 for head in module.model.latent_heads)
    assert module.model.person_codec.encoder[0].weight.grad.abs().sum() > 0


def test_variant_checkpoint_restores_without_initializer_or_calibration_assets_and_rejects_corruption(tmp_path):
    module = ready_pipeline()
    record = dict(state_dict=module.state_dict(),hyper_parameters=module.hparams,
        **{'pytorch-lightning_version':pl.__version__})
    module.on_save_checkpoint(record)
    expected_rng = torch.get_rng_state().clone()
    path = tmp_path/'balanced.ckpt'
    torch.save(record,path)
    restored = PoseFlowV49BalancedPipeline.load_from_checkpoint(path,weights_only=False)
    assert torch.equal(expected_rng,torch.get_rng_state())
    assert restored.calibration_provenance == module.calibration_provenance
    assert all(torch.equal(value,restored.state_dict()[name]) for name,value in module.state_dict().items())
    for key in ('contract','scales','provenance'):
        broken = deepcopy(record)
        if key == 'contract':
            broken['v49_contract']['architecture'] = 'V49'
        elif key == 'scales':
            broken['state_dict']['gradient_scales'][0] *= 2
        else:
            broken['v49_balanced_calibration']['norms'][0][0] = 10000.
        with pytest.raises(ValueError,match='V49|calibrat'):
            restored.on_load_checkpoint(broken)


def test_checkpointed_and_uncheckpointed_balanced_gradients_match():
    checkpointed = ready_pipeline().train()
    direct = deepcopy(checkpointed)
    direct.model.activation_checkpointing = False
    sample = batch((2,1))
    for instance in (checkpointed,direct):
        torch.manual_seed(7342)
        instance._losses_for_batch(sample)['total'].backward()
    for name,parameter in checkpointed.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad,dict(direct.named_parameters())[name].grad,atol=3e-6,rtol=3e-4)
