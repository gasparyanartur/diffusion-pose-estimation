from copy import deepcopy

import pytest
import pytorch_lightning as pl
import torch

from src.models.pipelines.pose_estimators.poseflow.training_v50 import PoseFlowV50Pipeline, PoseFlowV50PipelineConfig
from src.models.pipelines.pose_estimators.poseflow.calibration_v49_balanced import calibrate
from tests.test_poseflow_v43_training import batch
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v50 import model


def pipeline():
    config = PoseFlowV50PipelineConfig(native=arguments(), rows=7, capacity=6, codec_width=32,
        latent_dim=8, pretrained_checkpoint=None, quick_eval_enabled=False, calibration_batches=2)
    return PoseFlowV50Pipeline(config=config, model=model())


def test_training_one_context_assignment_balanced_losses_and_gradients(monkeypatch):
    module = pipeline().train()
    module.set_calibration([[1.,1.,1.,1.]]*2, dict(source='unit'))
    calls = []
    original_context = module.model.prepare_context
    original_match = module.slot_matcher.forward
    def context(*args, **kwargs):
        calls.append('context')
        return original_context(*args, **kwargs)
    def match(*args, **kwargs):
        calls.append('assignment')
        return original_match(*args, **kwargs)
    monkeypatch.setattr(module.model, 'prepare_context', context)
    monkeypatch.setattr(module.slot_matcher, 'forward', match)
    losses = module._losses_for_batch(batch((2,1)))
    losses['total'].backward()
    assert calls == ['context', 'assignment']
    assert all(head.weight.grad.abs().sum() > 0 for head in module.model.residual_heads)
    assert all(head.weight.grad.abs().sum() > 0 for head in module.model.latent_heads)
    assert module.model.selector.key.weight.grad.abs().sum() > 0
    assert module.model.person_codec.encoder[0].weight.grad.abs().sum() > 0
    assert losses['frontend'] > 0 and losses['selection'] > 0


def test_empty_and_calibration_restore_without_assets(tmp_path):
    module = pipeline().train()
    before = deepcopy(module.model.state_dict())
    samples = [batch((2,1)), batch((1,))]
    rng = torch.get_rng_state().clone()
    calibrate(module, iter(samples))
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, module.model.state_dict()[name]) for name, value in before.items())
    losses = module._losses_for_batch(batch((0,)))
    assert torch.isfinite(losses['total'])
    assert losses['selection'] == 0
    record = dict(state_dict=module.state_dict(), hyper_parameters=module.hparams,
        **{'pytorch-lightning_version':pl.__version__})
    module.on_save_checkpoint(record)
    path = tmp_path/'v50.ckpt'
    torch.save(record, path)
    restored = PoseFlowV50Pipeline.load_from_checkpoint(path, weights_only=False)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in module.state_dict().items())
    broken = deepcopy(record)
    broken['v49_contract']['architecture'] = 'V49-balanced'
    with pytest.raises(ValueError):
        restored.on_load_checkpoint(broken)


def test_checkpointed_pipeline_gradient_parity():
    checkpointed = pipeline().train()
    checkpointed.set_calibration([[1.,1.,1.,1.]]*2, dict(source='unit'))
    direct = deepcopy(checkpointed)
    direct.model.activation_checkpointing = False
    sample = batch((2,1))
    for instance in (checkpointed, direct):
        torch.manual_seed(7342)
        instance._losses_for_batch(sample)['total'].backward()
    for name, parameter in checkpointed.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, dict(direct.named_parameters())[name].grad, atol=3e-5, rtol=3e-4)


def test_backbone_rate_uses_new_frontend_path():
    module = pipeline()
    optimizer = module.configure_optimizers()['optimizer']
    assert [group['lr'] for group in optimizer.param_groups] == [2e-4, 1e-4]
    backbone_ids = {id(parameter) for parameter in module.model.frontend.native.backbone.parameters() if parameter.requires_grad}
    assert {id(parameter) for parameter in optimizer.param_groups[1]['params']} == backbone_ids
