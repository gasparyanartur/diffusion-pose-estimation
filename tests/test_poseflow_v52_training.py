from copy import deepcopy

import pytest
import pytorch_lightning as pl
import torch

from src.models.pipelines.pose_estimators.poseflow.calibration_v49_balanced import calibrate
from src.models.pipelines.pose_estimators.poseflow.losses_v49_balanced import TASK_NAMES
from src.models.pipelines.pose_estimators.poseflow.training_v52 import PoseFlowV52Pipeline, PoseFlowV52PipelineConfig
from tests.test_poseflow_v43_training import batch
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v52 import model


def pipeline():
    config = PoseFlowV52PipelineConfig(native=arguments(), rows=7, capacity=6, codec_width=32,
        latent_dim=32, pretrained_checkpoint=None, quick_eval_enabled=False, calibration_batches=2)
    return PoseFlowV52Pipeline(config=config, model=model())


def test_training_context_assignment_both_layers_and_fixed_task_priorities(monkeypatch):
    module = pipeline().train()
    module.set_calibration([[1., 2., 3., 4.]]*2, dict(source='unit'))
    calls = []
    prepare, match = module.model.prepare_context, module.slot_matcher.forward
    def context(*args, **kwargs):
        calls.append('context')
        return prepare(*args, **kwargs)
    def assignment(*args, **kwargs):
        calls.append('assignment')
        return match(*args, **kwargs)
    monkeypatch.setattr(module.model, 'prepare_context', context)
    monkeypatch.setattr(module.slot_matcher, 'forward', assignment)
    losses = module._losses_for_batch(batch((2, 1)))
    for index, name in enumerate(TASK_NAMES):
        torch.testing.assert_close(losses[f'{name}_contribution'], .25*losses[name]/module.gradient_scales[index])
    losses['total'].backward()
    assert calls == ['context', 'assignment']
    assert all(head[-1].weight.grad.abs().sum() > 0 for head in module.model.geometry_heads)
    assert module.model.coordinate_input[0].weight.grad.abs().sum() > 0
    for layer in module.model.layers:
        assert layer.time_modulation[-1].weight.grad.abs().sum() > 0
        assert layer.image_attention.sampling_offsets.weight.grad.abs().sum() > 0
    assert losses['frontend'] > 0 and losses['selection'] > 0


def test_calibration_empty_missing_joints_restore_and_corrupt_contract(tmp_path):
    module = pipeline().train()
    before = deepcopy(module.model.state_dict())
    samples = [batch((2, 1)), batch((1,))]
    rng = torch.get_rng_state().clone()
    calibrate(module, iter(samples))
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, module.model.state_dict()[name]) for name, value in before.items())
    for counts in ((0,), (2, 0)):
        sample = batch(counts)
        sample['gt_labeled'].zero_()
        losses = module._losses_for_batch(sample)
        assert all(torch.isfinite(value).all() for value in losses.values())
        assert losses['pose_birth'] == 0 and losses['pose_refine'] == 0
    record = dict(state_dict=module.state_dict(), hyper_parameters=module.hparams,
        **{'pytorch-lightning_version': pl.__version__})
    module.on_save_checkpoint(record)
    path = tmp_path/'v52.ckpt'
    torch.save(record, path)
    restored = PoseFlowV52Pipeline.load_from_checkpoint(path, weights_only=False)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in module.state_dict().items())
    broken = deepcopy(record)
    broken['v52_contract']['attention']['points'] += 1
    with pytest.raises(ValueError):
        restored.on_load_checkpoint(broken)
    broken = deepcopy(record)
    broken['state_dict']['gradient_scales'][0] *= 2
    with pytest.raises(ValueError):
        restored.on_load_checkpoint(broken)


def test_native_checkpoint_gradient_parity_and_backbone_rates():
    checkpointed = pipeline().train()
    checkpointed.set_calibration([[1., 1., 1., 1.]]*2, dict(source='unit'))
    direct = deepcopy(checkpointed)
    direct.model.activation_checkpointing = False
    sample = batch((2, 1))
    for instance in (checkpointed, direct):
        torch.manual_seed(4452)
        instance._losses_for_batch(sample)['total'].backward()
    for name, parameter in checkpointed.named_parameters():
        other = dict(direct.named_parameters())[name]
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=4e-5, rtol=4e-4)
    configuration = direct.configure_optimizers()
    assert [group['lr'] for group in configuration['optimizer'].param_groups] == [2e-4, 1e-4]
    assert set(configuration['lr_scheduler']['scheduler'].milestones) == {31, 61}
