from copy import deepcopy
import json

import pytest
import pytorch_lightning as pl
import torch

from src.models.pipelines.pose_estimators.poseflow.training_v53 import PoseFlowV53Pipeline, PoseFlowV53PipelineConfig
from src.models.pipelines.pose_estimators.poseflow.calibration_v53 import calibrate
from tests.test_poseflow_v43_training import batch
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v53 import model


def pipeline():
    config = PoseFlowV53PipelineConfig(native=arguments(), capacity=6, rows=7, pretrained_checkpoint=None,
                                       quick_eval_enabled=False, calibration_batches=2)
    network = model()
    for head in (*network.pose_decoder.pose_heads, *network.pose_decoder.box_heads):
        torch.nn.init.normal_(head.layers[-1].weight, std=.01)
    return PoseFlowV53Pipeline(config=config, model=network)


def test_birth_training_one_context_assignment_and_no_refinement(monkeypatch):
    module = pipeline().train()
    assert json.loads(json.dumps(module.checkpoint_contract())) == module.checkpoint_contract()
    module.set_calibration([[1., 2.]]*2, dict(source='unit'))
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
    assert calls == ['context', 'assignment']
    assert not any('refine' in name for name in losses)
    for index, name in enumerate(('pose_birth', 'box_birth')):
        torch.testing.assert_close(losses[f'{name}_contribution'], .5*losses[name]/module.gradient_scales[index])
    losses['total'].backward()
    for head in module.model.pose_decoder.pose_heads:
        assert head.layers[-1].weight.grad.abs().sum() > 0
    assert module.model.pose_decoder.keypoint_embed.weight.grad.abs().sum() > 0
    assert losses['frontend'] > 0 and losses['selection'] > 0


def test_empty_missing_joints_calibration_restore_and_gradient_parity(tmp_path):
    module = pipeline().train()
    samples = [batch(), batch((1,))]
    state, rng = deepcopy(module.model.state_dict()), torch.get_rng_state().clone()
    calibrate(module, iter(samples))
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, module.model.state_dict()[name]) for name, value in state.items())
    for counts in ((0,), (2, 0)):
        sample = batch(counts)
        sample['gt_labeled'].zero_()
        losses = module._losses_for_batch(sample)
        assert losses['pose_birth'] == 0
        assert all(torch.isfinite(value).all() for value in losses.values())
    direct = deepcopy(module)
    direct.model.activation_checkpointing = False
    sample = batch()
    for instance in (module, direct):
        instance.zero_grad(set_to_none=True)
        torch.manual_seed(53)
        instance._losses_for_batch(sample)['total'].backward()
    for name, parameter in module.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, dict(direct.named_parameters())[name].grad, atol=3e-5, rtol=3e-4)
    record = dict(state_dict=module.state_dict(), hyper_parameters=module.hparams,
                  **{'pytorch-lightning_version': pl.__version__})
    module.on_save_checkpoint(record)
    path = tmp_path/'v53.ckpt'
    torch.save(record, path)
    restored = PoseFlowV53Pipeline.load_from_checkpoint(path, weights_only=False)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in module.state_dict().items())
    broken = deepcopy(record)
    broken['v53_contract']['noise'] = 'Gaussian'
    with pytest.raises(ValueError):
        restored.on_load_checkpoint(broken)
    broken = deepcopy(record)
    broken['state_dict']['gradient_scales'][0] *= 2
    with pytest.raises(ValueError):
        restored.on_load_checkpoint(broken)


def test_launch_hashes_native_sources_and_forbids_fallback(monkeypatch, tmp_path):
    from scripts.training.train_poseflow_v53 import main, source_hashes
    from src.models.pipelines.pose_estimators.diffusionregpose import transformer
    monkeypatch.chdir(tmp_path)
    kernel = tmp_path/'src/kernel.cu'
    kernel.parent.mkdir()
    kernel.write_text('native source')
    initial = source_hashes()
    kernel.write_text('changed native source')
    assert initial['src/kernel.cu'] != source_hashes()['src/kernel.cu']
    monkeypatch.setattr(transformer, '_load_native_msda', lambda: None)
    with pytest.raises(RuntimeError, match='fallback is forbidden'):
        main.__wrapped__(None)
