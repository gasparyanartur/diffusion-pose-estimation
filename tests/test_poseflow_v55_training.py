from copy import deepcopy
import json

import pytest
import pytorch_lightning as pl
import torch

from src.models.pipelines.pose_estimators.poseflow.training_v55 import PoseFlowV55Pipeline, PoseFlowV55PipelineConfig
from src.models.pipelines.pose_estimators.poseflow.balance_v51 import gradient_families
from tests.test_poseflow_v43_training import batch
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v55 import model, opened


def pipeline():
    network = model()
    for head in (*network.pose_decoder.pose_heads, *network.pose_decoder.box_heads):
        torch.nn.init.normal_(head.layers[-1].weight, std=.01)
    config = PoseFlowV55PipelineConfig(native=arguments(), capacity=6, rows=7, pretrained_checkpoint=None,
        quick_eval_enabled=False, calibration_batches=2, residual_bank_capacity=8)
    module = PoseFlowV55Pipeline(config=config, model=network)
    module.set_residual_bank(torch.full((4, 32), .1), torch.ones(4, 14, dtype=torch.bool), dict(split='train', image_ids=[1, 2]))
    module.set_calibration([[1., 2.]]*2, dict(source='unit'))
    return module


def test_task_gradient_routes_family_ownership_and_one_assignment(monkeypatch):
    module = pipeline().train()
    opened(module.model.proposal_context)
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
    groups = gradient_families(module.model)
    assert {id(value) for value in module.model.proposal_context.parameters()} <= {id(value) for value in groups['geometry']}
    identities = [id(value) for values in groups.values() for value in values]
    assert len(identities) == len(set(identities))
    probes = (module.model.proposal_context.attention.out_proj.weight, module.model.selector.candidate_score[-1].weight,
              module.model.coordinate_input[0].weight, next(module.model.frontend.parameters()))
    for task in ('pose_birth', 'box_birth', 'selection', 'eos'):
        gradients = torch.autograd.grad(losses[task], probes, retain_graph=True, allow_unused=True)
        assert gradients[0] is not None and gradients[0].abs().sum() > 0
        if task in ('selection', 'eos'):
            assert gradients[1] is not None and gradients[1].abs().sum() > 0
            assert gradients[2] is None and gradients[3] is None
        else:
            assert gradients[1] is None
            assert gradients[2] is not None and gradients[2].abs().sum() > 0
    losses['total'].backward()
    assert all(head.layers[-1].weight.grad.abs().sum() > 0 for head in module.model.pose_decoder.pose_heads)


def test_checkpoint_parity_empty_missing_and_strict_restore(tmp_path):
    module = pipeline().train()
    opened(module.model.proposal_context)
    direct = deepcopy(module)
    direct.model.activation_checkpointing = False
    sample = batch((2, 0))
    for instance in (module, direct):
        torch.manual_seed(55)
        instance._losses_for_batch(sample)['total'].backward()
    for name, parameter in module.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, dict(direct.named_parameters())[name].grad, atol=3e-5, rtol=3e-4)
    missing = batch((2, 0))
    missing['gt_labeled'].zero_()
    assert module._losses_for_batch(missing)['pose_birth'] == 0
    empty = module._losses_for_batch(batch((0,)))
    assert empty['pose_birth'] == 0 and empty['box_birth'] == 0
    assert json.loads(json.dumps(module.checkpoint_contract())) == module.checkpoint_contract()
    record = dict(state_dict=deepcopy(module.state_dict()), hyper_parameters=module.hparams,
                  **{'pytorch-lightning_version': pl.__version__})
    module.on_save_checkpoint(record)
    expected = module._losses_for_batch(sample)['total']
    path = tmp_path/'v55.ckpt'
    torch.save(record, path)
    restored = PoseFlowV55Pipeline.load_from_checkpoint(path, weights_only=False).train()
    torch.testing.assert_close(restored._losses_for_batch(sample)['total'], expected, atol=0, rtol=0)
    for key in ('v55_contract', 'v55_prefix_bank'):
        broken = deepcopy(record)
        broken.pop(key)
        with pytest.raises(ValueError):
            restored.on_load_checkpoint(broken)
    broken = deepcopy(record)
    broken['v55_contract']['proposal_context']['heads'] += 1
    with pytest.raises(ValueError, match='contract'):
        restored.on_load_checkpoint(broken)
