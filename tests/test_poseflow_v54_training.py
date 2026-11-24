from copy import deepcopy
import json

import pytest
import pytorch_lightning as pl
import torch

from src.models.pipelines.pose_estimators.poseflow.training_v54 import PoseFlowV54Pipeline, PoseFlowV54PipelineConfig
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v43_training import batch
from tests.test_poseflow_v53 import model
from tests.test_poseflow_v53_training import pipeline as clean_pipeline


def pipeline():
    network = model()
    for head in (*network.pose_decoder.pose_heads, *network.pose_decoder.box_heads):
        torch.nn.init.normal_(head.layers[-1].weight, std=.01)
    config = PoseFlowV54PipelineConfig(native=arguments(), capacity=6, rows=7, pretrained_checkpoint=None,
        quick_eval_enabled=False, calibration_batches=2, residual_bank_capacity=8)
    module = PoseFlowV54Pipeline(config=config, model=network)
    module.set_residual_bank(torch.full((4, 32), .1), torch.ones(4, 14, dtype=torch.bool),
                             dict(split='train', image_ids=[1, 2], source_checkpoint_sha256='unit'))
    module.set_calibration([[1., 2.]]*2, dict(source='unit'))
    return module


def test_training_augmentation_clean_loss_parity_and_no_evaluation_rng(monkeypatch):
    module = pipeline().eval()
    reference = clean_pipeline().eval()
    reference.model.load_state_dict(module.model.state_dict())
    reference.set_calibration([[1., 2.]]*2, dict(source='unit'))
    sample = batch((2, 1))
    state = torch.get_rng_state().clone()
    actual = module._losses_for_batch(sample, training=False)
    expected = reference._losses_for_batch(sample, training=False)
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, atol=0, rtol=0)
    assert torch.equal(state, torch.get_rng_state())
    assert json.loads(json.dumps(module.checkpoint_contract())) == module.checkpoint_contract()
    seen = []
    forward = module.model.forward_rows
    def observe(context, **rows):
        seen.append(rows['prefix_coordinates'].clone())
        return forward(context, **rows)
    monkeypatch.setattr(module.model, 'forward_rows', observe)
    frozen = deepcopy(module.state_dict())
    module.train()
    torch.manual_seed(54)
    losses = module._losses_for_batch(sample)
    assert losses['prefix_corruption_fraction'] > 0
    losses['total'].backward()
    assert module.model.coordinate_input[0].weight.grad.abs().sum() > 0
    assert module.model.pose_decoder.keypoint_embed.weight.grad.abs().sum() > 0
    assert torch.equal(frozen['prefix_residuals'], module.prefix_residuals)
    empty = module._losses_for_batch(batch((0,)))
    assert empty['pose_birth'] == 0 and empty['prefix_corruption_fraction'] == 0


def test_strict_bank_checkpoint_restore_and_seeded_next_loss(tmp_path):
    module = pipeline().train()
    sample = batch((2, 1))
    record = dict(state_dict=deepcopy(module.state_dict()), hyper_parameters=module.hparams,
                  **{'pytorch-lightning_version': pl.__version__})
    module.on_save_checkpoint(record)
    expected = module._losses_for_batch(sample)['total']
    path = tmp_path/'v54.ckpt'
    torch.save(record, path)
    restored = PoseFlowV54Pipeline.load_from_checkpoint(path, weights_only=False).train()
    actual = restored._losses_for_batch(sample)['total']
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(module.prefix_residuals, restored.prefix_residuals)
    broken = deepcopy(record)
    broken['state_dict']['prefix_residuals'][0, 0] += 1
    with pytest.raises(ValueError, match='bank'):
        restored.on_load_checkpoint(broken)
    broken = deepcopy(record)
    broken['v54_contract']['noise']['clean_probability'] = .9
    with pytest.raises(ValueError, match='contract'):
        restored.on_load_checkpoint(broken)
