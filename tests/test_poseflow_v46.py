from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_estimators.diffusionregpose.model import DiffusionRegPoseModel
from src.models.pipelines.pose_estimators.poseflow.codec_v43 import geometry_from_batch
from src.models.pipelines.pose_estimators.poseflow.model_v46 import PoseFlowV46Model
from src.models.pipelines.pose_estimators.poseflow.losses_v46 import row_losses, task_average
from src.models.pipelines.pose_estimators.poseflow.rows_v46 import build_training_rows, ordered_sources
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v43_codec import small_codec
from tests.test_poseflow_v43_training import batch


def model():
    return PoseFlowV46Model(codec=small_codec(), native=DiffusionRegPoseModel(**arguments()),
                           rows=8, capacity=6, hidden_dim=32, num_heads=4, feedforward_dim=64).eval()


def context(batch_size=1):
    return dict(memory=torch.randn(batch_size, 7, 32), position=torch.randn(batch_size, 7, 32),
                padding_mask=torch.zeros(batch_size, 7, dtype=torch.bool))


def test_parallel_rows_match_independent_outputs_and_gradients():
    network = model()
    single = deepcopy(network)
    inputs = build_training_rows(torch.randn(1, 6, 8), torch.tensor([3]), rows=8)
    image = context()
    output = network.forward_rows(image, **{k: inputs[k] for k in ('noisy_latents', 'row_times', 'token_types', 'padding_mask')})
    separate = []
    for row in range(8):
        separate.append(single.forward_rows(image, inputs['noisy_latents'][:, row:row+1], inputs['row_times'][row:row+1],
            inputs['token_types'][:, row:row+1], inputs['padding_mask'][:, row:row+1]))
    for key in ('pred_latents', 'eos_logits'):
        expected = torch.cat([item[key] for item in separate], 2)
        torch.testing.assert_close(output[key], expected, atol=2e-6, rtol=2e-5)
    output['pred_latents'].square().mean().backward()
    torch.cat([item['pred_latents'] for item in separate], 2).square().mean().backward()
    for (name, value), (_, other) in zip(network.named_parameters(), single.named_parameters()):
        assert (value.grad is None) == (other.grad is None), name
        if value.grad is not None:
            torch.testing.assert_close(value.grad, other.grad, atol=3e-6, rtol=3e-4, msg=name)


def test_bidirectional_attention_padding_exclusion_and_row_isolation():
    network, image = model(), context()
    rows = build_training_rows(torch.randn(1, 6, 8), torch.tensor([3]), rows=8)
    def run(value):
        return network.forward_rows(image, value, rows['row_times'], rows['token_types'], rows['padding_mask'])['pred_latents']
    baseline = run(rows['noisy_latents'])
    changed = rows['noisy_latents'].clone()
    changed[0, 3, 2] += 10
    output = run(changed)
    assert not torch.allclose(baseline[:, 0, 3, 0], output[:, 0, 3, 0])
    torch.testing.assert_close(baseline[:, :, :3], output[:, :, :3], rtol=0, atol=0)
    changed = torch.where(rows['padding_mask'][..., None], 1e6, rows['noisy_latents'])
    torch.testing.assert_close(baseline, run(changed), rtol=0, atol=0)


@pytest.mark.parametrize('counts', [(2, 1), (0,)])
def test_losses_both_heads_frozen_codec_and_checkpoint_gradient_parity(counts):
    network = model().train()
    plain = deepcopy(network)
    plain.activation_checkpointing = False
    sample = batch(counts)
    sample['gt_labeled'] = sample['gt_labeled'].clone()
    sample['gt_labeled'][..., 3] = False
    geometry, locations = geometry_from_batch(sample)
    sources = ordered_sources(network.codec, geometry, locations, batch_size=len(counts), capacity=6)
    rows = build_training_rows(sources['latents'], sources['counts'], rows=8)
    image = context(len(counts))
    before = deepcopy(network.codec.state_dict())
    losses = []
    for instance in (network, plain):
        output = instance.forward_rows(image, **{k: rows[k] for k in ('noisy_latents', 'row_times', 'token_types', 'padding_mask')})
        loss = row_losses(output, rows, sources, geometry, instance.codec, sample['img_hw'].flip(-1))
        assert torch.isfinite(loss['total'])
        loss['total'].backward()
        losses.append(loss['total'])
        assert all(head.weight.grad.abs().sum() > 0 for head in instance.eos_heads)
        if sum(counts):
            assert all(head.weight.grad.abs().sum() > 0 for head in instance.latent_heads)
        else:
            assert loss['latent_birth'] == loss['latent_refine'] == loss['geometry_birth'] == loss['geometry_refine'] == 0
    torch.testing.assert_close(*losses, rtol=0, atol=0)
    for (name, value), (_, other) in zip(network.named_parameters(), plain.named_parameters()):
        if value.grad is not None:
            torch.testing.assert_close(value.grad, other.grad, atol=1e-5, rtol=1e-4, msg=name)
    for name, value in before.items():
        assert torch.equal(value, network.codec.state_dict()[name])
    assert all(value.grad is None for value in network.codec.parameters())


def test_equal_birth_and_refinement_task_normalization():
    rows = build_training_rows(torch.zeros(1, 6, 8), torch.tensor([3]), rows=8)
    error = torch.full((2, 1, 8, 8), 3.)
    assert task_average(error, rows['birth_mask']) == 3
    assert task_average(error, rows['refinement_mask']) == 3
    for rank in range(3):
        isolated = torch.zeros_like(error)
        isolated[:, :, :, rank] = 1
        torch.testing.assert_close(task_average(isolated, rows['refinement_mask']), torch.tensor(1/3))


def test_retained_image_frontend_exact_parity_and_no_detection_heads():
    native = DiffusionRegPoseModel(**arguments()).eval()
    network = PoseFlowV46Model(codec=small_codec(), native=deepcopy(native), rows=8, capacity=6,
                              hidden_dim=32, num_heads=4, feedforward_dim=64).eval()
    images = torch.randn(1, 3, 128, 96)
    mask = torch.zeros(1, 128, 96, dtype=torch.bool)
    inputs = native.prepare_transformer_inputs(images, mask)
    expected = native.encode_features(inputs)
    actual = network.prepare_context(images, mask)
    torch.testing.assert_close(expected, actual['memory'], atol=0, rtol=0)
    assert not hasattr(network.transformer, 'decoder')
    assert not any('bbox_embed' in name or 'class_embed' in name for name in network.state_dict())
