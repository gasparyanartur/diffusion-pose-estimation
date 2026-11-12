from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_estimators.diffusionregpose.model import DiffusionRegPoseModel
from src.models.pipelines.pose_estimators.poseflow.codec_v43 import geometry_from_batch
from src.models.pipelines.pose_estimators.poseflow.coordinates_v47 import ordered_coordinates, training_rows
from src.models.pipelines.pose_estimators.poseflow.losses_v47 import row_losses
from src.models.pipelines.pose_estimators.poseflow.model_v46 import DenseRowLayer
from src.models.pipelines.pose_estimators.poseflow.model_v47 import CachedRowLayer, PoseFlowV47Model
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v43_training import batch


def model():
    return PoseFlowV47Model(native=DiffusionRegPoseModel(**arguments()), rows=8, capacity=6,
        hidden_dim=32, num_heads=4, feedforward_dim=64, codec_width=32, latent_dim=8).eval()


def context(network, batch_size=1):
    return network._project_image_context(torch.randn(batch_size, 7, 32), torch.randn(batch_size, 7, 32),
                                         torch.zeros(batch_size, 7, dtype=torch.bool))


def test_cached_image_attention_matches_original_attention_and_gradients():
    original = DenseRowLayer(32, 4, 64).eval()
    cached = CachedRowLayer(32, 4, 64).eval()
    cached.load_state_dict(original.state_dict(), strict=True)
    state, memory, position = torch.randn(2, 3, 5, 32), torch.randn(2, 9, 32), torch.randn(2, 9, 32)
    mask = torch.zeros(2, 3, 5, dtype=torch.bool)
    mask[:, :, -2:] = True
    image_mask = torch.zeros(2, 9, dtype=torch.bool)
    image_mask[:, -1] = True
    expected = original(state, mask, memory, position, image_mask)
    actual = cached(state, mask, *cached.project_image(memory, position), image_mask)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for (name, value), (_, other) in zip(original.named_parameters(), cached.named_parameters()):
        torch.testing.assert_close(value.grad, other.grad, atol=3e-6, rtol=3e-4, msg=name)


def test_batched_rows_equal_independent_rows_with_bidirectional_influence_and_padding():
    network = model()
    rows = training_rows(torch.rand(2, 6, 32), torch.tensor([3, 1]), rows=8)
    image = context(network, 2)
    output = network.forward_rows(image, **{k: rows[k] for k in network.row_input_fields})
    separate = [network.forward_rows(image, rows['noisy_coordinates'][:, r:r+1], rows['row_times'][r:r+1],
                rows['token_types'][:, r:r+1], rows['padding_mask'][:, r:r+1]) for r in range(8)]
    for key in output:
        torch.testing.assert_close(output[key], torch.cat([item[key] for item in separate], 2), atol=2e-6, rtol=2e-5)
    changed = rows['noisy_coordinates'].clone()
    changed[0, 3, 2] += 20
    second = network.forward_rows(image, changed, rows['row_times'], rows['token_types'], rows['padding_mask'])
    assert not torch.allclose(output['pred_coordinates'][:, 0, 3, 0], second['pred_coordinates'][:, 0, 3, 0])
    torch.testing.assert_close(output['pred_coordinates'][:, :, :3], second['pred_coordinates'][:, :, :3], atol=0, rtol=0)
    padded = torch.where(rows['padding_mask'][..., None], float('nan'), rows['noisy_coordinates'])
    third = network.forward_rows(image, padded, rows['row_times'], rows['token_types'], rows['padding_mask'])
    torch.testing.assert_close(output['pred_coordinates'], third['pred_coordinates'], atol=0, rtol=0)


@pytest.mark.parametrize('counts', [(2, 1), (0,)])
def test_decoded_losses_gradients_and_checkpoint_parity(counts):
    network = model().train()
    plain = deepcopy(network)
    plain.activation_checkpointing = False
    sample = batch(counts)
    geometry, locations = geometry_from_batch(sample)
    sources = ordered_coordinates(geometry, locations, batch_size=len(counts), capacity=6)
    rows = training_rows(sources['coordinates'], sources['counts'], rows=8)
    raw = dict(memory=torch.randn(len(counts), 7, 32), position=torch.randn(len(counts), 7, 32),
               padding_mask=torch.zeros(len(counts), 7, dtype=torch.bool))
    losses = []
    for instance in (network, plain):
        image = instance._project_image_context(**raw)
        output = instance.forward_rows(image, **{k: rows[k] for k in instance.row_input_fields})
        loss = row_losses(output, rows, sources, geometry, sample['img_hw'].flip(-1))
        torch.testing.assert_close(loss['total'], 5*(loss['coordinate_birth']+loss['coordinate_refine'])+
                                   .25*(loss['geometry_birth']+loss['geometry_refine'])+loss['eos'])
        loss['total'].backward()
        losses.append(loss['total'])
        assert all(head.weight.grad.abs().sum() > 0 for head in instance.eos_heads)
        if sum(counts):
            assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0
                       for parameter in instance.person_codec.parameters())
            assert all(head.weight.grad.abs().sum() > 0 for head in instance.latent_heads)
        else:
            assert loss['coordinate_birth'] == loss['geometry_refine'] == 0
    torch.testing.assert_close(*losses, atol=0, rtol=0)
    for (name, value), (_, other) in zip(network.named_parameters(), plain.named_parameters()):
        if value.grad is not None:
            torch.testing.assert_close(value.grad, other.grad, atol=1e-5, rtol=1e-4, msg=name)


def test_original_frontend_parity_and_single_image_projection(monkeypatch):
    native = DiffusionRegPoseModel(**arguments()).eval()
    network = PoseFlowV47Model(native=deepcopy(native), rows=8, capacity=6,
        hidden_dim=32, num_heads=4, feedforward_dim=64, codec_width=32, latent_dim=8).eval()
    image = torch.randn(1, 3, 128, 96)
    mask = torch.zeros(1, 128, 96, dtype=torch.bool)
    expected = native.encode_features(native.prepare_transformer_inputs(image, mask))
    calls = []
    for layer in network.layers:
        original = layer.project_image
        def project(*args, original=original):
            calls.append(1)
            return original(*args)
        monkeypatch.setattr(layer, 'project_image', project)
    cached = network.prepare_context(image, mask)
    torch.testing.assert_close(expected, cached['memory'], atol=0, rtol=0)
    rows = training_rows(torch.rand(1, 6, 32), torch.tensor([2]), rows=8)
    for _ in range(2):
        network.forward_rows(cached, **{k: rows[k] for k in network.row_input_fields})
    assert len(calls) == 2
    assert not hasattr(network.transformer, 'decoder')


def test_coordinate_loss_masked_mean_and_equal_person_weights_independent_of_row_count():
    sample = batch((2,))
    sample['gt_labeled'] = sample['gt_labeled'].clone()
    sample['gt_labeled'][0, 0, :7] = False
    geometry, locations = geometry_from_batch(sample)
    sources = ordered_coordinates(geometry, locations, batch_size=1, capacity=6)
    rows = training_rows(sources['coordinates'], sources['counts'], rows=8)
    target = torch.nn.functional.pad(sources['coordinates'], (0, 0, 0, 2))
    prediction = target[None, :, None].expand(2, 1, 8, 8, 32).clone()
    prediction[:, :, :, 0] += 1
    prediction[:, :, :, 1] += 3
    prediction[:, :, :, 0, :14] += 1000  # Unobserved coordinates cannot enter either pose objective.
    output = dict(pred_coordinates=prediction, eos_logits=torch.zeros(2, 1, 8, 8))
    loss = row_losses(output, rows, sources, geometry, sample['img_hw'].flip(-1))
    assert loss['coordinate_birth'] == loss['coordinate_refine'] == 5  # (1^2+3^2)/2 people.
    torch.testing.assert_close(loss['eos'], torch.log(torch.tensor(2.)))


def test_independent_row_gradients_match_parallel_with_shared_learned_encoder():
    network = model()
    independent = deepcopy(network)
    rows = training_rows(torch.rand(2, 6, 32), torch.tensor([3, 1]), rows=8)
    memory, position = torch.randn(2, 7, 32), torch.randn(2, 7, 32)
    mask = torch.zeros(2, 7, dtype=torch.bool)
    image = network._project_image_context(memory, position, mask)
    other_image = independent._project_image_context(memory, position, mask)
    output = network.forward_rows(image, **{k: rows[k] for k in network.row_input_fields})
    separate = [independent.forward_rows(other_image, rows['noisy_coordinates'][:, r:r+1], rows['row_times'][r:r+1],
                rows['token_types'][:, r:r+1], rows['padding_mask'][:, r:r+1])['pred_coordinates'] for r in range(8)]
    output['pred_coordinates'].square().mean().backward()
    torch.cat(separate, 2).square().mean().backward()
    for (name, value), (_, other) in zip(network.named_parameters(), independent.named_parameters()):
        if value.grad is not None:
            torch.testing.assert_close(value.grad, other.grad, atol=3e-6, rtol=3e-4, msg=name)
