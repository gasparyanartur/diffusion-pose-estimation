from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_estimators.diffusionregpose.model import DiffusionRegPoseModel
from src.models.pipelines.pose_estimators.poseflow.codec_v43 import geometry_from_batch
from src.models.pipelines.pose_estimators.poseflow.coordinates_v47 import ordered_coordinates
from src.models.pipelines.pose_estimators.poseflow.losses_v48 import row_losses
from src.models.pipelines.pose_estimators.poseflow.model_v48 import PoseFlowV48Model
from src.models.pipelines.pose_estimators.poseflow.rows_v48 import training_rows
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v43_training import batch


def model():
    return PoseFlowV48Model(native=DiffusionRegPoseModel(**arguments()), capacity=6,
        hidden_dim=32, num_heads=4, feedforward_dim=64, codec_width=32, latent_dim=8).eval()


def context(network, batch_size=1):
    return network._project_image_context(torch.randn(batch_size, 7, 32), torch.randn(batch_size, 7, 32),
                                         torch.zeros(batch_size, 7, dtype=torch.bool))


def test_parallel_rows_gradient_parity_bidirectional_influence_and_padding():
    network = model()
    independent = deepcopy(network)
    rows = training_rows(torch.rand(2, 6, 32), torch.tensor([3, 1]))
    raw = dict(memory=torch.randn(2, 7, 32), position=torch.randn(2, 7, 32), padding_mask=torch.zeros(2, 7, dtype=torch.bool))
    image = network._project_image_context(**raw)
    second_image = independent._project_image_context(**raw)
    output = network.forward_rows(image, **{key: rows[key] for key in network.row_input_fields})
    separate = [independent.forward_rows(second_image, **{key: rows[key][:, row:row+1] for key in network.row_input_fields})
                for row in range(4)]
    for key in output:
        torch.testing.assert_close(output[key], torch.cat([value[key] for value in separate], 2), atol=2e-6, rtol=2e-5)
    output['pred_coordinates'].square().mean().backward()
    torch.cat([value['pred_coordinates'] for value in separate], 2).square().mean().backward()
    for (name, value), (_, other) in zip(network.named_parameters(), independent.named_parameters()):
        if value.grad is not None:
            torch.testing.assert_close(value.grad, other.grad, atol=3e-6, rtol=3e-4, msg=name)
    changed = rows['prefix_coordinates'].clone()
    changed[0, 3, 2] += 10
    result = network.forward_rows(image, changed, rows['token_types'], rows['padding_mask'])
    assert not torch.allclose(output['pred_coordinates'][:, 0, 3, 0], result['pred_coordinates'][:, 0, 3, 0])
    torch.testing.assert_close(output['pred_coordinates'][:, :, :3], result['pred_coordinates'][:, :, :3], atol=0, rtol=0)
    padded = torch.where(rows['padding_mask'][..., None], float('nan'), rows['prefix_coordinates'])
    safe = network.forward_rows(image, padded, rows['token_types'], rows['padding_mask'])
    torch.testing.assert_close(output['pred_coordinates'], safe['pred_coordinates'], atol=0, rtol=0)
    assert not hasattr(network, 'time_embedding')


@pytest.mark.parametrize('counts', [(2, 1), (0,)])
def test_checkpoint_gradient_parity_and_both_layer_codec_gradients(counts):
    network = model().train()
    plain = deepcopy(network)
    plain.activation_checkpointing = False
    sample = batch(counts)
    geometry, locations = geometry_from_batch(sample)
    sources = ordered_coordinates(geometry, locations, batch_size=len(counts), capacity=6)
    rows = training_rows(sources['coordinates'], sources['counts'])
    raw = dict(memory=torch.randn(len(counts), 7, 32), position=torch.randn(len(counts), 7, 32),
               padding_mask=torch.zeros(len(counts), 7, dtype=torch.bool))
    for instance in (network, plain):
        output = instance.forward_rows(instance._project_image_context(**raw), **{key: rows[key] for key in instance.row_input_fields})
        losses = row_losses(output, rows, sources, geometry, sample['img_hw'].flip(-1))
        losses['total'].backward()
        assert all(head.weight.grad.abs().sum() > 0 for head in instance.eos_heads)
        if sum(counts):
            assert all(head.weight.grad.abs().sum() > 0 for head in instance.latent_heads)
            assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in instance.person_codec.parameters())
    for (name, value), (_, other) in zip(network.named_parameters(), plain.named_parameters()):
        if value.grad is not None:
            torch.testing.assert_close(value.grad, other.grad, atol=1e-5, rtol=1e-4, msg=name)


def test_frontend_legacy_exact_parity_and_one_context_projection(monkeypatch):
    native = DiffusionRegPoseModel(**arguments()).eval()
    network = PoseFlowV48Model(native=deepcopy(native), capacity=6, hidden_dim=32, num_heads=4,
                              feedforward_dim=64, codec_width=32, latent_dim=8).eval()
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
    rows = training_rows(torch.rand(1, 6, 32), torch.tensor([2]))
    network.forward_rows(cached, **{key: rows[key] for key in network.row_input_fields})
    network.forward_rows(cached, **{key: rows[key] for key in network.row_input_fields})
    assert len(calls) == 2


@pytest.mark.parametrize('eos_at,count,reason', [(0, 0, 1), (2, 2, 1), (99, 6, 2)])
def test_early_stop_clean_prefix_and_immutable_births(monkeypatch, eos_at, count, reason):
    network, seen = model(), []
    def forward(image, prefix_coordinates, token_types, padding_mask):
        step = len(seen)
        seen.append(prefix_coordinates.clone())
        coordinates = prefix_coordinates.new_full((*token_types.shape, 32), .1*(step+1))
        eos = prefix_coordinates.new_full(token_types.shape, 20. if step == eos_at else -20.)
        return dict(pred_coordinates=torch.stack([coordinates/2, coordinates]), eos_logits=torch.stack([eos, eos]))
    monkeypatch.setattr(network, 'forward_rows', forward)
    output = network.predict(context(network), torch.tensor([[200., 100.]]), collect_trace=True)
    assert output['person_counts'].tolist() == [count] and output['stop_reason'].tolist() == [reason]
    assert len(seen) == (count+1 if reason == 1 else count)
    assert len(output['trace']) == 2*len(seen)
    for rank in range(count):
        torch.testing.assert_close(output['pred_coordinates'][0, rank], torch.full((32,), .1*(rank+1)))
        for step in range(rank+1, len(seen)):
            torch.testing.assert_close(seen[step][0, 0, rank], output['pred_coordinates'][0, rank], atol=0, rtol=0)
    assert output['sequence_indices'][0, :count].tolist() == list(range(count))
    assert not output['valid_mask'][0, count:].any()


def test_mixed_batch_stopped_images_leave_decoder_and_eval_is_batch_independent(monkeypatch):
    network, seen = model(), []
    image = context(network, 2)
    image['memory'][:, 0, 0] = torch.tensor([0., 2.])
    def forward(cache, coordinates, token_types, padding_mask):
        rank = token_types.shape[-1]-1
        seen.append(len(coordinates))
        eos = torch.where(cache['memory'][:, 0, 0] <= rank, 20., -20.)
        logits = eos[:, None, None].expand_as(token_types).to(coordinates)
        return dict(pred_coordinates=coordinates.new_full((2, *coordinates.shape), .25), eos_logits=torch.stack([logits, logits]))
    monkeypatch.setattr(network, 'forward_rows', forward)
    result = network.predict(image, torch.tensor([[100., 200.], [170., 90.]]))
    assert seen == [2, 1, 1] and result['person_counts'].tolist() == [0, 2]
    assert result['eos_row'].tolist() == [1, 3] and 'trace' not in result
