from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_estimators.diffusionregpose.model import DiffusionRegPoseModel
from src.models.pipelines.pose_estimators.poseflow.full_depth_birth import FullDepthBirthDecoder
from src.models.pipelines.pose_estimators.poseflow.human_context import NativeHumanContext
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v53 import native_pose


def native_model():
    torch.manual_seed(560)
    native = DiffusionRegPoseModel(**dict(arguments(), num_encoder_layers=6,
        num_decoder_layers=6, num_box_decoder_layers=2)).eval()
    for head in (*native.pose_embed, *native.bbox_embed):
        torch.nn.init.normal_(head.layers[-1].weight, std=.01)
    return native


def cached_context(network, batch=2):
    torch.manual_seed(561)
    memory = torch.randn(batch, 7, 32, requires_grad=True)
    padding = torch.zeros(batch, 7, dtype=torch.bool)
    from types import SimpleNamespace
    inputs = SimpleNamespace(padding_mask=padding,
        spatial_shapes=torch.tensor([[2, 2], [1, 2], [1, 1]]),
        level_start_index=torch.tensor([0, 4, 6]), valid_ratios=torch.ones(batch, 3, 2))
    return dict(memory=memory, features=torch.randn(batch, 6, 32, requires_grad=True),
        boxes=torch.rand(batch, 6, 4)*.6+.2,
        indices=torch.arange(6).expand(batch, -1)+100,
        pred_logits=torch.randn(batch, 6, 2), inputs=inputs,
        image_values=network.pose_decoder.project_memory(memory, padding))


def test_full_frontend_cached_once_and_native_context_parity(monkeypatch):
    native = native_model()
    reference = deepcopy(native)
    network = FullDepthBirthDecoder(native=native).eval()
    images = torch.randn(2, 3, 96, 128)
    padding = torch.zeros(2, 96, 128, dtype=torch.bool)
    padding[1, 80:] = True
    class NativeReference(NativeHumanContext):
        def __getattr__(self, name):
            return getattr(reference, name)

    expected = NativeReference().prepare_context(images, padding)
    calls = []
    handle = network.frontend.native.transformer.encoder.register_forward_hook(
        lambda *args: calls.append(1))
    actual = network.prepare_context(images, padding)
    for name in ('memory', 'features', 'boxes', 'indices', 'pred_logits'):
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    network.decode_proposals(actual, torch.tensor([[0, 1], [2, 3]]))
    network.decode_proposals(actual, torch.tensor([[4], [5]]))
    handle.remove()
    assert calls == [1]
    assert len(network.frontend.native.transformer.encoder.layers) == 6
    assert len(network.frontend.native.transformer.decoder.layers) == 2


def test_every_pose_layer_matches_native_and_preserves_expansion_head_tie():
    native = native_model()
    reference = deepcopy(native)
    network = FullDepthBirthDecoder(native=native).eval()
    context = cached_context(network)
    identities = torch.tensor([[2, 1, 5], [3, 4, 0]])
    expected = []
    for row in range(3):
        batch = torch.arange(2)
        features = context['features'][batch, identities[:, row]][:, None]
        boxes = context['boxes'][batch, identities[:, row]][:, None]
        expected.append(native_pose(reference, features, boxes, context['memory'],
            context['inputs'].spatial_shapes, context['inputs'].level_start_index,
            context['inputs'].valid_ratios))
    expected = torch.stack(expected, 2)
    output = network.decode_proposals(context, identities, collect_trace=True)
    actual = torch.stack([item['pred_coordinates'] for item in output['trace']])
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(output['pred_coordinates'], expected[-1], atol=2e-6, rtol=2e-5)
    assert network.pose_decoder.expansion_head is network.pose_decoder.pose_heads[0]
    assert [item['layer'] for item in output['trace']] == [1, 2, 3, 4]
    assert torch.equal(output['proposal_indices'], identities)
    assert torch.equal(output['query_indices'], identities+100)
    assert (output['pred_logits'][..., 0] == -30).all()
    assert 'trace' not in network.decode_proposals(context, identities)


def test_zero_prefix_correction_exact_parity_and_trainable_gradient():
    network = FullDepthBirthDecoder(native=native_model()).eval()
    context = cached_context(network)
    identities = torch.tensor([[0, 1], [1, 2]])
    prefix = torch.randn(2, 2, 3, 32)
    valid = torch.tensor([[[True, False, False], [True, True, False]],
                          [[False, False, False], [True, False, False]]])
    expected = network.decode_proposals(context, identities)['pred_coordinates']
    actual = network.decode_proposals(context, identities, prefix=prefix,
        prefix_valid=valid)['pred_coordinates']
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().mean().backward()
    gradient = network.proposal_context.attention.out_proj.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert context['features'].grad.abs().sum() > 0
    assert network.pose_decoder.layers[0].cross_attn.sampling_offsets.weight.grad.abs().sum() > 0
    with torch.no_grad():
        network.proposal_context.attention.out_proj.weight.normal_(std=.01)
    changed = network.decode_proposals(context, identities, prefix=prefix,
        prefix_valid=valid)['pred_coordinates']
    assert not torch.equal(changed, expected)
    perturbed = prefix.clone()
    perturbed[~valid] = float('nan')
    same = network.decode_proposals(context, identities, prefix=perturbed,
        prefix_valid=valid)['pred_coordinates']
    torch.testing.assert_close(same, changed, atol=0, rtol=0)


def test_independent_rows_masks_empty_and_strict_state_restore():
    network = FullDepthBirthDecoder(native=native_model()).eval()
    other = FullDepthBirthDecoder(native=native_model()).eval()
    other.load_state_dict(network.state_dict(), strict=True)
    context = cached_context(network)
    identities = torch.tensor([[1, 1, -1], [2, -1, -1]])
    valid = identities >= 0
    output = network.decode_proposals(context, identities, valid_mask=valid, collect_trace=True)
    assert output['person_counts'].tolist() == [2, 1]
    assert output['pred_coordinates'].shape == (2, 3, 32)
    assert not output['pred_coordinates'][~valid].any()
    assert not output['scores'][~valid].any()
    torch.testing.assert_close(output['pred_coordinates'][:, 0], output['pred_coordinates'][:, 1].where(
        valid[:, 1, None], output['pred_coordinates'][:, 0]), atol=0, rtol=0)
    for row in range(3):
        alone = network.decode_proposals(context, identities[:, row:row+1], valid_mask=valid[:, row:row+1])
        torch.testing.assert_close(alone['pred_coordinates'], output['pred_coordinates'][:, row:row+1], atol=2e-6, rtol=2e-5)
    empty = network.decode_proposals(context, identities[:, :0], valid_mask=valid[:, :0], collect_trace=True)
    assert empty['pred_coordinates'].shape == (2, 0, 32) and len(empty['trace']) == 4
    restored = other.decode_proposals(context, identities, valid_mask=valid)
    torch.testing.assert_close(restored['pred_coordinates'], output['pred_coordinates'], atol=0, rtol=0)
    with pytest.raises(RuntimeError):
        other.load_state_dict({key: value for key, value in network.state_dict().items()
                               if key != 'pose_decoder.pose_heads.0.layers.0.weight'}, strict=True)


def test_reject_sparse_native_and_invalid_proposal_or_prefix():
    with pytest.raises(ValueError, match='full-depth'):
        FullDepthBirthDecoder(native=DiffusionRegPoseModel(**arguments()))
    network = FullDepthBirthDecoder(native=native_model()).eval()
    context = cached_context(network, batch=1)
    for indices in (torch.tensor([[6]]), torch.tensor([[-1]]), torch.tensor([0]), torch.tensor([[0.]])):
        with pytest.raises(ValueError, match='proposal'):
            network.decode_proposals(context, indices)
    with pytest.raises(ValueError, match='prefix'):
        network.decode_proposals(context, torch.tensor([[0]]), prefix=torch.zeros(1, 1, 2, 32))
    with pytest.raises(ValueError, match='prefix'):
        network.decode_proposals(context, torch.tensor([[0]]), prefix=torch.full((1, 1, 2, 32), float('nan')),
            prefix_valid=torch.ones(1, 1, 2, dtype=torch.bool))


def test_independent_row_gradient_parity():
    batched = FullDepthBirthDecoder(native=native_model()).eval()
    separate = deepcopy(batched)
    context = cached_context(batched)
    other_context = cached_context(separate)
    identities = torch.tensor([[0, 2, 1], [3, 5, 2]])
    output = batched.decode_proposals(context, identities, collect_trace=True)
    sum(item['pred_coordinates'].square().sum() for item in output['trace']).backward()
    outputs = [separate.decode_proposals(other_context, identities[:, row:row+1], collect_trace=True)
               for row in range(3)]
    sum(item['pred_coordinates'].square().sum() for result in outputs for item in result['trace']).backward()
    other_parameters = dict(separate.named_parameters())
    for name, parameter in batched.named_parameters():
        other = other_parameters[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-4, rtol=2e-4, msg=name)
    torch.testing.assert_close(context['features'].grad, other_context['features'].grad, atol=2e-5, rtol=2e-4)
