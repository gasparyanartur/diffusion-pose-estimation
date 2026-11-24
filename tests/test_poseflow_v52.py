from copy import deepcopy
from types import SimpleNamespace

import torch

from src.models.pipelines.pose_estimators.diffusionregpose.model import DiffusionRegPoseModel
from src.models.pipelines.pose_estimators.poseflow.model_v52 import PoseFlowV52Model, decode_joint_geometry
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v50 import inputs


def model():
    return PoseFlowV52Model(native=DiffusionRegPoseModel(**arguments()), rows=7, capacity=6,
        hidden_dim=32, num_heads=4, feedforward_dim=64, latent_dim=32, codec_width=32).eval()


def context(network, batch=2):
    memory = torch.randn(batch, 7, 32)
    spatial_shapes = torch.tensor([[2, 2], [1, 2], [1, 1]])
    result = network._project_image_context(memory, torch.zeros_like(memory), torch.zeros(batch, 7, dtype=torch.bool))
    result.update(inputs=SimpleNamespace(spatial_shapes=spatial_shapes,
        level_start_index=torch.tensor([0, 4, 6]), valid_ratios=torch.ones(batch, 3, 2)),
        features=torch.randn(batch, 6, 32, requires_grad=True), boxes=torch.rand(batch, 6, 4),
        indices=torch.arange(6).repeat(batch, 1), existence_logits=torch.randn(batch, 6))
    return result


def test_joint_clean_readout_has_no_codec_and_relative_pose_uses_center():
    network = model()
    assert not hasattr(network, 'person_codec') and not hasattr(network, 'residual_heads')
    raw = torch.zeros(2, 32, requires_grad=True)
    decoded = decode_joint_geometry(raw)
    torch.testing.assert_close(decoded[:, :28], decoded[:, 28:30].repeat(1, 14))
    assert (decoded[:, 30:] > 0).all()
    decoded[:, :28].sum().backward()
    assert raw.grad[:, 28:30].abs().sum() > 0
    result = network.forward_rows(context(network), **inputs())
    assert result['pred_features'].shape == (2, 2, 3, 3, 32)
    assert result['pred_coordinates'].shape == (2, 2, 3, 3, 32)


def test_current_boxes_and_time_drive_sampling_and_layer_handoff():
    network = model()
    cached, values = context(network), inputs()
    observed = []
    def capture(layer, args):
        observed.append((args[1].detach().clone(), args[0].detach().clone()))
    handles = [layer.image_attention.register_forward_pre_hook(capture) for layer in network.layers]
    result = network.forward_rows(cached, **values)
    first_references = observed[0][0]
    torch.testing.assert_close(observed[1][0][:, :, 0], result['pred_coordinates'][0, ..., 28:].reshape(2, -1, 4))
    changed = {name: value.clone() for name, value in values.items()}
    changed['noisy_coordinates'][:, 2, 0, 28] += .5
    network.forward_rows(cached, **changed)
    assert not torch.equal(first_references[:, 6], observed[2][0][:, 6])
    old_query = observed[0][1]
    changed = {name: value.clone() for name, value in values.items()}
    changed['noise_sigmas'][changed['token_types'] == 1] = .05
    network.forward_rows(cached, **changed)
    assert not torch.equal(old_query, observed[4][1])
    for handle in handles:
        handle.remove()


def test_parallel_row_and_gradient_parity_padding_and_classifier_boundary():
    torch.manual_seed(73)
    network = model().train()
    cached, values = context(network), inputs()
    output = network.forward_rows(cached, **values)
    for row in range(3):
        one = network.forward_rows(cached, **{name: value[:, row:row+1] for name, value in values.items()})
        torch.testing.assert_close(output['pred_coordinates'][:, :, row:row+1], one['pred_coordinates'], atol=2e-6, rtol=2e-5)
    changed = {name: value.clone() for name, value in values.items()}
    changed['noisy_coordinates'][values['padding_mask']] = float('nan')
    changed['proposal_indices'][values['padding_mask']] = 5
    other = network.forward_rows(cached, **changed)
    torch.testing.assert_close(output['pred_coordinates'], other['pred_coordinates'], rtol=0, atol=0)
    direct = deepcopy(network)
    direct.activation_checkpointing = False
    copied = dict(cached, **direct._project_image_context(cached['memory'], cached['position'], cached['padding_mask']))
    output['pred_coordinates'].square().mean().backward()
    direct.forward_rows(copied, **values)['pred_coordinates'].square().mean().backward()
    for name, parameter in network.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, dict(direct.named_parameters())[name].grad, atol=3e-5, rtol=3e-4)
    assert all(head[-1].weight.grad.abs().sum() > 0 for head in network.geometry_heads)
    assert cached['features'].grad is None
    network.zero_grad(set_to_none=True)
    output = network.forward_rows(cached, **values)
    (output['selection_logits'].square().mean()+output['eos_logits'].square().mean()).backward()
    assert network.selector.key.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in network.coordinate_input.parameters())


def test_sampling_eos_capacity_and_seeded_batch_independence():
    network = model()
    cached = context(network)
    network.selector.eos_bias.data.fill_(-1e6)
    sizes = torch.tensor([[80., 64.], [64., 80.]])
    result = network.predict(cached, sizes, generator=[torch.Generator().manual_seed(i) for i in (42, 61)], collect_trace=True)
    assert result['person_counts'].tolist() == [6, 6]
    assert result['stop_reason'].tolist() == [2, 2]
    assert len(result['trace']) == 14 and torch.isfinite(result['pred_coordinates']).all()
    for image, seed in enumerate((42, 61)):
        single = {name: tuple(item[image:image+1] for item in value) if isinstance(value, tuple)
            else value if name == 'inputs' else value[image:image+1] for name, value in cached.items()}
        single['inputs'] = SimpleNamespace(spatial_shapes=cached['inputs'].spatial_shapes,
            level_start_index=cached['inputs'].level_start_index, valid_ratios=cached['inputs'].valid_ratios[image:image+1])
        alone = network.predict(single, sizes[image:image+1], generator=torch.Generator().manual_seed(seed))
        torch.testing.assert_close(result['pred_coordinates'][image:image+1], alone['pred_coordinates'], atol=5e-6, rtol=3e-5)
        assert torch.equal(result['proposal_indices'][image:image+1], alone['proposal_indices'])
    network.selector.eos_bias.data.fill_(1e6)
    assert network.predict(cached, sizes)['person_counts'].tolist() == [0, 0]


def test_native_frontend_parity_and_cached_deformable_read_parity():
    from types import MethodType
    from src.models.pipelines.pose_estimators.poseflow.human_context import NativeHumanContext
    from src.models.pipelines.pose_estimators.diffusionregpose.transformer import MultiScaleDeformableAttention
    native = DiffusionRegPoseModel(**arguments()).eval()
    legacy = deepcopy(native)
    legacy._decoder_layer = MethodType(NativeHumanContext._decoder_layer, legacy)
    network = PoseFlowV52Model(native=native, rows=7, capacity=6,
        hidden_dim=32, num_heads=4, feedforward_dim=64)
    image = torch.randn(1, 3, 64, 80)
    network.eval()
    with torch.no_grad():
        expected = NativeHumanContext.prepare_context(legacy, image)
        actual = network.prepare_context(image)
    for name in ('memory', 'features', 'boxes', 'indices', 'pred_logits'):
        assert torch.equal(expected[name], actual[name]), name
    cached = context(network)
    layer = network.layers[0]
    original = MultiScaleDeformableAttention(32, 3, 4, 4)
    original.load_state_dict(dict(layer.image_attention.state_dict(),
        **{f'value_proj.{name}': value for name, value in layer.image_projection.state_dict().items()}))
    query = torch.randn(2, 4, 32)
    references = torch.rand(2, 4, 3, 4)
    spatial = cached['inputs']
    full = original(query, references, cached['memory'], spatial.spatial_shapes, spatial.level_start_index, cached['padding_mask'])
    projected = layer.image_attention(query, references, cached['image_values'][0], spatial.spatial_shapes, spatial.level_start_index)
    torch.testing.assert_close(full, projected, rtol=0, atol=0)


def test_first_eos_snapshot_uses_same_trajectory_and_birth_scores():
    from src.models.pipelines.pose_estimators.poseflow.quick_eval_v50 import first_eos_geometry_output
    network = model()
    coordinates = torch.rand(1, 6, 32)
    counts = torch.tensor([2])
    scores = torch.rand(1, 6)
    sizes = torch.tensor([[80., 64.]])
    output = network._geometry_output(coordinates, counts, scores, sizes)
    snapshot = coordinates+.1
    output.update(eos_row=torch.tensor([3]), trace=[dict(row=3, layer=2, pred_coordinates=snapshot)])
    stopped = first_eos_geometry_output(network, output, sizes)
    torch.testing.assert_close(stopped['pred_coordinates'], snapshot)
    torch.testing.assert_close(stopped['scores'], output['scores'])
