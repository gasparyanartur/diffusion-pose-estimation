from copy import deepcopy

import torch

from src.models.pipelines.pose_estimators.poseflow.model_v48_adaln import PoseFlowV48AdaLNModel
from src.models.pipelines.pose_estimators.poseflow.model_v48 import PoseFlowV48Model
from src.models.pipelines.pose_estimators.poseflow.rows_v48 import training_rows
from src.models.pipelines.pose_estimators.diffusionregpose.model import DiffusionRegPoseModel
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v48 import context


def pair():
    native = DiffusionRegPoseModel(**arguments()).eval()
    options = dict(capacity=6, hidden_dim=32, num_heads=4, feedforward_dim=64, codec_width=32, latent_dim=8)
    start = torch.get_rng_state()
    baseline = PoseFlowV48Model(native=deepcopy(native), **options).eval()
    after_baseline = torch.get_rng_state()
    torch.set_rng_state(start)
    variant = PoseFlowV48AdaLNModel(native=deepcopy(native), **options).eval()
    assert torch.equal(after_baseline, torch.get_rng_state())
    return baseline, variant


def test_initial_common_weights_forward_and_gradients_equal_baseline_exactly():
    base, variant = pair()
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, variant.state_dict()[name], atol=0, rtol=0)
    rows = training_rows(torch.rand(2, 6, 32), torch.tensor([3, 0]))
    raw = dict(memory=torch.randn(2, 7, 32), position=torch.randn(2, 7, 32), padding_mask=torch.zeros(2, 7, dtype=torch.bool))
    outputs = []
    for network in (base, variant):
        output = network.forward_rows(network._project_image_context(**raw), **{key: rows[key] for key in network.row_input_fields})
        sum(value.square().mean() for value in output.values()).backward()
        outputs.append(output)
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=0, rtol=0)
    other = dict(variant.named_parameters())
    for name, value in base.named_parameters():
        if value.grad is not None:
            torch.testing.assert_close(value.grad, other[name].grad, atol=1e-6, rtol=1e-5, msg=name)
    for layer in variant.layers:
        assert layer.image_modulation[-1].weight.grad.abs().sum() > 0
    assert variant.generation_step_embedding.weight.grad.abs().sum() == 0


def test_step_is_prefix_length_not_row_index_and_conditioning_affects_image_attention():
    _, network = pair()
    for layer in network.layers:
        torch.nn.init.normal_(layer.image_modulation[-1].weight, std=.1)
    rows = training_rows(torch.rand(2, 6, 32), torch.tensor([3, 1]))
    image = context(network, 2)
    result = network.forward_rows(image, **{key: rows[key] for key in network.row_input_fields})
    separate = [network.forward_rows(image, **{key: rows[key][:, rank:rank+1] for key in network.row_input_fields})
                for rank in range(4)]
    for key in result:
        torch.testing.assert_close(result[key], torch.cat([value[key] for value in separate], 2), atol=2e-6, rtol=2e-5)
    with torch.no_grad():
        network.generation_step_embedding.weight[2] += 1
    changed = network.forward_rows(image, **{key: rows[key] for key in network.row_input_fields})
    assert not torch.allclose(result['pred_coordinates'][:, 0, 2], changed['pred_coordinates'][:, 0, 2])
    torch.testing.assert_close(result['pred_coordinates'][:, 0, :2], changed['pred_coordinates'][:, 0, :2], atol=0, rtol=0)
    torch.testing.assert_close(result['pred_coordinates'][:, 1], changed['pred_coordinates'][:, 1], atol=0, rtol=0)
    changed['pred_coordinates'].square().mean().backward()
    assert network.generation_step_embedding.weight.grad.abs().sum() > 0


def test_conditioned_generation_remains_deterministic_and_batch_independent():
    from src.models.pipelines.pose_estimators.poseflow.model_v48 import select_context
    _, network = pair()
    for layer in network.layers:
        torch.nn.init.normal_(layer.image_modulation[-1].weight, std=.05)
    for head in network.eos_heads:
        head.weight.data.zero_();head.bias.data.fill_(-5.)
    image = context(network, 2)
    sizes = torch.tensor([[100., 200.], [170., 90.]])
    output = network.predict(image, sizes, collect_trace=True)
    repeated = network.predict(image, sizes)
    torch.testing.assert_close(output['pred_coordinates'], repeated['pred_coordinates'], atol=0, rtol=0)
    assert output['person_counts'].tolist() == [6, 6]
    for index in range(2):
        single = network.predict(select_context(image, torch.tensor([index])), sizes[index:index+1])
        torch.testing.assert_close(output['pred_coordinates'][index], single['pred_coordinates'][0], atol=2e-6, rtol=2e-5)
    for row in output['trace']:
        if row['layer'] == 2:
            rank = row['step']-1
            torch.testing.assert_close(output['pred_coordinates'][:, rank], row['pred_coordinates'], atol=0, rtol=0)
    for head in network.eos_heads:
        head.bias.data.fill_(5.)
    empty = network.predict(image, sizes)
    assert empty['person_counts'].tolist() == [0, 0] and empty['generation_steps'].tolist() == [1, 1]
