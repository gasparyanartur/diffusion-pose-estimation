from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_estimators.diffusionregpose.model import DiffusionRegPoseModel
from src.models.pipelines.pose_estimators.poseflow.model_v49 import PoseFlowV49Model
from src.models.pipelines.pose_estimators.poseflow.sampling_v49 import ddim_transition
from src.models.pipelines.pose_estimators.poseflow.rows_token_age import build_training_rows
from tests.test_poseflow_v44 import arguments


def model():
    return PoseFlowV49Model(native=DiffusionRegPoseModel(**arguments()), rows=7, capacity=6,
        hidden_dim=32, num_heads=4, feedforward_dim=64, codec_width=32, latent_dim=8).eval()


def context(network, batch=1):
    return network._project_image_context(torch.randn(batch, 7, 32), torch.randn(batch, 7, 32),
                                         torch.zeros(batch, 7, dtype=torch.bool))


def test_ddim_oracle_pure_noise_and_tiny_sigma_endpoints():
    clean, noise = torch.randn(3, 32), torch.randn(3, 32)
    current, following = torch.tensor([1., .1, 1e-12])[:, None], torch.tensor([.1, .05, 0.])[:, None]
    state = (1-current.square()).sqrt()*clean+current*noise
    expected = (1-following.square()).sqrt()*clean+following*noise
    actual = ddim_transition(state, clean, current, following)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(ddim_transition(clean, clean, 0., 0.), clean)
    with pytest.raises(ValueError, match='sigma'):
        ddim_transition(state, clean, .1, .2)


def test_parallel_rows_and_gradients_match_separate_rows_and_exclude_padding():
    network = model()
    other = deepcopy(network)
    rows = build_training_rows(torch.rand(2, 6, 32), torch.tensor([3, 1]), row_count=7)
    raw = dict(memory=torch.randn(2, 7, 32), position=torch.randn(2, 7, 32), padding_mask=torch.zeros(2, 7, dtype=torch.bool))
    output = network.forward_rows(network._project_image_context(**raw), **{k: rows[k] for k in network.row_input_fields})
    second_context = other._project_image_context(**raw)
    separate = [other.forward_rows(second_context, **{k: rows[k][:, r:r+1] for k in other.row_input_fields}) for r in range(7)]
    for key in output:
        torch.testing.assert_close(output[key], torch.cat([item[key] for item in separate], 2), atol=2e-6, rtol=2e-5)
    output['pred_coordinates'].square().mean().backward()
    torch.cat([item['pred_coordinates'] for item in separate], 2).square().mean().backward()
    for name, parameter in network.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, dict(other.named_parameters())[name].grad, atol=3e-6, rtol=3e-4)
    assert network.time_embedding[1].weight.grad.abs().sum() > 0
    bad = dict(rows, noisy_coordinates=rows['noisy_coordinates'].masked_fill(rows['padding_mask'][..., None], float('nan')))
    safe = network.forward_rows(network._project_image_context(**raw), **{k: bad[k] for k in network.row_input_fields})
    torch.testing.assert_close(output['pred_coordinates'], safe['pred_coordinates'], atol=0, rtol=0)
    changed = rows['noisy_coordinates'].clone()
    changed[0, 3, 2] += 10
    result = network.forward_rows(network._project_image_context(**raw), changed, rows['noise_sigmas'], rows['token_types'], rows['padding_mask'])
    assert not torch.allclose(result['pred_coordinates'][:, 0, 3, 0], output['pred_coordinates'][:, 0, 3, 0])
    torch.testing.assert_close(result['pred_coordinates'][:, :, :3], output['pred_coordinates'][:, :, :3], atol=0, rtol=0)


def test_checkpointed_gradients_and_newborn_noise_influence():
    network = model().train()
    other = deepcopy(network)
    other.activation_checkpointing = False
    rows = build_training_rows(torch.rand(1, 6, 32), torch.tensor([2]), row_count=7)
    raw = dict(memory=torch.randn(1, 7, 32), position=torch.randn(1, 7, 32), padding_mask=torch.zeros(1, 7, dtype=torch.bool))
    outputs = []
    for instance in (network, other):
        out = instance.forward_rows(instance._project_image_context(**raw), **{k: rows[k] for k in instance.row_input_fields})
        out['pred_coordinates'].square().mean().backward()
        outputs.append(out)
    for name, parameter in network.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, dict(other.named_parameters())[name].grad, atol=3e-6, rtol=3e-4)
    for head in network.latent_heads:
        assert head.weight.grad.abs().sum() > 0
    assert network.person_codec.encoder[0].weight.grad.abs().sum() > 0
    altered = rows['noisy_coordinates'].clone()
    altered[:, 0, 0] += 10
    changed = other.forward_rows(other._project_image_context(**raw), altered, rows['noise_sigmas'], rows['token_types'], rows['padding_mask'])
    assert not torch.allclose(changed['pred_coordinates'][:, :, 0, 0], outputs[1]['pred_coordinates'][:, :, 0, 0])


def test_seeded_sampler_is_batch_independent_runs_tail_and_honors_capacity():
    network = model()
    for head in network.eos_heads:
        head.weight.data.zero_()
        head.bias.data.fill_(-20.)
    image = context(network, 2)
    sizes = torch.tensor([[100., 200.], [170., 90.]])
    generate = lambda: [torch.Generator().manual_seed(10+i) for i in range(2)]
    output = network.predict(image, sizes, generator=generate(), collect_trace=True)
    repeated = network.predict(image, sizes, generator=generate())
    assert output['person_counts'].tolist() == [6, 6]
    assert output['stop_reason'].tolist() == [2, 2]
    assert len(output['trace']) == 14 and output['generation_steps'].tolist() == [7, 7]
    assert not output['trace'][-1]['birth'].any()
    assert 'trace' not in repeated
    torch.testing.assert_close(output['pred_coordinates'], repeated['pred_coordinates'], atol=0, rtol=0)
    for index in range(2):
        selected = {k: tuple(x[index:index+1] for x in v) if isinstance(v, tuple) else v[index:index+1] for k,v in image.items()}
        one = network.predict(selected, sizes[index:index+1], generator=torch.Generator().manual_seed(10+index))
        torch.testing.assert_close(one['pred_coordinates'][0], output['pred_coordinates'][index], atol=3e-6, rtol=3e-5)


def test_eos_latches_even_if_later_logits_request_births(monkeypatch):
    network = model()
    calls = []
    def controlled(context, noisy_coordinates, noise_sigmas, token_types, padding_mask):
        calls.append(noisy_coordinates.clone())
        shape = noisy_coordinates.shape
        return dict(pred_coordinates=torch.ones(2,*shape),
                    eos_logits=torch.full((2,*shape[:-1]), 20. if len(calls)==2 else -20.))
    monkeypatch.setattr(network, 'forward_rows', controlled)
    output = network.predict(context(network), torch.tensor([[100., 200.]]), generator=torch.Generator().manual_seed(9), collect_trace=True)
    assert output['person_counts'].item() == 1 and output['eos_row'].item() == 2
    assert output['stop_reason'].item() == 1 and len(calls) == 7
    assert output['trace'][0]['birth'].item()
    assert not any(item['birth'].item() for item in output['trace'][2:])
    initial = calls[0][0, 0, 0]
    expected = (1-.1**2)**.5*6+initial*.1
    torch.testing.assert_close(calls[1][0, 0, 0], expected)
