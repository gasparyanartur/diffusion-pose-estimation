from copy import deepcopy

import torch

from src.models.pipelines.pose_estimators.diffusionregpose.model import DiffusionRegPoseModel
from src.models.pipelines.pose_estimators.poseflow.model_v50 import PoseFlowV50Model
from tests.test_poseflow_v44 import arguments


def model():
    return PoseFlowV50Model(native=DiffusionRegPoseModel(**arguments()), rows=7, capacity=6,
        hidden_dim=32, num_heads=4, feedforward_dim=64, codec_width=32, latent_dim=8).eval()


def context(instance, batch=2):
    memory = torch.randn(batch, 7, 32)
    result = instance._project_image_context(memory, torch.randn_like(memory), torch.zeros(batch, 7, dtype=torch.bool))
    result.update(features=torch.randn(batch, 6, 32, requires_grad=True), boxes=torch.rand(batch, 6, 4),
        indices=torch.arange(6).repeat(batch, 1), existence_logits=torch.randn(batch, 6))
    return result


def inputs():
    types = torch.tensor([[[2, 0, 0], [1, 2, 0], [1, 1, 2]]]).repeat(2, 1, 1)
    return dict(noisy_coordinates=torch.randn(2, 3, 3, 32), noise_sigmas=torch.where(types == 1, .1, 1.),
        token_types=types, padding_mask=types == 0,
        proposal_indices=torch.tensor([[[0,-1,-1],[0,1,-1],[0,1,2]]]).repeat(2,1,1))


def test_rows_no_future_identity_leak_and_checkpoint_gradients():
    torch.manual_seed(5)
    instance = model().train()
    cached = context(instance)
    values = inputs()
    output = instance.forward_rows(cached, **values)
    altered = {name: value.clone() for name, value in values.items()}
    altered['proposal_indices'][:, 0, 0] = 3
    other = instance.forward_rows(cached, **altered)
    torch.testing.assert_close(output['selection_logits'][:, 0], other['selection_logits'][:, 0])
    assert not torch.allclose(output['pred_coordinates'][:, :, 0, 0], other['pred_coordinates'][:, :, 0, 0])
    direct = deepcopy(instance)
    direct.activation_checkpointing = False
    output['pred_coordinates'].sum().backward()
    direct_context = context_copy(cached)
    direct_context.update(direct._project_image_context(cached['memory'], cached['position'], cached['padding_mask']))
    direct.forward_rows(direct_context, **values)['pred_coordinates'].sum().backward()
    for name, parameter in instance.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, dict(direct.named_parameters())[name].grad, atol=2e-5, rtol=2e-4)
    assert cached['features'].grad is None
    assert all(head.weight.grad.abs().sum() > 0 for head in instance.residual_heads)


def context_copy(cached):
    return {name: tuple(item.detach().clone() for item in value) if isinstance(value, tuple)
        else value.detach().clone() for name, value in cached.items()}


def test_sampling_capacity_ids_determinism_and_exact_terminal_bypass():
    instance = model()
    cached = context(instance, 1)
    with torch.no_grad():
        instance.selector.eos_bias.fill_(-1e6)
    sizes = torch.tensor([[80., 64.]])
    result = instance.predict(cached, sizes, generator=torch.Generator().manual_seed(42), collect_trace=True)
    repeated = instance.predict(cached, sizes, generator=torch.Generator().manual_seed(42))
    assert result['person_counts'].tolist() == [6]
    assert result['stop_reason'].tolist() == [2]
    assert result['proposal_indices'][0].unique().numel() == 6
    torch.testing.assert_close(result['pred_coordinates'], repeated['pred_coordinates'])
    assert len(result['trace']) == 14
    with torch.no_grad():
        instance.selector.eos_bias.fill_(1e6)
    empty = instance.predict(cached, sizes)
    assert empty['person_counts'].tolist() == [0] and empty['eos_row'].tolist() == [1]


def test_native_context_legacy_parity_and_row_isolation():
    from types import MethodType
    from src.models.pipelines.pose_estimators.poseflow.human_context import NativeHumanContext
    torch.manual_seed(83)
    native = DiffusionRegPoseModel(**arguments()).eval()
    legacy = deepcopy(native)
    legacy._decoder_layer = MethodType(NativeHumanContext._decoder_layer, legacy)
    instance = PoseFlowV50Model(native=native, rows=7, capacity=6,
        hidden_dim=32, num_heads=4, feedforward_dim=64, codec_width=32, latent_dim=8).eval()
    assert not any(name.startswith(('transformer.time_mlp.', 'transformer.block_time_mlp.')) for name in instance.frontend.native.state_dict())
    images = torch.randn(1, 3, 64, 80)
    with torch.no_grad():
        expected = NativeHumanContext.prepare_context(legacy, images)
        actual = instance.prepare_context(images)
    for name in ('memory','features','boxes','indices','pred_logits','existence_logits'):
        assert torch.equal(expected[name], actual[name]), name
    instance = model()
    cached, values = context(instance), inputs()
    output = instance.forward_rows(cached, **values)
    for row in range(3):
        one = instance.forward_rows(cached, **{name: value[:,row:row+1] for name,value in values.items()})
        torch.testing.assert_close(output['pred_coordinates'][:,:,row:row+1], one['pred_coordinates'], atol=2e-6, rtol=2e-5)
    changed = {name: value.clone() for name,value in values.items()}
    changed['noisy_coordinates'][values['padding_mask']] = 1e6
    changed['proposal_indices'][values['padding_mask']] = 5
    padded = instance.forward_rows(cached, **changed)
    torch.testing.assert_close(output['pred_coordinates'], padded['pred_coordinates'])


def test_bidirectional_influence_and_per_image_seeded_batch_independence():
    instance = model()
    cached, values = context(instance), inputs()
    first = instance.forward_rows(cached, **values)
    values['noisy_coordinates'][:,2,1] += 1
    changed = instance.forward_rows(cached, **values)
    assert not torch.equal(first['pred_coordinates'][:,:,2,0], changed['pred_coordinates'][:,:,2,0])
    with torch.no_grad():
        instance.selector.eos_bias.fill_(-1e6)
    seeds = [51, 92]
    sizes = torch.tensor([[80.,64.],[64.,80.]])
    together = instance.predict(cached, sizes, generator=[torch.Generator().manual_seed(seed) for seed in seeds])
    for image, seed in enumerate(seeds):
        one_context = {name: tuple(item[image:image+1] for item in value) if isinstance(value,tuple)
            else value[image:image+1] for name,value in cached.items()}
        alone = instance.predict(one_context, sizes[image:image+1], generator=torch.Generator().manual_seed(seed))
        torch.testing.assert_close(together['pred_coordinates'][image:image+1], alone['pred_coordinates'], atol=3e-6, rtol=3e-5)
        assert torch.equal(together['proposal_indices'][image:image+1], alone['proposal_indices'])
