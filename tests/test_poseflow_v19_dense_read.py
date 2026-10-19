from dataclasses import replace

import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.dense_read_v19 import (
    DenseHypothesisRead, PoseFlowV19DenseReadModel,
)
from src.models.pipelines.pose_estimators.poseflow.model_v14 import V14EncodedImage
from src.models.pipelines.pose_estimators.poseflow.model_v19 import PoseFlowV19Model
from tests.test_poseflow_v19 import source_model


def image_bundle(batch=3, width=32):
    return V14EncodedImage(
        memory=torch.randn(batch, 7, width),
        padding_mask=torch.tensor([[False]*5+[True]*2]*batch),
        spatial_shapes=torch.tensor([[1, 7]]),
        level_start_index=torch.tensor([0]), valid_ratios=torch.ones(batch, 1, 2),
        position=torch.randn(batch, 7, width),
    )


def model_pair():
    source = source_model()
    for head in list(source.bbox_embed)+list(source.pose_embed):
        torch.nn.init.normal_(head.layers[-1].weight, std=.01)
    baseline = PoseFlowV19Model(source).eval()
    baseline.detach_decoder_geometry = True
    dense = PoseFlowV19DenseReadModel(source).eval()
    report = dense.load_baseline_state_dict(baseline.state_dict())
    assert report['new_keys'] and all(key.startswith('dense_read.') for key in report['new_keys'])
    assert report['loaded_tensors'] == len(baseline.state_dict())
    return baseline, dense


def test_dense_read_exact_time_arithmetic_and_masked_positioned_keys():
    torch.manual_seed(42)
    read = DenseHypothesisRead(32, 4).eval()
    read.gate.data.fill_(.25)
    image = image_bundle()
    memory = torch.randn(1, 3, 32).expand(3, -1, -1).clone()
    image = replace(image, memory=image.memory[:1].expand_as(image.memory).clone(),
                    position=image.position[:1].expand_as(image.position).clone())
    times = torch.tensor([0., .5, 1.])
    output, diagnostics = read(memory, times, image, collect_diagnostics=True)
    message = read.attention(read.query_norm(memory), image.memory+image.position, image.memory,
                             key_padding_mask=image.padding_mask, need_weights=False)[0]
    expected = memory+.25*torch.tensor([1., .55, .1])[:, None, None]*message
    torch.testing.assert_close(output, expected)
    assert diagnostics['dense_read/gate'] == .25
    corrupted = replace(image, memory=image.memory.clone(), position=image.position.clone())
    corrupted.memory[:, 5:] = 1000
    corrupted.position[:, 5:] = -1000
    actual, _ = read(memory, times, corrupted)
    torch.testing.assert_close(actual, output)
    shifted, _ = read(memory, times, replace(image, position=torch.zeros_like(image.position)))
    assert not torch.allclose(shifted, output)
    changed = image.memory.clone()
    changed[0, 0] += 10
    isolated, _ = read(memory, times, replace(image, memory=changed), collect_diagnostics=True)
    torch.testing.assert_close(isolated[1:], output[1:], atol=0, rtol=0)
    assert not torch.allclose(isolated[0], output[0])
    with pytest.raises(ValueError, match='valid image token'):
        read(memory, times, replace(image, padding_mask=torch.ones_like(image.padding_mask)))
    with pytest.raises(ValueError, match='position'):
        read(memory, times, replace(image, position=None))


@pytest.mark.parametrize('collect_diagnostics', [False, True])
def test_zero_gate_preserves_baseline_outputs_and_baseline_gradients(collect_diagnostics):
    torch.manual_seed(42)
    baseline, dense = model_pair()
    image, state, times = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 8), torch.ones(1)
    expected = baseline.denoise(state, times, baseline.encode_image(image),
                                collect_diagnostics=collect_diagnostics)
    actual = dense.denoise(state, times, dense.encode_image(image),
                          collect_diagnostics=collect_diagnostics)
    for left, right in zip(expected['aux_outputs']+[expected], actual['aux_outputs']+[actual]):
        for key in ('pred_coordinates', 'pred_boxes', 'pred_keypoints', 'pred_logits'):
            torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
    torch.testing.assert_close(expected['hypothesis_memory'], actual['hypothesis_memory'], atol=0, rtol=0)
    def objective(output):
        return sum(stage['pred_coordinates'].square().mean()+stage['pred_logits'].square().mean()
                   for stage in output['aux_outputs']+[output])
    objective(expected).backward()
    objective(actual).backward()
    for name, parameter in baseline.named_parameters():
        other = dict(dense.named_parameters())[name]
        if parameter.grad is None:
            assert other.grad is None
        else:
            torch.testing.assert_close(parameter.grad, other.grad, atol=0, rtol=0)
    assert dense.dense_read.gate.item() == 0
    assert dense.dense_read.gate.grad.abs() > 0
    assert dense.dense_read.attention.in_proj_weight.grad.count_nonzero() == 0
    dense.zero_grad(set_to_none=True)
    dense.dense_read.gate.data.fill_(.1)
    updated = dense.denoise(state, times, dense.encode_image(image))
    objective(updated).backward()
    assert dense.dense_read.attention.in_proj_weight.grad.abs().sum() > 0
    assert dense.dense_read.attention.out_proj.weight.grad.abs().sum() > 0


def test_dense_read_once_per_call_and_memory_carry_without_leakage():
    from src.models.pipelines.pose_estimators.poseflow.latent_v12 import ddim_update
    _, model = model_pair()
    model.dense_read.gate.data.fill_(.2)
    image, state = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 8)
    encoded = model.encode_image(image)
    reads = []
    hook = model.dense_read.register_forward_pre_hook(lambda module, args: reads.append(args[0]))
    memory, current = None, state
    for index in range(3):
        times = torch.tensor([1-index/3])
        output = model.denoise(current, times, encoded, hypothesis_memory=memory)
        if index == 0:
            first = output
        else:
            assert reads[-1] is memory
        memory = output['hypothesis_memory']
        current = ddim_update(current, output['pred_coordinates'], times, torch.tensor([1-(index+1)/3]))
    assert len(reads) == 3
    gradients = torch.autograd.grad(output['pred_coordinates'].square().sum(),
                                    (first['pred_coordinates'], first['hypothesis_memory']))
    assert all(gradient.abs().sum() > 0 for gradient in gradients)
    with torch.no_grad():
        sampled = model.sample_keypoints(image, initial_latents=state, sampling_steps=3, encoded_image=encoded)
        repeated = model.sample_keypoints(image, initial_latents=state, sampling_steps=3, encoded_image=encoded)
    torch.testing.assert_close(sampled['coordinate_state'], current)
    torch.testing.assert_close(sampled['hypothesis_memory'], memory)
    torch.testing.assert_close(sampled['hypothesis_memory'], repeated['hypothesis_memory'], atol=0, rtol=0)
    hook.remove()
    invalid = dict(model.state_dict())
    with pytest.raises(ValueError, match='baseline'):
        model.load_baseline_state_dict(invalid)


def test_open_dense_read_candidate_equivariance_and_gate_only_changes_features():
    _, model = model_pair()
    model.dense_read.gate.data.fill_(.3)
    image, state = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 8)
    encoded = model.encode_image(image)
    bank, order = torch.randn(1, 3, 32), torch.tensor([2, 0, 1])
    references = []
    handle = model.layers[0].register_forward_pre_hook(lambda module, args: references.append(args[2]))
    output = model.denoise(state, torch.ones(1), encoded, hypothesis_memory=bank)
    permuted = model.denoise(state[:, order], torch.ones(1), encoded, hypothesis_memory=bank[:, order])
    for key in ('pred_coordinates', 'pred_logits', 'hypothesis_memory'):
        torch.testing.assert_close(output[key][:, order], permuted[key], atol=3e-5, rtol=3e-5)
    model.dense_read.gate.data.zero_()
    model.denoise(state, torch.ones(1), encoded, hypothesis_memory=bank)
    torch.testing.assert_close(references[0], references[-1], atol=0, rtol=0)
    handle.remove()
