import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.models.pipelines.pose_estimators.poseflow.training_v22 import PoseFlowV22PipelineConfig
from tests.test_poseflow import _batch
from tests.test_poseflow_v21 import configuration as v21_configuration


def configuration(**overrides):
    return PoseFlowV22PipelineConfig(**(v21_configuration(hypothesis_update_order='h_then_g').model_dump() | overrides))


def test_identity_adapters_preserve_v21_with_matching_time_scale():
    base = v21_configuration(hypothesis_update_order='h_then_g')._build_poseflow_model().eval()
    model = configuration(time_embedding_scale=1)._build_poseflow_model().eval()
    result = model.load_state_dict(base.state_dict(), strict=False)
    assert result.missing_keys and not result.unexpected_keys
    image, state = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 4)
    times = torch.tensor([.2, .7])
    expected, actual = base(image, state, times), model(image, state, times)
    for name in ('pred_logits', 'pred_boxes', 'pred_keypoints', 'hypothesis_memory'):
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


def test_coordinate_attention_explicit_time_and_adapter_gradients():
    model = configuration()._build_poseflow_model()
    noisy, reference = torch.rand(3, 2, 2, 2), torch.rand(3, 2, 2, 2)
    times = torch.tensor([.1, .8])
    original = model.transformer.decoder._refine_diffusion_keypoints(noisy, reference)
    actual = model.refine_diffusion_coordinates(noisy, reference, times)
    torch.testing.assert_close(actual, original, atol=3e-7, rtol=1e-6)
    with torch.no_grad():
        model.coordinate_conditioners['self_q'].affine.weight.normal_(std=.05)
        model.coordinate_conditioners['fusion_q'].affine.weight.normal_(std=.05)
    early = model.refine_diffusion_coordinates(noisy, reference, torch.full((2,), .1))
    late = model.refine_diffusion_coordinates(noisy, reference, torch.full((2,), .8))
    assert not torch.allclose(early, late)
    actual = model.refine_diffusion_coordinates(noisy, reference, times)
    actual.square().sum().backward()
    for adapter in model.coordinate_conditioners.values():
        assert adapter.affine.weight.grad is not None
        assert torch.isfinite(adapter.affine.weight.grad).all()
        assert adapter.affine.weight.grad.abs().sum() > 0


def test_real_criterion_final_h_and_conditioning_gradients():
    pipeline = configuration(dn_number=2).build(max_epochs=2).train()
    with torch.no_grad():
        pipeline.model.dense_read.gate.fill_(.1)
    captured = []
    hook = pipeline.model.register_forward_hook(lambda module, args, output: captured.append(output))
    loss = pipeline._losses_for_batch(_batch() | {'image': torch.randn(1, 3, 64, 64)})['total']
    hook.remove()
    gradient = torch.autograd.grad(loss, captured[0]['hypothesis_memory'], retain_graph=True)[0]
    assert gradient.abs().sum() > 0 and torch.isfinite(gradient).all()
    loss.backward()
    for group in (pipeline.model.conditioners, pipeline.model.proposal_conditioner,
                  pipeline.model.dense_query_conditioner, pipeline.model.coordinate_conditioners):
        gradients = [p.grad for p in group.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    for name in ('emb_SQ', 'emb_SK', 'emb_SV', 'emb_Q', 'emb_K', 'emb_V'):
        assert all(p.grad is None for p in getattr(pipeline.model.transformer.decoder, name).parameters())


def test_time_scale_and_strict_restore_and_slot_carry(tmp_path):
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    config = configuration()
    assert config.time_embedding_scale == 999
    with pytest.raises(ValueError):
        configuration(time_embedding_scale=0)
    pipeline = config.build(max_epochs=2).eval()
    torch.save({'state_dict': pipeline.state_dict()}, tmp_path/'model.ckpt')
    OmegaConf.save(OmegaConf.create({'pipeline': config.model_dump() | {
        '_target_': 'src.models.pipelines.pose_estimators.poseflow.PoseFlowV22PipelineConfig'}}), tmp_path/'config.yaml')
    restored, _ = load_poseflow_model(tmp_path/'model.ckpt', tmp_path/'config.yaml')
    image = torch.randn(1, 3, 64, 64)
    expected_memory = [None]
    def trace(module, args, kwargs):
        assert kwargs.get('hypothesis_memory') is expected_memory[0]
    def remember(module, args, output):
        expected_memory[0] = output['hypothesis_memory']
    pre = restored.register_forward_pre_hook(trace, with_kwargs=True)
    post = restored.register_forward_hook(remember)
    restored.eval(); restored.flow_sampling_steps = 3
    output = restored.sample_keypoints(image, generator=torch.Generator().manual_seed(42))
    pre.remove(); post.remove()
    pipeline.model.flow_sampling_steps = 3
    native = pipeline.model.sample_keypoints(image, generator=torch.Generator().manual_seed(42))
    for name in ('pred_boxes', 'pred_keypoints', 'hypothesis_memory', 'flow_keypoints'):
        torch.testing.assert_close(output[name], native[name], atol=0, rtol=0)


def test_decoder_time_response_with_fixed_image_geometry_and_hypothesis():
    model = configuration()._build_poseflow_model().eval()
    image = torch.randn(1, 3, 64, 64)
    inputs = model.prepare_transformer_inputs(image, torch.zeros(1, 64, 64, dtype=torch.bool))
    memory = model.encode_features(inputs, torch.tensor([.4]))
    geometry, hypotheses = torch.randn(1, 3, 32), torch.randn(1, 3, 32)
    references = torch.rand(1, 3, 4)*.5+.25
    layer = model.transformer.decoder.layers[0]
    mask = model.hypothesis_attention_mask(batch_size=1, dn_count=0, expanded=False, device=image.device)
    def refine(time):
        return model.refine_with_hypotheses(layer, geometry, references, hypotheses, inputs, memory,
                                            mask, timesteps=torch.tensor([time]))
    early, late = refine(.1), refine(.8)
    for first, second in zip(early, late):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    with torch.no_grad():
        for conditioner in model.conditioners[0].values():
            conditioner.affine.weight.normal_(std=.01)
    early, late = refine(.1), refine(.8)
    assert all(not torch.allclose(first, second) for first, second in zip(early, late))


def test_only_embedding_time_is_rescaled():
    model = configuration()._build_poseflow_model().eval()
    seen = []
    hook = model.transformer.time_mlp.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    model(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 4), torch.tensor([.25]))
    hook.remove()
    assert len(seen) > 1 and all(torch.equal(value, torch.tensor([249.75])) for value in seen)
    torch.testing.assert_close(model.dense_strength(torch.tensor([.25])), torch.tensor([.325]))
