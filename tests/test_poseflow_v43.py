from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.model_v43 import PoseFlowV43Model
from tests.test_poseflow_v43_codec import small_codec


def model():
    return PoseFlowV43Model(codec=small_codec(), backbone_pretrained=False,
        hidden_dim=32, num_heads=4, feedforward_dim=64, num_proposals=6,
        reference_candidates=12, reference_encoder_layers=1).eval()


def test_frozen_decoder_is_only_geometry_path_and_accepts_heterogeneous_metadata():
    network = model()
    images = torch.randn(2, 3, 128, 128)
    image_wh = torch.tensor([[128., 100.], [90., 128.]])
    state = torch.randn(2, 6, 8, requires_grad=True)
    output = network(images, state, torch.tensor([0, 999]), image_wh=image_wh)
    assert output['pred_keypoints'].shape == (2, 6, 42)
    assert output['pred_latents'].shape == (2, 6, 8)
    assert (output['pred_boxes'][..., 2:] > 0).all()
    expected = network.codec.decode_latents(output['pred_latents'], image_wh[:, None].expand(-1, 6, -1))
    torch.testing.assert_close(output['pred_boxes'], expected['boxes_cxcywh_px']/image_wh[:, None].repeat(1, 1, 2))
    output['keypoints_px'].square().mean().backward()
    assert torch.isfinite(state.grad).all() and state.grad.abs().sum() > 0
    assert network.latent_output.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in network.codec.parameters())
    assert not hasattr(network, 'clean_layers') and not hasattr(network, 'box_head')
    with pytest.raises(ValueError, match='shape'):
        network(images, state[..., :4], torch.tensor([0, 999]), image_wh=image_wh)


def test_sampler_caches_frontend_scores_and_never_calls_encoder(monkeypatch):
    network = model()
    images = torch.randn(1, 3, 128, 128)
    image_wh = torch.tensor([[128., 128.]])
    calls, references, outputs = [], [], []
    original_propose, original_denoise = network.propose, network.denoise
    def propose(*args, **kwargs):
        calls.append(1)
        return original_propose(*args, **kwargs)
    def denoise(*args, **kwargs):
        references.append(kwargs['references'])
        return original_denoise(*args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('Inference must not encode geometry')
    monkeypatch.setattr(network, 'propose', propose)
    monkeypatch.setattr(network, 'denoise', denoise)
    monkeypatch.setattr(network.codec.model, 'encode', forbidden)
    for steps, seed in ((1, 42), (4, 42), (8, 1042)):
        calls.clear(); references.clear()
        network.flow_sampling_steps = steps
        output = network.sample_keypoints(images, image_wh=image_wh,
            generator=torch.Generator().manual_seed(seed))
        assert len(calls) == 1 and len(references) == steps
        assert all(reference is references[0] for reference in references)
        assert output['latent_state'].shape == (1, 6, 8)
        assert all(torch.isfinite(output[key]).all() for key in ('pred_logits', 'pred_boxes', 'pred_keypoints'))
        outputs.append(output)
    for output in outputs[1:]:
        torch.testing.assert_close(output['pred_logits'], outputs[0]['pred_logits'], rtol=0, atol=0)


def test_diffusion_oracle_and_embedded_codec_restore():
    network = model()
    clean, noise = torch.randn(2, 6, 8), torch.randn(2, 6, 8)
    times = torch.tensor([0, 999])
    noisy = network.q_sample(clean, times, noise)
    recovered = network.predict_noise_from_start(noisy, times, clean)
    torch.testing.assert_close(recovered.float(), noise, rtol=2e-3, atol=2e-3)
    for time, next_time in network.ddim_timestep_pairs():
        alpha = network.alphas_cumprod[time].float()
        state = alpha.sqrt()*clean+(1-alpha).sqrt()*noise
        predicted_noise = network.predict_noise_from_start(state, torch.full((2,), time), clean).float()
        advanced = network.ddim_step(clean, predicted_noise, time, next_time, noise=torch.zeros_like(clean))
        expected = clean if next_time == -1 else (
            network.alphas_cumprod[next_time].float().sqrt()*clean
            +(1-network.alphas_cumprod[next_time].float()).sqrt()*noise)
        torch.testing.assert_close(advanced, expected, atol=1e-5, rtol=1e-5)
    restored = model()
    restored.load_state_dict(deepcopy(network.state_dict()), strict=True)
    for key, value in network.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)
