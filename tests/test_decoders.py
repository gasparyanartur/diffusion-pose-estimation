"""Unit tests for features, embedders, decoders, denoise networks."""

from __future__ import annotations

import torch

from src.data.graph import prepare_pose_graph_batch, prepare_sample
from src.decoders.denoise.graph_network import (
    FiLMLayer,
    GraphDenoiseNetwork,
    GraphDenoiseNetworkConfig,
    PoseGraphLayer,
)
from src.decoders.denoise.mlp_network import (
    MlpDenoiseNetwork,
    MlpDenoiseNetworkConfig,
)
from src.decoders.diffusion.base import TimestepSinusoidalEmbedding
from src.decoders.diffusion.graph import (
    GraphDiffusionDecoder,
    GraphDiffusionDecoderConfig,
)
from src.dummy import DUMMY_KEYPOINTS_PIXEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_graph(B: int = 2, N: int = 1, K: int = 14):
    keypoints = DUMMY_KEYPOINTS_PIXEL.view(1, 1, K, 2).expand(B, N, K, 2).contiguous()
    visibility = torch.full((B, N, K), 2.0)
    batch = {
        "keypoints": keypoints,
        "visibility": visibility,
        "image_id": torch.arange(B, dtype=torch.long),
        "img_path": [f"<dummy:{i}>" for i in range(B)],
    }
    sample = prepare_sample(batch)
    img_wh = torch.full((B, 2), 256.0)
    return sample, prepare_pose_graph_batch(sample, img_size=img_wh)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def test_timestep_embedding_shape_and_finite() -> None:
    emb = TimestepSinusoidalEmbedding(dim=32)
    out = emb(torch.tensor([0, 1, 50]))
    assert out.shape == (3, 32)
    assert torch.isfinite(out).all()


def test_film_layer_zero_cond_is_identity_when_proj_zero() -> None:
    film = FiLMLayer(cond_dim=4, feature_dim=8)
    film.proj.weight.data.zero_()
    film.proj.bias.data.zero_()
    h = torch.randn(3, 8)
    g = torch.randn(3, 4)
    assert torch.allclose(film(h, g), h)


# ---------------------------------------------------------------------------
# Denoise networks
# ---------------------------------------------------------------------------


def test_mlp_denoise_zero_init_returns_zero() -> None:
    net = MlpDenoiseNetworkConfig(
        hidden_dim=16, num_layers=2, dim_vis=8, dim_global=8
    ).build()
    M = 7
    h = torch.randn(M, 16)
    e = torch.randn(0, 64)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    v = torch.randn(M, 8)
    g = torch.randn(M, 8)
    out = net(h, edge_index, e, v, g)
    assert out.shape == (M, 2)
    assert torch.allclose(out, torch.zeros_like(out))


def test_graph_denoise_runs_and_zero_init() -> None:
    cfg = GraphDenoiseNetworkConfig(
        hidden_dim=16, num_layers=2, dim_edge=8, dim_vis=8, dim_global=8, num_heads=2
    )
    net = cfg.build()
    M, E = 6, 5
    h = torch.randn(M, 16)
    edge_index = torch.randint(0, M, (2, E))
    e = torch.randn(E, 8)
    v = torch.randn(M, 8)
    g = torch.randn(M, 8)
    out = net(h, edge_index, e, v, g)
    assert out.shape == (M, 2)
    assert torch.allclose(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# Diffusion decoder
# ---------------------------------------------------------------------------


def test_diffusion_decoder_noise_schedule_monotonic() -> None:
    cfg = GraphDiffusionDecoderConfig(
        denoise_network=MlpDenoiseNetworkConfig(num_layers=1),
        num_train_steps=10,
        sigma_min=0.01,
        sigma_max=0.5,
    )
    dec = cfg.build()
    sigmas = dec._sigmas
    assert sigmas[0].item() == 0.0
    assert torch.all(sigmas[1:] >= sigmas[:-1] - 1e-8) or sigmas[1] == sigmas[1]
    # log-linear from sigma_min to sigma_max
    assert abs(sigmas[1].item() - 0.01) < 1e-4
    assert abs(sigmas[-1].item() - 0.5) < 1e-4


def test_diffusion_decoder_add_noise_zero_at_t0() -> None:
    cfg = GraphDiffusionDecoderConfig(
        denoise_network=MlpDenoiseNetworkConfig(num_layers=1),
        num_train_steps=10,
    )
    dec = cfg.build()
    _, graph = _tiny_graph(B=2)
    timestep = torch.zeros(2, dtype=torch.long)
    noisy, noise = dec.add_noise(graph.node_pos, timestep, graph.node_batch)
    assert torch.allclose(noisy, graph.node_pos)


def test_diffusion_decoder_predict_delta_shape() -> None:
    cfg = GraphDiffusionDecoderConfig(
        denoise_network=MlpDenoiseNetworkConfig(num_layers=1, hidden_dim=32),
        hidden_dim=32,
        num_train_steps=10,
    )
    dec = cfg.build()
    _, graph = _tiny_graph(B=2)
    timestep = torch.full((2,), 5, dtype=torch.long)
    delta = dec.predict_delta(graph.node_pos, graph, {}, timestep)
    assert delta.shape == graph.node_pos.shape


def test_diffusion_decoder_ddim_sample_returns_finite() -> None:
    cfg = GraphDiffusionDecoderConfig(
        denoise_network=MlpDenoiseNetworkConfig(num_layers=1),
        num_train_steps=4,
    )
    dec = cfg.build()
    dec.eval()
    _, graph = _tiny_graph(B=2)
    out = dec.ddim_sample(
        graph=graph, embeddings={}, num_nodes=graph.num_nodes,
    )
    assert out.shape == (graph.num_nodes, 2)
    assert torch.isfinite(out).all()
