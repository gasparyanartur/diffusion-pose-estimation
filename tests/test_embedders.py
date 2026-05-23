"""Unit tests for feature embedders."""

from __future__ import annotations

import torch

from src.embedders import (
    BilinearSampleEmbedder,
    BilinearSampleEmbedderConfig,
    NoneEmbedder,
    NoneEmbedderConfig,
)
from src.embedders._sampling import pixel_to_grid_sample_coords


def test_pixel_to_grid_sample_coords_corners() -> None:
    img_hw = torch.tensor([[100.0, 200.0]])
    pts = torch.tensor([[[[0.0, 0.0], [200.0, 100.0]]]])  # (1,1,2,2)
    out = pixel_to_grid_sample_coords(pts, img_hw.view(1, 1, 1, 2))
    assert out.shape == (1, 1, 2, 2)
    # (0, 0)         -> (-1, -1)
    # (W=200, H=100) -> (+1, +1)
    assert torch.allclose(out[0, 0, 0], torch.tensor([-1.0, -1.0]))
    assert torch.allclose(out[0, 0, 1], torch.tensor([+1.0, +1.0]))


def test_none_embedder_returns_empty_dict() -> None:
    emb = NoneEmbedderConfig().build()
    out = emb({}, torch.zeros(1, 1, 14, 2), torch.zeros(1, 2))
    assert out == {}


def test_bilinear_embedder_shape() -> None:
    spatial_dim, hidden_dim, out_dim = 4, 8, 16
    cfg = BilinearSampleEmbedderConfig(
        spatial_dim=spatial_dim, hidden_dim=hidden_dim, out_dim=out_dim
    )
    emb = cfg.build()
    B, N, K = 2, 1, 14
    spatial = [torch.randn(spatial_dim, 32, 32) for _ in range(B)]
    semantic = [torch.randn(hidden_dim, 8, 8) for _ in range(B)]
    features = {"spatial": spatial, "semantic": semantic}
    kpt_pixel = torch.rand(B, N, K, 2) * 256
    img_hw = torch.full((B, 2), 256.0)
    out = emb(features, kpt_pixel, img_hw)
    assert out["node_feats"].shape == (B, N, K, out_dim)
