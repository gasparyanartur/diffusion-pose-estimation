"""Unit tests for src.model — KeypointDiffusionModel and PoseDiffusionModule."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data import (
    CANONICAL_POSE_GRAPH_EDGE_LIST,
    make_edge_index,
    prepare_pose_graph_batch,
    SceneGraphSampler,
    PoseDataset,
)
from src.model import (
    KeypointDiffusionConfig,
    KeypointDiffusionModel,
    PoseDiffusionModule,
    TimestepSinusoidalEmbedding,
)

# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

NUM_KEYPOINTS_CROWDPOSE = 14   # raw CrowdPose joints
NUM_KEYPOINTS_CANONICAL = 15   # after virtual root is prepended
BATCH_SIZE = 4
N_POSES = 3  # people per image (held constant for uniform batches)
DEVICE = torch.device("cpu")


def _make_pose_samples(
    n_images: int,
    n_poses: int = N_POSES,
    num_kp: int = NUM_KEYPOINTS_CROWDPOSE,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n_images):
        samples.append({
            "image_id":   i,
            "img_path":   f"/fake/img_{i}.jpg",
            "keypoints":  torch.from_numpy(
                rng.random((n_poses, num_kp, 2)).astype(np.float32) * 640
            ),
            "visibility": torch.from_numpy(
                rng.integers(0, 3, (n_poses, num_kp)).astype(np.float32)
            ),
        })
    return samples


def _make_batch(
    B: int = BATCH_SIZE,
    N: int = N_POSES,
    K: int = NUM_KEYPOINTS_CROWDPOSE,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Build a synthetic batch of shape (B, N, K, 2)."""
    rng = np.random.default_rng(seed)
    return {
        "image_id":   list(range(B)),
        "keypoints":  torch.from_numpy(rng.random((B, N, K, 2)).astype(np.float32) * 640),
        "visibility": torch.from_numpy(rng.integers(0, 3, (B, N, K)).astype(np.float32)),
    }


@pytest.fixture
def config() -> KeypointDiffusionConfig:
    """Tiny config that keeps tests fast."""
    return KeypointDiffusionConfig(
        name="test",
        dim_pos=16,
        dim_index=16,
        dim_time=16,
        dim_bone=16,
        dim_hidden=16,
        num_layers=2,
        num_train_steps=50,
    )


@pytest.fixture
def model(config) -> KeypointDiffusionModel:
    return KeypointDiffusionModel(config=config, num_keypoints=NUM_KEYPOINTS_CANONICAL)


@pytest.fixture
def module(model) -> PoseDiffusionModule:
    return PoseDiffusionModule(model=model)


# ---------------------------------------------------------------------------
# TimestepSinusoidalEmbedding
# ---------------------------------------------------------------------------


class TestTimestepSinusoidalEmbedding:
    def test_output_shape(self):
        B, dim = 8, 64
        emb = TimestepSinusoidalEmbedding(dim=dim)
        t = torch.randint(0, 100, (B,))
        out = emb(t)
        assert out.shape == (B, dim), f"Expected ({B}, {dim}), got {out.shape}"

    def test_output_dtype_float32(self):
        emb = TimestepSinusoidalEmbedding(dim=32)
        out = emb(torch.tensor([0, 1, 5, 99]))
        assert out.dtype == torch.float32

    def test_different_timesteps_differ(self):
        """Two distinct timesteps should produce different embeddings."""
        emb = TimestepSinusoidalEmbedding(dim=64)
        t0 = emb(torch.tensor([0]))
        t1 = emb(torch.tensor([1]))
        assert not torch.allclose(t0, t1)

    def test_same_timestep_is_reproducible(self):
        """Embedding is deterministic — same input, same output."""
        emb = TimestepSinusoidalEmbedding(dim=64)
        t = torch.tensor([7, 42])
        assert torch.allclose(emb(t), emb(t))

    def test_odd_dim_raises(self):
        with pytest.raises(ValueError, match="even"):
            TimestepSinusoidalEmbedding(dim=33)

    def test_values_bounded(self):
        """cos/sin outputs are in [-1, 1]."""
        emb = TimestepSinusoidalEmbedding(dim=128)
        out = emb(torch.arange(50))
        assert out.abs().max().item() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# KeypointDiffusionModel.add_noise
# ---------------------------------------------------------------------------


class TestAddNoise:
    def test_output_shapes(self, model):
        B, N, K = 3, N_POSES, NUM_KEYPOINTS_CANONICAL
        node_pos = torch.rand(B * N * K, 2)
        batch_idx = torch.arange(B).repeat_interleave(N * K)
        timestep = torch.randint(0, model.config.num_train_steps, (B,))

        noisy, noise = model.add_noise(node_pos, timestep, batch_idx)

        assert noisy.shape == node_pos.shape, "noisy_node_pos shape mismatch"
        assert noise.shape == node_pos.shape, "noise shape mismatch"

    def test_zero_timestep_leaves_positions_unchanged(self, model):
        """At t=0 sqrt(1-alpha_bar)=0, so noisy positions must equal clean positions.

        The returned ``noise`` tensor is the raw Gaussian epsilon (VP convention)
        and is NOT zero at t=0; only the *noise contribution to the noisy
        positions* (b * epsilon, b=0) is zero.
        """
        B, N, K = 2, N_POSES, NUM_KEYPOINTS_CANONICAL
        node_pos = torch.rand(B * N * K, 2)
        batch_idx = torch.arange(B).repeat_interleave(N * K)
        timestep = torch.zeros(B, dtype=torch.long)

        noisy, noise = model.add_noise(node_pos, timestep, batch_idx)

        assert torch.allclose(noisy, node_pos, atol=1e-6), \
            "Zero timestep should add no noise to the positions (b=0 in VP schedule)"
        # noise is raw epsilon (std~1), not the scaled contribution
        assert noise.shape == node_pos.shape

    def test_max_timestep_adds_full_noise(self, model):
        """At t=T the noise level is 1, so |noise| should be approximately std(randn)."""
        T = model.config.num_train_steps
        B, N, K = 2, N_POSES, NUM_KEYPOINTS_CANONICAL
        node_pos = torch.zeros(B * N * K, 2)
        batch_idx = torch.arange(B).repeat_interleave(N * K)
        timestep = torch.full((B,), T - 1, dtype=torch.long)  # t = T-1

        noisy, noise = model.add_noise(node_pos, timestep, batch_idx)

        # noise should have magnitude proportional to (T-1)/T ≈ 1
        assert noise.abs().mean().item() > 0.1, "Max timestep should add meaningful noise"

    def test_noise_scales_with_timestep(self, model):
        """Higher timesteps produce larger deviation from the clean positions.

        In the VP schedule ``x_t = sqrt(ab_t)*x0 + sqrt(1-ab_t)*eps``, the
        noise contribution ``sqrt(1-ab_t)*eps`` grows with t, so
        ``(noisy_high - x0).abs()`` > ``(noisy_low - x0).abs()``.
        The raw returned epsilon is the same for both calls (same seed), so
        comparing ``noise_high`` vs ``noise_low`` would give equality.
        """
        B, N, K = 4, N_POSES, NUM_KEYPOINTS_CANONICAL
        node_pos = torch.zeros(B * N * K, 2)  # x0 = 0, so noisy == noise contribution
        batch_idx = torch.arange(B).repeat_interleave(N * K)

        torch.manual_seed(0)
        noisy_low, _ = model.add_noise(node_pos, torch.full((B,), 10, dtype=torch.long), batch_idx)
        torch.manual_seed(0)
        noisy_high, _ = model.add_noise(node_pos, torch.full((B,), 40, dtype=torch.long), batch_idx)

        assert noisy_high.abs().mean() > noisy_low.abs().mean(), \
            "Higher timestep should produce larger deviation from clean positions"


# ---------------------------------------------------------------------------
# KeypointDiffusionModel.backward_process
# ---------------------------------------------------------------------------


class TestBackwardProcess:
    def _make_graph(self, B: int, N: int) -> dict[str, torch.Tensor]:
        """Build a minimal pose graph for B images with N people each."""
        K = NUM_KEYPOINTS_CANONICAL
        kpts = torch.rand(B, N, K, 2) * 100
        vis = torch.ones(B, N, NUM_KEYPOINTS_CROWDPOSE)
        # Use raw CrowdPose keypoints for prepare_pose_graph_batch (expects 14 kp)
        raw_kpts = kpts[:, :, :NUM_KEYPOINTS_CROWDPOSE, :]
        return prepare_pose_graph_batch(raw_kpts, vis)

    def test_output_shape(self, model):
        B, N = 3, N_POSES
        K = NUM_KEYPOINTS_CANONICAL
        g = self._make_graph(B, N)

        node_pos = g["node_pos"]
        batch_idx = g["batch"]
        timestep = torch.randint(0, model.config.num_train_steps, (B,))

        pred = model.backward_process(
            noisy_node_pos=node_pos,
            edge_index=g["edge_index"],
            kpt_type_idx=g["kpt_type_idx"].reshape(-1),
            batch_idx=batch_idx,
            timestep=timestep,
        )
        assert pred.shape == (B * N * K, 2), f"Expected ({B*N*K}, 2), got {pred.shape}"

    def test_output_is_float32(self, model):
        B, N = 2, N_POSES
        g = self._make_graph(B, N)
        batch_idx = g["batch"]
        timestep = torch.randint(0, model.config.num_train_steps, (B,))

        pred = model.backward_process(
            noisy_node_pos=g["node_pos"],
            edge_index=g["edge_index"],
            kpt_type_idx=g["kpt_type_idx"].reshape(-1),
            batch_idx=batch_idx,
            timestep=timestep,
        )
        assert pred.dtype == torch.float32

    def test_gradients_flow_through_prediction(self, model):
        """Loss computed on backward_process output should allow gradients to flow."""
        B, N = 2, N_POSES
        g = self._make_graph(B, N)
        batch_idx = g["batch"]
        timestep = torch.randint(0, model.config.num_train_steps, (B,))

        pred = model.backward_process(
            noisy_node_pos=g["node_pos"],
            edge_index=g["edge_index"],
            kpt_type_idx=g["kpt_type_idx"].reshape(-1),
            batch_idx=batch_idx,
            timestep=timestep,
        )
        loss = pred.mean()
        loss.backward()

        # At least one parameter should have a non-None, non-zero gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No gradients reached model parameters"


# ---------------------------------------------------------------------------
# PoseDiffusionModule (end-to-end)
# ---------------------------------------------------------------------------


class TestPoseDiffusionModule:
    def _make_loader(self, n_images: int = 8, n_poses: int = N_POSES) -> DataLoader:
        samples = _make_pose_samples(n_images=n_images, n_poses=n_poses)
        ds = PoseDataset(samples=samples)
        sampler = SceneGraphSampler(ds, batch_size=BATCH_SIZE, shuffle=False)
        return DataLoader(ds, batch_sampler=sampler)

    def test_training_step_returns_scalar_loss(self, module):
        batch = _make_batch(B=BATCH_SIZE, N=N_POSES)
        loss = module.training_step(batch, batch_idx=0)
        assert loss.ndim == 0, "Loss should be a scalar tensor"
        assert loss.item() >= 0.0, "MSE loss must be non-negative"

    def test_training_step_loss_is_finite(self, module):
        batch = _make_batch(B=BATCH_SIZE, N=N_POSES)
        loss = module.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_training_step_gradients_propagate(self, module):
        """Calling backward() on the training-step loss should populate all parameter gradients."""
        batch = _make_batch(B=BATCH_SIZE, N=N_POSES)
        loss = module.training_step(batch, batch_idx=0)
        loss.backward()

        for name, param in module.named_parameters():
            assert param.grad is not None, f"Parameter '{name}' has no gradient"

    def test_training_step_loss_decreases_over_steps(self, module):
        """A few gradient steps with Adam should reduce the training loss."""
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-2)

        torch.manual_seed(0)
        batch = _make_batch(B=BATCH_SIZE, N=N_POSES, seed=0)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            loss = module.training_step(batch, batch_idx=0)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease over 5 steps: {losses}"
        )

    def test_configure_optimizers_returns_adamw(self, module):
        opt = module.configure_optimizers()
        assert isinstance(opt, torch.optim.AdamW)

    def test_training_step_different_n_poses(self, config):
        """Module handles batches with a different number of people per image."""
        for n_poses in [1, 2, 5]:
            model = KeypointDiffusionModel(config=config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
            module = PoseDiffusionModule(model=model)
            batch = _make_batch(B=2, N=n_poses)
            loss = module.training_step(batch, batch_idx=0)
            assert torch.isfinite(loss), f"Non-finite loss with N={n_poses}"
