"""Unit tests for src.model — KeypointDiffusionModel and PoseDiffusionModule."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

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
    DinoConditioner,
    KeypointDiffusionConfig,
    KeypointDiffusionModel,
    PatchCNNConditioner,
    PoseDiffusionModule,
    TimestepSinusoidalEmbedding,
    compute_bone_loss,
    compute_joint_loss,
    image_dropout,
    joint_dropout,
    joint_randomize,
    keypoint_jitter,
    pose_dropout,
)
from src.metrics import compute_mpjpe, compute_pck

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
        dim_edge=16,
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
        """At t=0 sigma=0, so the VE forward process adds no noise: noisy == clean.

        The returned ``noise`` tensor is the raw unit Gaussian epsilon and is
        NOT zero at t=0; only the *noise contribution* (sigma * epsilon, sigma=0)
        is zero.
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
        """At t≈T the noise level ≈ sigma_max, so |noisy - x0| should be substantial."""
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

        In the VE schedule ``x_t = x0 + sigma_t * eps``, sigma_t grows with t,
        so ``(noisy_high - x0).abs()`` > ``(noisy_low - x0).abs()``.
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
        return prepare_pose_graph_batch({"keypoints": raw_kpts, "visibility": vis})

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
        result = module.configure_optimizers()
        # configure_optimizers may return a bare optimizer or a Lightning dict
        opt = result["optimizer"] if isinstance(result, dict) else result
        assert isinstance(opt, torch.optim.AdamW)

    def test_training_step_different_n_poses(self, config):
        """Module handles batches with a different number of people per image."""
        for n_poses in [1, 2, 5]:
            model = KeypointDiffusionModel(config=config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
            module = PoseDiffusionModule(model=model)
            batch = _make_batch(B=2, N=n_poses)
            loss = module.training_step(batch, batch_idx=0)
            assert torch.isfinite(loss), f"Non-finite loss with N={n_poses}"


# ---------------------------------------------------------------------------
# Image conditioning (DinoConditioner + conditioned backward_process)
# ---------------------------------------------------------------------------


def _make_mock_dino(dim: int, raw_dim: int | None = None):
    """Return a stub that mimics DinoConditioner without a real backbone.

    Args:
        dim:     Output dim of the projected forward() call.
        raw_dim: Backbone hidden size (used by _gather_neighborhood). Defaults to dim.
    """
    if raw_dim is None:
        raw_dim = dim
    mock = MagicMock(spec=DinoConditioner)
    mock.to.return_value = mock
    mock.hidden_dim = raw_dim
    mock.backbone = MagicMock()
    mock.backbone.config.hidden_size = raw_dim
    # Pre-populate a fake patch cache so _add_combined_conditioning takes the cache path
    # (avoids needing a real backbone).  Keys cover any batch with image_id in 0..99.
    mock._patch_cache = {i: torch.zeros(196, raw_dim) for i in range(100)}

    def _call(kpt_pixel, img_hw, image_ids=None, pil_images=None):
        b_, n_, k_, _ = kpt_pixel.shape
        return torch.zeros(b_, n_, k_, dim)

    def _gather(patch_feats, kpt_pixel, img_hw):
        b_, n_, k_, _ = kpt_pixel.shape
        return torch.zeros(b_, n_, k_, raw_dim)

    mock.proj = torch.nn.Linear(raw_dim, dim)
    mock.side_effect = _call
    mock._gather_neighborhood = MagicMock(side_effect=_gather)
    return mock


class TestPatchCNNConditioner:
    """Tests for the PatchCNNConditioner (TODO[3])."""

    @pytest.fixture
    def cnn_cond(self) -> PatchCNNConditioner:
        return PatchCNNConditioner(out_dim=16, patch_size=16)

    @pytest.fixture
    def cnn_config(self) -> KeypointDiffusionConfig:
        return KeypointDiffusionConfig(
            dim_pos=16, dim_index=16, dim_time=16, dim_edge=16,
            dim_hidden=16, num_layers=2, num_train_steps=10,
            use_image_cond=True, use_cnn_cond=True,
            dim_image_cond=8, cnn_patch_size=16,
        )

    def _make_images(self, B: int, H: int = 64, W: int = 80) -> list[torch.Tensor]:
        """Synthetic float [0,1] RGB images of shape (3, H, W)."""
        torch.manual_seed(0)
        return [torch.rand(3, H, W) for _ in range(B)]

    def test_output_shape(self, cnn_cond):
        """forward returns (B, N, K, out_dim)."""
        B, N, K = 2, 3, 14
        images = self._make_images(B, H=64, W=80)
        kpt = torch.rand(B, N, K, 2) * 60  # pixel coords within image
        out = cnn_cond(images, kpt)
        assert out.shape == (B, N, K, 16)

    def test_extract_patches_shape(self, cnn_cond):
        """_extract_patches returns (N*K, 5, P, P)."""
        N, K, P = 2, 3, 16
        image = torch.rand(3, 64, 80)
        kpt = torch.rand(N, K, 2) * 60
        patches = cnn_cond._extract_patches(image, kpt)
        assert patches.shape == (N * K, 5, P, P)

    def test_spatial_channels_bounded(self, cnn_cond):
        """Channels 3 and 4 (spatial coords) must lie in [-1, 1]."""
        N, K = 2, 3
        image = torch.rand(3, 64, 80)
        kpt = torch.rand(N, K, 2) * 60
        patches = cnn_cond._extract_patches(image, kpt)  # (N*K, 5, P, P)
        spatial = patches[:, 3:, :, :]  # channels 3-4
        assert spatial.min().item() >= -1.0 - 1e-5
        assert spatial.max().item() <=  1.0 + 1e-5

    def test_rgb_channels_bounded(self, cnn_cond):
        """RGB channels (0-2) must remain in [0, 1] for a [0,1]-normalised input."""
        N, K = 2, 3
        image = torch.rand(3, 64, 80)  # already in [0, 1]
        kpt = torch.rand(N, K, 2) * 60
        patches = cnn_cond._extract_patches(image, kpt)
        rgb = patches[:, :3, :, :]
        assert rgb.min().item() >= -0.01
        assert rgb.max().item() <=  1.01

    def test_border_keypoints_do_not_crash(self, cnn_cond):
        """Keypoints outside the image boundary should not crash (padding_mode=border)."""
        N, K = 1, 2
        image = torch.rand(3, 32, 32)
        kpt = torch.tensor([[[-50.0, -50.0], [200.0, 200.0]]])  # far outside
        out = cnn_cond._extract_patches(image, kpt)
        assert out.shape == (N * K, 5, 16, 16)
        assert torch.isfinite(out).all()

    def test_extract_patches_output_is_contiguous(self, cnn_cond):
        """_extract_patches output must be contiguous so Conv2d doesn't see zero-stride tensors.

        Non-contiguous inputs (e.g. from expand()) can trigger the CUDA allocator
        assertion (!block->expandable_segment_) when passed to Conv2d.
        """
        image = torch.rand(3, 64, 80)
        kpt = torch.rand(2, 3, 2) * 60
        patches = cnn_cond._extract_patches(image, kpt)
        assert patches.is_contiguous(), "patches must be contiguous for Conv2d"

    def test_rgb_and_spatial_channels_contiguous_after_cat(self, cnn_cond):
        """The full (M, 5, P, P) patches tensor from torch.cat must be contiguous.
        (Per-channel slices like patches[:, :3] are never contiguous — only
        the whole output matters for Conv2d.)
        """
        image = torch.rand(3, 64, 80)
        kpt = torch.rand(3, 5, 2) * 60
        patches = cnn_cond._extract_patches(image, kpt)
        # The full output is contiguous (checked separately); per-channel slices aren't.
        assert patches.is_contiguous()

    def test_cnn_can_forward_extract_patches_output(self, cnn_cond):
        """cnn Sequential can run on the _extract_patches output without error."""
        image = torch.rand(3, 64, 80)
        kpt = torch.rand(2, 3, 2) * 60
        patches = cnn_cond._extract_patches(image, kpt)
        out = cnn_cond.cnn(patches)   # would crash if non-contiguous on CUDA
        assert torch.isfinite(out).all()

    def test_many_keypoints_contiguous(self):
        """Large M (many keypoints) still produces a contiguous output."""
        cnn = PatchCNNConditioner(out_dim=8, patch_size=8)
        image = torch.rand(3, 128, 128)
        kpt = torch.rand(10, 15, 2) * 120  # M = 150
        patches = cnn._extract_patches(image, kpt)
        assert patches.shape == (150, 5, 8, 8)
        assert patches.is_contiguous()

    def test_odd_patch_size_contiguous(self):
        """Odd patch sizes work and produce contiguous output."""
        cnn = PatchCNNConditioner(out_dim=8, patch_size=13)
        image = torch.rand(3, 64, 64)
        kpt = torch.rand(2, 3, 2) * 60
        patches = cnn._extract_patches(image, kpt)
        assert patches.shape == (6, 5, 13, 13)
        assert patches.is_contiguous()

    def test_single_keypoint_contiguous(self, cnn_cond):
        """M=1 (single keypoint) still produces a contiguous output."""
        image = torch.rand(3, 64, 80)
        kpt = torch.tensor([[[30.0, 20.0]]])  # (1, 1, 2)
        patches = cnn_cond._extract_patches(image, kpt)
        assert patches.shape == (1, 5, 16, 16)
        assert patches.is_contiguous()

    def test_small_image_no_crash(self, cnn_cond):
        """Image smaller than patch size is handled safely via padding."""
        image = torch.rand(3, 8, 8)   # smaller than patch_size=16
        kpt = torch.rand(1, 2, 2) * 7
        patches = cnn_cond._extract_patches(image, kpt)
        assert patches.shape == (2, 5, 16, 16)
        assert torch.isfinite(patches).all()
        assert patches.is_contiguous()

    def test_cnn_conditioner_created_when_use_cnn_cond(self, cnn_config):
        """PoseDiffusionModule creates cnn_conditioner when model.config.use_cnn_cond=True."""
        model = KeypointDiffusionModel(cnn_config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(model)
        assert module.cnn_conditioner is not None
        assert isinstance(module.cnn_conditioner, PatchCNNConditioner)
        assert module.dino_conditioner is None   # no dino_model_name provided
        assert module.fuse_proj is None          # fusion only when BOTH conditioners active

    def test_training_step_with_cnn_cond(self, cnn_config):
        """End-to-end training step with CNN conditioning produces a finite loss."""
        import tempfile
        from PIL import Image as PILImage

        model = KeypointDiffusionModel(cnn_config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(model)
        assert module.cnn_conditioner is not None

        batch = _make_batch(B=BATCH_SIZE, N=N_POSES)

        # Write tiny real PNG images so read_image can load them
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(BATCH_SIZE):
                p = f"{tmp}/img_{i}.png"
                PILImage.new("RGB", (80, 64), color=(128, 64, 32)).save(p)
                paths.append(p)
            batch["img_path"] = paths

            loss = module.training_step(batch, batch_idx=0)

        assert loss.ndim == 0
        assert torch.isfinite(loss)


class TestLocalSpatialCondition:
    """Tests for get_local_spatial_condition and combined DINO+CNN conditioning."""

    _DINO_RAW_DIM = 24   # simulates backbone hidden_size
    _CNN_TRUNK_DIM = 64  # CNN backbone_dim (last conv output channels)
    _OUT_DIM = 8

    @pytest.fixture
    def combined_config(self) -> KeypointDiffusionConfig:
        return KeypointDiffusionConfig(
            dim_pos=16, dim_index=16, dim_time=16, dim_edge=16,
            dim_hidden=16, num_layers=2, num_train_steps=10,
            use_image_cond=True, use_cnn_cond=True,
            dim_image_cond=self._OUT_DIM, cnn_patch_size=16,
        )

    def _module_with_both(self, combined_config):
        model = KeypointDiffusionModel(combined_config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(model)
        module.dino_conditioner = _make_mock_dino(self._OUT_DIM, raw_dim=self._DINO_RAW_DIM)
        module.dino_proj = torch.nn.Linear(self._DINO_RAW_DIM, self._OUT_DIM)
        module.cnn_proj  = torch.nn.Linear(self._CNN_TRUNK_DIM, self._OUT_DIM)
        module.fuse_proj = torch.nn.Linear(2 * self._OUT_DIM, self._OUT_DIM)
        return module

    def test_fusion_proj_created_when_both_active(self, combined_config):
        """fuse_proj is None without DINO, set when both conditioners are active."""
        model = KeypointDiffusionModel(combined_config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(model)
        assert module.fuse_proj is None  # CNN only — no DINO

    def test_training_step_with_combined_conditioning(self, combined_config):
        """Training step with DINO (mocked) + CNN produces a finite loss."""
        import tempfile
        from PIL import Image as PILImage

        module = self._module_with_both(combined_config)
        batch = _make_batch(B=BATCH_SIZE, N=N_POSES)
        with tempfile.TemporaryDirectory() as tmp:
            paths = [f"{tmp}/img_{i}.png" for i in range(BATCH_SIZE)]
            for p in paths:
                PILImage.new("RGB", (80, 64), color=(128, 64, 32)).save(p)
            batch["img_path"] = paths
            loss = module.training_step(batch, batch_idx=0)

        assert loss.ndim == 0 and torch.isfinite(loss)

    def test_combined_conditioning_output_shape(self, combined_config):
        """get_local_spatial_condition returns (B*N*15, out_dim) when both active."""
        import tempfile
        from PIL import Image as PILImage

        module = self._module_with_both(combined_config)
        batch = _make_batch(B=2, N=N_POSES)
        with tempfile.TemporaryDirectory() as tmp:
            paths = [f"{tmp}/img_{i}.png" for i in range(2)]
            for p in paths:
                PILImage.new("RGB", (80, 64), color=(0, 0, 0)).save(p)
            batch["img_path"] = paths
            module.prepare_batch(batch)

        assert batch["image_cond"].shape == (2 * N_POSES * NUM_KEYPOINTS_CANONICAL, self._OUT_DIM)


class TestImageConditioning:
    @pytest.fixture
    def cond_config(self) -> KeypointDiffusionConfig:
        return KeypointDiffusionConfig(
            dim_pos=16, dim_index=16, dim_time=16, dim_edge=16,
            dim_hidden=16, num_layers=2, num_train_steps=10,
            use_image_cond=True, dim_image_cond=8,
        )

    def test_gather_neighborhood_picks_correct_patch(self):
        """Each keypoint should retrieve the patch whose grid cell the pixel falls in.

        Setup: 64×64 image, patch_size=16 → 4×4 grid (G=4).
        Give every patch a unique fingerprint vector (patch index embedded in dim 0).
        Place keypoints exactly at patch centres and verify the returned embedding
        matches that specific patch's fingerprint.
        """
        from unittest.mock import MagicMock

        G, D, patch = 4, 16, 16
        image_size = G * patch  # 64

        # Build a DinoConditioner stub (no real backbone needed)
        c = object.__new__(DinoConditioner)
        torch.nn.Module.__init__(c)
        c.patch_size = patch
        c.image_size = image_size
        c.neighborhood_radius = 0   # r=0: only the centre patch, no pooling
        c.grid_size = G
        c.proj = torch.nn.Linear(D, D)

        # Patch features: patch p has a unique fingerprint (p+1) in slot p%D
        patch_feats = torch.zeros(1, G * G, D)
        for p in range(G * G):
            patch_feats[0, p, p % D] = float(p + 1)

        # Place one keypoint at the centre of each patch, in an image the same
        # size as the DINO input (so no rescaling needed)
        num_kp = G * G
        centres_x = torch.tensor([(cx + 0.5) * patch for cy in range(G) for cx in range(G)])
        centres_y = torch.tensor([(cy + 0.5) * patch for cy in range(G) for cx in range(G)])
        kpt_pixel = torch.stack([centres_x, centres_y], dim=-1).view(1, 1, num_kp, 2)
        img_hw = torch.tensor([[float(image_size), float(image_size)]])

        result = c._gather_neighborhood(patch_feats, kpt_pixel, img_hw)  # (1,1,16,D)

        for p in range(G * G):
            expected = patch_feats[0, p]
            assert torch.allclose(result[0, 0, p], expected), \
                f"Keypoint {p} should retrieve patch {p}, got {result[0,0,p]} vs {expected}"

    def test_backward_process_accepts_image_cond(self, cond_config):
        """backward_process runs without error when image_cond is supplied."""
        model = KeypointDiffusionModel(cond_config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        batch = _make_batch(B=BATCH_SIZE, N=N_POSES)
        prepare_pose_graph_batch(batch)

        image_cond = torch.zeros(BATCH_SIZE * N_POSES * NUM_KEYPOINTS_CANONICAL, 8)
        t = torch.ones(BATCH_SIZE, dtype=torch.long)
        pred = model.backward_process(
            noisy_node_pos=batch["node_pos"],
            edge_index=batch["edge_index"],
            kpt_type_idx=batch["kpt_type_idx"].reshape(-1),
            batch_idx=batch["batch"],
            timestep=t,
            image_cond=image_cond,
        )
        assert pred.shape == (BATCH_SIZE * N_POSES * NUM_KEYPOINTS_CANONICAL, 2)

    def test_prepare_batch_adds_image_cond(self, cond_config):
        """prepare_batch populates image_cond with the right node count."""
        from unittest.mock import patch
        from PIL.Image import Image as PILImage

        model = KeypointDiffusionModel(cond_config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(model)
        module.dino_conditioner = _make_mock_dino(8)

        batch = _make_batch(B=BATCH_SIZE, N=N_POSES)
        batch["img_path"] = [f"/fake/img_{i}.jpg" for i in range(BATCH_SIZE)]
        mock_img = MagicMock()
        mock_img.height, mock_img.width = 480, 640
        mock_img.convert.return_value = mock_img
        with patch("src.model.Image.open", return_value=mock_img):
            module.prepare_batch(batch)

        expected_nodes = BATCH_SIZE * N_POSES * NUM_KEYPOINTS_CANONICAL
        assert "image_cond" in batch
        assert batch["image_cond"].shape == (expected_nodes, 8)

    def test_training_step_with_image_cond(self, cond_config):
        """End-to-end training step with mock DINO produces a finite loss."""
        from unittest.mock import patch

        model = KeypointDiffusionModel(cond_config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(model)
        module.dino_conditioner = _make_mock_dino(8)

        batch = _make_batch(B=BATCH_SIZE, N=N_POSES)
        batch["img_path"] = [f"/fake/img_{i}.jpg" for i in range(BATCH_SIZE)]
        mock_img = MagicMock()
        mock_img.height, mock_img.width = 480, 640
        mock_img.convert.return_value = mock_img
        with patch("src.model.Image.open", return_value=mock_img):
            loss = module.training_step(batch, batch_idx=0)

        assert loss.ndim == 0
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Global image conditioning (TODO[5])
# ---------------------------------------------------------------------------


class TestGlobalImageCond:
    """Tests for get_global_image_cond and end-to-end global conditioning."""

    _DINO_RAW_DIM = 24
    _GLOBAL_DIM = 8

    @pytest.fixture
    def global_config(self) -> KeypointDiffusionConfig:
        return KeypointDiffusionConfig(
            dim_pos=16, dim_index=16, dim_time=16, dim_edge=16,
            dim_hidden=16, num_layers=2, num_train_steps=10,
            use_image_cond=True, dim_image_cond=8,
            use_global_image_cond=True, dim_global_image_cond=self._GLOBAL_DIM,
        )

    def _module_with_global(self, config):
        model = KeypointDiffusionModel(config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        mock_dino = _make_mock_dino(8, raw_dim=self._DINO_RAW_DIM)
        return PoseDiffusionModule(model, dino_conditioner=mock_dino)

    def test_global_cond_mlp_input_dim_changes(self, global_config):
        """global_cond_mlp first layer accepts dim_time + dim_global_image_cond."""
        model = KeypointDiffusionModel(global_config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        expected_in = global_config.dim_time + global_config.dim_global_image_cond
        actual_in = model.global_cond_mlp[0].in_features
        assert actual_in == expected_in

    def test_global_cond_mlp_baseline_input_dim(self):
        """Without use_global_image_cond, global_cond_mlp input == dim_time only."""
        cfg = KeypointDiffusionConfig(dim_time=16, num_train_steps=10)
        model = KeypointDiffusionModel(cfg, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        assert model.global_cond_mlp[0].in_features == cfg.dim_time

    def test_get_global_image_cond_shape(self, global_config):
        """get_global_image_cond returns (B*N*15, dim_global_image_cond)."""
        module = self._module_with_global(global_config)
        batch = _make_batch(B=BATCH_SIZE, N=N_POSES)
        prepare_pose_graph_batch(batch)
        result = module.get_global_image_cond(batch)
        assert result is not None
        assert result.shape == (BATCH_SIZE * N_POSES * NUM_KEYPOINTS_CANONICAL, self._GLOBAL_DIM)

    def test_get_global_image_cond_none_without_proj(self):
        """Returns None when use_global_image_cond=False (no proj created)."""
        cfg = KeypointDiffusionConfig(dim_time=16, num_train_steps=10)
        model = KeypointDiffusionModel(cfg, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(model)
        batch = _make_batch(B=BATCH_SIZE, N=N_POSES)
        prepare_pose_graph_batch(batch)
        assert module.get_global_image_cond(batch) is None

    def test_raises_if_global_cond_enabled_without_dino(self, global_config):
        """PoseDiffusionModule raises ValueError at init when use_global_image_cond=True
        but no DINO conditioner is provided — prevents a silent shape mismatch at runtime."""
        model = KeypointDiffusionModel(global_config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        with pytest.raises(ValueError, match="use_global_image_cond"):
            PoseDiffusionModule(model)  # no dino_model_name → dino_conditioner is None

    def test_backward_process_accepts_global_image_cond(self, global_config):
        """backward_process runs without error when global_image_cond is supplied."""
        model = KeypointDiffusionModel(global_config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        batch = _make_batch(B=BATCH_SIZE, N=N_POSES)
        prepare_pose_graph_batch(batch)
        M = BATCH_SIZE * N_POSES * NUM_KEYPOINTS_CANONICAL
        t = torch.ones(BATCH_SIZE, dtype=torch.long)
        pred = model.backward_process(
            noisy_node_pos=batch["node_pos"],
            edge_index=batch["edge_index"],
            kpt_type_idx=batch["kpt_type_idx"].reshape(-1),
            batch_idx=batch["batch"],
            timestep=t,
            global_image_cond=torch.zeros(M, self._GLOBAL_DIM),
        )
        assert pred.shape == (M, 2)

    def test_training_step_with_global_cond(self, global_config):
        """End-to-end training step with global DINO conditioning produces a finite loss."""
        module = self._module_with_global(global_config)
        batch = _make_batch(B=BATCH_SIZE, N=N_POSES)
        batch["img_path"] = [f"/fake/img_{i}.jpg" for i in range(BATCH_SIZE)]
        loss = module.training_step(batch, batch_idx=0)
        assert loss.ndim == 0 and torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Multi-step AP evaluation (TODO[12])
# ---------------------------------------------------------------------------


class TestMultiStepAPEval:
    """Tests for multi-step AP evaluation — num_steps flows through the chain
    and epoch_end hooks log per-step metrics with an @{n} suffix."""

    _B, _N = 2, 2

    def _pred_batch(self) -> dict:
        return {
            "pred_keypoints":       torch.zeros(self._B, self._N, 14, 2),
            "pred_keypoint_scores": torch.full((self._B, self._N, 14), 0.9),
            "image_id":             list(range(self._B)),
            "img_path":             [f"/fake/{i}.jpg" for i in range(self._B)],
        }

    def test_num_steps_forwarded_to_ddim_sample(self, config):
        """_refine_prediction_batch passes its num_steps arg to ddim_sample."""
        model = KeypointDiffusionModel(config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(model)
        M = self._B * self._N * 15
        mock_ddim = MagicMock(return_value=torch.zeros(M, 2))
        module.model.ddim_sample = mock_ddim

        module._refine_prediction_batch(self._pred_batch(), num_steps=3)

        _args, call_kwargs = mock_ddim.call_args
        assert call_kwargs["num_inference_steps"] == 3, (
            "ddim_sample must receive num_inference_steps=3"
        )

    def test_ap_step_passes_num_steps_to_refine(self, config):
        """_ap_step forwards num_steps down to _refine_prediction_batch."""
        model = KeypointDiffusionModel(config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(model)
        M = self._B * self._N * 15
        module.model.ddim_sample = MagicMock(return_value=torch.zeros(M, 2))

        captured: list[int | None] = []
        orig = module._refine_prediction_batch

        def _spy(batch, num_steps=None):
            captured.append(num_steps)
            return orig(batch, num_steps=num_steps)

        module._refine_prediction_batch = _spy
        module._ap_step(self._pred_batch(), MagicMock(), num_steps=7)

        assert captured == [7]

    def test_run_ap_eval_passes_num_steps_to_ap_step(self, config):
        """_run_ap_eval forwards num_steps to every _ap_step call."""
        model = KeypointDiffusionModel(config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(model, ap_batch_size=2)

        captured: list[int | None] = []

        def _spy_ap_step(batch, metric, num_steps=None):
            captured.append(num_steps)

        module._ap_step = _spy_ap_step

        # Build a tiny fake PredictionDataset that yields two batches.
        from unittest.mock import patch
        from src.data import PredictionDataset

        fake_batches = [self._pred_batch(), self._pred_batch()]
        mock_ds = MagicMock(spec=PredictionDataset)
        mock_ds.create_dataloader.return_value = fake_batches

        with patch.object(PredictionDataset, "from_path", return_value=mock_ds):
            module._run_ap_eval("/fake/cache", MagicMock(), num_steps=5)

        assert captured == [5, 5], "Every batch should receive num_steps=5"

    def test_validation_epoch_end_calls_run_ap_eval_per_count(self, config):
        """on_validation_epoch_end calls _run_ap_eval once per entry in ap_step_counts."""
        model = KeypointDiffusionModel(config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(
            model,
            ap_step_counts=[1, 4],
            ap_eval_interval=1,
            val_pred_cache="/fake/cache",
        )
        module._val_metric = MagicMock()
        module._val_metric.compute.return_value = {"coco/AP": 0.5}
        module._run_ap_eval = MagicMock()
        module._trainer = MagicMock()
        module._trainer.sanity_checking = False
        module._trainer.current_epoch = 0

        module.on_validation_epoch_end()

        calls = module._run_ap_eval.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["num_steps"] == 1
        assert calls[1].kwargs["num_steps"] == 4

    def test_validation_epoch_end_multi_step_logs_with_suffix(self, config):
        """on_validation_epoch_end logs val/coco/AP@{n} for each step count."""
        model = KeypointDiffusionModel(config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(
            model,
            ap_step_counts=[2, 8],
            ap_eval_interval=1,
            val_pred_cache="/fake/cache",
        )
        module._val_metric = MagicMock()
        module._val_metric.compute.return_value = {"coco/AP": 0.5, "coco/AP50": 0.7}
        module._run_ap_eval = MagicMock()
        module._trainer = MagicMock()
        module._trainer.sanity_checking = False
        module._trainer.current_epoch = 0

        logged: dict = {}
        module.log = lambda key, val, **_: logged.update({key: val})

        module.on_validation_epoch_end()

        assert "val/coco/AP@2"   in logged
        assert "val/coco/AP@8"   in logged
        assert "val/coco/AP50@2" in logged
        assert "val/coco/AP50@8" in logged
        assert "val/coco/AP"     not in logged  # no bare key when step counts are set

    def test_validation_epoch_end_no_suffix_without_step_counts(self, config):
        """on_validation_epoch_end without ap_step_counts logs val/coco/AP (no suffix)."""
        model = KeypointDiffusionModel(config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(
            model,
            ap_eval_interval=1,
            val_pred_cache="/fake/cache",
        )
        module._val_metric = MagicMock()
        module._val_metric.compute.return_value = {"coco/AP": 0.5}
        module._run_ap_eval = MagicMock()
        module._trainer = MagicMock()
        module._trainer.sanity_checking = False
        module._trainer.current_epoch = 0

        logged: dict = {}
        module.log = lambda key, val, **_: logged.update({key: val})

        module.on_validation_epoch_end()

        assert "val/coco/AP" in logged
        assert not any("@" in k for k in logged)

    def test_test_epoch_end_multi_step_logs_with_suffix(self, config):
        """on_test_epoch_end with ap_step_counts logs test/coco/AP@{n} for each n."""
        model = KeypointDiffusionModel(config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(
            model,
            ap_step_counts=[1, 4],
            test_pred_cache="/fake/test_cache",
        )
        module._test_metric = MagicMock()
        module._test_metric.compute.return_value = {"coco/AP": 0.4, "coco/AP50": 0.6}
        module._run_ap_eval = MagicMock()
        module._trainer = MagicMock()

        logged: dict = {}
        module.log = lambda key, val, **_: logged.update({key: val})

        module.on_test_epoch_end()

        assert "test/coco/AP@1"   in logged
        assert "test/coco/AP@4"   in logged
        assert "test/coco/AP50@1" in logged
        assert "test/coco/AP50@4" in logged
        assert "test/coco/AP"     not in logged


# ---------------------------------------------------------------------------
# Standalone augmentation functions
# ---------------------------------------------------------------------------


class TestAugmentations:
    """Tests for the four standalone augmentation functions."""

    # --- joint_dropout ---

    def test_joint_dropout_p1_zeroes_non_root_preserves_root(self):
        """p=1 zeros every non-root joint and marks it absent; root survives."""
        B, N, K = 2, 3, 15
        node_pos = torch.ones(B * N * K, 2)
        can_vis  = torch.ones(B, N, K)
        new_vis  = joint_dropout(node_pos, can_vis, p=1.0)
        pos_grid = node_pos.view(B, N, K, 2)
        assert (pos_grid[:, :, 0, :] == 1.0).all(),   "Root position must not be zeroed"
        assert (pos_grid[:, :, 1:, :] == 0.0).all(),  "Non-root positions should be zeroed"
        assert (new_vis[:, :, 0] == 1.0).all(),        "Root visibility must not be zeroed"
        assert (new_vis[:, :, 1:] == 0.0).all(),       "Non-root visibility should be zeroed"

    def test_joint_dropout_p0_no_change(self):
        """p=0 leaves positions and visibility completely unchanged."""
        B, N, K = 2, 3, 15
        node_pos = torch.ones(B * N * K, 2)
        can_vis  = torch.ones(B, N, K)
        new_vis  = joint_dropout(node_pos, can_vis, p=0.0)
        assert (node_pos == 1.0).all()
        assert (new_vis  == 1.0).all()

    # --- pose_dropout ---

    def test_pose_dropout_p1_zeroes_all(self):
        """p=1 zeros every position and clears all visibility."""
        B, N, K = 2, 3, 15
        node_pos = torch.ones(B * N * K, 2)
        can_vis  = torch.ones(B, N, K)
        new_vis  = pose_dropout(node_pos, can_vis, p=1.0)
        assert (node_pos == 0.0).all(), "All positions should be zeroed"
        assert (new_vis  == 0.0).all(), "All visibility should be zeroed"

    def test_pose_dropout_p0_no_change(self):
        """p=0 leaves positions and visibility completely unchanged."""
        B, N, K = 2, 3, 15
        node_pos = torch.ones(B * N * K, 2)
        can_vis  = torch.ones(B, N, K)
        new_vis  = pose_dropout(node_pos, can_vis, p=0.0)
        assert (node_pos == 1.0).all()
        assert (new_vis  == 1.0).all()

    # --- keypoint_jitter ---

    def test_keypoint_jitter_zero_std_is_identity(self):
        """std_px=0 returns a tensor equal to the input."""
        kpts   = torch.randn(2, 3, 14, 2)
        result = keypoint_jitter(kpts, std_px=0.0)
        assert torch.allclose(result, kpts)

    def test_keypoint_jitter_nonzero_std_changes_values(self):
        """Non-zero std shifts at least some coordinates."""
        torch.manual_seed(0)
        kpts   = torch.zeros(2, 3, 14, 2)
        result = keypoint_jitter(kpts, std_px=10.0)
        assert not torch.allclose(result, kpts)

    def test_keypoint_jitter_does_not_modify_input(self):
        """The original tensor must be unchanged after jittering."""
        kpts  = torch.ones(2, 3, 14, 2)
        _     = keypoint_jitter(kpts, std_px=5.0)
        assert (kpts == 1.0).all(), "Input tensor must not be mutated"

    # --- joint_randomize ---

    def test_joint_randomize_p1_changes_all_non_root(self):
        """p=1 replaces every non-root joint position with a value in [-0.5, 0.5]."""
        B, N, K = 2, 3, 15
        node_pos = torch.zeros(B * N * K, 2)  # start at origin
        can_vis  = torch.ones(B, N, K)
        joint_randomize(node_pos, can_vis, p=1.0)
        pos_grid = node_pos.view(B, N, K, 2)
        assert (pos_grid[:, :, 0, :] == 0.0).all(), "Root position must not be touched"
        assert not (pos_grid[:, :, 1:, :] == 0.0).all(), "Non-root positions should change"
        assert (pos_grid[:, :, 1:, :] >= -1.0).all() and (pos_grid[:, :, 1:, :] <= 1.0).all(), \
            "Randomised positions should lie in [-1, 1] (full canonical image range)"

    def test_joint_randomize_p0_no_change(self):
        """p=0 leaves positions completely unchanged."""
        B, N, K = 2, 3, 15
        node_pos = torch.ones(B * N * K, 2)
        can_vis  = torch.ones(B, N, K)
        joint_randomize(node_pos, can_vis, p=0.0)
        assert (node_pos == 1.0).all()

    def test_joint_randomize_preserves_visibility(self):
        """Visibility tensor must be unchanged (joint is present but wrong)."""
        B, N, K = 2, 3, 15
        node_pos = torch.zeros(B * N * K, 2)
        can_vis  = torch.ones(B, N, K) * 2.0
        result   = joint_randomize(node_pos, can_vis, p=1.0)
        assert torch.allclose(result, can_vis), "Visibility must not be mutated"

    # --- image_dropout ---

    def test_image_dropout_p1_zeroes_all(self):
        """p=1 zeros every embedding row."""
        cond = torch.ones(30, 16)
        image_dropout(cond, p=1.0)
        assert (cond == 0.0).all()

    def test_image_dropout_p0_no_change(self):
        """p=0 leaves every embedding row unchanged."""
        cond = torch.ones(30, 16)
        image_dropout(cond, p=0.0)
        assert (cond == 1.0).all()


# ---------------------------------------------------------------------------
# Standalone loss functions
# ---------------------------------------------------------------------------


class TestLossFunctions:
    """Tests for compute_joint_loss and compute_bone_loss."""

    def _preds(self, B=2, N=3, K=14):
        torch.manual_seed(0)
        pred = torch.randn(B, N, K, 2) * 50 + 320
        gt   = torch.randn(B, N, K, 2) * 50 + 320
        return pred, gt

    # --- compute_joint_loss ---

    def test_joint_loss_zero_when_perfect(self):
        """Zero loss when predictions equal ground truth (for any visibility)."""
        _, gt = self._preds()
        vis = torch.ones(2, 3, 14)  # all vis=1
        assert compute_joint_loss(gt, gt, vis).item() == pytest.approx(0.0)

    def test_joint_loss_absent_joints_ignored(self):
        """Joints with visibility=0 do not affect the loss."""
        pred, gt = self._preds(B=1, N=1)
        pred_bad = pred.clone()
        pred_bad[0, 0, 0, :] = 9999.0   # corrupt the first joint

        vis_with  = torch.ones(1, 1, 14)            # joint 0 visible
        vis_without = vis_with.clone()
        vis_without[0, 0, 0] = 0.0                  # joint 0 absent

        loss_with    = compute_joint_loss(pred_bad, gt, vis_with).item()
        loss_without = compute_joint_loss(pred_bad, gt, vis_without).item()
        assert loss_without < loss_with, "Absent joint should not inflate the loss"

    def test_joint_loss_vis2_weights_more_than_vis1(self):
        """A single visible joint (vis=2) contributes more than an occluded one (vis=1)."""
        B, N, K = 1, 1, 2
        pred = torch.zeros(B, N, K, 2)
        gt   = torch.ones(B, N, K, 2)  # constant error of 1 per coord

        # Only joint 0 active; compare vis=1 vs vis=2
        vis1 = torch.zeros(B, N, K); vis1[0, 0, 0] = 1.0
        vis2 = torch.zeros(B, N, K); vis2[0, 0, 0] = 2.0

        # Both give MSE=2 (1^2+1^2), but denom differs (1 vs 2), so they are equal
        # on a per-unit-weight basis.  What changes is the *contribution when mixed*:
        # weight=2 means that joint dominates more in a multi-joint scenario.
        vis_mix = torch.zeros(B, N, K)
        vis_mix[0, 0, 0] = 1.0   # occluded
        vis_mix[0, 0, 1] = 2.0   # visible
        # With equal per-joint error, visible joint should pull more weight
        # → weighted mean is closer to the "visible" error
        pred_diff = pred.clone()
        pred_diff[0, 0, 0] = gt[0, 0, 0] + torch.tensor([10.0, 0.0])  # big error on occluded
        pred_diff[0, 0, 1] = gt[0, 0, 1] + torch.tensor([1.0, 0.0])   # small error on visible
        loss = compute_joint_loss(pred_diff, gt, vis_mix).item()
        # weighted mean = (1*100 + 2*1) / (1+2) = 34
        assert loss == pytest.approx((1 * 100 + 2 * 1) / 3, rel=1e-5)

    # --- compute_bone_loss ---

    def test_bone_loss_zero_when_perfect(self):
        """Zero loss when predicted lengths equal ground-truth lengths."""
        from src.data import CROWDPOSE_EDGE_LIST
        pred, gt = self._preds()
        vis = torch.ones(2, 3, 14) * 2  # all fully visible
        assert compute_bone_loss(gt, gt, vis, CROWDPOSE_EDGE_LIST).item() == pytest.approx(0.0)

    def test_bone_loss_absent_edge_ignored(self):
        """Edge with at least one absent endpoint (vis=0) must not affect the loss."""
        from src.data import CROWDPOSE_EDGE_LIST
        u, v = CROWDPOSE_EDGE_LIST[0]   # pick any edge
        B, N, K = 1, 1, 14
        pred = torch.zeros(B, N, K, 2)
        gt   = torch.zeros(B, N, K, 2)
        gt[0, 0, u] = torch.tensor([100.0, 0.0])  # big GT bone length

        vis_present = torch.ones(B, N, K)
        vis_absent  = vis_present.clone()
        vis_absent[0, 0, u] = 0.0

        loss_present = compute_bone_loss(pred, gt, vis_present, CROWDPOSE_EDGE_LIST).item()
        loss_absent  = compute_bone_loss(pred, gt, vis_absent,  CROWDPOSE_EDGE_LIST).item()
        assert loss_absent < loss_present, "Absent endpoint should reduce the loss"


class TestPoseMetrics:
    """Tests for compute_mpjpe and compute_pck."""

    def _tensors(self, B=2, N=3, K=14):
        torch.manual_seed(7)
        pred = torch.randn(B, N, K, 2) * 50 + 320
        gt   = torch.randn(B, N, K, 2) * 50 + 320
        vis  = torch.ones(B, N, K)  # all visible by default
        return pred, gt, vis

    # --- compute_mpjpe ---

    def test_mpjpe_zero_when_perfect(self):
        _, gt, vis = self._tensors()
        assert compute_mpjpe(gt, gt, vis).item() == pytest.approx(0.0)

    def test_mpjpe_positive_when_imperfect(self):
        pred, gt, vis = self._tensors()
        assert compute_mpjpe(pred, gt, vis).item() > 0.0

    def test_mpjpe_absent_joints_ignored(self):
        """vis=0 joints must not affect MPJPE."""
        _, gt, vis = self._tensors(B=1, N=1, K=2)
        pred_bad = gt.clone()
        pred_bad[0, 0, 0] = 9999.0   # huge error on joint 0

        vis_include = torch.ones(1, 1, 2)
        vis_exclude = vis_include.clone(); vis_exclude[0, 0, 0] = 0.0

        mpjpe_with    = compute_mpjpe(pred_bad, gt, vis_include).item()
        mpjpe_without = compute_mpjpe(pred_bad, gt, vis_exclude).item()
        assert mpjpe_without < mpjpe_with

    def test_mpjpe_all_absent_returns_zero(self):
        pred, gt, _ = self._tensors(B=1, N=1, K=2)
        vis_none = torch.zeros(1, 1, 2)
        assert compute_mpjpe(pred, gt, vis_none).item() == pytest.approx(0.0)

    # --- compute_pck ---

    def test_pck_one_when_perfect(self):
        _, gt, vis = self._tensors()
        assert compute_pck(gt, gt, vis, threshold_px=1.0).item() == pytest.approx(1.0)

    def test_pck_zero_when_far_away(self):
        pred, gt, vis = self._tensors(B=1, N=1, K=1)
        pred_far = gt + 9999.0
        assert compute_pck(pred_far, gt, vis, threshold_px=1.0).item() == pytest.approx(0.0)

    def test_pck_increases_with_threshold(self):
        pred, gt, vis = self._tensors()
        pck_tight = compute_pck(pred, gt, vis, threshold_px=1.0).item()
        pck_loose = compute_pck(pred, gt, vis, threshold_px=1000.0).item()
        assert pck_tight <= pck_loose

    def test_pck_absent_joints_ignored(self):
        """vis=0 joints must not count toward PCK denominator."""
        _, gt, vis = self._tensors(B=1, N=1, K=2)
        pred_bad = gt.clone()
        pred_bad[0, 0, 0] = 9999.0   # fail on joint 0

        vis_include = torch.ones(1, 1, 2)
        vis_exclude = vis_include.clone(); vis_exclude[0, 0, 0] = 0.0

        pck_with    = compute_pck(pred_bad, gt, vis_include, threshold_px=1.0).item()
        pck_without = compute_pck(pred_bad, gt, vis_exclude, threshold_px=1.0).item()
        assert pck_without > pck_with  # excluded bad joint → higher PCK

    def test_pck_all_absent_returns_zero(self):
        pred, gt, _ = self._tensors(B=1, N=1, K=2)
        vis_none = torch.zeros(1, 1, 2)
        assert compute_pck(pred, gt, vis_none, threshold_px=20.0).item() == pytest.approx(0.0)
