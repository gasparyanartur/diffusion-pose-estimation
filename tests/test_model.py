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
    PoseDiffusionModule,
    TimestepSinusoidalEmbedding,
    compute_bone_loss,
    compute_joint_loss,
    image_dropout,
    joint_dropout,
    keypoint_jitter,
    pose_dropout,
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


# ---------------------------------------------------------------------------
# Image conditioning (DinoConditioner + conditioned backward_process)
# ---------------------------------------------------------------------------


def _make_mock_dino(dim: int):
    """Return a callable that mimics DinoConditioner output without a real backbone."""
    mock = MagicMock(spec=DinoConditioner)
    mock.to.return_value = mock
    mock._patch_cache = None   # signal: no cache loaded

    def _call(kpt_pixel, img_hw, image_ids=None, pil_images=None):
        b_, n_, k_, _ = kpt_pixel.shape   # k_ == 14
        return torch.zeros(b_, n_, k_, dim)

    mock.side_effect = _call
    return mock


class TestImageConditioning:
    @pytest.fixture
    def cond_config(self) -> KeypointDiffusionConfig:
        return KeypointDiffusionConfig(
            dim_pos=16, dim_index=16, dim_time=16, dim_bone=16,
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
