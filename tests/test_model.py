"""Unit tests for src.model — KeypointDiffusionModel and PoseDiffusionModule."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data import (
    PoseSample,
    PoseGraphBatch,
    make_edge_index,
    prepare_sample,
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
    pixel_to_grid_sample_coords,
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
NUM_KEYPOINTS_CANONICAL = 14   # no virtual root; same as CrowdPose
BATCH_SIZE = 2
N_POSES = 2  # people per image
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
        num_layers=1,
        num_train_steps=50,
    )


@pytest.fixture
def model(config) -> KeypointDiffusionModel:
    return KeypointDiffusionModel(config=config, num_keypoints=NUM_KEYPOINTS_CANONICAL)


@pytest.fixture
def module(model) -> PoseDiffusionModule:
    return PoseDiffusionModule(model=model)


@pytest.fixture
def batch_b4n3() -> dict[str, Any]:
    """Canonical B=4, N=3 batch dict."""
    return _make_batch(B=BATCH_SIZE, N=N_POSES)


@pytest.fixture
def sample_b4n3(batch_b4n3) -> PoseSample:
    """Canonical B=4, N=3 PoseSample."""
    return prepare_sample(batch_b4n3)


@pytest.fixture
def graph_b4n3(sample_b4n3) -> PoseGraphBatch:
    """Canonical B=4, N=3 PoseGraphBatch."""
    return prepare_pose_graph_batch(sample_b4n3)


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
    def test_output_shapes(self, model, graph_b4n3):
        g = graph_b4n3
        timestep = torch.randint(0, model.config.num_train_steps, (g.B,))
        noisy, noise = model.add_noise(g.node_pos, timestep, g.node_batch)
        assert noisy.shape == g.node_pos.shape
        assert noise.shape == g.node_pos.shape

    def test_zero_timestep_leaves_positions_unchanged(self, model, graph_b4n3):
        """At t=0 sigma=0 so add_noise must return x unchanged (noise contribution = sigma*eps = 0)."""
        g = graph_b4n3
        timestep = torch.zeros(g.B, dtype=torch.long)
        noisy, noise = model.add_noise(g.node_pos, timestep, g.node_batch)
        assert torch.allclose(noisy, g.node_pos, atol=1e-6)
        assert noise.shape == g.node_pos.shape

    def test_max_timestep_adds_full_noise(self, model, graph_b4n3):
        """At t≈T the noise level ≈ sigma_max; with x0=0 the noisy output equals sigma*eps."""
        g = graph_b4n3
        T = model.config.num_train_steps
        node_pos = torch.zeros_like(g.node_pos)  # x0=0: noisy − 0 = noise contribution
        timestep = torch.full((g.B,), T - 1, dtype=torch.long)
        noisy, noise = model.add_noise(node_pos, timestep, g.node_batch)
        assert noise.abs().mean().item() > 0.1

    def test_noise_scales_with_timestep(self, model, graph_b4n3):
        """sigma_t grows with t: deviation from clean positions must increase with timestep."""
        g = graph_b4n3
        node_pos = torch.zeros_like(g.node_pos)  # x0=0: deviation = sigma_t * eps
        torch.manual_seed(0)
        noisy_low,  _ = model.add_noise(node_pos, torch.full((g.B,), 10, dtype=torch.long), g.node_batch)
        torch.manual_seed(0)
        noisy_high, _ = model.add_noise(node_pos, torch.full((g.B,), 40, dtype=torch.long), g.node_batch)
        assert noisy_high.abs().mean() > noisy_low.abs().mean()


# ---------------------------------------------------------------------------
# KeypointDiffusionModel.backward_process
# ---------------------------------------------------------------------------


class TestBackwardProcess:
    def test_output_shape(self, model, graph_b4n3):
        g = graph_b4n3
        timestep = torch.randint(0, model.config.num_train_steps, (g.B,))
        pred = model.backward_process(
            noisy_node_pos=g.node_pos,
            edge_index=g.edge_index,
            kpt_type_idx=g.flat_kpt_type_idx,
            batch_idx=g.node_batch,
            timestep=timestep,
        )
        assert pred.shape == (g.num_nodes, 2)

    def test_output_is_float32(self, model, graph_b4n3):
        g = graph_b4n3
        timestep = torch.randint(0, model.config.num_train_steps, (g.B,))
        pred = model.backward_process(
            noisy_node_pos=g.node_pos,
            edge_index=g.edge_index,
            kpt_type_idx=g.flat_kpt_type_idx,
            batch_idx=g.node_batch,
            timestep=timestep,
        )
        assert pred.dtype == torch.float32

    def test_gradients_flow_through_prediction(self, model, graph_b4n3):
        """Loss computed on backward_process output should allow gradients to flow."""
        g = graph_b4n3
        timestep = torch.randint(0, model.config.num_train_steps, (g.B,))
        pred = model.backward_process(
            noisy_node_pos=g.node_pos,
            edge_index=g.edge_index,
            kpt_type_idx=g.flat_kpt_type_idx,
            batch_idx=g.node_batch,
            timestep=timestep,
        )
        pred.mean().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, "No gradients reached model parameters"

    def test_zero_delta_returns_zeros(self, config, graph_b4n3):
        """zero_delta=True must return an all-zero delta regardless of input."""
        config.zero_delta = True
        model = KeypointDiffusionModel(config=config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        g = graph_b4n3
        timestep = torch.randint(1, config.num_train_steps + 1, (g.B,))
        pred = model.backward_process(
            noisy_node_pos=g.node_pos,
            edge_index=g.edge_index,
            kpt_type_idx=g.flat_kpt_type_idx,
            batch_idx=g.node_batch,
            timestep=timestep,
        )
        assert pred.shape == (g.num_nodes, 2)
        assert pred.eq(0).all(), "Expected all-zero delta when zero_delta=True"


# ---------------------------------------------------------------------------
# PoseDiffusionModule.disable_noise
# ---------------------------------------------------------------------------


class TestDisableNoise:
    def test_disable_noise_uses_clean_init(self, model, batch_b4n3):
        """_disable_noise=True must pass clean node_pos as x_init to ddim_sample."""
        module = PoseDiffusionModule(model=model, disable_noise=True)

        # _refine_prediction_batch expects pred_keypoints / pred_keypoint_scores keys
        refine_batch = {
            "image_id":              batch_b4n3["image_id"],
            "pred_keypoints":        batch_b4n3["keypoints"],
            "pred_keypoint_scores":  batch_b4n3["visibility"],
        }

        sample, graph = module.prepare_batch({
            **batch_b4n3,
            "keypoints": batch_b4n3["keypoints"],
            "visibility": batch_b4n3["visibility"],
        })

        captured = {}
        original_ddim = model.ddim_sample

        def mock_ddim(*args, x_init=None, **kwargs):
            captured["x_init"] = x_init
            return original_ddim(*args, x_init=x_init, **kwargs)

        model.ddim_sample = mock_ddim
        try:
            module._refine_prediction_batch(refine_batch)
        finally:
            model.ddim_sample = original_ddim

        assert captured["x_init"] is not None
        assert torch.allclose(captured["x_init"], graph.node_pos), (
            "Expected x_init to equal clean graph.node_pos when disable_noise=True"
        )


# ---------------------------------------------------------------------------
# PoseDiffusionModule (end-to-end)
# ---------------------------------------------------------------------------


class TestPoseDiffusionModule:
    def _make_loader(self, n_images: int = 8, n_poses: int = N_POSES) -> DataLoader:
        samples = _make_pose_samples(n_images=n_images, n_poses=n_poses)
        ds = PoseDataset(samples=samples)
        sampler = SceneGraphSampler(ds, batch_size=BATCH_SIZE, shuffle=False)
        return DataLoader(ds, batch_sampler=sampler)

    def test_training_step_returns_scalar_loss(self, module, batch_b4n3):
        loss = module.training_step(batch_b4n3, batch_idx=0)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_training_step_loss_is_finite(self, module, batch_b4n3):
        loss = module.training_step(batch_b4n3, batch_idx=0)
        assert torch.isfinite(loss)

    def test_training_step_gradients_propagate(self, module, batch_b4n3):
        """backward() on the training-step loss must populate all parameter gradients."""
        loss = module.training_step(batch_b4n3, batch_idx=0)
        loss.backward()
        for name, param in module.named_parameters():
            assert param.grad is not None, f"Parameter '{name}' has no gradient"

    def test_training_step_loss_decreases_over_steps(self, module, batch_b4n3):
        """A few gradient steps with Adam should reduce the training loss."""
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-2)
        torch.manual_seed(0)
        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            loss = module.training_step(batch_b4n3, batch_idx=0)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], f"Loss did not decrease over 5 steps: {losses}"

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
    """Return a stub that mimics DinoConditioner without a real backbone or cache.

    Args:
        dim:     Output dim of the projected forward() call.
        raw_dim: hidden_dim (pooler / stage4 channels). Defaults to dim.
    """
    if raw_dim is None:
        raw_dim = dim
    mock = MagicMock(spec=DinoConditioner)
    mock.to.return_value = mock
    mock.hidden_dim = raw_dim  # used when constructing global_dino_proj

    def _forward(kpt_pixel, img_hw, image_ids=None):
        b_, n_, k_, _ = kpt_pixel.shape
        return torch.zeros(b_, n_, k_, dim)

    def _fetch_global(image_ids, device=None):
        return torch.zeros(len(image_ids), raw_dim)

    mock.side_effect = _forward
    mock.fetch_global_feats = MagicMock(side_effect=_fetch_global)
    return mock



# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# pixel_to_grid_sample_coords
# ---------------------------------------------------------------------------


def test_pixel_to_grid_sample_coords_round_trip():
    # Image 640×426 (landscape), short side 426, resize to shortest_edge=224.
    # Scale s = 224/426.  Stride for stage1 = 4.
    # Feature map size after resize: H_feat = 224/4 = 56, W_feat = ceil(640*s/4).
    # For each feature cell (i, j), its centre in original pixel space is:
    #   x = (j + 0.5) * stride / s = (j + 0.5) * 4 * 426 / 224
    #   y = (i + 0.5) * stride / s = (i + 0.5) * 4 * 426 / 224
    # pixel_to_grid_sample_coords should map that pixel back to exactly
    # the grid_sample coord that samples cell (i, j).
    W_orig, H_orig = 640, 426
    shortest_edge = 224
    stride = 4
    s = shortest_edge / min(H_orig, W_orig)  # 224/426
    H_feat = int(H_orig * s / stride)  # 26
    W_feat = int(W_orig * s / stride)  # 37

    # Feature map where each cell holds its own flattened index.
    feat = torch.arange(H_feat * W_feat, dtype=torch.float32).view(1, 1, H_feat, W_feat)

    img_hw = torch.tensor([[float(H_orig), float(W_orig)]])

    # Sample a grid of 5×5 cells spread across the map.
    sample_rows = torch.linspace(0, H_feat - 1, 5).long()
    sample_cols = torch.linspace(0, W_feat - 1, 5).long()

    for i in sample_rows.tolist():
        for j in sample_cols.tolist():
            expected_val = float(i * W_feat + j)

            # Centre of cell (i, j) in original pixel coordinates.
            x_px = (j + 0.5) * stride / s
            y_px = (i + 0.5) * stride / s
            kpt = torch.tensor([[[[x_px, y_px]]]])  # (1, 1, 1, 2)

            grid_coord = pixel_to_grid_sample_coords(kpt, img_hw.view(1, 1, 1, 2))  # (1,1,1,2)
            grid = grid_coord.unsqueeze(0).view(1, 1, 1, 2)  # already (1,1,1,2)

            sampled = torch.nn.functional.grid_sample(
                feat, grid, mode="nearest", align_corners=False, padding_mode="zeros"
            )  # (1, 1, 1, 1)
            assert abs(sampled.item() - expected_val) < 1e-3, (
                f"cell ({i},{j}): expected {expected_val}, got {sampled.item()}"
            )


# Real DinoConditioner unit tests (no backbone, synthetic cache)
# ---------------------------------------------------------------------------


def _make_synthetic_cache(image_ids: list[int]) -> dict:
    """Build a minimal {image_id: dict} cache matching ConvNeXt-Small shapes."""
    return {
        iid: {
            "spatial":  torch.zeros(96, 56, 56),
            "semantic": torch.zeros(768, 7, 7),
            "global":   torch.zeros(768),
        }
        for iid in image_ids
    }


def _make_variable_size_cache(specs: list[tuple[int, int, int]]) -> dict:
    """Build a cache where each image has a different spatial size.

    specs: [(image_id, H_orig, W_orig), ...]
    Spatial dim = H//4 x W//4 (stride-4 stage1), semantic = H//32 x W//32.
    """
    cache = {}
    for iid, h, w in specs:
        cache[iid] = {
            "spatial":  torch.zeros(96,  h // 4,  w // 4),
            "semantic": torch.zeros(768, h // 32, w // 32),
            "global":   torch.zeros(768),
        }
    return cache


class TestDinoConditionerReal:
    """Tests that exercise the actual DinoConditioner code (no mocking)."""

    def _make_conditioner(
        self,
        image_ids: list[int],
        out_dim: int = 16,
        cache_dir: Path | None = None,
    ) -> DinoConditioner:
        cond = DinoConditioner(out_dim=out_dim)
        cond._cache = _make_synthetic_cache(image_ids)
        # Register a (possibly empty) cache dir so _load_image can reach the
        # 'not found' branch when an unknown id is requested.
        if cache_dir is not None:
            cond._cache_dirs = [cache_dir]
        return cond

    def test_forward_output_shape(self):
        B, N, K, out_dim = 2, 3, 14, 16
        cond = self._make_conditioner(list(range(B)), out_dim=out_dim)
        kpt_pixel = torch.rand(B, N, K, 2) * 640
        img_hw = torch.tensor([[480.0, 640.0]] * B)
        out = cond(kpt_pixel, img_hw, image_ids=list(range(B)))
        assert out.shape == (B, N, K, out_dim)

    def test_forward_variable_image_sizes(self):
        # Two images with different aspect ratios → different spatial map sizes.
        N, K, out_dim = 2, 14, 16
        specs = [(0, 224, 224), (1, 224, 336)]  # square and landscape
        cond = DinoConditioner(out_dim=out_dim)
        cond._cache = _make_variable_size_cache(specs)
        img_hw = torch.tensor([[224.0, 224.0], [224.0, 336.0]])
        kpt_pixel = torch.rand(2, N, K, 2) * 224
        out = cond(kpt_pixel, img_hw, image_ids=[0, 1])
        assert out.shape == (2, N, K, out_dim)

    def test_forward_output_is_finite(self):
        B, N, K = 2, 2, 14
        cond = self._make_conditioner(list(range(B)))
        kpt_pixel = torch.rand(B, N, K, 2) * 640
        img_hw = torch.tensor([[480.0, 640.0]] * B)
        out = cond(kpt_pixel, img_hw, image_ids=list(range(B)))
        assert torch.isfinite(out).all()

    def test_fetch_global_feats_shape(self):
        B = 3
        cond = self._make_conditioner(list(range(B)))
        out = cond.fetch_global_feats(list(range(B)), device=torch.device("cpu"))
        assert out.shape == (B, 768)

    def test_missing_image_id_raises(self, tmp_path):
        cond = self._make_conditioner([0, 1], cache_dir=tmp_path)
        kpt_pixel = torch.rand(1, 2, 14, 2) * 640
        img_hw = torch.tensor([[480.0, 640.0]])
        with pytest.raises(RuntimeError, match="not found"):
            cond(kpt_pixel, img_hw, image_ids=[999])

    def test_no_cache_raises(self):
        cond = DinoConditioner(out_dim=16)
        kpt_pixel = torch.rand(1, 2, 14, 2)
        img_hw = torch.tensor([[480.0, 640.0]])
        with pytest.raises(RuntimeError, match="cache not loaded|Call load_cache"):
            cond(kpt_pixel, img_hw, image_ids=[0])

    def test_load_cache_wrong_format_raises(self, tmp_path):
        # load_cache now expects a directory; a plain file should raise
        bad_file = tmp_path / "vit.pt"
        bad_file.write_bytes(b"dummy")
        cond = DinoConditioner(out_dim=16)
        with pytest.raises(RuntimeError, match="not a directory"):
            cond.load_cache(bad_file)

    def test_load_cache_skips_missing_dir(self, tmp_path, capsys):
        cond = DinoConditioner(out_dim=16)
        cond.load_cache(tmp_path / "nonexistent")
        captured = capsys.readouterr()
        assert "skipping" in captured.out
        assert cond._cache_dirs == []


class TestImageConditioning:
    @pytest.fixture
    def cond_config(self) -> KeypointDiffusionConfig:
        return KeypointDiffusionConfig(
            dim_pos=16, dim_index=16, dim_time=16, dim_edge=16,
            dim_hidden=16, num_layers=2, num_train_steps=10,
            use_image_cond=True, dim_image_cond=8,
        )

    def test_backward_process_accepts_image_cond(self, cond_config, graph_b4n3):
        """backward_process runs without error when image_cond is supplied."""
        model = KeypointDiffusionModel(cond_config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        g = graph_b4n3
        image_cond = torch.zeros(g.num_nodes, 8)
        t = torch.ones(g.B, dtype=torch.long)
        pred = model.backward_process(
            noisy_node_pos=g.node_pos,
            edge_index=g.edge_index,
            kpt_type_idx=g.flat_kpt_type_idx,
            batch_idx=g.node_batch,
            timestep=t,
            image_cond=image_cond,
        )
        assert pred.shape == (g.num_nodes, 2)

    def test_prepare_batch_adds_image_cond(self, cond_config, batch_b4n3):
        """prepare_batch populates image_cond with the right node count."""
        model = KeypointDiffusionModel(cond_config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(model)
        module.dino_conditioner = _make_mock_dino(8)

        batch = {**batch_b4n3, "img_path": [f"/fake/img_{i}.jpg" for i in range(BATCH_SIZE)]}
        with patch("src.model.read_image_size", return_value=(480, 640)):
            _, graph = module.prepare_batch(batch)

        assert graph.image_cond is not None
        assert graph.image_cond.shape == (graph.num_nodes, 8)

    def test_training_step_with_image_cond(self, cond_config, batch_b4n3):
        """End-to-end training step with mock DINO produces a finite loss."""
        model = KeypointDiffusionModel(cond_config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(model)
        module.dino_conditioner = _make_mock_dino(8)

        batch = {**batch_b4n3, "img_path": [f"/fake/img_{i}.jpg" for i in range(BATCH_SIZE)]}
        with patch("src.model.read_image_size", return_value=(480, 640)):
            loss = module.training_step(batch, batch_idx=0)

        assert loss.ndim == 0 and torch.isfinite(loss)


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

    def test_get_global_image_cond_shape(self, global_config, sample_b4n3, graph_b4n3):
        """get_global_image_cond returns (num_nodes, dim_global_image_cond)."""
        module = self._module_with_global(global_config)
        result = module.get_global_image_cond(sample_b4n3, graph_b4n3)
        assert result is not None
        assert result.shape == (graph_b4n3.num_nodes, self._GLOBAL_DIM)

    def test_get_global_image_cond_none_without_proj(self, sample_b4n3, graph_b4n3):
        """Returns None when use_global_image_cond=False (no proj created)."""
        cfg = KeypointDiffusionConfig(dim_time=16, num_train_steps=10)
        model = KeypointDiffusionModel(cfg, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(model)
        assert module.get_global_image_cond(sample_b4n3, graph_b4n3) is None

    def test_raises_if_global_cond_enabled_without_dino(self, global_config):
        """PoseDiffusionModule raises ValueError at init when use_global_image_cond=True
        but no DINO conditioner is provided — prevents a silent shape mismatch at runtime."""
        model = KeypointDiffusionModel(global_config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        with pytest.raises(ValueError, match="use_global_image_cond"):
            PoseDiffusionModule(model)  # no dino_model_name → dino_conditioner is None

    def test_backward_process_accepts_global_image_cond(self, global_config, graph_b4n3):
        """backward_process runs without error when global_image_cond is supplied."""
        model = KeypointDiffusionModel(global_config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        g = graph_b4n3
        t = torch.ones(g.B, dtype=torch.long)
        pred = model.backward_process(
            noisy_node_pos=g.node_pos,
            edge_index=g.edge_index,
            kpt_type_idx=g.flat_kpt_type_idx,
            batch_idx=g.node_batch,
            timestep=t,
            global_image_cond=torch.zeros(g.num_nodes, self._GLOBAL_DIM),
        )
        assert pred.shape == (g.num_nodes, 2)

    def test_training_step_with_global_cond(self, global_config, batch_b4n3):
        """End-to-end training step with global DINO conditioning produces a finite loss."""
        module = self._module_with_global(global_config)
        batch = {**batch_b4n3, "img_path": [f"/fake/img_{i}.jpg" for i in range(BATCH_SIZE)]}
        with patch("src.model.read_image_size", return_value=(480, 640)):
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
        M = self._B * self._N * 14
        mock_ddim = MagicMock(return_value=torch.zeros(M, 2))
        module.model.ddim_sample = mock_ddim

        module._refine_prediction_batch(self._pred_batch(), num_steps=3)

        _args, call_kwargs = mock_ddim.call_args
        assert call_kwargs["num_inference_steps"] == 3, (
            "ddim_sample must receive num_inference_steps=3"
        )

    def test_ap_step_passes_num_steps_as_n_passes(self, config):
        """_ap_step forwards num_steps as n_passes to _iterative_refine_batch."""
        model = KeypointDiffusionModel(config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(model)
        M = self._B * self._N * 14
        module.model.ddim_sample = MagicMock(return_value=torch.zeros(M, 2))

        captured: list[int] = []
        orig = module._iterative_refine_batch

        def _spy(batch, n_passes):
            captured.append(n_passes)
            return orig(batch, n_passes=n_passes)

        module._iterative_refine_batch = _spy
        module._ap_step(self._pred_batch(), MagicMock(), num_steps=7)

        assert captured == [7]

    def test_raw_refiner_skips_noise_and_uses_one_ddim_step(self, config):
        """raw_refiner=True: add_noise is never called and ddim_sample gets num_inference_steps=1."""
        model = KeypointDiffusionModel(config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(model, raw_refiner=True)
        M = self._B * self._N * 14

        add_noise_calls: list = []
        orig_add_noise = module.model.add_noise
        def _spy_add_noise(**kw):
            add_noise_calls.append(kw)
            return orig_add_noise(**kw)
        module.model.add_noise = _spy_add_noise

        ddim_calls: list = []
        module.model.ddim_sample = MagicMock(
            side_effect=lambda **kw: (ddim_calls.append(kw) or torch.zeros(M, 2))
        )

        module._refine_prediction_batch(self._pred_batch())

        assert len(add_noise_calls) == 0, "raw_refiner must not call add_noise"
        assert len(ddim_calls) == 1
        assert ddim_calls[0]["num_inference_steps"] == 1, (
            f"raw_refiner must use 1 DDIM step, got {ddim_calls[0]['num_inference_steps']}"
        )

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
        module._val_metric.compute.return_value = {"crowdpose/AP": 0.5}
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
        module._val_metric.compute.return_value = {"crowdpose/AP": 0.5, "coco/AP50": 0.7}
        module._run_ap_eval = MagicMock()
        module._trainer = MagicMock()
        module._trainer.sanity_checking = False
        module._trainer.current_epoch = 0

        logged: dict = {}
        module.log = lambda key, val, **_: logged.update({key: val})

        module.on_validation_epoch_end()

        assert "val/crowdpose/AP@2"   in logged
        assert "val/crowdpose/AP@8"   in logged
        assert "val/coco/AP50@2" in logged
        assert "val/coco/AP50@8" in logged
        assert "val/crowdpose/AP"     not in logged  # no bare key when step counts are set

    def test_validation_epoch_end_no_suffix_without_step_counts(self, config):
        """on_validation_epoch_end without ap_step_counts logs val/coco/AP (no suffix)."""
        model = KeypointDiffusionModel(config, num_keypoints=NUM_KEYPOINTS_CANONICAL)
        module = PoseDiffusionModule(
            model,
            ap_eval_interval=1,
            val_pred_cache="/fake/cache",
        )
        module._val_metric = MagicMock()
        module._val_metric.compute.return_value = {"crowdpose/AP": 0.5}
        module._run_ap_eval = MagicMock()
        module._trainer = MagicMock()
        module._trainer.sanity_checking = False
        module._trainer.current_epoch = 0

        logged: dict = {}
        module.log = lambda key, val, **_: logged.update({key: val})

        module.on_validation_epoch_end()

        assert "val/crowdpose/AP" in logged
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
        module._test_metric.compute.return_value = {"crowdpose/AP": 0.4, "coco/AP50": 0.6}
        module._run_ap_eval = MagicMock()
        module._trainer = MagicMock()

        logged: dict = {}
        module.log = lambda key, val, **_: logged.update({key: val})

        module.on_test_epoch_end()

        assert "test/crowdpose/AP@1"   in logged
        assert "test/crowdpose/AP@4"   in logged
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

    # --- _augment_sample and prepare_batch augmentation ---

    def test_augment_sample_creates_new_keypoint_tensor(self, module, sample_b4n3):
        """keypoint_jitter returns a new tensor; _augment_sample must replace sample.keypoints."""
        module.model.config.aug_jitter = True
        original_id = id(sample_b4n3.keypoints)
        module._augment_sample(sample_b4n3)
        assert id(sample_b4n3.keypoints) != original_id

    def test_augment_sample_gt_keypoints_unchanged(self, module, sample_b4n3):
        """gt_keypoints must not be reassigned by _augment_sample; it always points to the original tensor."""
        module.model.config.aug_jitter = True
        original_gt = sample_b4n3.gt_keypoints
        module._augment_sample(sample_b4n3)
        assert sample_b4n3.gt_keypoints is original_gt

    def test_prepare_batch_augment_changes_node_pos(self, module, batch_b4n3):
        """prepare_batch(augment=True) builds node_pos from jittered keypoints; node_pos must differ from the clean run."""
        module.model.config.aug_jitter = True
        _, graph_clean = module.prepare_batch(batch_b4n3, augment=False)
        torch.manual_seed(0)
        _, graph_noisy = module.prepare_batch(batch_b4n3, augment=True)
        assert not torch.allclose(graph_clean.node_pos, graph_noisy.node_pos), (
            "node_pos should differ with augment=True (jitter applied before graph construction)"
        )


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
