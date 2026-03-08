"""Tests for src.metrics — coordinate helpers and KeypointAPMetric.

These tests cover three layers of the AP pipeline in isolation, making it
easy to pinpoint where a regression occurs:

  1. canonical_to_pixel  — coordinate-space roundtrip and broadcasting.
  2. KeypointAPMetric.build_samples_for_image  — CocoMetric sample format.
  3. KeypointAPMetric (integration)  — full accumulate → compute cycle
     using the real CrowdPose val annotation file.  Tests in this class are
     skipped when the annotation file is not present (CI / fresh clone).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import prepare_pose_graph_batch
from src.metrics import KeypointAPMetric, canonical_to_pixel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_KP = 14   # raw CrowdPose joints (no virtual root)
NUM_KP_CANONICAL = 15  # after virtual root is prepended

VAL_ANN_FILE  = Path("data/crowdpose/annotations/crowdpose_val.json")
PRED_CACHE    = Path("results/cache/rtmpose-m_crowdpose_val_predictions.pkl")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_pixel_kpts(B: int = 2, N: int = 3, K: int = NUM_KP, scale: float = 640.0, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.rand(B, N, K, 2, generator=g) * scale


def _ones_vis(B: int, N: int, K: int = NUM_KP) -> torch.Tensor:
    return torch.ones(B, N, K)


# ===========================================================================
# 1.  canonical_to_pixel
# ===========================================================================


class TestCanonicalToPixel:
    """Verify that the coordinate inversion is exact and handles all input shapes."""

    def test_roundtrip_recovers_original_keypoints(self):
        """Encoding via prepare_pose_graph_batch then decoding via
        canonical_to_pixel should give back the exact pixel keypoints."""
        B, N = 3, 4
        kpts = _rand_pixel_kpts(B, N)
        vis  = _ones_vis(B, N)

        graph     = prepare_pose_graph_batch(kpts, vis)
        canonical = graph["node_pos"].view(B, N, NUM_KP_CANONICAL, 2)  # (B, N, 15, 2)
        recovered = canonical_to_pixel(canonical, graph["root_pos"], scale=graph["scale"])

        assert recovered.shape == (B, N, NUM_KP, 2)
        torch.testing.assert_close(recovered, kpts, atol=1e-4, rtol=1e-4,
                                   msg="canonical→pixel roundtrip must recover original keypoints")

    def test_virtual_root_is_dropped(self):
        """Output must have 14 joints (root index 0 is always dropped)."""
        B, N = 2, 3
        canonical = torch.rand(B, N, NUM_KP_CANONICAL, 2)
        root_pos  = torch.zeros(B, N, 2)
        out = canonical_to_pixel(canonical, root_pos, scale=1.0)
        assert out.shape == (B, N, NUM_KP, 2), (
            f"Expected (B={B}, N={N}, 14, 2), got {out.shape}"
        )

    def test_scale_one_with_zero_root_is_slice(self):
        """With scale=1 and root=[0,0] the output is exactly canonical[:,1:,:]."""
        B, N = 2, 3
        canonical = torch.rand(B, N, NUM_KP_CANONICAL, 2)
        root_pos  = torch.zeros(B, N, 2)
        out = canonical_to_pixel(canonical, root_pos, scale=1.0)
        torch.testing.assert_close(out, canonical[:, :, 1:, :])

    def test_tensor_scale_matches_float_scale(self):
        """Per-image Tensor scale must give the same result as the equivalent scalar."""
        B, N = 3, 2
        canonical = torch.rand(B, N, NUM_KP_CANONICAL, 2)
        root_pos  = torch.zeros(B, N, 2)
        scale_val = 512.0

        out_float  = canonical_to_pixel(canonical, root_pos, scale=scale_val)
        out_tensor = canonical_to_pixel(canonical, root_pos,
                                        scale=torch.full((B,), scale_val))

        torch.testing.assert_close(out_float, out_tensor)

    def test_tensor_scale_per_image(self):
        """Different per-image scales must be applied independently."""
        B, N = 2, 1
        canonical = torch.ones(B, N, NUM_KP_CANONICAL, 2)  # all ones
        root_pos  = torch.zeros(B, N, 2)
        scales = torch.tensor([100.0, 200.0])

        out = canonical_to_pixel(canonical, root_pos, scale=scales)
        # After dropping root: joint positions = scale * 1 + 0 = scale
        assert out[0, 0, 0, 0].item() == pytest.approx(100.0)
        assert out[1, 0, 0, 0].item() == pytest.approx(200.0)

    def test_zero_relative_position_maps_to_root(self):
        """A canonical position of [0, 0] (relative) should map back to root_pos."""
        B, N = 1, 1
        # canonical[0,0,1,:] = [0,0] means joint 0 (first real joint) is at the root
        canonical = torch.zeros(B, N, NUM_KP_CANONICAL, 2)
        root_pos  = torch.tensor([[[42.0, 99.0]]])  # (1, 1, 2)
        out = canonical_to_pixel(canonical, root_pos, scale=800.0)
        # Every joint is at [0,0] canonical → should map to root_pos
        torch.testing.assert_close(out[0, 0, 0], root_pos[0, 0])

    def test_root_pos_shift_applied_correctly(self):
        """Changing root_pos by delta should shift all output joints by the same delta."""
        B, N = 2, 3
        canonical = torch.rand(B, N, NUM_KP_CANONICAL, 2)
        root_a = torch.zeros(B, N, 2)
        root_b = torch.ones(B, N, 2) * 50.0

        out_a = canonical_to_pixel(canonical, root_a, scale=800.0)
        out_b = canonical_to_pixel(canonical, root_b, scale=800.0)

        torch.testing.assert_close(out_b - out_a, torch.ones_like(out_a) * 50.0)

    def test_roundtrip_with_partial_visibility(self):
        """Roundtrip still works correctly when some joints are invisible (vis=0)."""
        B, N = 2, 3
        kpts = _rand_pixel_kpts(B, N)
        vis  = (torch.rand(B, N, NUM_KP) > 0.4).float()  # ~60 % visible

        graph     = prepare_pose_graph_batch(kpts, vis)
        canonical = graph["node_pos"].view(B, N, NUM_KP_CANONICAL, 2)
        recovered = canonical_to_pixel(canonical, graph["root_pos"], scale=graph["scale"])

        torch.testing.assert_close(recovered, kpts, atol=1e-4, rtol=1e-4)


# ===========================================================================
# 2.  KeypointAPMetric.build_samples_for_image
# ===========================================================================


class TestBuildSamplesForImage:
    """Verify the CocoMetric data-sample format produced for a single image.

    This is the level at which the "all N people share one id → N-1 deleted"
    bug used to live.  These tests ensure the invariant is maintained.
    """

    def _make_inputs(self, N: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        kpts   = rng.random((N, NUM_KP, 2)).astype(np.float32) * 640
        scores = rng.random((N, NUM_KP)).astype(np.float32)
        return kpts, scores

    # --- count ---

    def test_one_sample_per_person(self):
        N = 5
        kpts, scores = self._make_inputs(N)
        samples = KeypointAPMetric.build_samples_for_image(
            img_id=1, kpts=kpts, scores=scores, ann_ids=list(range(N))
        )
        assert len(samples) == N

    def test_zero_people_returns_empty_list(self):
        kpts   = np.zeros((0, NUM_KP, 2), dtype=np.float32)
        scores = np.zeros((0, NUM_KP),    dtype=np.float32)
        samples = KeypointAPMetric.build_samples_for_image(1, kpts, scores, [])
        assert samples == []

    # --- id uniqueness (the key anti-regression test) ---

    def test_all_ids_are_unique(self):
        N = 6
        kpts, scores = self._make_inputs(N)
        ann_ids = [10, 20, 30, 40, 50, 60]
        samples = KeypointAPMetric.build_samples_for_image(1, kpts, scores, ann_ids)
        ids = [s["id"] for s in samples]
        assert len(set(ids)) == N, f"Duplicate ids: {ids}"

    def test_fallback_ids_are_unique_when_ann_ids_too_short(self):
        """When fewer ann_ids than people, synthetic ids must still be unique."""
        N = 5
        img_id = 99
        kpts, scores = self._make_inputs(N)
        ann_ids = [7, 8]   # only 2 for 5 people
        samples = KeypointAPMetric.build_samples_for_image(img_id, kpts, scores, ann_ids)
        ids = [s["id"] for s in samples]
        # First 2 use ann_ids, rest use synthetic — all must be distinct
        assert ids[:2] == [7, 8]
        assert len(set(ids)) == N, f"Ids not all unique: {ids}"

    def test_fallback_ids_are_unique_when_no_ann_ids(self):
        N = 4
        img_id = 42
        kpts, scores = self._make_inputs(N)
        samples = KeypointAPMetric.build_samples_for_image(img_id, kpts, scores, [])
        ids = [s["id"] for s in samples]
        assert len(set(ids)) == N

    # --- keypoint shape ---

    def test_each_sample_has_single_person_keypoints(self):
        """Shape must be (1, 14, 2) / (1, 14) — not (N, 14, 2)."""
        N = 4
        kpts, scores = self._make_inputs(N)
        samples = KeypointAPMetric.build_samples_for_image(1, kpts, scores, [])
        for i, s in enumerate(samples):
            kp = s["pred_instances"]["keypoints"]
            sc = s["pred_instances"]["keypoint_scores"]
            assert kp.shape == (1, NUM_KP, 2), \
                f"Sample {i}: expected (1,{NUM_KP},2), got {kp.shape}"
            assert sc.shape == (1, NUM_KP), \
                f"Sample {i}: expected (1,{NUM_KP}), got {sc.shape}"

    def test_keypoints_match_input_row(self):
        """Sample n must contain exactly the n-th row of the input arrays."""
        N = 3
        kpts, scores = self._make_inputs(N)
        ann_ids = [100, 200, 300]
        samples = KeypointAPMetric.build_samples_for_image(5, kpts, scores, ann_ids)
        for n in range(N):
            np.testing.assert_array_equal(
                samples[n]["pred_instances"]["keypoints"][0],
                kpts[n],
                err_msg=f"Person {n} keypoints mismatch"
            )

    # --- metadata ---

    def test_img_id_is_set_on_every_sample(self):
        img_id = 777
        kpts, scores = self._make_inputs(3)
        samples = KeypointAPMetric.build_samples_for_image(img_id, kpts, scores, [])
        for s in samples:
            assert s["img_id"] == img_id

    def test_bbox_scores_shape_is_one(self):
        """CocoMetric reads bbox_scores to determine instance confidence."""
        kpts, scores = self._make_inputs(3)
        samples = KeypointAPMetric.build_samples_for_image(1, kpts, scores, [])
        for s in samples:
            bs = s["gt_instances"]["bbox_scores"]
            assert bs.shape == (1,), f"Expected (1,), got {bs.shape}"
            assert bs[0] == pytest.approx(1.0)


# ===========================================================================
# 3.  KeypointAPMetric  (integration — requires real annotation file)
# ===========================================================================


@pytest.fixture(scope="module")
def val_metric():
    """Real KeypointAPMetric backed by the CrowdPose val annotations.

    All tests in TestKeypointAPMetricIntegration are skipped if the file is
    absent (fresh clone, CI environment without data).
    """
    if not VAL_ANN_FILE.exists():
        pytest.skip(f"CrowdPose val annotations not found at {VAL_ANN_FILE}")
    import sys
    sys.path.insert(0, "external/mmpose")
    return KeypointAPMetric(str(VAL_ANN_FILE))


class TestKeypointAPMetricIntegration:
    """Integration tests that need the real annotation file."""

    # --- reset ---

    def test_reset_clears_state(self, val_metric):
        m = val_metric
        m.reset()
        valid_id = list(m._valid_img_ids)[0]
        kpts   = torch.rand(1, 1, NUM_KP, 2) * 640
        scores = torch.ones(1, 1, NUM_KP)
        m.update(kpts, scores, torch.tensor([valid_id]))
        assert m._n_images > 0
        m.reset()
        assert m._n_images == 0

    # --- _n_images counting ---

    def test_n_images_equals_persons_processed(self, val_metric):
        """_n_images must equal the total number of person samples sent to
        CocoMetric.process, not the number of images (they differ by N people)."""
        m = val_metric
        m.reset()

        valid_ids = list(m._valid_img_ids)[:3]
        B, N = len(valid_ids), 4
        kpts   = torch.rand(B, N, NUM_KP, 2) * 640
        scores = torch.ones(B, N, NUM_KP)
        m.update(kpts, scores, torch.tensor(valid_ids))

        assert m._n_images == B * N, (
            f"Expected _n_images={B*N} (one per person), got {m._n_images}"
        )

    def test_n_images_matches_metric_results_length(self, val_metric):
        """_n_images must equal len(metric.results) so collect_results doesn't truncate."""
        m = val_metric
        m.reset()

        valid_ids = list(m._valid_img_ids)[:4]
        B, N = len(valid_ids), 3
        kpts   = torch.rand(B, N, NUM_KP, 2) * 640
        scores = torch.ones(B, N, NUM_KP)
        m.update(kpts, scores, torch.tensor(valid_ids))

        assert m._n_images == len(m._metric.results), (
            "_n_images must equal len(metric.results) or collect_results will truncate"
        )

    def test_invalid_image_ids_are_skipped(self, val_metric):
        m = val_metric
        m.reset()
        # IDs that are definitely not in the val set
        bad_ids = torch.tensor([-1, -2, -3])
        kpts   = torch.rand(3, 2, NUM_KP, 2) * 640
        scores = torch.ones(3, 2, NUM_KP)
        m.update(kpts, scores, bad_ids)
        assert m._n_images == 0

    def test_mixed_valid_invalid_ids(self, val_metric):
        m = val_metric
        m.reset()
        valid_id = list(m._valid_img_ids)[0]
        N = 2
        mixed_ids = torch.tensor([valid_id, -999])
        kpts   = torch.rand(2, N, NUM_KP, 2) * 640
        scores = torch.ones(2, N, NUM_KP)
        m.update(kpts, scores, mixed_ids)
        # Only the valid image contributes → N persons accumulated
        assert m._n_images == N

    # --- compute returns empty dict when no predictions ---

    def test_compute_returns_empty_dict_when_no_predictions(self, val_metric):
        m = val_metric
        m.reset()
        result = m.compute()
        assert result == {}, f"Expected empty dict, got {result}"

    # --- AP smoke test: perfect GT predictions ---

    def test_perfect_gt_predictions_give_nonzero_ap(self, val_metric):
        """Feeding exact GT keypoints back through the metric should yield AP > 0.

        This is the end-to-end sanity check: if this fails with a trained model
        but passes here, the issue is model quality, not a metric bug.
        """
        m = val_metric
        m.reset()

        with open(VAL_ANN_FILE) as f:
            data = json.load(f)

        # Group annotations by image; take the first 10 images with ≥1 person
        by_image: dict[int, list] = {}
        for ann in data["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann)
        test_images = [(iid, anns) for iid, anns in by_image.items() if anns][:10]

        for img_id, anns in test_images:
            raw = np.array([a["keypoints"] for a in anns], dtype=np.float32)
            raw = raw.reshape(len(anns), NUM_KP, 3)      # (N, 14, 3)
            kpts_t   = torch.from_numpy(raw[:, :, :2]).unsqueeze(0)  # (1, N, 14, 2)
            scores_t = torch.from_numpy(raw[:, :, 2 ]).unsqueeze(0)  # (1, N, 14)
            m.update(kpts_t, scores_t, torch.tensor([img_id]))

        results = m.compute()
        assert results, "compute() returned empty dict — no predictions accumulated"
        ap = results.get("coco/AP", 0.0)
        # With only 10 images predicted out of ~2000 in the val set, pycocotools
        # divides credit over all GT instances, so AP is low (~0.01) even with
        # exact GT predictions.  The important invariant is AP > 0: the old
        # id-collision bug caused all but one person per image to be deleted,
        # giving AP = 0.0 exactly.
        assert ap > 0.0, (
            f"Perfect GT predictions yielded AP={ap:.4f} (expected > 0). "
            "This indicates a metric pipeline bug such as id-collision "
            "deleting all but one person per image."
        )

    # --- multiple people per image all survive (anti-regression for the id-collision bug) ---

    def test_all_persons_survive_deduplication(self, val_metric):
        """N people per image must all be present in metric.results — not deduplicated."""
        m = val_metric
        m.reset()

        valid_id = list(m._valid_img_ids)[0]
        N = 5
        kpts   = torch.rand(1, N, NUM_KP, 2) * 640
        scores = torch.ones(1, N, NUM_KP)
        m.update(kpts, scores, torch.tensor([valid_id]))

        # Each person should be stored as a separate result entry
        assert len(m._metric.results) == N, (
            f"Expected {N} result entries (one per person), "
            f"got {len(m._metric.results)}. "
            "Likely id-collision bug: multiple persons sharing one sample id."
        )

    # --- AP pipeline roundtrip with cached detector predictions ---

    def test_raw_detector_predictions_give_expected_ap(self):
        """The unrefined RTMPose-M predictions should score ≈0.24 AP on the val set.

        This test doubles as a regression test: if someone accidentally breaks
        the metric pipeline the AP would drop back to 0.
        """
        if not VAL_ANN_FILE.exists() or not PRED_CACHE.exists():
            pytest.skip("CrowdPose data or prediction cache not available")

        import sys
        sys.path.insert(0, "external/mmpose")
        from src.data import PredictionDataset

        dataset = PredictionDataset.from_path(str(PRED_CACHE))
        loader  = dataset.create_dataloader(batch_size=64, shuffle=False, num_workers=0)
        m = KeypointAPMetric(str(VAL_ANN_FILE))

        for batch in loader:
            m.update(batch["pred_keypoints"], batch["pred_keypoint_scores"], batch["image_id"])

        results = m.compute()
        ap = results.get("coco/AP", 0.0)
        # RTMPose-M on CrowdPose val in the bottom-up / oracle-box mode gives ~0.24
        assert ap > 0.15, (
            f"Raw detector AP={ap:.4f} is unexpectedly low. "
            "Expected ≥0.15 for RTMPose-M on CrowdPose val."
        )

    def test_canonical_roundtrip_preserves_ap(self):
        """canonical→pixel roundtrip through prepare_pose_graph_batch must be lossless.

        This is the 'perfect denoiser' floor: if DDIM returns the original
        canonical detector predictions unchanged (delta=0 model), the resulting
        AP must equal the raw detector AP to within floating-point error.
        A failure here means the coordinate conversion is wrong and AP will
        be depressed for any model, regardless of quality.
        """
        if not VAL_ANN_FILE.exists() or not PRED_CACHE.exists():
            pytest.skip("CrowdPose data or prediction cache not available")

        import sys
        sys.path.insert(0, "external/mmpose")
        from src.data import PredictionDataset, prepare_pose_graph_batch

        dataset   = PredictionDataset.from_path(str(PRED_CACHE))
        loader    = dataset.create_dataloader(batch_size=64, shuffle=False, num_workers=0)
        m_raw     = KeypointAPMetric(str(VAL_ANN_FILE))
        m_roundtrip = KeypointAPMetric(str(VAL_ANN_FILE))

        for batch in loader:
            pred_kpts   = batch["pred_keypoints"]
            pred_scores = batch["pred_keypoint_scores"]
            vis = (pred_scores > 0.1).float()

            # Raw predictions
            m_raw.update(pred_kpts, pred_scores, batch["image_id"])

            # Canonical roundtrip (simulate a perfect delta=0 model)
            g = prepare_pose_graph_batch(pred_kpts, vis)
            B2, N2 = pred_kpts.shape[:2]
            canonical = g["node_pos"].view(B2, N2, 15, 2)
            recovered = canonical_to_pixel(canonical, g["root_pos"], scale=g["scale"])
            m_roundtrip.update(recovered, pred_scores, batch["image_id"])

        ap_raw       = m_raw.compute().get("coco/AP", -1.0)
        ap_roundtrip = m_roundtrip.compute().get("coco/AP", -1.0)

        assert abs(ap_roundtrip - ap_raw) < 1e-4, (
            f"Canonical roundtrip changed AP: raw={ap_raw:.6f}, "
            f"roundtrip={ap_roundtrip:.6f} (diff={abs(ap_roundtrip - ap_raw):.6f}). "
            "The coordinate conversion has a lossy bug."
        )

    def test_identity_ddim_gives_baseline_ap(self):
        """When DDIM is monkeypatched to return the input unchanged (delta=0),
        AP must match the raw detector baseline.

        This tests the full _refine_batch → metric pipeline without any
        dependency on model quality.  A failure here means the coordinate
        conversion or the noise injection in _refine_batch is incorrect.
        """
        if not VAL_ANN_FILE.exists() or not PRED_CACHE.exists():
            pytest.skip("CrowdPose data or prediction cache not available")

        import sys
        sys.path.insert(0, "external/mmpose")
        from unittest.mock import patch
        from src.data import PredictionDataset
        from src.model import (
            KeypointDiffusionConfig,
            KeypointDiffusionModel,
            PoseDiffusionModule,
        )

        cfg    = KeypointDiffusionConfig(dim_pos=16, dim_index=16, dim_time=16,
                                         dim_bone=16, dim_hidden=16, num_layers=1,
                                         num_train_steps=100)
        model  = KeypointDiffusionModel(config=cfg, num_keypoints=15)
        module = PoseDiffusionModule(model=model, inference_noise_frac=0.0)  # no noise
        module.eval()

        dataset = PredictionDataset.from_path(str(PRED_CACHE))
        loader  = dataset.create_dataloader(batch_size=64, shuffle=False, num_workers=0)

        # Raw detector baseline
        m_raw = KeypointAPMetric(str(VAL_ANN_FILE))
        for batch in loader:
            m_raw.update(batch["pred_keypoints"], batch["pred_keypoint_scores"],
                         batch["image_id"])
        ap_baseline = m_raw.compute().get("coco/AP", -1.0)

        # Identity DDIM: patch add_noise → no-op AND ddim_sample → return x_init.
        # Together these make _refine_batch a pure canonical ↔ pixel roundtrip,
        # which must preserve AP exactly (tolerance 1e-3 covers float rounding).
        import torch as _torch
        def _noop_add_noise(**kw):
            return kw["node_pos"], _torch.zeros_like(kw["node_pos"])

        m_id = KeypointAPMetric(str(VAL_ANN_FILE))
        with (
            patch.object(module.model, "add_noise",    side_effect=_noop_add_noise),
            patch.object(module.model, "ddim_sample",  side_effect=lambda **kw: kw["x_init"]),
        ):
            for batch in loader:
                pred_pixel, pred_scores, image_ids = module._refine_batch(batch)
                m_id.update(pred_pixel, pred_scores, image_ids)
        ap_identity = m_id.compute().get("coco/AP", -1.0)

        assert abs(ap_identity - ap_baseline) < 1e-3, (
            f"Identity DDIM AP={ap_identity:.6f} differs from "
            f"detector baseline={ap_baseline:.6f}.  "
            "The _refine_batch pipeline has a lossy bug "
            "(check root_pos / scale passing or root-keypoint dropping)."
        )

    def test_inference_noise_frac_controls_starting_timestep(self):
        """_refine_batch must add noise at t_start=round(T*frac), not at T=100.

        We verify the mechanism (not AP value) by monkeypatching add_noise and
        ddim_sample so we can inspect the timestep and start_t arguments.
        A random model's delta accumulation makes AP meaningless here — what
        matters is that the noise is small (frac*T, not T) so a TRAINED model
        that predicts delta≈0 can gracefully fall back to the detector output.
        """
        import sys
        sys.path.insert(0, "external/mmpose")
        from unittest.mock import patch
        from src.data import PredictionDataset
        from src.model import (
            KeypointDiffusionConfig,
            KeypointDiffusionModel,
            PoseDiffusionModule,
        )

        frac   = 0.15
        T      = 100
        cfg    = KeypointDiffusionConfig(dim_pos=16, dim_index=16, dim_time=16,
                                         dim_bone=16, dim_hidden=16, num_layers=1,
                                         num_train_steps=T)
        model  = KeypointDiffusionModel(config=cfg, num_keypoints=15)
        module = PoseDiffusionModule(model=model, inference_noise_frac=frac)
        module.eval()

        # Build a minimal single-batch dataset from a few dummy samples
        # (We only need one batch to check the timestep values)
        import torch
        B = 4
        dummy_batch = {
            "pred_keypoints":       torch.rand(B, 2, 14, 2) * 800,
            "pred_keypoint_scores": torch.rand(B, 2, 14),
            "image_id":             torch.arange(B, dtype=torch.long),
        }

        captured: dict = {}

        def capture_add_noise(**kw):
            captured["timestep"] = kw["timestep"].clone()
            return kw["node_pos"], torch.zeros_like(kw["node_pos"])

        def capture_ddim(**kw):
            captured["start_t"]  = kw.get("start_t")
            captured["num_inf"]  = kw.get("num_inference_steps")
            return kw["x_init"]

        with (
            patch.object(module.model, "add_noise",   side_effect=capture_add_noise),
            patch.object(module.model, "ddim_sample", side_effect=capture_ddim),
        ):
            module._refine_batch(dummy_batch)

        expected_t_start = max(1, round(T * frac))          # 15
        expected_S       = expected_t_start                  # default: S = t_start

        assert "timestep" in captured, "_refine_batch did not call add_noise"
        assert "start_t"  in captured, "_refine_batch did not call ddim_sample"

        assert (captured["timestep"] == expected_t_start).all(), (
            f"add_noise timestep={captured['timestep'].unique().tolist()} "
            f"but expected {expected_t_start} (= round(T*frac)={round(T*frac)}).  "
            "Large noise at t=T would make AP=0 for any undertrained model."
        )
        assert captured["start_t"] == expected_t_start, (
            f"ddim_sample start_t={captured['start_t']} but expected {expected_t_start}."
        )
        assert captured["num_inf"] == expected_S, (
            f"ddim_sample num_inference_steps={captured['num_inf']} expected {expected_S}."
        )
