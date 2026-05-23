"""Tests for set-prediction concrete classes (CNN / SpatialAttention / PoseSlot)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.decoders.pose_slot import PoseSlotDecoder, PoseSlotDecoderConfig
from src.embedders.spatial_attention import (
    SpatialAttentionEmbedder,
    SpatialAttentionEmbedderConfig,
    make_2d_sinusoidal_pe,
)
from src.features.cnn import CNNFeatureExtractor, CNNFeatureExtractorConfig
from src.pose_estimator import PoseEstimatorConfig


CONF_DIR = str(_ROOT / "conf")


# ---------------------------------------------------------------------------
# CNNFeatureExtractor
# ---------------------------------------------------------------------------


def test_cnn_extractor_total_stride_16():
    extractor = CNNFeatureExtractor(out_channels=64)
    image = torch.rand(2, 3, 256, 192)
    out = extractor(image=image)
    assert "spatial" in out
    assert out["spatial"].shape == (2, 64, 16, 12)  # stride 16


def test_cnn_extractor_requires_image():
    extractor = CNNFeatureExtractor(out_channels=64)
    try:
        extractor(image=None, image_ids=[1, 2])
    except ValueError:
        return
    raise AssertionError("expected ValueError when image is None")


# ---------------------------------------------------------------------------
# Sinusoidal PE
# ---------------------------------------------------------------------------


def test_sinusoidal_pe_shape_and_finite():
    pe = make_2d_sinusoidal_pe(8, 6, dim=32, device=torch.device("cpu"))
    assert pe.shape == (48, 32)
    assert torch.isfinite(pe).all()
    # different positions produce different encodings
    assert not torch.allclose(pe[0], pe[1])


def test_sinusoidal_pe_rejects_bad_dim():
    try:
        make_2d_sinusoidal_pe(4, 4, dim=10, device=torch.device("cpu"))
    except ValueError:
        return
    raise AssertionError("expected ValueError for dim not divisible by 4")


# ---------------------------------------------------------------------------
# SpatialAttentionEmbedder (size-independence)
# ---------------------------------------------------------------------------


def test_spatial_attention_embedder_emits_global_feature():
    emb = SpatialAttentionEmbedder(in_channels=64, feature_dim=32, num_heads=4)
    feat = torch.rand(2, 64, 16, 12)
    out = emb({"spatial": feat})
    assert out["feature"].shape == (2, 32)


def test_spatial_attention_embedder_size_independent():
    """Same module produces the same output dim for two different spatial sizes."""
    emb = SpatialAttentionEmbedder(in_channels=64, feature_dim=32, num_heads=4)
    feat_a = torch.rand(2, 64, 16, 12)
    feat_b = torch.rand(2, 64, 8, 6)
    out_a = emb({"spatial": feat_a})
    out_b = emb({"spatial": feat_b})
    assert out_a["feature"].shape == (2, 32)
    assert out_b["feature"].shape == (2, 32)


# ---------------------------------------------------------------------------
# PoseSlotDecoder (canonical (-1,1) output)
# ---------------------------------------------------------------------------


def test_pose_slot_decoder_output_keys_and_shapes():
    decoder = PoseSlotDecoder(in_dim=32, num_slots=5, num_keypoints=14, hidden_dim=64)
    feature = torch.randn(3, 32)
    out = decoder({"feature": feature})
    assert out["keypoints"].shape == (3, 5, 14, 2)
    assert out["scores"].shape == (3, 5)
    assert out["scores_logits"].shape == (3, 5)
    assert out["keypoint_scores"].shape == (3, 5, 14)
    assert out["vis_logits"].shape == (3, 5, 14)


def test_pose_slot_decoder_keypoints_in_canonical_range():
    decoder = PoseSlotDecoder(in_dim=32, num_slots=5, num_keypoints=14, hidden_dim=64)
    feature = torch.randn(4, 32) * 100  # large magnitude logits
    out = decoder({"feature": feature})
    kp = out["keypoints"]
    assert (kp >= -1.0).all() and (kp <= 1.0).all()


def test_pose_slot_decoder_scores_in_unit_range():
    decoder = PoseSlotDecoder(in_dim=32, num_slots=3, num_keypoints=14, hidden_dim=64)
    out = decoder({"feature": torch.randn(2, 32)})
    assert (out["scores"] >= 0).all() and (out["scores"] <= 1).all()
    assert (out["keypoint_scores"] >= 0).all() and (out["keypoint_scores"] <= 1).all()


# ---------------------------------------------------------------------------
# End-to-end PoseEstimator from the new Hydra config
# ---------------------------------------------------------------------------


def test_cnn_set_prediction_hydra_config_builds_and_runs():
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(
            config_name="config",
            overrides=["pose_estimator=cnn_set_prediction"],
        )
    pose_cfg = PoseEstimatorConfig.from_hydra_config(cfg.pose_estimator)
    estimator = pose_cfg.build()
    image = torch.rand(2, 3, 256, 192)
    out = estimator(image=image, image_ids=[0, 1])
    N = estimator.decoder.num_slots
    # PoseEstimator forwards the decoder's dict back; downstream consumers
    # rely on `keypoints` being canonical (-1, 1).
    assert out["keypoints"].shape == (2, N, 14, 2)
    assert out["scores"].shape == (2, N)
    assert (out["keypoints"] >= -1).all() and (out["keypoints"] <= 1).all()


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


def test_configs_round_trip():
    # build directly without Hydra
    extractor = CNNFeatureExtractorConfig(out_channels=64).build()
    assert isinstance(extractor, CNNFeatureExtractor)
    embedder = SpatialAttentionEmbedderConfig(
        in_channels=64, feature_dim=32, num_heads=4,
    ).build()
    assert isinstance(embedder, SpatialAttentionEmbedder)
    decoder = PoseSlotDecoderConfig(
        in_dim=32, num_slots=5, num_keypoints=14, hidden_dim=64,
    ).build()
    assert isinstance(decoder, PoseSlotDecoder)
