"""End-to-end pipeline tests with dummy and real components."""

from __future__ import annotations

import torch

from src.data.graph import prepare_pose_graph_batch, prepare_sample
from src.decoders import (
    GraphDenoiseNetworkConfig,
    GraphDiffusionDecoderConfig,
    MlpDenoiseNetworkConfig,
)
from src.dummy import (
    DUMMY_KEYPOINTS_PIXEL,
    BroadcastEmbedderConfig,
    CenterPatchExtractor,
    CenterPatchExtractorConfig,
    DummyPoseDataset,
    ShiftDecoderConfig,
)
from src.embedders import NoneEmbedderConfig
from src.features import NoneFeatureExtractorConfig
from src.pose_estimator import PoseEstimatorConfig


def _collate(items: list[dict]) -> dict:
    return {
        "keypoints":  torch.stack([it["keypoints"] for it in items]),
        "visibility": torch.stack([it["visibility"] for it in items]),
        "image_id":   torch.tensor([it["image_id"] for it in items]),
        "img_path":   [it["img_path"] for it in items],
        "image":      [it["image"] for it in items],
    }


def test_dummy_dataset_emits_fixed_pose() -> None:
    ds = DummyPoseDataset(length=4)
    item = ds[0]
    assert item["keypoints"].shape == (1, 14, 2)
    assert torch.equal(item["keypoints"][0], DUMMY_KEYPOINTS_PIXEL)


def test_dummy_pipeline_end_to_end() -> None:
    """Center-patch extractor → broadcast embedder → shift decoder."""
    ds = DummyPoseDataset(length=2, image_size=256)
    batch = _collate([ds[0], ds[1]])

    cfg = PoseEstimatorConfig(
        feature_extractor=CenterPatchExtractorConfig(patch_size=32, out_dim=32),
        feature_embedder=BroadcastEmbedderConfig(in_dim=32, out_dim=32),
        decoder=ShiftDecoderConfig(in_dim=32, max_shift=0.05),
    )
    model = cfg.build()

    # Manually register images on the extractor (no on-disk cache for dummy).
    for iid, img in zip(batch["image_id"].tolist(), batch["image"]):
        model.feature_extractor.register_image(iid, img)

    sample = prepare_sample(batch)
    img_hw = torch.tensor([[256.0, 256.0]] * sample.B)
    img_wh = img_hw[:, [1, 0]]
    graph = prepare_pose_graph_batch(sample, img_size=img_wh)

    out = model(
        graph=graph,
        image_ids=sample.image_id,
        kpt_pixel=sample.keypoints,
        img_hw=img_hw,
        initial_keypoints=graph.node_pos,
    )
    assert isinstance(out, dict)
    assert out["keypoints"].shape == (graph.B, graph.N, graph.K, 2)
    # Shift is at most max_shift; with zero-init head, equal to identity.
    expected = graph.node_pos.view(graph.B, graph.N, graph.K, 2)
    assert torch.allclose(out["keypoints"], expected, atol=1e-5)
    assert out["scores"].shape == (graph.B, graph.N)
    assert out["keypoint_scores"].shape == (graph.B, graph.N, graph.K)


def test_graph_diffusion_pipeline_end_to_end() -> None:
    """Full graph-diffusion pipeline runs DDIM sampling on a tiny batch."""
    ds = DummyPoseDataset(length=2, image_size=256)
    batch = _collate([ds[0], ds[1]])

    cfg = PoseEstimatorConfig(
        feature_extractor=NoneFeatureExtractorConfig(),
        feature_embedder=NoneEmbedderConfig(),
        decoder=GraphDiffusionDecoderConfig(
            denoise_network=MlpDenoiseNetworkConfig(num_layers=2),
            num_train_steps=4,
        ),
    )
    model = cfg.build()
    model.eval()

    sample = prepare_sample(batch)
    img_hw = torch.tensor([[256.0, 256.0]] * sample.B)
    img_wh = img_hw[:, [1, 0]]
    graph = prepare_pose_graph_batch(sample, img_size=img_wh)

    with torch.no_grad():
        out = model(
            graph=graph,
            image_ids=sample.image_id,
            kpt_pixel=sample.keypoints,
            img_hw=img_hw,
            initial_keypoints=graph.node_pos,
        )
    assert isinstance(out, dict)
    assert out["keypoints"].shape == (graph.B, graph.N, graph.K, 2)
    assert torch.isfinite(out["keypoints"]).all()


def test_graph_diffusion_predict_delta_shape() -> None:
    cfg = GraphDiffusionDecoderConfig(
        denoise_network=GraphDenoiseNetworkConfig(num_layers=2),
        num_train_steps=4,
    )
    decoder = cfg.build()

    ds = DummyPoseDataset(length=2)
    batch = _collate([ds[0], ds[1]])
    sample = prepare_sample(batch)
    img_wh = torch.tensor([[256.0, 256.0]] * sample.B)
    graph = prepare_pose_graph_batch(sample, img_size=img_wh)

    timestep = torch.full((graph.B,), 2, dtype=torch.long)
    delta = decoder.predict_delta(graph.node_pos, graph, {}, timestep)
    assert delta.shape == graph.node_pos.shape
    # Zero-init final layer in denoise network → exactly zero output.
    assert torch.allclose(delta, torch.zeros_like(delta))
