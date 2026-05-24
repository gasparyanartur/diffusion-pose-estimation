"""Smoke tests for ``BoxSetDiffusionModule`` (I6)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from pycocotools.coco import COCO

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.transforms import (
    boxes_padded_canonical_to_pixel_xywh,
    boxes_xywh_to_padded_canonical,
)
from src.decoders.diffusion.box_set import BoxSetDiffusionDecoder
from src.features.cnn import CNNFeatureExtractor
from src.training.box_diffusion_module import (
    BoxSetDiffusionModule,
    _image_tokens_and_centers,
)
from src.training.losses import SetBoxDiffusionLossWeights


N_SLOTS = 4
FEAT_C = 16


def _make_module(
    val_coco: COCO | None = None,
    test_coco: COCO | None = None,
    **module_kwargs,
) -> BoxSetDiffusionModule:
    backbone = CNNFeatureExtractor(out_channels=FEAT_C)
    decoder = BoxSetDiffusionDecoder(
        num_slots=N_SLOTS, dim=32, num_heads=4, num_blocks=2,
        time_dim=32, image_token_dim=FEAT_C,
        fourier_box_bands=4, fourier_sigma_bands=4,
    )
    module = BoxSetDiffusionModule(
        backbone=backbone, decoder=decoder,
        weights=SetBoxDiffusionLossWeights(),
        lr=1e-3, **module_kwargs,
    )
    # Inject AP metrics directly when test wants them populated.
    if val_coco is not None:
        from src.metrics import BoxAPMetric
        module._val_ap_metric = BoxAPMetric(coco=val_coco)
        module._val_ap_metric_1step = BoxAPMetric(coco=val_coco)
    if test_coco is not None:
        from src.metrics import BoxAPMetric
        module._test_ap_metric = BoxAPMetric(coco=test_coco)
        module._test_ap_metric_1step = BoxAPMetric(coco=test_coco)
    return module


def _fake_batch(B: int = 2, P_max: int = 3, H: int = 32, W: int = 32) -> dict:
    gt_boxes_canon = torch.zeros(B, P_max, 4)
    gt_box_labeled = torch.zeros(B, P_max)
    for b in range(B):
        n_real = b + 1  # 1, 2, ...
        for p in range(n_real):
            gt_boxes_canon[b, p] = torch.tensor([0.0, 0.0, 0.4, 0.4])
            gt_box_labeled[b, p] = 1.0
    return {
        "image": torch.rand(B, 3, H, W),
        "gt_boxes_canon": gt_boxes_canon,
        "gt_box_labeled": gt_box_labeled,
        "img_hw": torch.tensor([[float(H), float(W)]] * B),
        "image_id": torch.arange(1, B + 1, dtype=torch.long),
    }


def _build_synthetic_coco() -> COCO:
    """Tiny COCO with images matching the synthetic batch image_ids."""
    coco_dict = {
        "images": [
            {"id": 1, "width": 32, "height": 32, "file_name": "a.jpg"},
            {"id": 2, "width": 32, "height": 32, "file_name": "b.jpg"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [10.0, 10.0, 12.0, 12.0], "area": 144.0, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    coco = COCO()
    coco.dataset = coco_dict
    coco.createIndex()
    return coco


# ---------------------------------------------------------------------------
# Helper coverage
# ---------------------------------------------------------------------------


def test_image_tokens_and_centers_accepts_bchw():
    feat = torch.randn(2, FEAT_C, 4, 5)
    centers = torch.randn(4, 5, 2)
    tokens, c = _image_tokens_and_centers(
        {"spatial": feat, "patch_centers": centers}, feature_key="spatial",
    )
    assert tokens.shape == (2, 20, FEAT_C)
    assert c.shape == (20, 2)


def test_image_tokens_and_centers_accepts_btc():
    feat = torch.randn(2, 7, FEAT_C)
    centers = torch.randn(7, 2)
    tokens, c = _image_tokens_and_centers(
        {"spatial": feat, "patch_centers": centers}, feature_key="spatial",
    )
    assert tokens.shape == (2, 7, FEAT_C)
    assert c.shape == (7, 2)


def test_boxes_canonical_to_pixel_round_trip():
    boxes_xywh = torch.tensor([
        [10.0, 20.0, 30.0, 40.0],
        [0.0, 0.0, 100.0, 50.0],
    ])
    canon = boxes_xywh_to_padded_canonical(boxes_xywh, original_h=100, original_w=100)
    img_hw = torch.tensor([100.0, 100.0])
    recon = boxes_padded_canonical_to_pixel_xywh(canon.unsqueeze(0), img_hw.unsqueeze(0))
    assert torch.allclose(recon.squeeze(0), boxes_xywh, atol=1e-4)


# ---------------------------------------------------------------------------
# Module smoke
# ---------------------------------------------------------------------------


def test_training_step_returns_finite_loss():
    module = _make_module()
    batch = _fake_batch(B=2)
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_training_step_with_augmentations_runs():
    module = _make_module(
        noise_std=0.05, brightness=0.2, contrast=0.2, saturation=0.2, hflip_p=1.0,
    )
    batch = _fake_batch(B=2)
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)


def test_validation_step_runs_without_ap_trigger():
    module = _make_module(val_ap_every_n_epochs=10)
    module.trainer = None  # current_epoch defaults to 0
    batch = _fake_batch(B=2)
    # epoch 0 → (0 + 1) % 10 != 0 → no AP work.
    module.validation_step(batch, batch_idx=0)


def test_validation_step_populates_ap_on_trigger_epoch():
    coco = _build_synthetic_coco()
    module = _make_module(val_coco=coco, val_ap_every_n_epochs=1)
    batch = _fake_batch(B=2)

    # Force the sampler to emit non-degenerate boxes so the metric records dets.
    def _fake_sample(self, batch, num_inference_steps):  # noqa: ARG001
        B = batch["image"].shape[0]
        N = self.num_slots
        return {
            "pred_boxes": torch.tensor([[[0.0, 0.0, 0.5, 0.5]] * N] * B),
            "obj_logits": torch.zeros(B, N),
        }

    import types
    module._sample_decoder = types.MethodType(_fake_sample, module)
    module.validation_step(batch, batch_idx=0)
    assert len(module._val_ap_metric._results) > 0
    assert len(module._val_ap_metric_1step._results) > 0


def test_test_step_populates_ap():
    coco = _build_synthetic_coco()
    module = _make_module(test_coco=coco)
    batch = _fake_batch(B=2)

    def _fake_sample(self, batch, num_inference_steps):  # noqa: ARG001
        B = batch["image"].shape[0]
        N = self.num_slots
        return {
            "pred_boxes": torch.tensor([[[0.0, 0.0, 0.5, 0.5]] * N] * B),
            "obj_logits": torch.zeros(B, N),
        }

    import types
    module._sample_decoder = types.MethodType(_fake_sample, module)
    module.test_step(batch, batch_idx=0)
    assert len(module._test_ap_metric._results) > 0
    assert len(module._test_ap_metric_1step._results) > 0


def test_configure_optimizers_returns_dict_with_scheduler():
    module = _make_module(lr_T_max=5)
    out = module.configure_optimizers()
    assert isinstance(out, dict)
    assert "optimizer" in out
    assert "lr_scheduler" in out


# ---------------------------------------------------------------------------
# Branch-block architecture
# ---------------------------------------------------------------------------


def test_decoder_branch_blocks_output_shapes():
    """Structured branches produce correct output shapes with non-default block counts."""
    B, T, C = 2, 9, FEAT_C
    decoder = BoxSetDiffusionDecoder(
        num_slots=N_SLOTS, dim=32, num_heads=4,
        num_blocks=1, num_obj_blocks=2, num_box_branch_blocks=2, num_box_refine_blocks=2,
        time_dim=32, image_token_dim=C,
        fourier_box_bands=4, fourier_sigma_bands=4,
    )
    noisy = torch.randn(B, N_SLOTS, 4)
    sigma = torch.rand(B).clamp_min(1e-3)
    tokens = torch.randn(B, T, C)
    centers = torch.zeros(T, 2)
    out = decoder(noisy_boxes=noisy, sigma=sigma, image_tokens=tokens, patch_centers=centers)
    assert out["pred_boxes"].shape == (B, N_SLOTS, 4)
    assert out["obj_logits"].shape == (B, N_SLOTS)


def test_decoder_branch_blocks_zero_counts_still_runs():
    """num_obj_blocks=0 / num_box_branch_blocks=0 / num_box_refine_blocks=0 are valid."""
    B, T, C = 2, 9, FEAT_C
    decoder = BoxSetDiffusionDecoder(
        num_slots=N_SLOTS, dim=32, num_heads=4,
        num_blocks=2, num_obj_blocks=0, num_box_branch_blocks=0, num_box_refine_blocks=0,
        time_dim=32, image_token_dim=C,
        fourier_box_bands=4, fourier_sigma_bands=4,
    )
    noisy = torch.randn(B, N_SLOTS, 4)
    sigma = torch.rand(B).clamp_min(1e-3)
    tokens = torch.randn(B, T, C)
    centers = torch.zeros(T, 2)
    out = decoder(noisy_boxes=noisy, sigma=sigma, image_tokens=tokens, patch_centers=centers)
    assert out["pred_boxes"].shape == (B, N_SLOTS, 4)
    assert out["obj_logits"].shape == (B, N_SLOTS)


def test_decoder_wider_trunk_with_branch_projection():
    """dim=512 trunk projects independently to obj_dim and box_dim; output shapes are correct."""
    B, T, C = 2, 9, FEAT_C
    decoder = BoxSetDiffusionDecoder(
        num_slots=N_SLOTS, dim=64, obj_dim=32, box_dim=48, num_heads=4,
        num_blocks=2, num_obj_blocks=1, num_box_branch_blocks=1, num_box_refine_blocks=1,
        time_dim=32, image_token_dim=C,
        fourier_box_bands=4, fourier_sigma_bands=4,
    )
    noisy = torch.randn(B, N_SLOTS, 4)
    sigma = torch.rand(B).clamp_min(1e-3)
    tokens = torch.randn(B, T, C)
    centers = torch.zeros(T, 2)
    out = decoder(noisy_boxes=noisy, sigma=sigma, image_tokens=tokens, patch_centers=centers)
    assert out["pred_boxes"].shape == (B, N_SLOTS, 4)
    assert out["obj_logits"].shape == (B, N_SLOTS)
