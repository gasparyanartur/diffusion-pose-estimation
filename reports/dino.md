# DINOv3 Feature Probe — Investigation Report

**Date:** 2026-04-26
**Notebook:** [notebook/dino_probe.ipynb](../notebook/dino_probe.ipynb)
**Cache:** [results/cache/dino_probe/](../results/cache/dino_probe/)

## 1. Motivation

The pose refiner uses a frozen **DINOv3 ConvNeXt-Small** backbone
(`facebook/dinov3-convnext-small-pretrain-lvd1689m`) as its visual encoder.
Each keypoint receives a feature sampled from this backbone via the
`AttentionDinoConditioner`. We need to know whether these features are
**granular enough for fine-grained sub-patch refinement**: can a probe read
*joint identity* and *exact pixel-level joint position* off them?

This question matters because if DINO patches do not encode sub-patch
position, the refiner cannot improve localization beyond the patch grid no
matter what head we put on top.

## 2. Setup

- **Data:** CrowdPose val (2000 images), deterministically split 80/20
  (1600 train / 400 val) → 60 704 / 16 220 visible joints.
- **Backbone:** DINOv3 ConvNeXt-Small, frozen, eval mode.
  Stage strides 4 / 8 / 16 / 32, channels 96 / 192 / 384 / 768; pooled
  global = 768.
- **Sampling:** bilinear `grid_sample` at GT joint pixel; regression target
  is the pixel offset from the stage's nearest patch center to the GT joint.
- **Linear probe:** `Linear(C, out)`, AdamW lr=1e-2, wd=1e-4, batch=4096,
  20 epochs, seed=0.
- **MLP probe:** `Linear(C, 256) → GELU → Dropout(0.1) → Linear(256, out)`,
  AdamW lr=1e-3, wd=1e-4, batch=4096, 30 epochs, seed=0.
- **Features standardized** on train mean/std for every probe.
- **Trained-conditioner features:** `AttentionDinoConditioner._fuse_patches`
  loaded from
  [results/sweeps/arch-message-passing/task_1/20260316_003813/checkpoints/epoch140-running_loss_5=0.0155.ckpt](../results/sweeps/arch-message-passing/task_1/20260316_003813/checkpoints/epoch140-running_loss_5=0.0155.ckpt)
  (`inner_dim=256`, frozen). Pipeline: per-image stage1 + stage4 →
  `_fuse_patches` → `(256, H_s, W_s)` → sample at joints.

## 3. What was probed

| Feature        | C    | Notes                                               |
|----------------|------|-----------------------------------------------------|
| `stage1`       |   96 | stride 4 — fine-grained texture                     |
| `stage2`       |  192 | stride 8                                            |
| `stage3`       |  384 | stride 16 — mid-level                               |
| `stage4`       |  768 | stride 32 — high-level semantics                    |
| `global`       |  768 | pooled, no spatial info (chance baseline for cls)   |
| `fused`        |  864 | concat(stage1, stage4) — math equivalent of model fusion |
| `trained_cond` |  256 | output of trained `_fuse_patches` (MLP-fused)       |

Two probe tasks per feature:

- **Joint-type classification** (14-way, chance = 1/14 ≈ 0.071).
- **Joint-position regression** (predict pixel offset from patch center to
  GT joint). Compared against the **predict-zero baseline** — always
  outputting the patch center, which is the irreducible quantization floor
  of that stage's stride.

## 4. Results

### 4.1 Linear probes — per stage

| Feature   | top-1 cls | reg L2 (px) | predict-zero (px) |
|-----------|----------:|------------:|------------------:|
| `stage1`  |     0.246 |        3.65 |              3.65 |
| `stage2`  |     0.310 |        7.28 |              7.30 |
| `stage3`  | **0.459** |       14.78 |             14.76 |
| `stage4`  |     0.336 |       29.94 |             29.88 |
| `global`  |     0.080 |           — |                 — |
| `fused`   |     0.393 |        3.69 |              3.65 |

### 4.2 Linear vs MLP — main feature variants

| Feature        | cls (lin → MLP)   | reg L2 (lin → MLP, baseline) |
|----------------|-------------------|------------------------------|
| `stage1`       | 0.246 → **0.280** | 3.65 → **3.46** (3.65)       |
| `stage3`       | 0.459 → **0.497** | 14.78 → 14.94 (14.76)        |
| `fused`        | 0.393 → **0.430** | 3.69 → 3.74 (3.65)           |
| `trained_cond` | 0.315 → 0.339     | 3.66 → 3.66 (3.65)           |

### 4.3 Per-joint classification (best stage = `stage3`)

- Easy: head 0.85, neck 0.69.
- Hard: wrists / knees ~0.27. Errors are dominated by left ↔ right
  symmetric confusions and "limb-tip looks like another limb-tip".

### 4.4 Spatial resolution in original-image pixels

Measured on a probe-val image:

| Stage    | grid (Hs × Ws) | ~ px / patch |
|----------|----------------|--------------|
| `stage1` | 84 × 56        |        ~11.2 |
| `stage2` | 42 × 28        |        ~22.3 |
| `stage3` | 21 × 14        |        ~44.6 |
| `stage4` | 10 × 7         |        ~89.3 |

`stage1`'s patch width (~11 px) is the same scale as the regression floor
(~3.65 px), which is consistent with predictions collapsing inside the
patch.

### 4.5 Offset visualization on a probe-val image

GT mean offset 4.16 px. Predicted mean offset:
`stage1 = 0.14 px`, `fused = 0.52 px`, `trained_cond ≈ 0`. The probe is
collapsing to "output 0", not learning the offset. Notebook Section G
shows the scatter plots, per-joint L2 bars, and quiver overlay.

## 5. Key findings

1. **Mid-level features (stage3) carry the most linearly-decodable joint
   identity** — top-1 ≈ 0.46 linear, 0.50 MLP. `stage1` is too low-level,
   `stage4` too coarse, `global` is at chance.

2. **No feature linearly recovers sub-patch position.** Every regression
   probe lands within < 0.1 px of the predict-zero baseline at every stage.
   Predictions collapse to a horizontal stripe at offset = 0.

3. **A shallow MLP gives only a marginal lift.** Classification gains
   +3–4 pp across stages. For regression, only `stage1` + MLP edges below
   baseline (3.46 vs 3.65, ≈ 0.2 px / 5%). Non-linearity does not unlock
   sub-patch localization.

4. **The model's `fused` features inherit `stage1`'s failure.** The 864-d
   concat used by `DinoConditioner` does *not* outperform `stage1` for
   regression, and is below `stage3` for classification. The extra channels
   carry no localization signal beyond what `stage1` already has.

5. **Surprising — training the `AttentionDinoConditioner` did not make
   the patch tokens more localizable.** `trained_cond` features sit *below*
   the frozen `fused` features for classification (0.34 vs 0.43 MLP) and
   are pinned to the predict-zero floor for regression. Whatever
   localization the trained refiner produces must come from the **cross-
   attention readout** (rel-pos bias + neighborhood) on top of the tokens,
   not from the per-token features themselves.

## 6. Implications for the refiner

- A **linear** refinement head on these features cannot improve sub-patch
  localization. This is consistent with the observed regression floor.
- **Sub-patch refinement gain has to come from**:
  1. cross-attention readout that aggregates *neighborhood* tokens with an
     explicit positional code (which is what the current
     `AttentionDinoConditioner` does — the rel-pos bias supplies the
     missing positional signal),
  2. or a higher-resolution, localization-trained backbone (HRNet,
     RTMPose),
  3. or an explicit coordinate channel appended to the features.
- The classification ceiling (~0.50) suggests the conditioner has a clean
  "what kind of body region" signal to lean on, but this is decoupled
  from precise placement.

## 7. Open follow-ups

These are recorded but not yet executed:

1. Probe the **post-cross-attention node features** (the actual output of
   `AttentionDinoConditioner.forward`, not just `_fuse_patches`). Does the
   cross-attention readout recover sub-patch position at the *node* level?
2. **HRNet / RTMPose** probe as a localization upper bound on the same
   joints / split.
3. **Coordinate-channel sanity check**: append `(x, y)` as extra channels
   and confirm the probe code recovers offset signal when it is present.

## 8. Where to look in the notebook

- **A–B** — setup, feature extraction utility.
- **C** — probe dataset + caching.
- **D–E** — per-stage linear probe results + per-joint breakdown.
- **F** — fused-feature probe.
- **G** — offset visualizations (scatter, per-joint L2, quiver overlay).
- **H** — spatial-resolution grid + PCA-RGB feature maps per stage.
- **I** — MLP probe on `stage1` / `stage3` / `fused`.
- **J** — trained `AttentionDinoConditioner` probe.
