# DiffusionDet vs. BoxSetDiffusionDecoder — Pipeline Comparison Report

**Date:** 2026-05-27  
**Commit:** `71f38bf` (branch `diffusion-det`, HEAD)  
**Author:** Artur Gasparyan  
**Purpose:** Diagnose why our multi-step diffusion sampling produces no AP improvement over single-step, by mapping both pipelines in detail against the DiffusionDet reference (Gu et al., 2022; `external/diffusiondet`).

---

## Background

Our model (`BoxSetDiffusionDecoder`, `src/decoders/diffusion/box_set.py`) is a set-prediction diffusion decoder for bounding boxes. It was expected that running more EDM Heun steps at inference would progressively refine the predicted boxes, as reported for DiffusionDet in the paper. In practice, `AP(1-step) ≈ AP(5-step)` — the multi-step refinement produces no measurable improvement.

This report maps both pipelines stage-by-stage, identifies the structural differences responsible for the discrepancy, and lists actionable fix candidates.

---

## 1. Data / Target Preparation

### 1.1 GT Box Coordinate Space

| | DiffusionDet | Ours |
|---|---|---|
| **Source format** | COCO/Detectron2: absolute `xyxy` pixel | COCO: `xywh` pixel |
| **Normalization** | Divide by `(W, H, W, H)` → `[0, 1]` | `boxes_xywh_to_padded_canonical`: scale longest side to `S`, pad square → `cxcywh` |
| **Final diffusion space** | `(2x − 1) × scale` where `scale = SNR_SCALE = 2.0` → **`cxcywh ∈ [-2, +2]`** | Padded-canonical `cxcywh`: centres `∈ [-1, 1]`, sizes `∈ [0, 2]` |

The coordinate ranges are similar in intent but our `σ_max = 10` is very large relative to the `[-1, 1]` centre range — the signal-to-noise ratio at training time is much worse than DiffusionDet's effective SNR at `t = 0` (where `√ᾱ₀ ≈ 1, √(1−ᾱ₀) ≈ 0`).

### 1.2 Padding Non-GT Slots to Fixed N

| | DiffusionDet (`prepare_diffusion_concat`, `detector.py:573`) | Ours (`_build_noisy_input`, `box_diffusion_module.py`) |
|---|---|---|
| **Strategy** | `N(0.5, (1/6)²)` clipped at `w,h ≥ 1e-4` then mapped to `[-2, +2]` — random plausible object-shaped boxes | `(cx, cy) ~ U(-1, 1)`, fixed `(w, h) = 1.0` — random centres with full-image-size boxes |
| **Effect** | Placeholders look like possible detections; model learns to reject them | Placeholders are all large, overlapping boxes — possibly learnable cue |

### 1.3 Noise Model

| | DiffusionDet | Ours |
|---|---|---|
| **Schedule** | **Cosine** DDPM, `T = 1000` discrete steps, `s = 0.008` | **Continuous EDM** (Karras et al., 2022), `σ ~ logN(P_mean=−1.2, P_std=1.2)`, clamped to `[σ_min=0.002, σ_max=10]` |
| **Forward process** | `x_t = √ᾱ_t · x_start + √(1−ᾱ_t) · ε` | `x_noisy = x_start + σ · ε` |
| **Timestep / σ sampling** | One `t ~ Uniform[0, T)` per image, broadcast to all N proposals | One `σ` per image from log-normal, broadcast to all N slots |

### 1.4 Self-Conditioning at Training

| | DiffusionDet | Ours |
|---|---|---|
| **Enabled?** | **No** — `self.self_condition = False` (`detector.py:99`) | **Yes** — 50% per-sample no-grad forward to produce `x0_prev` |

DiffusionDet does not use self-conditioning; we implemented it in an attempt to improve multi-step AP, but it did not help (see Section 4).

---

## 2. Network Forward Pass

### 2.1 How Noisy Boxes Enter the Network (Critical)

| | DiffusionDet (`head.py:240`) | Ours (`box_set.py:185`) |
|---|---|---|
| **Mechanism** | Noisy boxes → convert to **absolute pixel `xyxy`** → **`ROIPooler` (RoIAlign)** → per-box spatial feature grid pooled from the FPN feature map at the box's location | Noisy boxes → **`FourierFeatures(cxcywh)`** → added to a **learned `nn.Embedding(N, D)` slot query** |
| **What the model sees per slot** | Actual image content extracted from the box region — physically different if the box moves | A positional embedding of the box coordinates — no image content is re-read |
| **Image feature map usage** | Multi-scale FPN `[P2..P5]`, each used as a spatial feature bank for RoIAlign | Single flat token sequence `[B, T, C]` from backbone (RADIO/DINO), with 2-D sinusoidal PE from patch centres |
| **Cross-image coupling** | None — each box is processed independently after pooling | Cross-attention: every slot attends to all image tokens |

**This is the primary structural difference.** In DiffusionDet the model literally reads a different crop of the image when the box has moved. In our model the image tokens are static; the box only shifts a query embedding.

### 2.2 Timestep Conditioning

| | DiffusionDet | Ours |
|---|---|---|
| **Embedding** | `SinusoidalPositionEmbeddings(t)` → 2-layer MLP → `time_dim = d_model × 4 = 1024` | `FourierFeatures(log σ)` → MLP |
| **Conditioning** | **Per-block FiLM** (scale + shift) inside each `RCNNHead`, applied to FC features after pooling | **Per-block FiLM** in every `TimeFiLMTransformerBlock` |

Both use FiLM-style conditioning; this is architecturally equivalent.

### 2.3 Iterative Refinement Inside One Forward Pass (Cascade)

| | DiffusionDet | Ours |
|---|---|---|
| **Cascade heads** | **6 `RCNNHead` modules** in sequence (`NUM_HEADS = 6`, `config.py:22`) | **1 unified block set** — no internal cascade |
| **Re-pooling between heads** | `bboxes = pred_bboxes.detach()` between heads → **each head re-pools at the refined boxes** | N/A |
| **Deep supervision** | Loss applied to all 6 head outputs (`DEEP_SUPERVISION = True`) | N/A |

The cascade is a second mechanism by which DiffusionDet uses box coordinates to select image features: 6 times per forward, the boxes are updated and RoIAlign is re-run. The model is explicitly trained by deep supervision to perform incremental refinement at each cascade stage.

---

## 3. Multi-Step Inference Sampling

### 3.1 Sampler

| | DiffusionDet (`ddim_sample`, `detector.py:170`) | Ours (`BoxSetDiffusionDecoder.sample`, `box_set.py:289`) |
|---|---|---|
| **Algorithm** | DDIM, `η = 1.0` (stochastic) | EDM Heun (2nd-order, deterministic given initial noise) |
| **Default steps** | `SAMPLE_STEP = 1` (paper ablations use 4 on COCO); DiffusionDet is also designed for 1-step use | `num_inference_steps = 5` |
| **Schedule** | Discrete cosine-schedule `t` indices, linearly spaced | Karras `σ` schedule, `ρ = 7` |

### 3.2 Image Features Across Steps

| | DiffusionDet | Ours |
|---|---|---|
| **Backbone re-run?** | No — `backbone_feats` computed once before the loop | No — `image_tokens` computed once before the loop |
| **RoIAlign re-run at new boxes?** | **Yes** — `model_predictions()` re-converts `img` to pixel boxes and re-pools at each step | **No** — `image_tokens` are the same tensor, no per-step re-extraction |

**The feedback loop in DiffusionDet:**

```
step i:  x_i  ─► pixel_boxes_i  ─► RoIAlign(FPN, pixel_boxes_i)  ─► refine  ─► x̂₀
DDIM update ─► x_{i+1}  (new box positions)
step i+1: x_{i+1}  ─► pixel_boxes_{i+1}  ─► RoIAlign(FPN, pixel_boxes_{i+1})  ─► refine  ─► x̂₀
                                                       ↑
                                          different pooled features
```

Each step genuinely changes the features seen by the network because the physical crop has shifted. In our pipeline, the features are identical every step; only the Fourier embedding of the box coordinates changes, which the network has learned to ignore when dominant image features are available.

### 3.3 Box Renewal

| | DiffusionDet | Ours |
|---|---|---|
| **Enabled** | **Yes** — `self.box_renewal = True` (`detector.py:100`) | Not implemented |
| **Mechanism** | After each denoising step, drop all slots with `max_class_score < 0.5`; replenish removed slots with fresh `N(0,1)` noise | — |
| **Effect** | Progressive filtering: multi-step explores box space; low-quality proposals are replaced with new samples | Recall is capped at what the first step produces |

### 3.4 Multi-Step Ensemble

| | DiffusionDet | Ours |
|---|---|---|
| **Enabled** | `self.use_ensemble = True` — active when `sampling_timesteps > 1` | Not implemented |
| **Mechanism** | Predictions from all intermediate steps are concatenated; one NMS pass over the union | Only the final step's predictions are used |
| **Effect** | Multi-step naturally has higher recall since it's a union; NMS collapses duplicates | Information from intermediate steps is discarded |

---

## 4. Loss, Matching, Evaluation

| | DiffusionDet | Ours |
|---|---|---|
| **Matcher** | `HungarianMatcherDynamicK` (OTA-style top-K ≈ 5 per GT; foreground filter by 2.5× radius) | Standard Hungarian 1-to-1 |
| **Classification loss** | Sigmoid focal (α=0.25, γ=2.0) | Objectness BCE + pairwise RankNet |
| **Box regression loss** | L1 (norm.) + GIoU (absolute), all matched predictions (up to K per GT) | L1 + GIoU + diffusion-L2, one-to-one matched |
| **Deep supervision** | All 6 cascade heads contribute | N/A |
| **Post-processing** | Top-K flatten → per-class NMS @ 0.5 → top-100 kept | `sigmoid(obj_logits)` as score, no NMS |

The absence of NMS is a minor factor for AP but means our model needs to rank proposals by itself and cannot merge nearby duplicates.

---

## 5. Key Hyperparameters

| Parameter | DiffusionDet | Ours | Notes |
|---|---|---|---|
| `num_proposals` | 300 | 20 (`num_slots`) | Ours is 15× smaller |
| `sampling_timesteps` | 1 (default) | 5 | DiffusionDet is designed for 1-step use |
| `scale` / `σ_max` | `scale = 2.0` (boxes in `[-2, +2]`) | `σ_max = 10.0` | Our σ_max is 5× the data range — very low SNR |
| Noise schedule | Cosine, discrete T=1000 | EDM continuous logN | Different families |
| Cascade depth | 6 internal heads | 0 | |
| `box_renewal` | True | False | |
| Ensemble | True (if steps > 1) | False | |
| Focal loss | Yes | No | |
| NMS | Yes, per-class @ 0.5 | No | |
| Deep supervision | Yes | No | |

---

## 6. Root-Cause Diagnosis

### 6.1 Primary Cause: No Box-Conditional Feature Extraction

**DiffusionDet's multi-step works because each step re-reads a different region of the feature map.** The noisy boxes are RoIAlign regions, not just positional embeddings. Moving boxes → different features → different prediction → better x̂₀. This is reinforced by the internal 6-head cascade, which repeats this logic within one forward pass.

In our model, **the image tokens are a fixed set of global tokens**. Box coordinates enter only as a Fourier embedding on the slot queries. The cross-attention mechanism attends globally and is not constrained by box positions. A well-trained network simply learns the best-possible image-driven answer and outputs it regardless of what `noisy_boxes` contains, because the image features dominate and the box signal does not provide anything the model cannot already infer from the image.

Self-conditioning (`x0_prev`) adds a second box-derived channel but has the same fundamental limitation: it does not change which image features the network reads.

### 6.2 Contributing Causes (Smaller Effect)

1. **`σ_max = 10` greatly exceeds the data scale `[-1, 1]`** — at high σ the noisy boxes are pure noise; the model is trained to ignore them at those noise levels, so the model relies on image features exclusively at the first (highest-σ) step.
2. **No box renewal** — low-quality proposals are never discarded and resampled; recall cannot improve beyond step 1.
3. **No multi-step ensemble + NMS** — information from intermediate steps is thrown away.
4. **No deep supervision** — the network is never trained to "given approximately-correct boxes, refine them"; it is only trained to "given random noisy boxes, output GT." The cascade in DiffusionDet explicitly teaches incremental refinement.
5. **Dynamic-K matching vs 1-to-1 Hungarian** — fewer positive assignments per image makes gradients noisier for rare/small objects (minor issue vs the architectural one).

---

## 7. Proposed Fix Candidates (Ranked)

1. **Box-conditional feature pooling** *(most important)*  
   At each decoder forward pass, for each slot, spatially sample image tokens in the neighbourhood of the slot's current box (e.g., bilinear sampling at 7×7 points inside the box, or RoIAlign-style if using a spatial backbone). Concatenate or add these to the slot input before the transformer blocks. This mirrors DiffusionDet's RoIPooler without requiring an FPN — can be implemented on top of RADIO spatial tokens.

2. **Internal cascade with deep supervision**  
   Stack multiple refinement blocks where each block reads the box produced by the previous block to re-pool features, and apply a box+objectness loss to every block's output. Teaches the model the "refine approximately-correct boxes" task.

3. **σ_max reduction to ~1.0**  
   Align with EDM paper convention (`σ_data ≈ 0.5` for coordinates in `[-1, 1]`). At `σ_max = 1` the noisy boxes still carry usable signal, so the model cannot trivially ignore them.

4. **Box renewal in `sample()`**  
   After each step, drop low-objectness slots and replace them with fresh noise; equivalent to importance-resampling.

5. **Multi-step ensemble + NMS**  
   Concatenate predictions from all sampling steps; apply per-class NMS. A cheap way to recover some of the recall lost by fixing the shortcut.

**Recommendation:** Option 1 is the structural fix that addresses the root cause. Options 3–5 are low-cost complementary improvements. Option 2 adds significant complexity and is worth considering only after Option 1 is validated.

---

## Appendix: File Index

| File | Role |
|---|---|
| `src/decoders/diffusion/box_set.py` | `BoxSetDiffusionDecoder`: forward, sample, EDM schedule |
| `src/training/box_diffusion_module.py` | Lightning module: training_step, _build_noisy_input, self-cond mask |
| `src/training/losses.py` | `SetBoxDiffusionLoss`: Hungarian matching, L1+GIoU+rank |
| `src/data/transforms.py` | `boxes_xywh_to_padded_canonical`, coordinate transforms |
| `src/metrics.py` | `BoxAPMetric`: COCO AP via pycocotools |
| `external/diffusiondet/detector.py` | DiffusionDet main model: prepare_targets, q_sample, ddim_sample, box_renewal |
| `external/diffusiondet/head.py` | `DynamicHead`, `RCNNHead`: RoIAlign pooling, 6-head cascade, FiLM conditioning |
| `external/diffusiondet/loss.py` | `HungarianMatcherDynamicK`, focal loss, GIoU loss |
| `external/diffusiondet/config.py` | Default hyperparameters (NUM_HEADS=6, scale=2, SAMPLE_STEP=1, …) |
