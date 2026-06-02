# Backbone Probe Design Report

**Date:** 2026-06-01
**Entry point:** `scripts/benchmark_backbone.py`
**Key sources:**
- `src/training/backbone_probe.py` — probe heads and Lightning module
- `src/features/` — backbone extractors
- `src/training/losses.py` — set-prediction losses
- `src/data/datasets.py` — `SetPoseDataset` and `set_pose_collate`
- `conf/benchmark_backbone.yaml` and `conf/training_module/backbone_probe*.yaml`

---

## 1. Purpose

The backbone probe benchmark answers a single question: *how much pose
information does a given feature extractor expose?* Every experiment uses the
same fixed probe head and training recipe; only the backbone changes. This
ensures that any difference in metric between runs is attributable entirely
to the backbone.

---

## 2. Data loading

### Dataset

The probe uses `SetPoseDataset` (`src/data/datasets.py`), a set-format wrapper
around `PoseDataset`. Each sample corresponds to one image with an arbitrary
number of people.

`__getitem__` returns:

| key              | shape / type          | note                                         |
|------------------|-----------------------|----------------------------------------------|
| `image`          | `uint8 [C, H, W]`     | read from disk with `torchvision.io.read_image` |
| `gt_keypoints`   | `[P, K, 2]`           | converted to canonical space (−1, 1) at load time |
| `gt_labeled`     | `[P, K]`              | `visibility > 0`                             |
| `gt_visible`     | `[P, K]`              | `visibility == 2`                            |
| `person_scale`   | `[P]`                 | canonical-space bounding-box extent per person |
| `image_id`       | int                   | CrowdPose image ID                           |
| `img_hw`         | `[2]`                 | original `(H, W)` in pixels                  |

Keypoints are converted to canonical space (`pixel / [W, H] × 2 − 1`) in
`__getitem__` so that loss and AP evaluation never need raw pixel dimensions.
`person_scale` is computed as the max canonical-space extent (width or height)
of each person's labeled-keypoint bounding box.

### Collation

`set_pose_collate` pads all per-image tensors to the batch's maximum person
count `P_max`, then emits a binary `person_mask [B, P_max]` whose value is 1
for real persons and 0 for padding. Images are kept as a `list[Tensor]` of
variable spatial size because they have not been resized yet.

### GPU-side preprocessing (`on_after_batch_transfer`)

`PoseDataModule.on_after_batch_transfer` runs on the accelerator after the
batch is transferred to GPU:

1. Each uint8 image is cast to float and divided by 255.0.
2. If `max_size` is set (default 128 for the backbone benchmark), each image
   is resized and zero-padded to a square of that side length using
   `resize_and_pad_to_square`. All images in the batch are then stacked into
   `[B, 3, S, S]` with a companion `padding_mask [B, S, S]`.
3. If explicit `(image_h, image_w)` are given, images are resized to that
   fixed size with bilinear interpolation.

This design keeps the CPU workers free from resizing and moves that cost to
the GPU, where it is parallelised across the batch.

### Annotation files

Configured by `conf/data/crowdpose_set_prediction.yaml`:
- Train: `crowdpose/annotations/crowdpose_train.json`
- Val:   `crowdpose/annotations/crowdpose_val.json`
- Test:  `crowdpose/annotations/crowdpose_test.json`
- Images: `crowdpose/images/`

Batch size: 64 with 8 data-loader workers.

---

## 3. Backbone extractors

All backbone extractors inherit from `BaseFeatureExtractor` (`src/features/base.py`).
They must implement:
- `forward(image, image_ids, **kwargs) -> dict[str, Any]` — returns at minimum
  `"spatial"` (a `[B, C, H', W']` feature map) and `"patch_centers"`.
- `get_output_size(input_size) -> int` — number of spatial positions on one axis.
- `out_channels: int` — number of feature channels C.

Upstream code feeds raw `[0, 1]` float tensors; each extractor applies its
own normalisation internally.

### Available extractors

| Config yaml                    | Class                         | C    | stride | Notes                                          |
|-------------------------------|-------------------------------|------|--------|------------------------------------------------|
| `cnn`                         | `CNNFeatureExtractor`         | 128  | 16     | 4-stage conv trained from scratch               |
| `resnet18`                    | `ResNetFeatureExtractor`      | 256  | 16     | torchvision ResNet-18, `layer3`, ImageNet pretrain |
| `resnet34`                    | `ResNetFeatureExtractor`      | 256  | 16     | torchvision ResNet-34, `layer3`, ImageNet pretrain |
| `vit-b16`                     | `ViTFeatureExtractor`         | 768  | 16     | torchvision ViT-B/16, resized to 224×224 if needed |
| `dino-v3-convnext-small`      | `DinoFeatureExtractor`        | 768  | 32     | HuggingFace `facebook/dinov3-convnext-small`, frozen by default, stage 4 |
| `dino-v3-convnext-small-s3`   | `DinoFeatureExtractor`        | 384  | 16     | same model, stage 3                            |
| `dino-v3-vit-s16`             | `DinoViTFeatureExtractor`     | 384  | 16     | DINOv3 ViT-S/16, patch tokens reshaped to `[B, D, G, G]` |
| `c-radio-v3-b`               | `RadioFeatureExtractor`       | auto | 16     | `nvidia/C-RADIOv3-B`, normalisation by model's `input_conditioner` |
| `c-radio-v4-h`               | `RadioFeatureExtractor`       | auto | 16     | `nvidia/C-RADIOv4-H`                           |
| `identity`                    | `IdentityFeatureExtractor`    | —    | —      | passthrough for ablations                      |

### Frozen vs. trainable

Pretrained vision-foundation-model extractors (`DinoFeatureExtractor`,
`DinoViTFeatureExtractor`, `RadioFeatureExtractor`) support a `frozen` flag
(default `True`). When frozen, `p.requires_grad_(False)` is called on all
backbone parameters during construction. The `BackboneProbeModule.state_dict`
override explicitly strips all keys starting with `"backbone."` from the
saved checkpoint, so frozen backbone weights are never written to disk.

---

## 4. ConcatFeatureExtractor: fusing multiple backbones

`ConcatFeatureExtractor` (`src/features/concat.py`) allows two or more
extractors to be combined into a single token sequence that the probe head
sees as one backbone.

**Construction:**

```python
ConcatFeatureExtractor(
    backbones=[backbone_a, backbone_b, ...],
    dim=256,          # shared projection dimension
    feature_key="spatial",
)
```

For each child backbone the module creates:
- a `LayerNorm(child.out_channels)`,
- a `Linear(child.out_channels → dim)`,
- a learned modality embedding `nn.Parameter [1, 1, dim]` (initialised to zero).

**Forward pass (per child backbone `i`):**

1. Run child `i` on the image to obtain `feat [B, C_i, H_i, W_i]`.
2. Flatten spatial: `tokens = feat.flatten(2).permute(0, 2, 1)` → `[B, H_i*W_i, C_i]`.
3. Apply `LayerNorm(C_i)`.
4. Project: `Linear(C_i, dim)` → `[B, H_i*W_i, dim]`.
5. Add the learned modality embedding `[1, 1, dim]` (broadcast).
6. Collect the child's `patch_centers [H_i*W_i, 2]`.

After all children are processed, the token sequences and patch center lists
are concatenated along the sequence axis:

```
output["spatial"]       = cat([tokens_0, tokens_1, ...], dim=1)   # [B, T, dim]
output["patch_centers"] = cat([centers_0, centers_1, ...], dim=0) # [T, 2]
```

where `T = sum_i(H_i × W_i)`.

The `out_channels` property returns `dim`, making `ConcatFeatureExtractor`
a drop-in for any single extractor.

**Key design decisions:**
- Each modality gets its own LayerNorm before the shared projection, so
  activations from very different backbones are brought to a comparable range
  before being projected.
- The learned modality embedding gives the downstream probe a way to
  distinguish tokens from different backbones without relying solely on
  positional cues.
- The sinusoidal positional encoding is *not* added inside
  `ConcatFeatureExtractor`; it is the probe head's responsibility, ensuring
  each token's position in its own backbone's grid is encoded correctly and
  independently.

**Concrete configs using concat:**

| Config yaml                     | Children                            |
|---------------------------------|-------------------------------------|
| `concat-cnn-resnet`             | `CNNFeatureExtractor` + `ResNet18/layer3` |
| `concat-resnet18-radio-v3b`     | `ResNet18/layer3` + `C-RADIOv3-B` (frozen) |
| `concat-resnet34-radio-v4h`     | `ResNet34/layer3` + `C-RADIOv4-H` (frozen) |

---

## 5. Probe heads

Two probe heads are implemented. Both are constructed internally by the
training config and are kept **identical across all backbone variants** (only
`in_channels` changes to match the backbone's `out_channels`). The backbone
is the only variable.

### 5.1 ConvProbe (default, `backbone_probe.yaml`)

`ConvProbe` (`src/training/backbone_probe.py`) processes a `[B, C, H, W]`
spatial feature map through a convolutional stack followed by a Transformer
readout.

**Architecture pipeline:**

```
feat [B, C, H, W]
  proj1(1×1): C → proj_dim=768          # widen to adapter dimension
  ReLU
  proj2(1×1): proj_dim → conv_dim=96    # compress to cheap working dim
+ sinusoidal 2D PE added at conv_dim resolution
  ×5  ConvNeXtBlock(conv_dim=96)
        DWConv(7×7) → LN → Linear(96→384) → GELU → Linear(384→96) → Dropout + residual
      [if max(H, W) > target_size=16: AvgPool2d(2,2)]
  readout_proj(1×1): conv_dim → feature_dim=256
  flatten → [B, H'×W', 256]
+ sinusoidal 2D PE added at feature_dim resolution
  ×2  TransformerBlock(256, heads=4)
        pre-norm MHSA + Add, FFN (expansion=2) + Add
  mean-pool over tokens → [B, 256]
  shared_proj: LN(256) → Linear(256→256) → ReLU → Dropout(0.2) → Linear(256→256) → LN
─────────────────────────────────────────────
  scores_head:    Linear(256 → N=20)          → scores_logits  [B, N]
  keypoints_head: Linear(256 → N×K×2=560)    → kpt_logits     [B, N, K, 2]
  vis_head:       Linear(256 → N×K=280)      → vis_logits      [B, N, K]
```

Output dict:

| key              | shape       | formula                              |
|------------------|-------------|--------------------------------------|
| `keypoints`      | `[B, N, K, 2]` | `sigmoid(kpt_logits) × 2 − 1`    |
| `scores`         | `[B, N]`   | `sigmoid(scores_logits)`             |
| `scores_logits`  | `[B, N]`   | raw logits                           |
| `keypoint_scores`| `[B, N, K]`| `sigmoid(vis_logits)`                |
| `vis_logits`     | `[B, N, K]`| raw logits                           |

Keypoints are in canonical space (−1, 1) by construction: the sigmoid maps
raw logits to (0, 1), then `× 2 − 1` maps to (−1, 1).

**Default hyperparameters (`backbone_probe.yaml`):**

| param             | value  |
|-------------------|--------|
| `feature_dim`     | 256    |
| `proj_dim`        | 768    |
| `conv_dim`        | 96     |
| `target_size`     | 16     |
| `num_conv_layers` | 5      |
| `num_attn_layers` | 2      |
| `num_heads`       | 4      |
| `ffn_mult`        | 2      |
| `conv_dropout`    | 0.1    |
| `attn_dropout`    | 0.3    |
| `ffn_dropout`     | 0.5    |
| `hidden_dim`      | 256    |
| `num_slots` (N)   | 20     |
| `proj_dropout`    | 0.2    |

### 5.2 QueryProbe (DETR-style, `backbone_probe_query.yaml`)

`QueryProbe` replaces the convolutional pooling stack with DETR-style
cross-attention. It operates on `[B, T, D]` token sequences and is therefore
compatible with both `[B, C, H, W]` and pre-flattened `[B, T, D]` backbones
(including `ConcatFeatureExtractor` output).

**Architecture pipeline:**

```
feat [B, C, H, W]  or  [B, T, C]
  if spatial: permute+flatten → [B, HW, C]
  proj: Linear(C → D=256) → [B, T, D]
+ positional encoding from patch_centers (sinusoidal, scale=1000)
= patch_tokens [B, T, D]

── Instance path ──────────────────────────────────────────────────────
  instance_queries  [N, D]  (learned Embedding)
  expand → [B, N, D]
  inst_cross_attn: Q=[B,N,D] attends to KV=patch_tokens [B,T,D]
  inst_self_attn:  self-attention on [B, N, D]
  objectness_head: LN → Linear(D→D/2) → GELU → Dropout → Linear(D/2→1)
  → scores_logits [B, N]

── Joint path ──────────────────────────────────────────────────────────
  joint_queries  [K, D]  (learned Embedding)
  expand → [B*N, K, D]
  joint_cross_inst: Q=[B*N,K,D] cross-attends to inst_latents [B*N,1,D]
  joint_self_attn: self-attention on [B*N, K, D]
  reshape → [B, N, K, D]
  final_norm: LayerNorm(D)

  vis_head (if predict_visibility):
      Linear(D→D/2) → Dropout → GELU → Dropout → Linear(D/2→1)
      → vis_logits [B, N, K]
  keypoints_head:
      Linear(D→D/2) → GELU → Linear(D/2→2)
      → kpt_logits [B, N, K, 2]
```

Output dict is identical to `ConvProbe`. The `predict_visibility` flag is
`False` by default (visibility loss weight is 0.0 in
`backbone_probe_query.yaml`); when disabled, `vis_logits` are fixed zeros.

**Default hyperparameters (`backbone_probe_query.yaml`):**

| param               | value  |
|---------------------|--------|
| `feature_dim` (D)   | 256    |
| `num_slots` (N)     | 20     |
| `num_heads`         | 4      |
| `ffn_mult`          | 2      |
| `attn_dropout`      | 0.0    |
| `ffn_dropout`       | 0.1    |
| `proj_dropout`      | 0.1    |
| `pe_scale`          | 1000.0 |
| `predict_visibility`| false  |

### 5.3 Parameter containment

The probe head is constructed entirely **inside the training config's `build()`
method** and stored in `BackboneProbeModule.probe_head`. The module never
touches the backbone's parameters:

- `BackboneProbeModule.state_dict` strips all keys that start with
  `"backbone."`, so checkpoints contain only probe-head weights.
- When a backbone is frozen (`frozen=True`), `requires_grad_(False)` is
  called on all its parameters at construction time; the optimiser in
  `BackboneProbeModule.configure_optimizers` iterates over
  `self.parameters()`, but frozen parameters have zero gradients and are
  skipped by AdamW.

This ensures:
1. No backbone gradients leak into the probe.
2. Checkpoints are small and backbone-independent (can be reloaded with a
   different backbone).
3. Metrics across runs are comparable because the probe architecture and
   initialisation conditions are identical.

---

## 6. Training module

`BackboneProbeModule` (`src/training/backbone_probe.py`) is a PyTorch
Lightning module. Its forward pass is:

```python
features = backbone(image=batch["image"])  # backbone-specific dict
pred     = probe_head(features)            # standardised pred dict
```

### Augmentations

Applied only in `training_step`, before the forward pass:

| augmentation        | default  | description                             |
|---------------------|----------|-----------------------------------------|
| Gaussian noise      | std=0.05 | additive noise on the `[0,1]` image     |
| Color jitter        | brightness=0.3, contrast=0.3, saturation=0.2 | applied per-image |

No spatial transforms are applied; keypoint canonical coordinates remain
valid.

### Optimiser and schedule

AdamW with:
- lr = 4e-4, weight_decay = 1e-3
- Cosine annealing over `max_epochs=100`, floor `lr × 0.1`
- Linear warmup for 1 epoch
- Gradient clip: 5.0

---

## 7. Losses

All losses are defined in `src/training/losses.py` and composed by
`SetPredictionLoss`. The loss is computed per-image in the batch; per-image
values are averaged.

### 7.1 Matching

Before any component loss is computed, predicted slots must be matched to GT
persons. This is done once per image by `hungarian_match`:

1. Compute the OKS matrix `[N, P]` between every predicted slot and every GT
   person (see §7.2 for OKS definition).
2. Build a cost matrix:
   ```
   cost[n, p] = λ_oks × (1 − oks[n, p]) + λ_obj × (−log σ(objectness_logit[n]))
   ```
   Default: `λ_oks = 1.0`, `λ_obj = 0.1`.
3. Solve the linear sum assignment problem with `batch_linear_assignment`
   (torch-linear-assignment). This yields `M = min(N, P)` matched
   `(slot_idx, gt_idx)` pairs.

GT persons with no labeled joints are dropped before matching.

### 7.2 Object Keypoint Similarity (OKS)

```
OKS(n, p) = Σ_k [ exp(−d²_k / (2 s²_p σ²_k)) × labeled_k ] / Σ_k labeled_k
```

where:
- `d²_k` = squared canonical-space distance between predicted and GT keypoint k,
- `s_p` = canonical-space scale of person p (bounding box extent),
- `σ_k` = per-keypoint sigma (CrowdPose values from `CROWDPOSE_SIGMAS`),
- `labeled_k ∈ {0, 1}` — whether joint k is annotated for person p.

OKS is computed between all N slots and all P GT persons simultaneously,
producing the `[N, P]` matrix used for matching.

### 7.3 Loss components

All component losses are averaged over the batch dimension.

**OKS loss** (`weight = 2.0`):

```
L_oks = mean over matched pairs { 1 − OKS(slot_n, gt_p) }
```

Only matched slots contribute; unmatched slots are not penalised here.

**Objectness loss** (`weight = 1.0`):

Binary cross-entropy on all N slots:
- Target = 1 for matched slots, 0 for unmatched slots.
- `pos_weight = min((N − M) / max(M, 1),  20)` compensates for the
  slot imbalance (N >> M in scenes with few people).

```
L_obj = BCE_with_logits(scores_logits [N], target [N], pos_weight)
```

**Visibility loss** (`weight = 0.5`, disabled in query probe by default):

BCE over matched slots, restricted to labeled joints:
```
L_vis = Σ_{matched n, labeled k} BCE(vis_logits[n,k], visible[gt_p, k])
        / number_of_labeled_joints
```

**Ranking loss** (`weight = 1.0`):

Pairwise RankNet loss encouraging matched (positive) slots to have higher
objectness logits than unmatched (negative) slots:
```
L_rank = −mean_{i∈matched, j∈unmatched} log σ(logit_i − logit_j)
```

**Total loss:**
```
L = 2.0 × L_oks  +  1.0 × L_obj  +  0.5 × L_vis  +  1.0 × L_rank
```

(Visibility loss weight is 0.0 for the query probe configuration.)

### 7.4 Diagnostic: mean OKS

`mean_oks` is the detached mean OKS over matched slots and is logged as a
training diagnostic, not added to the total loss.

---

## 8. Evaluation

AP is computed using `KeypointAPMetric` (wrapping MMPose `CocoMetric`) on the
CrowdPose val and test sets. AP evaluation runs every `val_ap_every_n_epochs =
10` epochs during training, and once on the test set after training.

Predicted keypoints are converted from canonical space back to pixel space
before AP submission: `pixel = (canonical + 1) / 2 × [W, H]`.
