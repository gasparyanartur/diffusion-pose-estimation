# Joint Box-and-Pose Diffusion Formulation

This document describes the `GraphDiffusionPipeline`: a unified diffusion model that
simultaneously predicts bounding boxes and skeleton keypoints for all persons in a
crowded image.  It is implemented across three files:

- `src/decoders/diffusion/graph_pipeline.py` — decoder / denoiser
- `src/training/graph_pipeline_module.py`  — Lightning training module
- `src/training/losses.py` (appended)       — end-to-end Hungarian loss

---

## 1. Coordinate System

All spatial quantities are in **canonical** (−1, 1) space computed from the original
pixel image:

| Quantity | Formula |
|---|---|
| keypoint x | `x_px / W × 2 − 1` |
| keypoint y | `y_px / H × 2 − 1` |
| box centre cx | `(x + w/2) / W × 2 − 1` |
| box centre cy | `(y + h/2) / H × 2 − 1` |
| box width cw | `w / W × 2` |
| box height ch | `h / H × 2` |

`W`, `H` are the original image dimensions.  This is the inverse of
`canonical_to_pixel` and is consistent with `pixel_to_canonical` so that box
centres and absolute keypoint positions are directly comparable.

Bounding boxes are stored as `(cx, cy, cw, ch)`.  Keypoints are stored in two
equivalent forms: **absolute** (`kpt_abs ∈ ℝ²`, same space as keypoints) and
**relative** (`kpt_rel = kpt_abs − box_ctr`).  The diffusion state uses relative
form; the loss operates in absolute form.

---

## 2. Dataset and Data Preprocessing

### Source

CrowdPose annotations (train / val / test splits).  The standard
`crowdpose_set_prediction` data config is used with `max_size: 128` (square
padding).  The `SetPoseDataset` / `set_pose_collate` pipeline in
`src/data/datasets.py` produces one dict per batch:

| Key | Shape | Description |
|---|---|---|
| `image` | `[B, 3, H, W]` | Normalised float image (resized + padded) |
| `gt_keypoints` | `[B, P, K, 2]` | Per-axis canonical keypoints |
| `gt_labeled` | `[B, P, K]` | `{0, 1}` — joint is annotated |
| `gt_visible` | `[B, P, K]` | `{0, 1}` — joint is unoccluded |
| `gt_boxes_pixel` | `[B, P, 4]` | `(x, y, w, h)` in original pixels |
| `gt_box_labeled` | `[B, P]` | `{0, 1}` — person slot is real |
| `img_hw` | `[B, 2]` | Original `(H, W)` |
| `image_id` | `[B]` | COCO image ID |

`P` is padded to the largest person count in the batch.

### Batch normalisation (`_normalize_batch`)

Before entering any step, the training module remaps keys and recomputes
canonical boxes with per-axis scaling (so box centres match keypoint canonical
coords):

```
gt_joints_canon  ← gt_keypoints
gt_visibility    ← gt_labeled
gt_boxes_canon   ← _boxes_pixel_to_canonical(gt_boxes_pixel, img_hw)
```

### Noisy input construction (`_build_noisy_input`)

For each training step:

1. **Pad** GT boxes to `N` slots.  Extra slots are filled with
   `cx, cy ~ U(−1, 1)`, `w, h ~ U(0.05, 0.5)`.
2. **Compute** `gt_joints_rel = gt_joints_canon − gt_box_ctr` for real slots;
   zero for padding slots.
3. **Shuffle** all `N` slots with a random permutation `π` (independently per
   image).  The loss uses Hungarian matching so it is order-invariant.
4. **Add noise**: sample `ε ~ N(0, I)` and form
   `x_boxes = x₀_boxes + σ·ε_boxes`, `x_joints = x₀_joints + σ·ε_joints`.

`σ` is drawn once per image from the EDM log-normal distribution (see §5).

---

## 3. Backbone: Frozen RADIO

The backbone is `nvidia/C-RADIOv3-B` (ViT-B/16, ~90 M params), loaded via
`RadioFeatureExtractor` in `src/features/radio.py`.  **All parameters are
frozen**; only the decoder is trained.

The backbone applies CLIP-style normalisation internally and takes raw `[0, 1]`
float images.

**Outputs** (after reshaping in `_radio_features`):

| Variable | Shape | Description |
|---|---|---|
| `image_tokens` | `[B, T, C]` | Flat patch tokens (`T = G_H × G_W`) |
| `patch_centers` | `[T, 2]` | Canonical centre of each patch |
| `spatial_feat` | `[B, C, G_H, G_W]` | 2-D spatial feature map |

For a 224 × 224 input: `G_H = G_W = 14`, `T = 196`, `C = 768`.

---

## 4. Denoiser Architecture (`GraphDiffusionPipelineDecoder`)

The denoiser is called once per diffusion step.  It takes noisy state
`(x_boxes, x_joints_rel, σ)` and image features, and predicts clean `x₀`.

### 4.1 Inputs

| Input | Shape | Embedding |
|---|---|---|
| `noisy_boxes` | `[B, N, 4]` | `FourierFeatures(4→D)` |
| `noisy_joints_rel` | `[B, N, K, 2]` | `FourierFeatures(2→D)` |
| `σ` | `[B]` | `FourierFeatures(1→T_dim)` on `log σ` |
| `image_tokens` | `[B, T, C]` | `Linear(C→D)` + sinusoidal PE |
| `spatial_feat` | `[B, C, G_H, G_W]` | bilinear sampling (Stage 2 only) |

`FourierFeatures` projects via random Fourier bands: `[sin(2π·W·x), cos(2π·W·x)]`
then a final linear to `D`.

### 4.2 Stage 1 — Attention Trunk (instance × image)

```
slot_tokens = box_in(noisy_boxes)          # [B, N, D]
slot_tokens += slot_query.weight           # learned per-slot bias
time_emb    = time_in(log σ)              # [B, T_dim]
img         = image_proj(image_tokens) + sinusoidal_PE   # [B, T, D]

for blk in trunk_blocks:   # num_trunk_blocks × TimeFiLMTransformerBlock
    slot_tokens = blk(slot_tokens, context=img, time_emb=time_emb)
    # self-attn over slots + cross-attn to image + FiLM time-modulation + FFN

inst_latents = slot_tokens                # [B, N, D]
obj_logits   = obj_head(inst_latents)     # [B, N]    (LayerNorm → Linear(D→1))
pred_boxes   = box_head(inst_latents)     # [B, N, 4] (LayerNorm → Linear(D→4))
```

### 4.3 Stage 1 — Joint Path (joints × instance)

```
joint_tokens = joint_in(noisy_joints_rel)       # [B, N, K, D]
joint_tokens += joint_query.weight              # per-keypoint-type bias [K, D]
joint_tokens = reshape(B*N, K, D)

inst_kv = reshape(inst_latents, B*N, 1, D)
time_per_slot = expand(time_emb, B*N, T_dim)

for blk in joint_blocks:   # num_joint_blocks × TimeFiLMTransformerBlock
    joint_tokens = blk(joint_tokens, context=inst_kv, time_emb=time_per_slot)
    # cross-attn to instance latent + FiLM + FFN

joint_tokens = joint_self_attn(joint_tokens, time_emb=time_per_slot)
# within-slot joint self-attention (all K joints of one person interact)

rough_joints_rel = joint_head(joint_tokens)  # [B, N, K, 2]
```

Stage 1 produces a rough prediction of `(boxes, joints_rel)`.

### 4.4 Stage 2 — GNN Skeleton Refiner

Stage 2 refines the rough keypoints using local skeleton structure and
image evidence at each joint location.

**Graph construction**: rough absolute joints `joints_abs = pred_box_ctr + rough_joints_rel`
are packed into a flat block-diagonal PyG graph using `create_graph_from_keypoints`.
This replicates the CrowdPose skeleton (`CANONICAL_EDGE_INDEX`, 14 nodes and 28
undirected edges) for every person in every image, yielding `M = B × N × K`
nodes.

```
# Node features: position + joint-type embedding
node_pos   = joints_abs.reshape(M, 2)
type_flat  = kpt_type_idx.reshape(M)        # which of 14 keypoint types
h = LayerNorm(
        refiner_pos_emb(node_pos)           # MLP: 2 → D
      + refiner_type_emb(type_flat)         # Embedding(K, D)
    )                                       # [M, D]

# Edge features: relative displacement between connected nodes
e = edge_encoder(node_pos[src] − node_pos[dst])   # [E, dim_edge]

# Visual features: bilinear sample from spatial_feat at each joint
grid = joints_abs.reshape(B, N*K, 1, 2)
v_raw = F.grid_sample(spatial_feat, grid)         # [B, C, N*K, 1]
v = vis_proj(v_raw.reshape(B*N*K, C))             # [M, dim_vis]

# Global conditioning: zero (FiLM learns bias only)
g = zeros(M, 1)

delta = refiner(h, edge_index, e, v, g)           # [M, 2]
pred_joints_rel = rough_joints_rel + delta.reshape(B, N, K, 2)
```

**`GraphDenoiseNetwork`** (`src/decoders/denoise/graph_network.py`) stacks
`num_refiner_layers` × `PoseGraphLayer`, then applies a zero-initialised linear
head to produce `(dx, dy)` per node.

#### What one `PoseGraphLayer` does

Think of each keypoint as a node holding a feature vector `h ∈ ℝᴰ`.
The layer updates every node by looking at its anatomical neighbours (e.g. the
elbow looks at the shoulder and the wrist), weighted by how relevant each
neighbour is, and conditioned on what the image looks like at each location.

Concretely, for each node `i`:

**Step 1 — global conditioning (FiLM)**

Scale and shift `h_i` by a learned affine function of the global signal `g`.
In our case `g = 0`, so this degenerates to a learned bias-only rescaling of
the features — a cheap form of layer-level normalisation with trainable gain.

```
h_i ← (1 + scale_i) * h_i + shift_i        # scale,shift = Linear(g)
```

**Step 2 — message computation**

For each edge `(j → i)` (neighbour `j` sends to node `i`), build a message
that combines three things:
- the neighbour's value state `v_state_j = W_v · h_j`
- the edge feature `e_ij` (relative spatial displacement between j and i)
- the visual feature sampled at neighbour j's location `vis_j`

```
msg_ij = MLP(LayerNorm([v_state_j ‖ e_ij ‖ vis_j]))   # dim D
```

**Step 3 — attention weighting**

Compute a per-head attention score for each incoming edge using the
query-key dot product, biased by the edge geometry:

```
score_ij  = (W_q·h_i) · (W_k·h_j) / √(D/H)  +  edge_bias(e_ij)
alpha_ij  = softmax over all j→i edges       # normalised per node
```

Then weight each message by its attention score and sum:

```
agg_i = Σ_j  alpha_ij · msg_ij              # [D]
```

**Step 4 — node update (residual)**

Combine the original features, the aggregated neighbour information, and the
node's own visual evidence into a residual update:

```
h_i ← h_i + MLP(LayerNorm([h_i ‖ agg_i ‖ vis_i]))
```

After `num_refiner_layers` such rounds, every node has propagated information
along the skeleton (e.g. a wrist has seen the shoulder two hops away).  The
final head `LayerNorm → Linear(D → 2)` converts the updated features into a
`(dx, dy)` displacement, added to Stage 1's rough joint position.

The output linear is **zero-initialised**: at the start of training the GNN
predicts zero displacement, so the model begins from Stage 1's rough estimate
and learns the refinement incrementally.

### 4.5 Default Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `num_slots` N | 20 | Instance slots per image |
| `num_keypoints` K | 14 | CrowdPose joints |
| `dim` D | 256 | Transformer hidden dim |
| `num_heads` | 8 | Attention heads |
| `num_trunk_blocks` | 3 | Instance ↔ image blocks |
| `num_joint_blocks` | 2 | Joint ↔ instance blocks |
| `time_dim` | 256 | σ-embedding dim |
| `ffn_mult` | 4.0 | FFN expansion (→ 4D) |
| `image_token_dim` | 768 | RADIO output channels (auto-set) |
| `spatial_feat_dim` | 768 | Same (auto-set) |
| `dim_vis` | 128 | Visual feature dim for GNN |
| `dim_edge` | 64 | Edge feature dim for GNN |
| `num_refiner_layers` | 3 | PoseGraphLayer count |
| `dropout` | 0.1 | Applied to FFN, edge encoder, attention |

Total trainable parameters (decoder only, with `dim=256`): ~10 M.

---

## 5. Diffusion Schedule

The pipeline uses the **EDM (Karras et al. 2022)** framework with
x₀-prediction parameterisation.  The decoder directly predicts clean `x₀` at
every step.

### Training noise distribution

σ is sampled from a log-normal distribution per image in a batch:

```
log σ ~ N(σ_p_mean, σ_p_std²)
      = N(−1.2, 1.2²)
```

This places most training samples at moderate noise levels while still covering
the full range `[σ_min, σ_max] = [0.002, 10.0]`.

### Karras σ-schedule (inference)

Inference uses `num_inference_steps` steps on a `ρ`-power schedule:

```
σᵢ = (σ_max^(1/ρ) + i/(S−1) × (σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ,  i = 0…S−1
σ_S = 0
```

Default: `ρ = 7.0`, `σ_max = 10.0`, `σ_min = 0.002`, `S = 5`.

### Heun second-order ODE solver

At each step `i` the solver computes:

1. **Euler step**: predict `x₀` at `σᵢ`, form derivative
   `d = (x − x₀) / σᵢ`, step `x_euler = x + (σ_{i+1} − σᵢ) · d`.
2. **Corrector** (if `σ_{i+1} > 0`): predict `x₀` at `σ_{i+1}` on the Euler
   point, form `d₂`, update
   `x ← x + ½ · (σ_{i+1} − σᵢ) · (d + d₂)`.

Both boxes and joints are stepped jointly under the same σ schedule.

Single-step inference (`num_inference_steps=1`) skips the Heun corrector and
runs a single forward pass at `σ_max`, functioning as a direct regression
(useful for speed benchmarking).

---

## 6. Loss

`GraphPipelineLoss` in `src/training/losses.py` computes a **per-image
Hungarian matching** between the `N` predicted slots and the `P` ground-truth
persons, then sums weighted components.

### 6.1 Matching cost

The assignment cost matrix `C ∈ ℝ^{N × P}` for a single image is:

```
C[n, p] = λ_L1 · ‖pred_box_n − gt_box_p‖₁
         + λ_GIoU · (1 − GIoU(pred_box_n, gt_box_p))
         + λ_kpt  · mean_{k: labeled_p[k]=1} ‖pred_kpt_abs_n,k − gt_kpt_p,k‖₂
         + λ_obj  · (−log sigmoid(obj_logit_n))
```

Default weights: `λ_L1 = 5`, `λ_GIoU = 2`, `λ_kpt = 1`, `λ_obj = 0.1`.

Matching is solved with `torch_linear_assignment` (batched Hungarian).

### 6.2 Loss components

After matching `M` pairs `(slot_idx[m], gt_idx[m])`:

| Component | Formula | Weight |
|---|---|---|
| `box_l1` | `mean_m ‖pred_box_m − gt_box_m‖₁` | 5.0 |
| `box_giou` | `mean_m (1 − GIoU(pred_box_m, gt_box_m))` | 2.0 |
| `box_l2` | `mean_m ‖pred_box_m − gt_box_m‖²` | 1.0 |
| `kpt_l2` | `Σ_{m,k: labeled} ‖pred_kpt_abs_m,k − gt_kpt_m,k‖² / Σ labeled` | 5.0 |
| `obj_bce` | BCE on all N slots with dynamic positive weight | 1.0 |
| `ranking` | Pairwise RankNet loss on matched vs. unmatched objectness | 1.0 |

The keypoint loss is computed on **absolute** canonical coordinates
(`pred_kpt_abs = pred_box_ctr + pred_kpt_rel`), and only on annotated joints
(`gt_visibility > 0`).

The objectness BCE uses a dynamic positive weight
`min((N − M) / M, 20)` to compensate for the large number of background slots.

### 6.3 Total loss

```
L = 5·box_l1 + 2·box_giou + 1·box_l2 + 5·kpt_l2 + 1·obj_bce + 1·ranking
```

---

## 7. Training Module (`GraphDiffusionPipelineModule`)

### Step loop

```
σ ~ LogNormal(−1.2, 1.2²)                   # [B]
(noisy_boxes, noisy_joints) = _build_noisy_input(GT, σ, N, K)
tokens, centers, spatial = RADIO(image)      # frozen backbone
pred = decoder(noisy_boxes, noisy_joints, σ, tokens, centers, spatial)
loss = GraphPipelineLoss(pred, targets)
```

### Optimiser

AdamW with cosine annealing:
- `lr = 2e-4`, `weight_decay = 1e-4`
- Warmup: 1 epoch linear ramp
- Cosine decay to `lr_min = 2e-6` over `max_epochs = 100`
- Gradient clipping: `max_norm = 1.0`

### Validation / test AP

Every `val_ap_every_n_epochs = 10` epochs the module runs full EDM sampling
(`num_inference_steps = 5`) on the validation set.  Keypoints are converted
from canonical to pixel space via `canonical_to_pixel(kpt_canon, img_wh)` (where
`img_wh = img_hw[:, [1, 0]]`) and evaluated with `KeypointAPMetric`
(OKS-based, backed by MMPose's `CocoMetric`).  A 1-step AP (direct regression)
is logged in parallel for comparison.

Logged metrics: `ap-val/5-step/keypoints/AP`, `ap-val/1-step/keypoints/AP`.

### Augmentations

Applied per training step (on the GPU, post-collation):
- Gaussian pixel noise: `σ = 0.02`
- ColorJitter: brightness 0.2, contrast 0.2, saturation 0.1
- (No spatial augmentations; the diffusion noise serves as implicit input augmentation)

---

## 8. Skeleton Graph

The CrowdPose skeleton (`src/data/skeleton.py`) has 14 keypoints:

```
left_shoulder(0)  right_shoulder(1)
left_elbow(2)     right_elbow(3)
left_wrist(4)     right_wrist(5)
left_hip(6)       right_hip(7)
left_knee(8)      right_knee(9)
left_ankle(10)    right_ankle(11)
head(12)          neck(13)
```

`CANONICAL_EDGE_INDEX` contains the 14 undirected anatomical edges (28
directed) listed in `CROWDPOSE_EDGE_LIST`, for example:
`neck(13) → left_shoulder(0) → left_elbow(2) → left_wrist(4)`, etc.

In the GNN, each person's 14 nodes are connected according to this skeleton
independently, and `create_graph_from_keypoints` packs all `B × N` people into
a single block-diagonal graph of `B × N × K` nodes.

---

## 9. Training Script

```bash
# Standard training on CrowdPose (downloads RADIO backbone on first run)
python scripts/train_graph_pipeline.py

# Smoke run on CPU with the debug data subset
python scripts/train_graph_pipeline.py \
    data=crowdpose_set_prediction_debug \
    training_module.max_epochs=2 \
    training_module.val_ap_every_n_epochs=1 \
    hardware.accelerator=cpu hardware.devices=1 hardware.precision=32

# Resume
python scripts/train_graph_pipeline.py \
    training_module.resume=results/graph_pipeline/.../checkpoints/last.ckpt

# Evaluate a saved checkpoint (skip training)
python scripts/train_graph_pipeline.py skip_training=true \
    training_module.resume=results/graph_pipeline/.../checkpoints/last.ckpt
```

Outputs go to `results/graph_pipeline/{run_name}/{timestamp}/`. TensorBoard
and CSV logs are written alongside the checkpoint.

---

## 10. File Index

| File | Role |
|---|---|
| `src/decoders/diffusion/graph_pipeline.py` | Decoder (denoiser) + config |
| `src/training/graph_pipeline_module.py` | Lightning module + configs |
| `src/training/losses.py` | `GraphPipelineLoss`, `GraphPipelineLossWeights` |
| `src/decoders/denoise/graph_network.py` | `GraphDenoiseNetwork`, `PoseGraphLayer` |
| `src/data/graph.py` | `create_graph_from_keypoints` |
| `src/data/skeleton.py` | CrowdPose joint names + `CANONICAL_EDGE_INDEX` |
| `src/features/radio.py` | `RadioFeatureExtractor`, `CRadioV3BFeatureExtractorConfig` |
| `src/training/diffusion_modules.py` | `FourierFeatures`, `TimeFiLMTransformerBlock` |
| `conf/benchmark_graph_pipeline.yaml` | Root Hydra config |
| `conf/decoder/graph_pipeline.yaml` | Decoder hyperparams |
| `conf/training_module/graph_pipeline.yaml` | Optimizer / diffusion / loss weights |
| `scripts/train_graph_pipeline.py` | Training entry point |
| `tests/test_graph_pipeline.py` | 18 unit tests |
