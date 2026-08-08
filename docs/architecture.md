# Architecture

## Overview

This project provides image-conditioned 2D human pose estimation and object
detection on the [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) benchmark.
All pose models share the composition
`image → features → pose`. An optional `initial_keypoints` input supports
debugging and refiner-style experiments, but it is not required by one-stage
models. The repository distinguishes the following model families:

- **One-stage pose** — image-to-pose prediction without requiring cached
  detector keypoints. Set prediction is a one-stage pose implementation.
- **Refiner pose** — the legacy graph diffusion variant, normally initialized
  from cached detector keypoints. It remains useful for explicit refinement
  experiments and debugging.
- **Object detection** — a separate image-to-box model family with no keypoint
  output.

All configuration is managed by [Hydra](https://hydra.cc/). Every major component
is a pydantic `BaseConfig` subclass with a `build()` factory method, so you can
swap implementations by changing a config group on the command line.

---

## Directory layout

```
conf/               Hydra config groups
  pose_estimator/   model architecture presets
  training_module/  optimiser, schedule, augmentations, losses
  data/             dataset paths and loader settings
  hardware/         device selection

src/                core library
  pose_estimator.py shared image → features → pose model + config
  base_config.py    BaseConfig (pydantic + Hydra dispatch)
  features/         image feature extractors
  embedders/        feature embedders
  decoders/         pose and box decoders
  training/         Lightning module, training config, augmentations, losses
  data/             datasets, data modules, data configs, graph utilities
  metrics.py        AP metric wrapping MMPose CocoMetric

scripts/            CLI entry points
  train_graph_pipeline.py    one-stage graph diffusion pose training
  train_pose_refinement.py   legacy graph diffusion refiner training
  train_box_diffusion.py     standalone object detection training
  benchmark_mmpose.py        run MMPose baseline
  extract_predictions.py     write prediction cache files
  precompute_dino.py         pre-extract DINO features to disk

tests/              unit tests (pytest)
results/            outputs: checkpoints, prediction caches, benchmark CSVs
data/crowdpose/     CrowdPose images and annotation JSONs
checkpoints/        pretrained weights (DEKR, RTMPose)
external/           vendored mmdet + mmpose (editable installs)
```

---

## Config system

[`BaseConfig`](../src/base_config.py) is the common base for every config object.
It is a `pydantic.BaseModel` and exposes two class-level helpers:

- `from_hydra_config(cfg)` — converts a Hydra `DictConfig` to the concrete Python
  type. If a node carries a `_target_` key, `hydra.utils.instantiate` is called
  recursively so that nested configs are also resolved to the correct subclass.
- `build()` — constructs the live Python object (model, data module, etc.).

Hydra config groups in `conf/` map directly to the three top-level config objects:

| Hydra group        | Python type                | Default yaml            |
|--------------------|---------------------------|-------------------------|
| `pose_estimator`   | `PoseEstimatorConfig`      | `cnn_set_prediction.yaml` |
| `training_module`  | one-stage/refiner training config | `set_prediction.yaml` / `graph_diffusion_refiner.yaml` |
| `data`             | `CrowdPoseDataConfig`      | `crowdpose.yaml`        |
| `hardware`         | *(plain dict)*             | `auto.yaml`             |

Override a group: `python scripts/train_pose_refinement.py pose_estimator=dino_bilinear`

Override individual values: `training_module.lr=3e-4 training_module.max_epochs=100`

---

## Shared pose model: PoseEstimator

[`PoseEstimator`](../src/pose_estimator.py) is an `nn.Module` built from three
sequential stages:

```
image      ──►  FeatureExtractor  ──►  feature dict
                                          │
optional initial_keypoints ─────►  FeatureEmbedder  ──►  embeddings
                                                               │
                                                         Decoder  ──►  predicted keypoints
```

One-stage decoders can ignore `initial_keypoints`; refiner decoders use it as
the initial pose state. Both remain valid implementations of the shared
`PoseEstimator` interface.

### Stage 1 — Feature Extractor

Abstract base: [`BaseFeatureExtractor`](../src/features/base.py)

Takes images for online extraction or image IDs for cached extraction and returns
a feature dictionary. One-stage configurations use an online image backbone;
the legacy refiner commonly uses [`CachedDinoFeatureExtractor`](../src/features/cached_dino.py),
which loads precomputed DINOv3-ConvNeXt-small features from per-image `.pt` files
on disk (produced by `scripts/precompute_dino.py`). Each cache file contains:

| key        | shape              | description                         |
|------------|-------------------|-------------------------------------|
| `spatial`  | `(96, H/4, W/4)`  | stage-1 ConvNeXt features (fine)    |
| `semantic` | `(768, H/32, W/32)` | stage-4 features (coarse, rich)   |
| `global`   | `(768,)`           | global average-pooled vector        |

Spatial dimensions vary per image because resizing is aspect-ratio-preserving,
so `spatial` and `semantic` are emitted as `list[Tensor]` (one per image in the
batch).

Config: [`CachedDinoFeatureExtractorConfig`](../src/features/cached_dino.py) —
`model_name` + `cache_paths` (list of directories to search).

### Stage 2 — Feature Embedder

Abstract base: [`BaseFeatureEmbedder`](../src/embedders/base.py)

Takes the feature dict and pixel-space keypoint coordinates `(B, N, K, 2)`, and
returns a dict of per-node tensors consumed by the decoder.

The production implementation is [`BilinearSampleEmbedder`](../src/embedders/bilinear.py):

1. For each image in the batch, bilinearly sample the `spatial` map (96-d) and the
   `semantic` map (768-d) at each of the `N×K` keypoint pixel locations.
2. Concatenate the two sampled vectors per node → `(96 + 768 = 864)` dims.
3. Project to `out_dim` (default 128) with a single linear layer.
4. Also forward the `global` vector (768-d) unchanged, keyed as `"global_feats"`.

Output dict keys: `"node_feats"` `(B, N, K, out_dim)`, optionally `"global_feats"`.

Config: [`BilinearSampleEmbedderConfig`](../src/embedders/bilinear.py) —
`spatial_dim`, `hidden_dim`, `out_dim`.

### Stage 3 — Decoder

Abstract base: [`BaseDecoder`](../src/decoders/base.py)

Takes per-node embeddings and an optional initial keypoint state, and returns
predicted keypoints in *canonical space* `(V, 2)` where `V = B × N × K`.

The legacy refiner implementation is [`GraphDiffusionRefinerDecoder`](../src/decoders/diffusion/graph.py).
It implements EDM-style continuous-time diffusion over the pose graph:

**During training** — a random noise level `σ` is sampled; the input keypoints
are corrupted with `N(0, σ²I)` noise; the network predicts the clean keypoints.

**During inference** — the model starts from the detector's noisy keypoints plus
a small amount of added noise, then runs a fixed number of denoising steps
(default 5).

The decoder builds per-node input features as:

```
h = LN( pos_emb(noisy keypoints)  +  idx_emb(joint_type_index) )
e = edge_encoder(noisy[src] − noisy[dst])          # edge features
v = embeddings["node_feats"]                        # visual conditioning
g = MLP(timestep_emb [+ embeddings["global_feats"]])
```

These are passed to a [`GraphDenoiseNetwork`](../src/decoders/denoise/) — a stack
of graph-transformer layers that attends over `(h, edge_index, e, v, g)` and
outputs a predicted delta `(V, 2)`.

Config: [`GraphDiffusionDecoderConfig`](../src/decoders/diffusion/graph.py) with a
nested `denoise_network: GraphDenoiseNetworkConfig`.

---

## Data pipeline

### Coordinate spaces

The model uses two coordinate systems:

- **Pixel space** — absolute `(x, y)` in pixels, shape `(B, N, K, 2)`.
- **Canonical space** — image-normalised, `canonical = pixel / [W, H] × 2 − 1`,
  so in-frame joints lie in `(−1, 1)`. The decoder inputs and outputs are in
  canonical space.

Conversion utilities live in [`src/metrics.py`](../src/metrics.py)
(`canonical_to_pixel`) and [`src/data/`](../src/data/).

### Graph representation

[`src/data/graph.py`](../src/data/graph.py) packs a batch of B images, each with
N people and K keypoints, into a single block-diagonal PyG graph.
The global node index for person `n`, keypoint `k` in image `b` is
`b×N×K + n×K + k`. Each batch provides:

- `edge_index` — block-diagonal adjacency from the CrowdPose skeleton
- `batch_idx`, `pose_idx`, `kpt_type_idx` — index tensors for scatter/gather
- `node_pos` — flattened keypoint coordinates `(B×N×K, 2)`

### Datasets and data modules

[`PoseDataset`](../src/data/datasets.py) reads a COCO-style annotation JSON and
groups annotations by image ID. One dataset item = one image, with N people and
K keypoints. It uses [`SceneGraphSampler`](../src/data/sampler.py) to produce
uniform-N batches.

[`PredictionDataset`](../src/data/datasets.py) is similar but wraps detector
prediction cache files (produced by `scripts/extract_predictions.py`) for
inference-time use.

[`CrowdPoseDataModule`](../src/data/data_module.py) is a `pl.LightningDataModule`
that owns three `PoseDataset` instances (train/val/test) and three `DataLoader`
instances. It also stores resolved annotation and prediction-cache paths, which
the training module reads to set up AP evaluation.

**`subset_frac`** (on `DataConfig`, default `1.0`) truncates each split to a
fraction of its samples — useful for quick smoke runs. E.g.
`data=crowdpose_debug` sets `subset_frac=0.001` (≈12 train images).

### DataConfig hierarchy

```
DataConfig (ABC)  ──  build() → PoseDataModule
  └─ CrowdPoseDataConfig  ──  build() → CrowdPoseDataModule
```

Key `CrowdPoseDataConfig` fields: `train_ann`, `val_ann`, `test_ann`, `img_dir`,
`val_pred_cache`, `test_pred_cache`, `batch_size`, `num_workers`, `subset_frac`.

---

## Training

### TrainingModuleConfig

[`TrainingModuleConfig`](../src/training/training_config.py) is the abstract
polymorphic base. The concrete subclass is
[`DiffusionTrainingModuleConfig`](../src/training/training_config.py), which owns:

| Category         | Fields                                                      |
|------------------|-------------------------------------------------------------|
| Optimiser        | `lr`, `weight_decay`, `cosine_lr`, `lr_warmup_epochs`, `lr_min_frac` |
| Inference / AP   | `num_inference_steps`, `ap_eval_interval`, `ap_step_counts`, `start_inference_noise_frac` |
| Trainer          | `max_epochs`, `seed`, `grad_clip`, `early_stop_patience`, `log_every_n_steps`, `compile`, `resume` |
| Augmentations    | nested `AugmentationConfig`                                 |
| Losses           | nested `LossConfig`                                         |

`build(pose_estimator, data)` returns a
[`PoseEstimatorTrainingModule`](../src/training/training_module.py). It reads
annotation paths and prediction-cache paths from the data module so the training
module never has to know the on-disk layout.

### PoseEstimatorTrainingModule

A `pl.LightningModule` wrapping `PoseEstimator`. Responsibilities:

- **Training step**: corrupt keypoints with random noise, run forward pass,
  compute joint loss + bone loss, return loss.
- **Validation step**: run inference (multi-step denoising), accumulate
  predictions, optionally compute AP via `KeypointAPMetric`.
- **Augmentations** (applied per batch):
  - `keypoint_jitter` — Gaussian noise on keypoint coordinates
  - `joint_dropout` — zero individual joints at probability `p_joint_drop`
  - `joint_randomize` — replace joints with random coordinates
  - `pose_dropout` — drop entire persons
  - `image_dropout` — zero visual conditioning for a fraction of nodes
- **Optimiser**: AdamW with optional cosine-annealing + linear warmup.

### Loss functions

Both losses are defined in [`src/training/losses.py`](../src/training/losses.py)
and weighted by keypoint visibility (0=absent, 1=occluded, 2=visible):

- **Joint loss** — visibility-weighted MSE between predicted and GT keypoints in
  pixel space.
- **Bone loss** — MSE on bone lengths (per skeleton edge), weighted by the
  product of the two endpoint visibilities.

### Evaluation metric

[`KeypointAPMetric`](../src/metrics.py) wraps MMPose's `CocoMetric`. It converts
from canonical space → pixel space, formats predictions in COCO format, and
returns standard AP / AP50 / AP75 / APM / APL numbers.

---

## Config presets and common run patterns

| Use case                     | Command                                                            |
|------------------------------|--------------------------------------------------------------------|
| One-stage graph pose smoke   | `python scripts/train_graph_pipeline.py data=crowdpose_set_prediction_debug training_module.max_epochs=2 hardware.accelerator=cpu hardware.devices=1 hardware.precision=32` |
| One-stage set prediction     | `pose_estimator=cnn_set_prediction training_module=set_prediction data=crowdpose_set_prediction` |
| Legacy refiner               | `pose_estimator=dino_diffusion_legacy training_module=graph_diffusion_refiner data=crowdpose` |
| Object detection             | `python scripts/train_box_diffusion.py`                            |
| Resume from checkpoint       | `training_module.resume=results/.../last.ckpt`                     |

Outputs land in `results/pose_estimator_training/<run_name>/<timestamp>/`, including
TensorBoard logs, CSV logs, and Lightning checkpoints (`last.ckpt`, `best.ckpt`).

---

## Entry points

| Script                             | Purpose                                                        |
|------------------------------------|----------------------------------------------------------------|
| `scripts/train_graph_pipeline.py` | One-stage graph diffusion pose training                      |
| `scripts/train_pose_refinement.py` | Legacy graph diffusion refiner training                     |
| `scripts/benchmark_mmpose.py`      | Run DEKR / RTMPose baselines via MMPose, write result CSVs     |
| `scripts/extract_predictions.py`   | Write per-image prediction caches for use as decoder input     |
| `scripts/precompute_dino.py`       | Pre-extract DINOv3-ConvNeXt features to `results/cache/`       |
| `scripts/benchmark_diffusion_refiner.py` | Benchmark the trained refiner end-to-end                 |
| `scripts/prepare_crowdpose.py`     | Download and prepare CrowdPose annotations                     |
| `scripts/train_box_diffusion.py`   | Train the set-prediction box-diffusion model (see below)       |

---

## Box-diffusion pipeline (object detection)

A parallel pipeline performs **set-prediction object detection** (boxes only,
no keypoints) using a DETR-style query set denoised by EDM continuous-time
diffusion. The pipeline shares the backbone + data module with the pose
pipeline but uses its own decoder, Lightning module, loss, and entry script.

| Stage                 | Class                                                     | File |
|-----------------------|-----------------------------------------------------------|------|
| Data (set format)     | `SetPoseDataset` + `set_pose_collate`                     | [src/data/datasets.py](../src/data/datasets.py) |
| Padded-canonical box transform | `_boxes_to_padded_canonical_batch` (in `on_after_batch_transfer`) | [src/data/data_module.py](../src/data/data_module.py) |
| Backbone (image tokens)| `BaseFeatureExtractor` (e.g. `CNNFeatureExtractor`)      | [src/features/](../src/features) |
| Diffusion decoder     | `BoxSetDiffusionDecoder`                                  | [src/decoders/diffusion/box_set.py](../src/decoders/diffusion/box_set.py) |
| Loss + Hungarian match| `SetBoxDiffusionLoss` (L1 + GIoU + diff-L2 + BCE-objectness) | [src/training/losses.py](../src/training/losses.py) |
| Lightning module      | `BoxSetDiffusionModule` (+ `Config`)                      | [src/training/box_diffusion_module.py](../src/training/box_diffusion_module.py) |
| AP metric             | `BoxAPMetric` (COCO bbox AP via `pycocotools`)            | [src/metrics.py](../src/metrics.py) |

Box parameterisation: `(cx, cy, w, h)` in padded-canvas canonical units,
`cx, cy ∈ [-1, 1]`. Targets predicted as `x₀`. Validation `val/bbox/AP` is
the `ModelCheckpoint` monitor; multi-step (Heun-style EDM) and single-step
inference are both reported.

Hydra root config: [conf/benchmark_box_diffusion.yaml](../conf/benchmark_box_diffusion.yaml)
(defaults: `backbone=cnn`, `decoder=box_set`, `training_module=box_diffusion`,
`data=crowdpose_set_prediction`). Run via:

```
python scripts/train_box_diffusion.py
```

Smoke run on a tiny CPU subset (used to validate the wiring):

```
python scripts/train_box_diffusion.py \
    data=crowdpose_set_prediction +data.subset_frac=0.001 \
    data.batch_size=2 data.num_workers=0 data.ap_batch_size=2 \
    training_module.max_epochs=2 training_module.val_ap_every_n_epochs=1 \
    hardware.accelerator=cpu hardware.devices=1 hardware.precision=32
```

Note: `data=crowdpose_debug` cannot be used directly because it inherits
`target_format=refinement` (no images in the batch). Use
`data=crowdpose_set_prediction` with `+data.subset_frac=...` overrides.


