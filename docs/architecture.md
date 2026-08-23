# Architecture

## Overview

This project provides image-conditioned 2D human pose estimation and object
detection on the [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) benchmark.
The canonical pose pipelines accept image tensors directly and own their full
Lightning lifecycle. The repository distinguishes the following model families:

- **Fixed-slot baseline pose** — image-to-pose prediction using Hungarian
  matching over a fixed set of slots.
- **One-stage graph pose** — image-to-box-and-pose diffusion prediction.
- **Refiner pose** — unconfigured legacy code without a supported compatibility
  path.
- **Object detection** — a separate image-to-box model family with no keypoint
  output.

All configuration is managed by [Hydra](https://hydra.cc/). Every major component
is a pydantic `BaseConfig` subclass with a `build()` factory method, so you can
swap implementations by changing a config group on the command line.

DiffusionRegPose is the deliberate exception to the repository-wide canonical
box convention: its model-internal boxes are sigmoid-normalized `(cx, cy, w, h)`
values in `[0, 1]`, matching the downloaded checkpoint and external criterion.
Keypoints still cross the data/model boundary in canonical `[-1, 1]` space.

DiffusionRegPose is the deliberate exception to the repository-wide canonical
box convention: its model-internal boxes are sigmoid-normalized `(cx, cy, w, h)`
values in `[0, 1]`, matching the downloaded checkpoint and external criterion.
Keypoints still cross the data/model boundary in canonical `[-1, 1]` space.

---

## Directory layout

```
conf/               Hydra config groups
  pipeline/         complete model-family pipeline presets
  data/             dataset paths and loader settings
  hardware/         device selection

src/                core library
  configs/          BaseConfig (pydantic + Hydra dispatch)
  models/features/   image-conditioned feature extractors
  models/common/     shared attention and positional utilities
  models/pipelines/  complete Lightning pipelines and private model-family heads
  data/             datasets, data modules, data configs, graph utilities
  eval/             AP metrics, diagnostics, and visualization helpers

scripts/            categorized CLI entry points
  data/             dataset preparation and cache generation
  training/         model training and sweep orchestration
  evaluation/       benchmarks and prediction-cache evaluation
  investigation/    probes and exploratory inspection
  utils/            environment and command utilities
  legacy/           retained examples for historical reference

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

The canonical baseline root config maps directly to a complete pipeline and data
module:

| Hydra group        | Python type                | Default yaml            |
|--------------------|---------------------------|-------------------------|
| `pipeline`         | `BaselinePosePipelineConfig` | `baseline_pose.yaml` |
| `data`             | `CrowdPoseDataConfig`      | `crowdpose.yaml`        |
| `hardware`         | *(plain dict)*             | `auto.yaml`             |
| `training`         | `TrainingConfig`           | `default.yaml`          |

Override a group: `python scripts/training/train_pose_refinement.py pipeline=baseline_pose`

Override individual values: `pipeline.lr=3e-4 training.max_epochs=100`

Runtime training policy is composed independently through the `training`
group, for example `training.max_epochs=2 training.test_after_fit=false`.
The typed runtime schema lives in [`src/training/config.py`](../src/training/config.py);
Hydra YAML remains under [`conf/training/`](../conf/training/).

---

## Fixed-slot baseline pose pipeline

[`BaselinePosePipeline`](../src/models/pipelines/pose_estimators/baseline.py)
is the canonical one-stage baseline `LightningModule`:

```
image ──► CNN feature extractor ──► spatial-attention pooling ──► fixed-slot heads
```

`forward(images)` returns canonical `keypoints` with raw and sigmoid slot and
visibility scores. The same pipeline owns matching loss, training/validation/
test steps, test AP accumulation, and optimizer configuration. The generic
The former extractor → embedder → decoder composition is not supported.

## Data pipeline

### Coordinate spaces

The model uses two coordinate systems:

- **Pixel space** — absolute `(x, y)` in pixels, shape `(B, N, K, 2)`.
- **Canonical space** — image-normalised, `canonical = pixel / [W, H] × 2 − 1`,
  so in-frame joints lie in `(−1, 1)`. Pose-pipeline inputs and outputs are in
  canonical space.

Conversion utilities live in [`src/eval/metrics.py`](../src/eval/metrics.py)
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
prediction cache files (produced by `scripts/evaluation/extract_predictions.py`) for
inference-time use.

[`CrowdPoseDataModule`](../src/data/data_module.py) is a `pl.LightningDataModule`
that owns three `PoseDataset` instances (train/val/test) and three `DataLoader`
instances. It also stores resolved annotation and prediction-cache paths, which
the active pipeline reads to set up AP evaluation.

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

Each canonical pipeline is a `pl.LightningModule` and owns its forward, loss,
metric, and optimizer behavior. The shared `TrainingRunner` owns the
cross-pipeline execution lifecycle. Shared losses and metrics are available
from [`src/training/losses.py`](../src/training/losses.py) and
[`src/eval/metrics.py`](../src/eval/metrics.py).

The cross-pipeline execution lifecycle is provided by
[`TrainingRunner`](../src/training/runner.py). It consumes typed
`TrainingConfig` values and remains independent of Hydra. Pipeline-specific
scripts provide the model/data objects and can supply specialized callbacks or
checkpoint I/O policies.

`DiffusionRegPosePipeline` wraps the checkpoint-compatible native model in the
same Lightning boundary. Its target builder creates the reference xyxy-plus-
visibility target layout and timestep-zero cosine diffusion inputs, while its
criterion owns Hungarian matching and final/auxiliary decoder losses. The
CrowdPose person target uses category label `1`, matching the annotation
format and the reference implementation; checkpoints trained with the old
label-`0` target contract must be retrained.
training entry point is
[`scripts/training/train_diffusionregpose.py`](../scripts/training/train_diffusionregpose.py)
with the `pipeline=diffusionregpose` and
`data=crowdpose_diffusionregpose` presets.

The DiffusionRegPose data preset follows the external training data and
target semantics: it samples the reference resize/crop policy, normalizes
images with ImageNet statistics, updates pixel targets before converting
keypoints to per-image canonical coordinates and boxes to normalized sigmoid
coordinates, and pads transformed images only to the largest height/width in
each batch. The default model freezes the ResNet stem
and `layer1` and shares the pose height/width prediction head across decoder
layers, matching the reference trainable parameter set of 49,304,924.
The training preset initializes from the EDPose CrowdPose checkpoint and uses
`batch_size=1` and
`training.accumulate_grad_batches=8`, giving an effective batch size of 8.
Its paper-compatible schedule runs for 80 epochs with AdamW at `2e-4`
(`1e-4` for the backbone), clips gradients at `0.1`, uses keypoint-loss weight
`10`, applies reference milestones `[31, 61]`, and includes 100 denoising
queries with their classification/box losses. Validation loss runs after each
epoch and validation AP runs every 10 epochs; `last.ckpt` is saved after every
training epoch and separate test/AP evaluation runs after fitting. The
published DiffusionRegPose checkpoint is an evaluation/parity target, not the
training initialization.

The training entry point has passed one-image CPU and CUDA Lightning smoke
runs, including checkpoint creation and restore. The reduced smoke
configuration validates wiring only; full paper-sized training remains a
separate experiment.

For wiring-only smoke tests, `benchmark_diffusionregpose_debug` reduces the
transformer/query dimensions, image sizes, and deterministic dataset subset.
Because those structural reductions change decoder semantics, its checkpoints
must not be used as submodels of the downloaded checkpoint or as AP gates. For
short AP feedback cycles, `benchmark_diffusionregpose_fast` preserves the full
checkpoint-compatible structure and reduces only the data/resolution. Its
checkpoint must be evaluated with the resolved Hydra config via `--config`;
subset metrics filter both predictions and ground truth and remain diagnostic,
not paper-scale benchmark results.

For a full-data learning gate, `benchmark_diffusionregpose_lowres` preserves the
full structure and all train/validation images while using 128/192-pixel
training and validation resizing and batch size 8 with no accumulation on the
current RTX 3090. It uses 80 epochs and milestones `[31, 61]`, matching the
paper recipe's effective batch and 1,500 optimizer updates per epoch. It must
not be confused with the paper benchmark because its image resolution is lower.
`CrowdPoseDataConfig` exposes `eval_resize_scale` and `eval_max_size`; their
defaults are 800/1333 so the paper-compatible preset remains unchanged. The
low-resolution run evaluates AP every 10 epochs.

Its evaluation path is independent of the external repository. CUDA training
and inference use a lazily compiled, repository-owned forward/backward
deformable-attention kernel whose scalar bilinear accumulation matches the
checkpoint operator. CPU runs and environments where the extension cannot be
built retain the differentiable PyTorch sampler. During training, validation
loss is calculated every epoch and validation AP is calculated every tenth
epoch through `DiffusionRegPoseAPMetric`. That metric accumulates the same
CrowdPose result records consumed by the standalone evaluator and calls
`crowdposetools.COCOeval` with `iou_type="keypoints"`, matching the external
repository. The generic `KeypointAPMetric` remains available to other pose
pipelines.

---

## Config presets and common run patterns

| Use case                     | Command                                                            |
|------------------------------|--------------------------------------------------------------------|
| One-stage graph pose smoke   | `python scripts/training/train_graph_pipeline.py data=crowdpose_set_prediction_debug training.max_epochs=2 hardware.accelerator=cpu hardware.devices=1 hardware.precision=32` |
| DiffusionRegPose training    | `python scripts/training/train_diffusionregpose.py` |
| Fixed-slot baseline          | `python scripts/training/train_pose_refinement.py pipeline=baseline_pose data=crowdpose_set_prediction` |
| Object detection             | `python scripts/training/train_box_diffusion.py`                            |
| Resume from checkpoint       | `training.resume=results/.../last.ckpt`                            |

Outputs land in `results/pose_estimator_training/<run_name>/<timestamp>/`, including
TensorBoard logs, CSV logs, and Lightning checkpoints (`last.ckpt`, `best.ckpt`).

---

## Entry points

| Script                             | Purpose                                                        |
|------------------------------------|----------------------------------------------------------------|
| `scripts/training/train_graph_pipeline.py` | One-stage graph diffusion pose training |
| `scripts/training/train_diffusionregpose.py` | Native DiffusionRegPose training |
| `scripts/training/train_pose_refinement.py` | Fixed-slot baseline pose training |
| `scripts/evaluation/benchmark_mmpose.py` | Run DEKR / RTMPose baselines via MMPose |
| `scripts/evaluation/extract_predictions.py` | Write prediction caches for evaluation |
| `scripts/evaluation/evaluate_diffusionregpose.py` | Evaluate a native DiffusionRegPose checkpoint on CrowdPose |
| `scripts/data/precompute_dino.py` | Pre-extract DINOv3-ConvNeXt features |
| `scripts/data/prepare_crowdpose.py` | Download and prepare CrowdPose annotations |
| `scripts/training/train_box_diffusion.py` | Train the set-prediction box-diffusion model |

## Graph-pose pipeline

`GraphPosePipeline` is the canonical one-stage pose-estimation Lightning
module. It owns the image feature extractor, attention-and-graph denoiser, EDM noisy
target preparation and sampling, Hungarian graph-pose loss, augmentations, AP
metrics, and optimizer lifecycle. Its tensor-only `forward(images,
noisy_boxes, noisy_joints_rel, sigma)` and `sample(images, ...)` methods return
canonical boxes, relative joints, and objectness logits.

The Hydra root [conf/benchmark_graph_pipeline.yaml](../conf/benchmark_graph_pipeline.yaml)
selects one `pipeline=graph_pose` config. The config at
[conf/pipeline/graph_pose.yaml](../conf/pipeline/graph_pose.yaml) contains the
nested feature extractor and graph-pose loss weights; use `pipeline.*` overrides for
training and inference settings. The legacy cached-keypoint graph refiner is a
separate model family and is not part of this pipeline.

---

## Box-diffusion pipeline (object detection)

A tensor-native pipeline performs **set-prediction object detection** (boxes
only, no keypoints) using a DETR-style query set denoised by EDM continuous-time
diffusion. `BoxDiffusionPipeline` owns its feature extractor, denoiser, Lightning
lifecycle, loss, augmentation, AP metrics, and optimizer configuration.

| Stage                 | Class                                                     | File |
|-----------------------|-----------------------------------------------------------|------|
| Data (set format)     | `SetPoseDataset` + `set_pose_collate`                     | [src/data/datasets.py](../src/data/datasets.py) |
| Padded-canonical box transform | `_boxes_to_padded_canonical_batch` (in `on_after_batch_transfer`) | [src/data/data_module.py](../src/data/data_module.py) |
| Complete model        | `BoxDiffusionPipeline` + `BoxDiffusionPipelineConfig`     | [src/models/pipelines/object_detectors/box_diffusion.py](../src/models/pipelines/object_detectors/box_diffusion.py) |
| Feature extractor (image tokens)| nested `BaseFeatureExtractorConfig` (e.g. CNN)            | [src/models/features/](../src/models/features/) |
| Loss + Hungarian match| `SetBoxDiffusionLoss` (L1 + GIoU + diff-L2 + BCE-objectness) | [src/training/losses.py](../src/training/losses.py) |
| AP metric             | `BoxAPMetric` (COCO bbox AP via `pycocotools`)            | [src/eval/metrics.py](../src/eval/metrics.py) |

Box parameterisation: `(cx, cy, w, h)` in padded-canvas canonical units,
`cx, cy ∈ [-1, 1]`. Targets predicted as `x₀`. Validation `val/bbox/AP` is
the `ModelCheckpoint` monitor; multi-step (Heun-style EDM) and single-step
inference are both reported.

Hydra root config: [conf/benchmark_box_diffusion.yaml](../conf/benchmark_box_diffusion.yaml)
(defaults: `pipeline=box_diffusion`, `data=crowdpose_set_prediction`). The
pipeline config builds the full model directly. Run via:

```
python scripts/training/train_box_diffusion.py
```

Smoke run on a tiny CPU subset (used to validate the wiring):

```
python scripts/training/train_box_diffusion.py \
    data=crowdpose_set_prediction +data.subset_frac=0.001 \
    data.batch_size=2 data.num_workers=0 data.ap_batch_size=2 \
    training.max_epochs=2 pipeline.val_ap_every_n_epochs=1 \
    hardware.accelerator=cpu hardware.devices=1 hardware.precision=32
```

Note: `data=crowdpose_debug` cannot be used directly because it inherits
`target_format=refinement` (no images in the batch). Use
`data=crowdpose_set_prediction` with `+data.subset_frac=...` overrides.
