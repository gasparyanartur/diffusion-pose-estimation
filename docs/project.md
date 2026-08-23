# Project assumptions and standards

This file records cross-cutting conventions so individual modules don't have to
re-document them. New modules should comply; deviations must be documented per-class.

## Coordinate spaces

The project uses three coordinate spaces. Conversions are explicit at the
boundary between them.

### 1. Pixel space (raw input)

- Origin at top-left of the image.
- Units: pixels.
- Range: `(0, 0)` to `(W, H)` (any positive float; out-of-frame allowed).
- Used at I/O boundaries: dataset loaders, AP-metric submission, visualization.
- Tensor shape for keypoints: typically `[B, N, K, 2]` with last-dim `(x, y)`.

### 2. Canonical space (model-internal, the standard)

- **All pose pipelines return keypoints in canonical space.**
- Definition: `canonical = pixel / [W, H] * 2 - 1`.
- Range: `(-1, 1)` for in-frame joints (matches PyTorch `grid_sample`).
- Inverse: `pixel = (canonical + 1) / 2 * [W, H]`.
- Helper: `src.eval.metrics.canonical_to_pixel(canonical, img_wh)`.
- Builder: `src.data.graph.prepare_pose_graph_batch` produces `node_pos` here.

### 3. Resized-image normalized `[0, 1]` (do not use across stages)

- Used inside per-image data preprocessing only (e.g. resizing for the CNN
  feature extractor in the benchmark notebook). Convert to canonical $(-1, 1)$
  at the pipeline boundary.
- Rationale: keeping $[0, 1]$ in some stages and $(-1, 1)$ in others creates
  silent off-by-half errors during loss / AP eval.

## Stage I/O contracts

### Feature extractors (`src/models/features/`)

```
forward(image=None, image_ids=None, **kwargs) -> dict[str, Any]
```

- `image`: optional pixel tensor `[B, 3, H, W]` for online extractors.
- `image_ids`: optional list of int for cached extractors.
- Concrete subclasses validate which one they require (raise on missing input).
- Feature extractors are consumed directly by complete pipelines; the former
  generic embedder and decoder stages are removed.

### Model composition (`src/models/`)

- `src/models/features/` contains image-conditioned feature extractors and
  their `BaseFeatureExtractorConfig` classes.
- `src/models/common/` contains shared attention and positional utilities.
- Graph message-passing and denoising layers that are specific to the graph
  pose model live privately in `src/models/pipelines/pose_estimators/graph.py`.
- `src/models/pipelines/` contains complete model-family Lightning modules.
  Each pipeline owns its heads, losses, training lifecycle, sampling, metrics,
  and optimizer configuration.

## Visibility encoding

Keypoint visibility is a single integer-valued tensor in `{0, 1, 2}`:

- `0` — absent / not labeled.
- `1` — labeled but occluded.
- `2` — visible.

Labeled-mask: `labeled = visibility > 0`.
Visible-mask: `visible = visibility == 2`.

## Configuration

- All swappable components inherit from `src.base_config.BaseConfig` (pydantic).
- Hydra YAML uses `_target_` to point at the config class.
- User code calls `cfg.build()` to get the network instance.
- Polymorphic dispatch: `BaseConfig.from_hydra_config` accepts a root `_target_`
  that resolves to a subclass instance.

## CrowdPose

- 14 keypoints; joint sigmas live in `src.eval.metrics.CROWDPOSE_SIGMAS`.
- Edge list in `src.data.skeleton.CROWDPOSE_EDGE_LIST`.
- General pose pipelines use `src.eval.metrics.KeypointAPMetric` with
  `iou_type="keypoints_crowd"`. DiffusionRegPose uses its
  `DiffusionRegPoseAPMetric`, which matches the external repository's
  `crowdposetools.COCOeval(iou_type="keypoints")` contract. Standard reported
  metrics are `AP`, `AP .5`, `AP .75`, `AP (E/M/H)`, and `AR (E/M/H)`.

## Model families

- `BaselinePosePipeline` is the direct image-to-fixed-pose-slot model. It owns
  its CNN feature extractor, spatial-attention pooling, Hungarian matching loss, AP
  lifecycle, and optimizer setup.
- `GraphPosePipeline` is the direct image-to-box-and-pose diffusion set model.
- `DiffusionRegPosePipeline` is the native checkpoint-compatible set-prediction
  pose model. Its training criterion reproduces the reference Hungarian,
  focal-classification, box, keypoint, OKS, and denoising losses; its model-only
  evaluation path loads compatible tensors from the downloaded reference
  checkpoint. Exact
  CUDA training and evaluation use the repository-owned native
  forward/backward deformable-attention kernel; the PyTorch sampler remains
  the differentiable fallback when the extension is unavailable. CrowdPose
  person targets use category label `1`, matching the dataset and reference
  implementation; native checkpoints trained before this correction require
  retraining.
- The native training wrapper has passed one-image CPU and CUDA smoke runs,
  including checkpoint restore. The external-compatible preset freezes the
  ResNet stem and `layer1`, shares the decoder pose-size head, and applies the
  reference multi-scale/crop training transforms with variable-size padding.
  It uses the `crowdpose_trainval.json` split, micro-batch size 1 with eight
  accumulated batches, FP32, the EDPose checkpoint initialization, and the
  external two-group AdamW policy. The paper-compatible 80-epoch preset uses
  AdamW at `2e-4` (`1e-4` for the backbone), gradient clipping `0.1`,
  keypoint-loss weight `10`, and reference milestones `[31, 61]`. Validation
  loss runs after every epoch and validation-set CrowdPose AP is calculated
  every 10 epochs; `last.ckpt` remains available after every epoch. The
  published DiffusionRegPose checkpoint is used only as an evaluation/parity
  target, not as the training initialization.
- `benchmark_diffusionregpose_debug` is a deliberately reduced end-to-end
  diagnostic preset. It is useful for detecting target/layout/checkpoint
  regressions quickly, but its reduced decoder is not a checkpoint-compatible
  submodel and its AP is not a learning/parity gate. For a short AP feedback
  loop, use `benchmark_diffusionregpose_fast`: it retains the full model
  structure and reduces only the data/resolution; a lower fine-tuning LR is
  appropriate for its tiny subset.
- `benchmark_diffusionregpose_lowres` retains the full model and full
  CrowdPose train/validation splits while resizing to 128 pixels with a
  192-pixel maximum dimension. It uses batch size 8 with no gradient
  accumulation, matching the paper recipe's effective batch and 1,500
  optimizer updates per epoch. Its 80 epochs and milestones `[31, 61]` thus
  preserve the reference optimization schedule. It remains a learning gate,
  not a paper benchmark, because of its lower image resolution. Evaluation
  resizing is configurable on `CrowdPoseDataConfig`; the paper-compatible
  defaults remain 800 and 1333.
- Cached-DINO graph-diffusion refinement is unsupported and has no
  configuration or compatibility path.
- Box diffusion and DiffusionDet are separate object-detection models whose
  outputs are boxes and objectness scores, not poses. Box diffusion is the
  tensor-native `BoxDiffusionPipeline`; it accepts image tensors and does not
  implement the path-based `BoxDetector` contract retained for DiffusionDet.

The root default is the cache-free fixed-slot baseline configuration. It accepts
image tensors and returns canonical keypoints plus slot and keypoint confidence
scores.
