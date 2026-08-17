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
- AP via `src.eval.metrics.KeypointAPMetric` with `iou_type="keypoints_crowd"`.
  Standard reported metrics: `AP`, `AP .5`, `AP .75`, `AP (E/M/H)`, `AR (E/M/H)`.

## Model families

- `BaselinePosePipeline` is the direct image-to-fixed-pose-slot model. It owns
  its CNN feature extractor, spatial-attention pooling, Hungarian matching loss, AP
  lifecycle, and optimizer setup.
- `GraphPosePipeline` is the direct image-to-box-and-pose diffusion set model.
- `DiffusionRegPosePipeline` is the native checkpoint-compatible set-prediction
  pose model. Its training criterion reproduces the reference Hungarian,
  focal-classification, box, keypoint, and OKS losses; its model-only
  evaluation path loads the downloaded reference checkpoint strictly. Exact
  CUDA evaluation uses the repository-owned native deformable-attention
  kernel; the PyTorch sampler remains the differentiable fallback.
- Cached-DINO graph-diffusion refinement is unsupported and has no
  configuration or compatibility path.
- Box diffusion and DiffusionDet are separate object-detection models whose
  outputs are boxes and objectness scores, not poses. Box diffusion is the
  tensor-native `BoxDiffusionPipeline`; it accepts image tensors and does not
  implement the path-based `BoxDetector` contract retained for DiffusionDet.

The root default is the cache-free fixed-slot baseline configuration. It accepts
image tensors and returns canonical keypoints plus slot and keypoint confidence
scores.
