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

- **All decoders return keypoints in canonical space.**
- Definition: `canonical = pixel / [W, H] * 2 - 1`.
- Range: `(-1, 1)` for in-frame joints (matches PyTorch `grid_sample`).
- Inverse: `pixel = (canonical + 1) / 2 * [W, H]`.
- Helper: `src.metrics.canonical_to_pixel(canonical, img_wh)`.
- Builder: `src.data.graph.prepare_pose_graph_batch` produces `node_pos` here.

### 3. Resized-image normalized `[0, 1]` (do not use across stages)

- Used inside per-image data preprocessing only (e.g. resizing for the CNN
  backbone in the benchmark notebook). Convert to canonical $(-1, 1)$ at the
  decoder boundary.
- Rationale: keeping $[0, 1]$ in some stages and $(-1, 1)$ in others creates
  silent off-by-half errors during loss / AP eval.

## Stage I/O contracts

### Feature extractors (`src/features/`)

```
forward(image=None, image_ids=None, **kwargs) -> dict[str, Any]
```

- `image`: optional pixel tensor `[B, 3, H, W]` for online extractors.
- `image_ids`: optional list of int for cached extractors.
- Concrete subclasses validate which one they require (raise on missing input).
- Output dict is consumed by an embedder; per-class keys are documented inline.

### Feature embedders (`src/embedders/`)

```
forward(features, kpt_pixel=None, img_hw=None, joint_mask=None, **kwargs)
    -> dict[str, Tensor]
```

Possible output keys (all optional unless required by the downstream consumer):

- `feature`: `[B, D]` global vector (set-prediction head).
- `tokens`: `[B, T, D]` spatial tokens (no pooling).
- `node_feats`: `[B, N, K, D]` per-node (refinement decoder).
- `global_feats`: `[B, D]` per-batch global (refinement).

Refinement embedders read `kpt_pixel` / `img_hw`; set-prediction embedders ignore them.

### Decoders (`src/decoders/`)

```
forward(embeddings, graph=None, initial_keypoints=None, **kwargs)
    -> dict[str, Tensor]
```

Required output key:

- `keypoints`: `[B, N, K, 2]` in canonical $(-1, 1)$.

Optional output keys:

- `scores`: `[B, N]` per-person/per-slot confidence in $[0, 1]$.
- `keypoint_scores`: `[B, N, K]` per-keypoint confidence in $[0, 1]$.

`graph` and `initial_keypoints` are only required by refinement decoders
(graph diffusion); set-prediction decoders ignore them.

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

- 14 keypoints; joint sigmas live in `src.metrics.CROWDPOSE_SIGMAS`.
- Edge list in `src.data.skeleton.CROWDPOSE_EDGE_LIST`.
- AP via `src.metrics.KeypointAPMetric` with `iou_type="keypoints_crowd"`.
  Standard reported metrics: `AP`, `AP .5`, `AP .75`, `AP (E/M/H)`, `AR (E/M/H)`.
