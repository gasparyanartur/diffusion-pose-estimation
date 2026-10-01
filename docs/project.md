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
  `DiffusionRegPoseAPMetric` and `DiffusionRegPoseBBoxAPMetric`. Periodic
  validation reports separate keypoint and bbox AP from the same sampled
  pose-NMS outputs. The bbox metric uses
  `crowdposetools.COCOeval(iou_type="bbox")`; the keypoint metric retains its
  CrowdPose keypoint contract. Standard keypoint metrics are `AP`, `AP .5`,
  `AP .75`, `AP (E/M/H)`, and `AR (E/M/H)`. The Lightning test step records
  losses only; the standalone DiffusionRegPose evaluator is the canonical
  full-test AP path and emits both pose and bbox metrics.

## Model families

- `BaselinePosePipeline` is the direct image-to-fixed-pose-slot model. It owns
  its CNN feature extractor, spatial-attention pooling, Hungarian matching loss, AP
  lifecycle, and optimizer setup.
- `GraphPosePipeline` is the direct image-to-box-and-pose diffusion set model.
- `PoseFlowPipeline` is the endpoint-flow variant of DiffusionRegPose. Its
  optional pose-hypothesis codebook is one shared learned token bank read by
  image-conditioned decoder queries after image attention. `pose_codebook_size=0`
  disables the branch. `pose_codebook_initial_gate` controls the learnable
  residual scale at initialization: zero preserves the image-conditioned
  baseline, while a nonzero value activates codebook gradients from the first
  backward pass. Denoising-query prefixes never use the codebook. A configured
  codebook can be reused at one or more decoder layers, including box-oriented
  layers when an experiment explicitly requests all-layer conditioning. Enabled
  PoseFlow runs also record detached clean-query codebook diagnostics: gate and
  query/attention/residual scales every epoch, plus attention entropy and
  effective slot count from the first validation batch only.
- PoseFlow V8 is a separate DINO-only ConvNeXt-Tiny feature family. Its fine
  path uses all four native stages `[96, 192, 384, 768]`; its coarse path uses
  stages 2/3/4 `[192, 384, 768]`. It does not generate another pyramid.
  Independent `1x1 Conv + GroupNorm(32)` adapter banks create the three-level
  encoder/query features and four-level decoder deformable-attention features.
  The DINO backbone remains frozen by default; only the adapters and model
  heads train from their respective branches.
- `DiffusionRegPosePipeline` is the native checkpoint-compatible set-prediction
  pose model. Its training criterion defaults to the reference Hungarian
  assignment and supports opt-in DiffusionDet-style Dynamic-K OTA while
  retaining the focal-classification, box, keypoint, OKS, and denoising losses;
  its model-only
  evaluation path loads compatible tensors from the downloaded reference
  checkpoint. Selected two-stage encoder proposals receive the reference
  intermediate classification, box, and GIoU supervision during training.
  Same-seed construction reproduces all shared external model tensors before
  EDPose loading, including transformer reset order and time conditioning.
  Crop augmentation removes cropped-out people from supervised targets while
  retaining their clipped keypoints for diffusion proposals. Exact CUDA
  training and evaluation use the vendored DINOv3 native
  forward/backward deformable-attention kernel at a pinned revision; its
  DINOv3 and Deformable DETR/SenseTime attribution and license are preserved
  beside the imported sources. The PyTorch sampler remains the differentiable
  fallback when the extension is unavailable. CrowdPose
  person targets use category label `1`, matching the dataset and reference
  implementation; native checkpoints trained before this correction require
  retraining. OTA assigns complete person groups rather than individual joints:
  box IoU controls spatial candidates and Dynamic-K cardinality, while
  pose-capable layers retain the visibility-masked keypoint and OKS matching
  costs. Each clean decoder/encoder output is matched independently and its OTA
  losses are normalized by that output's positive assignments. Denoising
  queries retain their explicit correspondence and are never OTA-matched.
  Its clean classification diagnostics separately report focal
  contributions from matched positives and all remaining negatives, matched
  and unmatched person scores, matched-query count, and matches-per-target
  statistics under `debug/` without changing the optimized loss.
- PoseFlowPipeline retains the DiffusionRegPose ED-Pose backbone, decoder,
  prediction heads, denoising path, task criterion, evaluation, and pose NMS,
  but uses pose-aware Dynamic-K OTA by default. Its generative state is only
  flattened keypoint XY. Training samples continuous per-image flow time and
  linearly interpolates from a Gaussian source to a padded endpoint set;
  source-prior padding is the default and is not a semantic ground-truth
  target. Sampling uses forward Euler with a fixed original source anchor:
  velocity is predicted endpoint minus source, not endpoint minus current
  state. Optional inference-only renewal replaces weak proposal slots and their
  fixed source anchors together. Optional ensemble selects candidates inside
  each flow step, aggregates those candidate records, then applies one
  cross-step pose NMS; it does not globally top-k concatenated raw queries.
  The standalone evaluator accepts optional `--flow-seed`; it reseeds each
  image from `seed + image_id` so step-count runs share an identical initial
  source for every image without changing the stochastic default.
  Both default off and renewal uses an explicit score threshold.
  DiffusionRegPose remains the checkpoint/external-parity reference; a
  trained PoseFlow model requires its own retraining provenance.
- PoseFlowV3Pipeline is a separate retraining-only checkpoint family. It
  transports normalized two-corner boxes and keypoint XY together, retains 50
  selected person states and their fixed sources between Euler calls, and
  refills the 200-slot box-decoder pool with 150 source-prior states. Its fixed
  state gate is zero at `t=0` and 0.5 at `t=1`, so image query content and at
  least half of image reference geometry remain at every time. The V3 control
  uses the identical pool, endpoint targets, optimizer, losses, OTA, and
  evaluator while disabling only state conditioning. V3 has no score-based
  renewal switch; refilling is its renewal contract.
- PoseFlowV4Pipeline is a separate retraining-only coordinate-flow family.
  Each Euler call independently builds 200 pose hypotheses from learned slot
  identities, the timestep embedding, Fourier features of the current joint
  box-and-pose state, and cross-attention over all position-aware encoder image
  tokens. It selects 50 complete initial two-corner-box-plus-14-joint states,
  refills 150 fresh coordinate/source pairs at later calls, and never carries a
  latent hypothesis between calls. The initial hypothesis and every later joint
  refiner receive complete independently OTA-matched class, box, keypoint, and
  OKS losses. A matched control zeros only the coordinate-prior path.
  Its `debug/{train,val}/` metrics are detached observations only; the
  TensorBoard run label identifies the PoseFlow version:
  gate/context/image-update/hypothesis RMS, selected initial confidence,
  initial-to-final box/pose movement, and named per-stage losses. They do not
  alter optimization, matching, sampling, or prove a causal state effect.
- PoseFlowV5Pipeline is a separate V4-derived checkpoint family for a changed
  optimization/source treatment. It uses two 40-epoch cosine cycles: main and
  backbone rates decay from `1e-4`/`5e-5` to `1e-5`, restart at `3e-5`, then
  decay to `2e-6`. During training, its 200 source states are split between
  same-image GT-person resamples with zero-mean `0.25` center jitter and
  log-scale jitter, and same-image interpolations of two distinct GT states.
  One-person images use resamples; empty images and inference/refill use
  uniform-center, empirical-mean-size sources. V5 predicts two corners plus
  an auxiliary center point at its hypothesis and refiner stages. Decoded
  boxes remain corner-derived; direct-center and corner-midpoint-to-GT-center
  losses each use `0.1 * GIoU-weight` and do not alter OTA or inference.
  The optional V5-only `decoder_loss_mode=first_last` ablation optimizes the
  hypothesis and final decoder outputs (including their center losses), final
  denoising losses, and the corner-order penalty. Intermediate refiner and
  denoising losses remain logged but do not contribute to the objective;
  `all` retains the original supervision at every decoder stage.
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
  every 10 epochs; validation preserves the training RNG state and the runner
  skips Lightning sanity validation by default. `last.ckpt` remains available
  after every epoch. The
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
- `benchmark_diffusionregpose_development` is the measured short-feedback
  model from the initial runtime audit. It uses a 10% deterministic split,
  128/192-pixel resizing, batch 16,
  three feature levels, two encoder layers, three decoder layers, 200 initial
  queries, 20 person/proposal groups, 20 denoising queries, and FFN width 1024.
  Its full-data 256-pixel run remained near zero AP because those reductions
  leave one pose layer and make most retained EDPose transformer tensors
  shape-incompatible. It is retained for runtime/wiring measurements only.
- `benchmark_diffusionregpose_development_compatible_256` is the replacement
  learning gate. It keeps four feature levels, FFN width 2048, and all four
  pose-refinement layers while reducing initial queries to 200 and
  person/proposal/DN groups to 50. All six pretrained encoder layers are
  retained because a two-layer initialization control reduced bbox AP from
  approximately 0.29 to zero. Its EDPose load requires
  at least 99% element coverage independently for each critical component and
  uses a fixed validation diffusion seed. It remains a development experiment,
  not a paper-scale result.
- `benchmark_diffusionregpose_development_compatible_128` keeps the same
  checkpoint-compatible architecture, optimizer, losses, and evaluator while
  using 128/192-pixel train and validation resizing. It is a full-data
  lower-resolution learning gate, not a paper-scale benchmark.
- `benchmark_diffusionregpose_development_compatible_256_freeze_backbone` is a
  controlled 256-pixel comparison preset that freezes every ResNet parameter
  while retaining the compatible transformer, diffusion, losses, optimizer,
  and evaluator. It has approximately 26.1M trainable parameters versus
  49.3M in the standard compatible preset; it is not the paper-default recipe.
- `benchmark_diffusionregpose_development_compatible_256_head_only` loads the
  EDPose checkpoint and freezes the backbone, input projections, label
  embedding, and transformer computation while retaining prediction and
  encoder proposal heads. It is a diagnostic transfer/runtime experiment,
  not a paper training recipe.
- `benchmark_diffusionregpose_development_compatible_256_keep_layers` is a
  reduced transfer experiment. It runs three encoder and three decoder layers,
  retaining EDPose checkpoint layers 2, 4, and 6 through explicit one-based
  checkpoint-key remapping. Decoder layer 2 supplies the box stage and layers
  4 and 6 supply the two pose-refinement stages; the original expansion pose
  head is retained separately. Because skipped layers change the activations,
  this preset cannot reproduce the full checkpoint's outputs or serve as a
  paper-parity evaluator.
- `benchmark_diffusionregpose_development_compatible_256_keep_layers_ota` is
  the matched OTA experiment for that selected-layer fast run. It inherits the
  same 3E/3D architecture, EDPose layer remapping, 256/427 resizing, batch 4,
  accumulation 2, initialization, optimizer, schedule, and evaluator; only
  clean-query assignment changes from Hungarian to pose-aware Dynamic-K OTA.
- Cached-DINO graph-diffusion refinement is unsupported and has no
  configuration or compatibility path.
- The 3D-prior research thread has closed its depth-based reconstruction
  route: inspected DA3 and SHARP point clouds did not provide reliable enough
  human geometry for skeletal reasoning. Future research in that thread uses
  DINO image features without an inferred 3D point cloud. This records an
  experimental direction, not a new supported pipeline or feature contract;
  see `reports/3d-prior/sharp-feasibility.md` for the visual evidence.
- Box diffusion and DiffusionDet are separate object-detection models whose
  outputs are boxes and objectness scores, not poses. Box diffusion is the
  tensor-native `BoxDiffusionPipeline`; it accepts image tensors and does not
  implement the path-based `BoxDetector` contract retained for DiffusionDet.

The root default is the cache-free fixed-slot baseline configuration. It accepts
image tensors and returns canonical keypoints plus slot and keypoint confidence
scores.
