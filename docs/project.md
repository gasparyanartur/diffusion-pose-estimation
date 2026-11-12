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
- Native pose postprocessing retains each selected candidate's original query ID
  through pose NMS, so its confidence and geometry come from the same query.
  Historical AP using the old candidate-index-to-raw-logit mapping remains
  legacy scoring; it is not directly interchangeable with corrected scores.

## Model families

- V20 extends V19 dense-read with a shared auxiliary `H -> 17x3` pose head.
  The student retains the dense image read, geometry-detach boundary and DDIM
  sampler, and trains its hypothesis/geometry interaction under the original
  losses plus final-call projected-pose supervision and frozen GFPose score
  distillation. GFPose's released state is image-plane XY plus scaled depth,
  not metric XYZ. Twelve common limb joints receive 2D supervision; head/neck
  definitions are excluded and teacher-private joints remain latent. The teacher
  is lazy, training-only, hash-pinned and excluded from student checkpoints and
  optimizers. A separate checkpointed noise generator preserves the student's
  RNG stream. This adds a structural training signal, not a calibrated 3D
  posterior, identity guarantee or established AP improvement.

- V19 replaces learned latent geometry with 32 coordinate logits per candidate:
  box center/size and 14 joint XY pairs. Fifty candidates have 15 geometry features
  plus their own persistent hypothesis in 16-token local attention; there is no
  global token. Separate decoder image attention, local attention/FFNs, reference
  position encoding, coordinate-correction heads and person-class weights initialize
  from compatible EDPose components. The shared hypothesis-bank block and coordinate
  adapters initialize freshly. Coordinate DDIM and hypotheses remain differentiable
  across rollout calls. The objective averages independent OTA geometry/confidence
  losses across iterations/calls; no learned codec, latent MSE or reconstruction
  loss is used. `benchmark_poseflow_v19` retains the 80-epoch rollout recipe in a
  separate checkpoint namespace. Transfer auditing distinguishes exact checkpoint
  tensors from tensors merely copied out of the initialized source model.
  `benchmark_poseflow_v19_detached` isolates a training gradient boundary:
  coordinates and joint extents detach between decoder layers, while features,
  hypotheses, final predictions and cross-call DDIM remain differentiable.
  It uses the same parameter count/recipe in a separate output namespace.
  `benchmark_poseflow_v19_sampling_detach` stops only later-layer sampling and
  positional reference gradients, retaining coordinate-residual gradients. The
  two detach flags are mutually exclusive; existing variants default to false
  for the new flag. This isolates the gradient routes without changing inference.
  `benchmark_poseflow_v19_dense_read` extends the full geometry-detach baseline
  with one dense image read into the 50 hypotheses at the start of every denoising
  call. Its residual is scaled by a learned scalar initialized at zero and by
  `.1+.9*(1-t)`, preserving the baseline initially while allowing a smaller update
  at high noise. Encoded images retain spatial/level positions for dense keys and
  padding masks. Local refinement, coordinate DDIM, rollout supervision and memory
  carry remain unchanged; confidence-ranking supervision is absent. It has a
  separate configuration/checkpoint namespace. Explicit baseline-state import
  reports copied coverage and new dense-read keys; variant checkpoints restore
  strictly. Implementation verification does not imply a training launch.

- V18 isolates pre-image representation processing: each decoder iteration applies
  a separate normalization/residual FFN/normalization block to V17's 16 nonmemory
  representations before decoding image-sampling references and reading evidence.
  Tied memory, global hypothesis attention, ungated updates and the rollout recipe
  stay unchanged. It uses fresh initialization and a separate namespace alongside
  V17; the old all-bank memory reader is not restored.

- V17 ties each candidate to its own persistent hypothesis token inside a 17-token
  local self-attention block (box, 14 joints, global, hypothesis). A shared
  hypothesis self-attention block exchanges context across candidates before local
  image evidence and local coordination. The local output directly updates memory;
  separate memory reading/writing and retention gating are removed. The existing
  rollout recipe and geometry codec/DDIM remain unchanged in a fresh namespace.
  Candidate memory can follow a candidate that changes its matched person; no
  identity-tracking guarantee or new identity supervision is introduced.

- `poseflow_v15_rollout` changes training only, retaining V15's architecture and
  inference model-state schema. Half the batches retain random-time single-call
  training; half draw two or three calls, starting from Gaussian latents at t=1
  with independent per-image descending distinct timestep indices. DDIM latents
  and hypothesis memory remain differentiable between calls; each image is encoded
  once and hidden globals reset per call. All calls/layers are supervised with
  independent OTA matching, losses average equally across calls, and reconstruction
  is computed once. Conditional validation remains single-call for comparison.
  It starts fresh alongside V15, with the same optimizer/data/schedule. Improvement
  is judged by persistent/reset multi-call gaps, not one-call AP alone.

- `poseflow_v15_rollout_query_norm` isolates writer-query normalization in a
  fresh rollout run. Non-affine LayerNorm transforms only the hypothesis queries;
  the original memory enters the retention interpolation unchanged. Parameter
  count and state keys match V15, so restoration must use this experiment's
  configuration to recover its behavior. Image initialization, rollout mixture,
  losses and schedule remain unchanged; no path dropout or decoder transfer is used.

- PoseFlow V15 retains V14's 15 box/joint diffusion tokens and adds a carried
  hidden global token to each candidate's decoder representations. Each iteration
  reads hypotheses, estimates sampling geometry, updates box/joint features directly
  from deformable image evidence, then coordinates all 16 tokens within the person.
  Global summaries update the persistent 50-slot memory through normalized attention
  evidence and a learned scalar interpolation gate initialized to retention 0.9.
  Global tokens reset per call; hypotheses reset per sampling invocation. There is
  no post-interpolation normalization, FFN, candidate-slot pairing or diversity loss.
  `benchmark_poseflow_v15` is a fresh family with V14's OTA/optimizer/data recipe;
  compatible image components transfer and revised decoder blocks initialize fresh.
  Single-call training is retained. Advanced SSM updates and rollout training are deferred.

- PoseFlow V14 uses 50 fixed candidates with 15 sampled latent tokens each:
  one box token and 14 joint tokens. The box token decodes center, size and
  confidence; confidence has no separately diffused token. Each denoising call
  creates one hidden global token per candidate, while 50 independent hypothesis
  slots persist between sampling calls. Global tokens persist only within a call.
  Each iteration gathers geometry-guided deformable image evidence, aggregates
  it within each person including the global token, lets hypothesis queries read
  global summaries, then lets all representation tokens read hypotheses.
  Evidence extraction and hypothesis updating share weights across iterations;
  hypotheses have no required correspondence to person slots.
  `benchmark_poseflow_v14` starts a fresh checkpoint family with the V13 OTA
  reconstruction-decay recipe (top-k 5, gradient cap 2.0). Box and labeled-joint
  latent losses supervise matched slots only; unmatched confidence remains
  supervised. Single-call training exercises recurrent memory within the decoder;
  cross-step carry-over is evaluated empirically, without training rollouts.
  Quick-test diagnostics pair persistent/reset memory with identical image/noise
  inputs, reporting raw box coverage, excess duplicates and confidence quality
  alongside pose/bbox AP and per-step/per-layer evidence-memory summaries.
  These are bounded subset diagnostics, not full-test results.

- PoseFlow V13 retains V12's latent-only fixed-group architecture, but each
  decoder layer independently matches predictions to real people (Hungarian by default).
  Its latent and geometry losses use the same assignment within a layer;
  unmatched slots have existence-zero classification and existence-token MSE,
  without geometry denoising supervision. Clean reconstruction remains paired
  and covers empty geometry. Decoder focal classification is normalized by
  matched-person count (clamped to one), while clean reconstruction retains
  all-slot averaging. Matching never reorders sampler states. V13 has
  its own config/checkpoint namespace and starts fresh for comparison with V12.
  Every epoch also evaluates 32 fixed test images across noise variance fractions
  25/50/100 percent and 1/2/3 DDIM calls. Partially noisy inputs use GT encodings;
  these are conditional recovery diagnostics. Only 100 percent noise is
  Gaussian-start generation. These preliminary test-subset metrics are distinct
  from full validation/test AP and must not be presented as full-test results.

  The separate `benchmark_poseflow_v13_reconstruction_decay` experiment keeps
  V13's architecture, matcher, data and learning rates, but starts fresh with
  reconstruction weight decaying geometrically from 1 at epoch 0 to 0.01 at
  epoch 39, then holding. Its global gradient norm cap is 2.0. This is a combined
  intervention; improvements cannot be attributed to either change alone.
  Raw reconstruction losses remain comparable; weighted reconstruction enters
  the total. Optional gradient probes report objective norms on the first
  training microbatch each epoch and accumulated preclip norms on the first
  and every 100th optimizer step. Ordinary V13 defaults keep weight 1 and probes
  disabled. Resuming uses the restored absolute epoch for the schedule.

  `benchmark_poseflow_v13_ota_reconstruction_decay` matches the fresh
  reconstruction-decay experiment but selects Dynamic-K OTA with candidate
  top-k 5. Multiple slots may target the same person; latent and geometry
  supervision share each layer's assignment, and classification/geometry
  normalize by positive assignments. Matcher choice does not change model
  parameter keys or sampling. It uses a separate output namespace; OTA's
  effects on confidence stability and duplicate predictions remain unproven.

- PoseFlow V12 is a separate fixed-slot latent-diffusion family. Its default
  50 groups carry 17 latent state tokens each; an additional global person token
  gathers group information and queries the image inside the decoder. Clean
  geometry is encoded and normalized before Gaussian corruption. Training keeps
  each noisy slot aligned with its own clean target, including random enclosed
  geometry for empty slots; it does not use proposal selection or matching.
  Clean reconstruction trains the encoder, while detached clean latents supervise
  layerwise denoising. Sampling uses latent DDIM; coordinate readouts never feed
  back into sampling or decoder image attention. V12 requires a fresh checkpoint
  and uses `benchmark_poseflow_v12` / `results/poseflow_v12_training`.

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
- PoseFlow V11 is a separate center/size-flow checkpoint family. A 256-wide
  coarse person latent conditioned on noisy boxes predicts existence and boxes,
  without a pose head or pose matching costs. The top 50 of 200 candidates expand
  into 17 learned roles: existence, center, size, and 14 CrowdPose joints. Noisy
  joints condition their corresponding selected tokens during training and
  direct conditional validation. Sampling defaults to
  `sampling_joint_conditioning=false`, skipping this extra feature addition
  while retaining noisy-box conditioning, timestep inputs, and state transport.
  Set it to `true` to reproduce the previous sampling behavior. Existing V11
  checkpoints restore strictly; missing config fields use the new default.
  This is a checkpoint-tested inference mitigation, not a change to training.
  Initial joint references
  are decoded from these tokens. Each of three refiners performs image attention,
  within-person attention, existence-only cross-person attention, then a 2048-wide
  FFN. Self-attention uses latent Q/K/V without geometric position embeddings or
  distance bias. One point MLP serves center/joints across stages, with separate
  shared size/existence MLPs. DN examples use three tokens and box/class losses
  with inherited isolation. Corner/order/midpoint and redundant center losses
  are absent; classification, center/size L1, GIoU, joint L1 and OKS remain.
  V11 retains V5's data, reusable EDPose initialization and two-cycle LR schedule;
  new role/readout/existence-attention weights train from fresh initialization.
  First-periodic-AP-batch diagnostics distinguish coarse box coverage from pose
  refinement and record detached token-role/existence updates. A two-epoch
  reduced-data smoke verifies wiring only; it does not establish model quality.
- PoseFlow V10 retains V5's image-conditioned hypothesis initialization, including
  learned slot identity, timestep and gated coordinate-state conditioning. Each
  selected person remains one attention-width latent rather than expanding into
  box/joint tokens. One shared absolute geometry MLP predicts both box corners,
  auxiliary center and all joints at every stage; one shared classifier predicts
  person scores. Decoder layers use decoded bbox references for 2D image sampling
  and keep references differentiable across stages. Coordinate-flow sampling and
  V5 losses are retained; the original V10 preset uses one cosine decay with main/backbone peak LRs
  1e-4/1e-5, warmups of 1/10 epochs, and group-relative 1% minima. V10 requires
  its own checkpoints. Auxiliary DN
  queries retain noisy label embeddings with a learned noisy-box conditioning MLP
  and bbox/class supervision, without joint targets. Future V10 processes also
  collect first-periodic-AP-validation-batch solution diagnostics for proposal
  coverage, fixed-pair refinement, flow turnover and score/NMS coverage; optional
  feature interventions preserve baseline RNG. These are bounded diagnostics,
  not full-validation AP. V10 additionally records clean-person latent updates,
  person spread, geometry saturation/movement and tiny boxes on that same batch,
  plus main/backbone effective LR each epoch. Existing processes keep their
  loaded instrumentation.
- PoseFlow V10-tiny is a separate V10 preset with attention/model width 128 and
  transformer FFN width 256. It keeps V10's data, depth, queries, losses, and
  schedule. The same EDPose checkpoint initializes the ResNet backbone exactly;
  most width-dependent transformer, projection, and head weights are retrained
  because their 256-wide checkpoint shapes do not fit. Its checkpoints are not
  interchangeable with 256-wide V10 checkpoints.
- The V10-tiny V5-LR preset selects V5's two-cycle cosine schedule through the
  shared V5-derived optimizer path: main/backbone start at 1e-4/5e-5, both
  restart at 3e-5 at epoch 40, and both finish at 2e-6. It has no warmup and
  starts a fresh optimizer state in a distinct output directory.
- PoseFlow V9 branches from V5 and retains its ResNet image features, flow
  state, optimizer, losses, and evaluator. In each expanded person group, its
  joint-query self-attention adds one learned per-refiner-layer logit bias,
  initialized to `-5`, only for nonneighbor joint pairs under the CrowdPose
  skeleton. It preserves self, skeleton-edge, and box-token interactions, plus
  the existing person-group and denoising isolation masks. Bias `0` recovers
  V5 attention for otherwise identical weights and input.
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
  and initial-to-final box/pose movement. Canonical loss tags already include
  every stage; future runs omit duplicate debug loss aliases and the gate mean
  that is fixed at half the flow time. These activity metrics do not
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

Quick-test comparison metrics use the same TensorBoard tags across V13/V14.
V14 persistent memory uses `quick-test/`, reset controls use `quick-test-reset/`,
and extra diagnostics use `debug/quick-test/`. JSON condition names retain memory
mode provenance. Existing trainers can use a separate live comparison event view;
source event files and training processes remain unchanged.

- PoseFlow V16 separates V15's box role into confidence, center and size,
  retaining 14 joint roles and the hidden global token (17 diffused/18 decoder
  tokens). The baseline remains single-call training with the V15 recipe,
  OTA5 and unchanged geometry/classification weights. Matched box-family latent
  supervision averages its three roles before equal weighting with joints;
  unmatched confidence classification and invisible-joint masking are retained.
  Use `benchmark_poseflow_v16` and its separate output namespace. Existing
  rollout runs remain independent. Bone loss is the next deferred iteration.
- CrowdPose validation overlaps trainval training and is an in-sample fit
  diagnostic. Generalization comparisons use the disjoint test split; fixed32
  test probes are bounded diagnostics rather than full-test AP.

- `poseflow_v15_rollout_path_dropout` is a separate V15 rollout experiment with
  .25 independent probabilities for dropping joint image and self-attention
  messages. Box/global receiving rows, direct hypothesis reads and joint
  representations remain active. Survival scaling preserves mean message size;
  evaluation disables path dropout. Parameters and strict model-state layout
  match V15. A batch-local generator keeps masks fresh despite fixed attention
  seeds. Existing V15 rollout/V16 jobs remain independent; bone loss is deferred.

- `benchmark_poseflow_v19_evidence_rank` is a separate full-detach V19-derived
  auxiliary-supervision family. Three RMS-relative Gaussian evidence levels
  `[0, .25, .5]` update copies of the same hypothesis prior; a shared linear
  ordinal head ranks them with variance-scaled margins. OTA positives select
  people, without using predicted pose quality as the confidence target. Only
  clean updates carry forward. Validation/inference stay clean-only. This adds
  256 parameters at width256 and changes strict checkpoint keys; old checkpoints
  and existing runs stay separate. Auxiliary corruption varies across training
  batches despite the inherited fixed attention RNG fork. The approved full run
  starts fresh from the same compatible source initialization as V19; it does not
  resume an existing V19 model or optimizer.

- V21 uses V1's joint-coordinate flow and detector recipe with a time-conditioned
  hypothesis-derived initial proposal stage. It requires equal query/proposal/group
  counts (50 in the preset), keeps candidate order through pose expansion, and
  carries one additional H per candidate. Its memory is context, not a diffused
  target or a ground-truth identity. DN and geometric outputs exclude H. The
  original V1 feature/reference seed detach boundaries remain, while the H branch
  stays attached to detector gradients. Default training remains single-call;
  no V20 rollout/3D objective or optimizer schedule is inherited. Source-copy
  initialization checks and wiring smokes do not establish AP improvement.

V21 quick-test monitoring uses the same fixed 32 held-out CrowdPose test images
(seed 42), native Gaussian-start Euler sampling at 1/2/3 calls, and persistent
versus reset hypotheses. It does not use ground-truth coordinates to initialize
sampling. These repeatedly used subset diagnostics are not full-test AP.
Monitoring adds no model state or training objective. The epoch-end hook restores
RNG, module modes and sampling call count. An existing trainer can receive the
same metrics from a separate checkpoint monitor without losing optimizer state.

The current V21 preset updates H before geometry in each decoder layer. Geometry
reads the fully updated H and returns that exact state for carry, so the current
criterion supervises the final returned H. Older saved configurations missing
`hypothesis_update_order` retain `parallel`; the current preset explicitly selects
`h_then_g`. This changes the graph without adding parameters or rollout training.
It does not establish that carried H generalizes across sampling calls.

V22 adds explicit FiLM/AdaLN timestep conditioning to V21 H-first while retaining
H influence on initial proposals and later box/pose representations. Time is
scaled only at embedding (default999), not in physical flow interpolation or
Euler updates. New modulation heads initialize to identity. The existing
coordinate-reference detach boundary is retained for inherited geometry weights;
only new coordinate-conditioner outputs retain a loss path into pose refinement.
Old V21 checkpoints keep their architecture and embedding convention. V22 uses
its own config/model/checkpoint namespace and the same fixed-slot pairing and
held-out quick-test protocol. More conditioning does not by itself prove stable
recurrent reuse or improved test AP.

The current V22 preset explicitly uses `coordinate_batch_layout: per_image`.
Noisy coordinates arrive as[group,batch,joint,XY]; batch moves to the leading
axis before flattening candidate/joint tokens. This prevents cross-image state
mixing and keeps each image's coordinates aligned with its own timestep and
reference coordinates. Missing layout fields in historical V22 saved configs
retain `legacy` for reproducible restoration. Single-image sampling is identical
between layouts; the correction changes batched training semantics.

V23 retains corrected V22 architecture and changes training to a50/50 mixture of
supervising the first or second of two ordered iid uniform flow times. The mixture
has uniform marginal supervised time. First-step batches make one training call;
second-step batches make an eval/no-grad warm-up without DN targets, carry its
detached H, advance coordinates with the native fixed-source endpoint increment,
and supervise only the later call. Every batch has one criterion and one
gradient-bearing forward; average compute is higher due to warm-ups. A private
checkpointed schedule RNG is independent of deterministic-forward RNG resets.
Validation and native held-out sampling are unchanged. This is detached recurrent
input exposure, not backpropagation through a multi-step trajectory.

## Matched one-step h experiment (V24)

Compare newly trained control/h arms with the same V1 spatial initializer,
200-to-50 selection, corrected per-image coordinate layout, original time
convention and single-call training. h only adds a zero-gated residual interaction
inside the pose decoder; it does not generate initial proposals. Closed-gate
prediction/loss/common-gradient parity and matched initialization/RNG are required
before launch. Gates must subsequently open and supply nonzero image/writer/reader
gradients. DN and other images/candidate groups cannot enter the local h branch.

Use one paired training seed42, 80 epochs, identical source weights, FP32
training, batch4/accumulate2 and the original optimizer/LR/loss recipe. Final
epoch079 on all8000 held-out test images with K1, Gaussian seed42+imageID,
resize256/427, FP32/TF32off and corrected native NMS is the primary endpoint.
Report the single paired difference as a pilot; replication requires a later
decision. Fixed32 is monitoring only;
validation overlaps trainval and cannot substitute for held-out performance.
No tuning or final-checkpoint selection based on fixed32 results is planned.
A negative result is specific to this residual h design. Removing h from a model
trained to use it is not evidence of the effect of adding h.

## Person-aligned h feature supervision (V25)

V25 extends the V24 h arm with an auxiliary decoder from the final person h to a
7×7×384 DINOv3 ViT-S/16 region grid. The decoder receives h and learned constant
spatial positions only. The target is ROIAlign of frozen DINO patch features at
that candidate's matched ground-truth box in the augmented image. These region
features retain full-image context; they are not independently encoded person
crops. The exact final pose assignments are shared by pose and feature losses;
the DINO objective does not change the matching cost or supervise unmatched/DN
queries. Per-cell cosine loss is averaged over actual valid positive assignments,
with default weight1.0 alongside the unchanged pose objective.

The teacher is revision/hash pinned, loaded locally on demand, permanently eval
and outside student checkpoints/optimizers. Native batches are already
ImageNet-normalized; target extraction uses each unpadded image and its augmented
size, without a second normalization. Only right/bottom patch padding is added.
The teacher and auxiliary decoder are not executed for ordinary pose sampling;
h continues to reset at every forward. Zero auxiliary weight follows V24's
original loss path. V25 has a distinct config/checkpoint namespace; importing a
V24 h model may initialize only h_decoder freshly, while V25 restores strictly.
Implementation and a short wiring smoke do not establish learned semantics or AP.

## Jointly learned person-latent diffusion (V26)

V26 is a new single-seed pilot using V24 control's ResNet extraction and training
recipe, with projected image features directly attended by separate clean and
noisy three-layer dense-attention stacks. There is no deformable image encoder,
query expansion, DN supervision, matching, intermediate loss or carried memory.
Each of50 proposals has one256D latent. Input14-joint pixel coordinates are divided
by100 before a two-layer MLP. Clean task supervision grounds the latent space;
its detached, non-affine-normalized output supplies the cosine diffusion target.
Both clean and recovered latents share final box/existence/joint heads.

The `benchmark_poseflow_v26_latent_only` ablation removes denoised task supervision:
`0.5 * clean_task + latent_MSE`, retaining the clean coefficient and latent weight1.
The encoder stays jointly trained. Denoised task losses remain detached monitoring
metrics; the denoiser receives only latent MSE gradients. Original V26 defaults
retain a denoised task coefficient0.5. Use a fresh source snapshot and separate
`poseflow_v26_latent_only` namespace with identical seed42 initialization/recipe.
Layer-conditioning/FiLM–AdaLN review and REPA-style DINO targets are later steps,
excluded from this loss-only comparison.

Every valid GT person appears once. Half the remaining slots (rounded down)
duplicate positives; the rest borrow poses from other batch images and have
existence zero. Borrowed poses are not overlap-filtered: false negatives are an
accepted limitation. Missing foreign donors use the mean of two available poses
plus sigma5-pixel Gaussian noise, including retained pre-augmentation poses when
necessary. Source assignments are constructed and preserved, not rematched.
Negatives have existence and latent supervision, with no geometry supervision.

K4 deterministic latent DDIM is primary; K1 is a diagnostic. Sampling consumes
images and independent Gaussian latents only. Fixed32 is monitoring; epoch079
full8000 held-out test AP is the endpoint. Validation overlaps trainval. Compare
against V24 control as a reference with changed architecture/compute, not as an
isolated causal ablation. Latent MSE reduction does not establish detection quality.

## Shared experiment dashboard

Reuse the existing shared TensorBoard on port33329 for new experiments. Add run
log directories to its shared launch configuration instead of creating another
TensorBoard per version. The current launcher is `tmp/tensorboard_shared.sh`, with
its run map in `tmp/tensorboard-shared/runs.json`. Preserve existing runs and labels
when updating it. Training processes and event files are independent of dashboard
restarts. The current host is reachable over Tailscale at
`http://lynx.tail481642.ts.net:33329/`; verify host/network state when changing access.

## Coarse-to-fine person-latent attention (V27)

V27 changes V26 latent-only image access: both independent three-block stacks
read the existing ResNet feature levels in stride32/16/8 order, one per block.
No image fusion encoder, image self-attention, stride64 level or multiscale token
concatenation is used. All50 proposal slots persist through the stack. The scale
schedule repeats inside each DDIM call, rather than across diffusion timesteps.

Keep0.5 clean-task + latent-MSE1, the jointly trained clean encoder, detached
latent source/target, existing timestep modulation and shared final task heads.
Denoised task losses remain detached monitoring-only metrics. V27 uses a distinct
model/config/checkpoint family and fresh seed42 initialization. Image transfer
explicitly selects source levels0/1/2; original V26 import/restore defaults stay
strict and unchanged. Dropping the fourth image projection changes parameter
count and initialization RNG consumption, which must be recorded in comparisons.
The standard80epoch data/optimizer recipe, fixed32 K1/K4 monitoring and final
full8000 evaluation are retained. Any RT/RF-DETR neck, identity skip or DINO target
is a separate future experiment.

## Clean-branch image stop-gradient (V28)

V28 uses V26 latent-only's four-level image memory, clean/denoiser stacks, heads,
initialization and training recipe. Only clean-branch backward routing changes:
image memory and positions are detached at the clean encoder input. The clean task
trains the latent encoder and readout; latent recovery trains the denoiser and
shared image extraction. Clean latents remain detached before corruption and as
latent targets. Denoised task loss stays monitoring-only, with objective
0.5 clean task + latent MSE1. The clean forward still depends on image features,
so targets can drift when those features change under denoising gradients.

Use a fresh seed42 run in the distinct V28 namespace and frozen snapshot. Compare
against V26 latent-only at the same age; original V26's additional denoised task
loss is a separate difference. All previous defaults/restoration remain unchanged.

## OTA person-latent targets (V29)

V29 is a parallel fresh pilot from V28. Architecture, initialization, sampling,
constructed clean inputs, stopgrad(F), detached h0 corruption and latent-only
objective remain unchanged. Existing Dynamic-K OTA (topk5, center-radius2.5)
assigns all recovered slots against unique valid receiving-image GT people.
Assigned slots target the corresponding detached original clean latent (firstM
slots), not a duplicated clean slot. Unmatched slots target their closest clean
negative latent from the same image's constructed negative slots, with all target
selection detached. Every slot contributes uniformly to latent MSE. M0 uses only
negative targets; MP needs no negative bank because eachGT must be assigned.

Clean supervision retains constructed labels. Denoised task/score diagnostics
use OTA labels and are monitoring-only; their values are not directly comparable
to fixed-correspondence denoised metrics. Existing OTA uses class/joint/OKS costs
when poses exist, box priors and box-IoU Dynamic-K; it is many-to-one, with conflict
repair preserving at least one query perGT. Nearest-negative target selection is
an explicit additional policy, not proof that assignment ambiguity caused failure.

## Latent four-image capability diagnostics

The temporary overfit4 harness runs existing V28/V29 on the same4 trainval images
with1/2/4/8people, no augmentation and fixed constructed proposals. Clean latents
remain jointly learned; noise/timesteps are resampled. Both runs use a bounded
5000updates with constant original AdamW LRs, batch4/accum2. Clean, conditional
and GaussianK1/K4/K16 evaluations are separated, including raw OKS50/75 coverage,
score/NMS effects and3 evaluation noise seeds. These are memorization diagnostics,
not held-out AP. Full-data jobs and production code are unchanged. See
`docs/reports/2026-10-25-latent-overfit4.md` for exact inputs/provenance and limits.

## Decoder tolerance from detached recovered latents (V30)

V30 branches from V28 with fixed constructed correspondence, not V29 OTA. It
adds recovered task supervision through the shared readout after detaching the
recovered latent: `0.5*clean_task + 0.5*task(R(sg(hhat0))) + latent_MSE`.
Clean task gradients train the clean encoder and shared heads; recovered task
gradients train the heads only. Image extraction and denoiser parameters still
learn exclusively through latent MSE. Clean image features and both corruption
source/latent targets remain detached as in V28. The readout remains jointly
trained; there is no frozen encoder or EMA teacher.

Architecture, initial model tensors, proposal/noise draws, sampling, final-only
loss definitions and original seed42/80epoch recipe are unchanged from V28. This
is a decoder-adaptation experiment; it does not train denoiser representations
with decoded task gradients and does not test OTA. Original V26 also differs
in its clean-image gradient route, so comparisons must state that difference.

`PoseFlowV30Pipeline` overrides the default identity task-prediction hook in
`PoseFlowV26Pipeline` to rerun shared heads on detached recovered latents. Old
families retain their previous behavior and checkpoint schema. Evaluation uses
Gaussian-start denoising and ordinary shared readout, without the training hook.
Strict pipeline restoration and initialization-free model loading are tested.

## Restored denoised task gradients with detached clean latents (V31)

V31 uses the V28 model and original V26 training pipeline with denoised task
weight0.5: `0.5*clean_task + 0.5*recovered_task + latent_MSE`. It omits V30's
recovered-latent task detach, allowing task gradients through the shared readout
into the denoiser and its live image conditioning. Both h0 detach points remain:
`q_sample(h0.detach(),t,noise)` and MSE against `h0.detach()`. Neither recovered
task loss nor MSE backpropagates into the clean latent encoder. The clean branch
still receives detached image features and trains the clean encoder/shared heads
through clean task supervision. No OTA, sampler or model parameter changes.

`PoseFlowV31PipelineConfig` subclasses the V28 config, restoring weight0.5 while
retaining its inherited default V26 pipeline/identity task-prediction hook.
Historical V26/V28/V29/V30 defaults remain unchanged. Fresh seed42 initialization
is used for a separate80epoch pilot; checkpoints restore strictly and evaluation
loads without the initialization checkpoint. V30's final retained checkpoint
is epoch025 (26 completed epochs), with last.ckpt preserved.

## Dual-branch DINO and live clean-image gradients (V32)

V32 jointly supervises clean h0 and recovered hhat0 against frozen person-region
DINOv3 ViT-S/16 features. One shared HFeatureGridDecoder predicts7x7x384 grids;
the pinned V25 teacher, unpadded augmented-image extraction, ROIAlign and patchwise
cosine loss are reused. Unique-person target boxes come from canonical firstM
constructed slots; original annotation rows (including gaps) are mapped to unique
target indices. Duplicate positives share their person's target and count in the
positive-proposal normalization. Negative slots have no DINO target. Empty images
and invalid boxes skip supervision; an all-negative fallback batch skips teacher
execution and yields differentiable zero auxiliary loss. Targets are computed
once per batch and reused for both branches.

Unlike V31/V28, V32 inherits the original V26 model's live clean image features.
Clean task and clean DINO gradients reach the clean encoder/image extraction;
recovered task and recovered DINO gradients reach the denoiser/image extraction.
h0 stays detached before q_sample and as latent-MSE target; neither recovered
branch objective nor MSE trains the clean encoder through h0. The teacher alone
is frozen. Total loss is0.5clean_task+0.5recovered_task+latent_MSE+
0.5*dino_loss_weight*(DINO_clean+DINO_recovered), with default DINO weight1.

The only new model parameters are dino_head.*, initialized within fork_rng to
preserve baseline tensors and global RNG. The lazy teacher is held by a plain
wrapper and excluded from optimizer/model state. DINO loss uses a default-noop
auxiliary hook in V26; old family behavior is unchanged. Weight0 bypasses teacher
and head execution exactly. Strict evaluation restores the DINO head state but
requires neither teacher nor initialization checkpoint; Gaussian inference runs
the existing denoiser/readout without DINO execution.

V32 combines two user-requested changes (DINO and clean-image gradients); it is
not an isolated DINO ablation. V31 remains active as a reference. Monitor clean/
recovered DINO loss, cosine, prediction variance, valid/skipped assignments and
correct-minus-wrong-person cosine (available for multiple valid people per image),
plus fixed32 Gaussian AP. High DINO cosine alone is not successful detection.

## Structured17-token person readout (V33)

V33 changes only V32's task readout. A role-specific linear expansion maps each
person latent h into17 width256 tokens (14joints, box, existence, global), with
learned role embeddings. Flattening[B,P] as independent sequences[B*P,17,D]
ensures no cross-person or cross-image attention inside this decoder. Three
pre-LN transformer encoder blocks use8heads, FFN2048, GELU and dropout0; final
LayerNorm precedes shared joint Linear256->2, box Linear256->4, existence
Linear256->1. Only final task outputs are supervised; the global token has no
direct readout. Box/joints use sigmoid; joints are normalized image coordinates
and are not calculated from predicted box center/size. Token features can still
exchange information through attention. No new image attention is added.

Diffusion state and pred_latents remain[B,50,256]. The expanded tokens are
readout-local, do not persist between DDIM calls and are not diffusion targets.
The shared task decoder reads both clean/recovered h; DINO continues to predict
from original person h through the unchanged shared DINO head. Clean image
gradients and both h0 detaches remain as V32. No OTA/loss/sampler changes.

Old task MLPs are replaced, not retained unused. New task-head/pose_decoder
initialization is enclosed in fork_rng; shared image/clean/denoiser/DINO tensors
and global RNG match V32. Initial Gaussian latent/DDIM trajectories match V32
exactly; task readout values intentionally differ. V33 checkpoint restoration is
strict and does not need teacher/init assets for inference. Earlier families
retain their defaults and state schema. The readout executes per denoiser call
in the inherited sampler, so inference compute increases with call count.

This tests structured readout capacity relative to V32 with the same seed42/
80epoch recipe and final8000 evaluation. A null result cannot isolate latent
representation failure from optimization, correspondence or sampling failures.

## OTA-supervised recovered outputs (V34)

V34 keeps the V33 model/readout and adds a distinct training pipeline. It calls
existing Dynamic-K OTA once against unique GT people after the recovered forward
pass, using the V29 helper. The same mapping supplies recovered task supervision,
detached clean-person latent targets and recovered DINO targets. Recovered
pose/box/existence losses are gradient-bearing, unlike V29 monitoring-only task
losses. Clean task/DINO retain known constructed correspondence.

Unmatched recovered predictions receive existence0, zero geometry/DINO supervision
and V29 nearest-clean-negative latent targets. Matched candidates can originate
from positive or negative source slots and may target a different person. Existing
OTA topk5/center_radius2.5, positive-assignment normalization, uniform all-slot
latent MSE and task weights remain. Many-to-one matches are permitted; at most
one target per prediction and target coverage repair are retained.

Objective:0.5clean_task+0.5OTA_recovered_task+OTA_latent_MSE+
0.5(DINO_clean+DINO_OTA_recovered). Corruption still uses detached source h0;
latent targets are detached clean h0 selected after OTA (or nearest background),
and therefore need not be the original corruption source. This is an explicit
target-remapping intervention, not unchanged fixed-pair denoising. No labels or
assignment metadata enter transformers or Gaussian inference. Clean-image
gradients, DINO teacher/head and sampler are unchanged.

The V32 auxiliary hook accepts optional denoised_matches, with the original
constructed mapping as its default; old families keep exact behavior. Teacher
regions are computed once per batch and reused with clean/recovered mappings.
Pipeline state adds matcher sigma buffers, but model parameters/state schema
match V33 exactly. Strict full-pipeline restoration and next draws are tested;
model-only evaluation requires no teacher, matcher or initialization checkpoint.

Diagnostics include OTA positive/unmatched counts, GT coverage, Dynamic-K mean,
source reassignment, negative-to-positive conversions, background-bank utilization,
matched/unmatched latent MSE and original-source latent MSE. Recovered confidence
and DINO metrics use OTA labels; they are not directly comparable to fixed-source
conditional metrics. Gaussian AP remains comparable with identical evaluation.

## V33 clipping and latent-weight controls

`benchmark_poseflow_v33_no_clip` retains V33 and sets training.grad_clip=0,
which the native runner passes to Lightning as no clipping. The parallel
`benchmark_poseflow_v33_no_clip_rescale` additionally sets latent_loss_weight=10
instead of1. Model family/state schema, constructed correspondence, both h0
stop-gradient points and task/DINO coefficients stay V33. Fresh seed42/80epoch
runs share the existing TensorBoard; run-local callbacks log accumulated gradient
norms and hypothetical0.1-clipping frequency. See the V33 no-clipping report.

## V35 box-only detector pair

V35-OTA and V35-spatial share box-input MLP4->256->256, original image/latent
stacks, direct linear box/existence heads, task focal2/L1 5/GIoU2 and latentMSE10.
No DINO or joint modules/objectives. Clean known correspondence and recovered
box-only OTA/latent targets remain detached through corruption/targets. Spare
queries split perturbed positive duplicates/foreign boxes; missing sources
fall back to positive/foreign/zero as available. Only spatial globally sorts
ALL proposals via Hungarian, uses explicit canonical slots, and adds denoiser-
only Fourier anchors plus matching cost1. Clipping remains0.1. Bbox evaluation
uses all50queries/noNMS, distinct from historical pose-NMS box AP. Fixed32/final8000
K1/K4 use the existing deterministic protocol. See V35 report for calibration.

V35-spatial-latent-only retains the complete spatial model/recipe but sets
recovered box/existence task weight0. Clean task coefficient0.5 and OTA-matched
latentMSE10 remain; recovered task values are no-grad monitoring only. Shared
heads train only via the clean task; denoiser learns only latent recovery.
OTA still decodes predictions to select targets, so this is not fixed-source
reconstruction. Existing supervised defaults and strict state schema unchanged.

## V36 independent clean box latents

V36-box and V36-box-c branch from V35-spatial-latent-only. The clean path is
pixel cxcywh/100 -> MLP -> non-affine LayerNorm; box-c appends unscaled binary
constructed existence before the MLP. There are no clean transformer blocks,
image features, timestep inputs or spatial query embeddings. Each clean code
is independent of the other proposals. Both clean task heads remain trained;
box-only cannot unambiguously infer existence from identical box geometry alone.
Denoiser, spatial sorting/query embedding, OTA target remapping and latent weight10
are unchanged. Objective is .5 clean box task +10 detached OTA latent MSE.
Only denoising gradients reach image extraction. Labels are clean training inputs
in box-c, never direct denoiser or Gaussian inference inputs. This remains joint
training, not a frozen autoencoder. Separate fresh seed42 runs use identical
shared module initialization, with only an additional input column for box-c.

V36-box-normalized is a fresh control of V36-box with
clean_box_input_format=normalized: clean MLP receives constructed relative
cx/W,cy/H,w/W,h/H directly, with no division by100. These coordinates refer to
resized unpadded image dimensions; padding and uniform resizing do not change
represented geometry. Foreign box inputs retain their relative coordinates;
perturbed centers remain unclamped. No c input, objective/OTA/sampler changes.
Historical V36 configs default to pixels/100. Box-only existence ambiguity and
OTA target remapping remain limitations; improved detection is not assumed.

V36-box-c-normalized applies the same normalized-input correction to V36-box-c:
clean MLP input is[cx/W,cy/H,w/W,h/H,c], with c retained unscaled0/1. No coordinates
or labels are divided by100. The c label belongs exclusively to clean training
encoding; denoising/Gaussian inference receive no direct c input. Objective,
spatial construction/matching, sampling and training recipe remain unchanged.

## V35 timestep-dependent spatial matching cost

The V35-spatial-timed-cost control retains decoded task supervision and the
image-conditioned clean transformer. Only the extra fixed-anchor OTA cost changes:
lambda(t)=lambda_max*max(0,1-t/500), with lambda_max1. It is zero for t>=500 and
reaches1 at t0, computed independently for each image's sampled noise timestep.
It does not follow epoch progress. Existing constant spatial cost is the default
for historical configs. Query conditioning, Hungarian proposal placement, native
OTA candidate penalties, both h0detaches and recovered-task/latent assignment
agreement remain unchanged. Objective.5clean+.5recovered+10latentMSE, clip.1.

V35-spatial-early-cost mirrors the late ramp exactly: g_early(t)=g_late(999-t)
for the1000step schedule and500step ramp. Spatial weight is1at999,.5at749,
zero at t<=499. Its uniform discrete mean.2505 equals the late ramp's, so timing
changes without changing maximum/mean coefficient. All other V35-spatial decoded
supervision/architecture/recipe settings retained. Both schedules remain optional;
constant cost is still the default for historical configs.

V35-spatial-early-cost-q200 is a fresh query-budget control:200 proposals on a10x20 normalized grid, versus50 on5x10. All200 proposals are retained through clean encoding, denoising and readout. The early ramp, objective, optimizer and data recipe stay unchanged. Extra slots retain the same positive-perturbation/foreign-negative split and fallbacks; their increased counts also change task normalization/competition. Bbox evaluation receives all200 predictions, noNMS, with the existing COCO maxDets[1,10,100].


## V37 aligned latent matching

V37 branches from V35-spatial with the same image-conditioned clean encoder,
box-only pixel/100 inputs, 50 proposals, 5x10 slot placement and spatial denoiser
queries. It changes the training pipeline, not the inference model. Mean squared
latent distance defines matching, denoising and clean-space alignment. All padded
real/fake clean codes are candidate targets. Dynamic-K uses exp(-distance),
top5 sum/floor/min1; conflict resolution and nearest fill assign every prediction.
Coverage repair protects one prediction per unique real person, not per padded
slot. Positive duplicates/fakes can be reused or unselected. No decoded geometry,
class score, anchor cost or candidate penalty enters matching.

Clean reconstruction retains constructed correspondence. The single assignment
controls recovered box/existence targets and detached clean latent targets.
Objective: .5clean_task + .5recovered_task + 10latent_MSE + alignment. Clean-clean
pairwise sigmoid alignment uses normalized position/size similarities
exp(-squared_distance/.25^2) and existence equality, weighted1/1/4. Geometry is
GT for positives and constructed input for fakes. Real-real/fake-fake/mixed
pair groups are averaged separately, then available groups/images equally.
Three trainable scalar scales/biases calibrate the same distance during training;
matching does not execute these calibrations. Clean codes retain gradients for
alignment while corruption and latent-target selection remain detached. This
combined intervention tests useful latent geometry, not an exact coordinate
metric or isolated causal effect. See docs/reports/poseflow_v37_latent_ota.md.


V37-reweight is a config-only fresh control of V37: alignment property weights
1:1:1 instead of1:1:4 and latent MSE100 instead of10. Clean/recovered decoded
task weights remain0.5 each, alignment overall weight1, clipping0.1. Matching,
geometry targets, both detach boundaries and model/sampler stay unchanged. Compare
raw component losses and matched-age Gaussian geometry/AP; weighted total loss
is intentionally not comparable. The combined intervention does not isolate
existence weighting from stronger latent supervision.

## V38 variational clean representation

V38 inherits the V37-reweight recipe and replaces explicit pairwise alignment
with a diagonal Gaussian clean posterior. Two linear heads follow the existing
clean LayerNorm: identity-initialized mean and zero-weight logvariance with
bias -4, clamped to [-10,2]. The mean receives no further normalization. Sampled
codes train clean reconstruction; detached means supply corruption and all-target
latent OTA matching. The deterministic denoiser predicts means. Objective:
0.5 sampled-clean task +0.5 recovered task +100 matched latent MSE +0.1 prior KL.
KL averages over batch, all slots and latent dimensions. Box geometry remains
real-only; existence and KL cover every padded real/fake slot. No prior KL is
applied to recovered latents. Both clean posterior heads and clean stack are
detached from recovered supervision; the shared image features still learn
through both branches. Training is joint from scratch with the same image import.

A separate checkpointed CPU generator (seed1042) supplies explicit posterior
noise. Validation uses seed1042+batch index without changing training state.
Original proposal, timestep and diffusion-noise draws are preserved. Inference
uses the unchanged [B,50,256] Gaussian/DDIM path and never executes clean heads.
This regularizes the clean representation while diffusing means, rather than
reproducing LDM sampled-code diffusion. KL alone does not establish useful box
neighborhoods. Fake geometry remains unsupervised and may collapse. Compare
matched-age Gaussian AP/coverage and recovered box geometry against V37-reweight;
better KL or clean reconstruction alone does not demonstrate detection progress.


## V39 one-to-one recovery

`benchmark_poseflow_v39` retains the nonspatial V35-OTA box-only model
and recipe. Each valid person appears once; spare slots borrow foreign-image
boxes or use random valid normalized boxes if donors are absent. A random joint
permutation of proposals and metadata removes fixed positive-slot ordering.
Source labels stay fixed through corruption and recovery; no matching is used.
Foreign/random boxes can overlap real people, so negative-to-GT overlap is
logged rather than assumed absent. Source labels describe provenance.

The objective is `0.5 clean_task + 0.5 recovered_task + 10 positive_latent_MSE`.
Both task losses use the original same-slot targets: all-slot focal classification
(coefficient2), positive-only normalized-box L1 (5) and GIoU (2). Latent MSE
averages over positive slots and256 channels only, with detached clean targets.
All-fake images have a connected zero latent/geometry loss and retain
classification. Fake states can receive indirect gradients through cross-slot
attention. Clean encoding remains image conditioned and jointly trained.

The classifier reads image-conditioned latents; labels are never model inputs.
Focal scores are not asserted to be calibrated probabilities. Heavy corruption
can make original identity/class ambiguous; unique training targets do not
guarantee duplicate-free inference. Masking fake latent targets also means this
is not an established full-scene score-matching objective. Fixed32 Gaussian
K1/K4 bbox AP, coverage and excess IoU hits are monitoring; final8000 is separate.


### V39-denoiser-grad

This fresh V39 subversion removes only the source detach before corruption:
`ht = sqrt(alpha_bar_t) * h0 + sqrt(1-alpha_bar_t) * noise`, retaining a live
`h0` in autograd. The positive-only latent target remains `stopgrad(h0)`. Both
clean and recovered decoded box/class losses retain their V39 weights. Fake
slots still have no direct latent/geometry reconstruction; source gradients
from classification and cross-slot interactions can now train their encoder.

The optional `detach_corruption_source` setting defaults true for V39 and all
historical model callers. The subversion sets it false, is initialized fresh
with the same seed42 parameters/draws and has a separate namespace. No sampler
or model-state changes occur. Gradients through this source route are scaled
by sqrt(alpha_bar_t), so they weaken at high noise. Decoded supervision opposes
collapse without guaranteeing useful latent separation. Real/fake latent
spread and readout norms accompany source recovery and Gaussian AP diagnostics.


## V40 reference-conditioned box diffusion

V40 branches from V35-OTA and adds the ED-Pose image encoder and encoder proposal
heads, stopping before the native decoder. Four projected image scales feed three
deformable encoder layers initialized from ED-Pose layers2/4/6. Native grid anchors
and encoder box/class heads produce candidates; native max-class ranking selects200
valid image positions. A single Hungarian assignment precedes latent corruption.
Training keeps all matched references and fills to50 with highest-scoring unmatched
candidates, ordered by score. Inference takes top50. Proposal focal supervision
covers both native class channels; matched candidates additionally receive L1/GIoU.

The V35 box codec/readout and dense full-image attention remain. Clean inputs are
matched GT pixelcxcywh/100 or zero placeholders. Each denoiser block adds its own
reference feature plus embedded normalized reference box to timestep conditioning
before modulation. References remain live in autograd; assignment/selection indices
are discrete. Corruption sources and latent targets stay detached. Geometry and
latent MSE cover matched slots only; all50 clean/recovered existence outputs are
supervised. No constructed duplicates, foreign negatives or post-denoising OTA.

Objective: proposal_task +0.5clean_task +0.5recovered_task +10positive_latent_MSE,
with focal/L1/GIoU coefficients2/5/2. Shared heads still predict denoised boxes and
existence; final scores do not multiply proposal scores. Proposal extraction is
independent of diffusion time and cached within each Gaussian-start sampling call.
The frontend is evaluated once per image; references stay fixed over DDIM calls.

V40 has separate reference_candidates200 and latent num_proposals50 settings.
Historical configuration defaults/state schemas are unchanged. Strict V40 model
and pipeline restoration needs no ED-Pose checkpoint; fresh training audits all
backbone/projection/encoder/proposal-head initialization tensors. Proposal coverage,
matched ranks, background scores, noise-binned recovery and final Gaussian bbox AP
are separate diagnostics. Conditional recovery or pretrained proposal quality alone
is not evidence of successful latent detection.

V40 verification/launch:31focused checks and native CUDA preflight pass; fresh
seed42/80epoch run `/storage/outputs/pose-graph-diff/results/poseflow_v40/20261101_235355` is active. See the V40 report for provenance;
first-epoch Gaussian AP is pending.


## V40-EMA clean-code targets

V40-EMA uses the unchanged online V40 model and adds a pipeline-only EMA clean
encoder. The teacher includes backbone/position extraction, input projections,
level embeddings, box-input MLP, clean transformer blocks, clean LayerNorm and
the shared time embedding evaluated at time0. It excludes proposal modules,
denoiser and task heads. It initializes from the online tensors after ED-Pose
initialization, stays eval/no_grad/requires_gradFalse and is excluded from AdamW.

Decay0.999 is applied once AFTER each optimizer update, including accumulation2;
validation does not update the teacher. Floating tensors use EMA; nonfloating
buffers copy their online values. Teacher weights and the EMA update counter
are full-pipeline checkpoint state. Model-only Gaussian inference restores the
unchanged online V40 model and requires neither teacher nor initialization assets.

The same detached teacher code supplies forward corruption and positive-only
latent MSE. Online clean reconstruction and recovered box/existence objectives
retain all V40 coefficients/masks. The clean online branch remains trainable.
No teacher task heads or additional loss are introduced. Diagnostic teacher-code
reconstruction uses the live shared readout without gradients, exposing possible
teacher/readout lag. Existing noise-bin latent recovery now uses the actual
teacher target, rather than silently comparing against online clean codes.

Diagnostics include positive/background online-to-teacher MSE/cosine/mean/RMS,
teacher-code live-readout losses, update count and decay. EMA slows target motion;
it does not establish correct reverse marginals or solve background trajectory
behavior by construction. Judge matched-age raw K4 bbox AP and the K4-K1 gap.
SigReg is deferred. `benchmark_poseflow_v40_ema` is the separate experiment preset.

V40-EMA verification/launch:37focused tests and native CUDA/teacher cadence/strict
restore checks pass. Freshseed42/80epoch run `/storage/outputs/pose-graph-diff/results/poseflow_v40_ema/20261102_112758` is active; teacher updates
match optimizer steps. See V40-EMA report. FirstepochAP pending; SigReg deferred.


## V40-SigReg single clean encoder

`benchmark_poseflow_v40_sigreg` replaces the EMA teacher with one trainable clean
encoder. Final clean LayerNorm becomes identity; internal norms remain. Clean
codes stay live through corruption, while matched latent-regression targets stay
detached. The original V40 objective gains0.05times SIGReg on matched clean codes
only. Default256 unit projections/17knots[0,3], symmetric weighted quadrature and
matched-count scaling follow pinned LeJEPA code. Background/recovered codes are
excluded; fewer than2matches and weight0 skip. Independent projection RNG restores
with the pipeline; validation/inference do not advance it. There is no teacher.

V40 proposals200/references50, assignment, losses/masks and cached-reference DDIM
remain. Model-only restoration uses the saved SigReg config to restore identity
clean output normalization; no initialization checkpoint is needed. Shared initial
parameter tensors/RNG match V40, not initial forward codes. Historical paths retain
their source detach and final LayerNorm. See `docs/reports/poseflow_v40_sigreg.md`
for implementation/provenance/verification and the approved one-run launch.

Gaussianity does not guarantee semantic stability or correct reverse trajectories.
Monitor matched-age K1/K4AP, background ranking, clean/recovered statistics and
fixed held-out per-person clean-code movement. No pure SIGReg causal claim is
possible against EMA alone because normalization and gradient routing also change.

V40-SigReg launch verified: `/storage/outputs/pose-graph-diff/results/poseflow_v40_sigreg/20261103_123752`, fresh seed42/80epochs, 380 finite optimizer updates at verification;45focused checks and native CUDA/restore/probe checks pass.607source/doc+5input hashes verified; existing dashboard history/pins preserved. AP pending; see the V40-SigReg report.

2026-11-03 V40-SigReg learned outcome FAILED: fixed32 K1/K4 AP0 and IoU0.5 coverage0 at epochs1/2; fixed held-out clean RMS .849->14.357->246.464->2136.947; run stopped duringepoch3 on nonfinite gradient-norm check(lastlogged4940). Predictions cover correct32images/1600outputs; frozen source matches current. SIGReg weighted penalty remains order1 while MSE explodes. Frozen target-gradient counterfactual removes outward latent component at high noise, but parameter-Jacobian check does not prove target detach alone caused AdamW scale growth. Production training rules unchanged; report docs/reports/poseflow_v40_sigreg_failure.md; propose SigReg-only live MSE target and bounded2epoch pilot, awaiting review.


## V40-SigReg live-target correction (two-epoch pilot)

The corrected preset explicitly sets detach_latent_target=false, restoring
latent-regression gradients through the online clean target as well as the
corruption source. Missing flags default true for historical checkpoint/config
restoration. All other families retain detached targets. Objective values,
normalization, SIGReg weight and sampling are unchanged at fixed weights.
See docs/reports/poseflow_v40_sigreg_live_target.md for tests and the approved
bounded2epoch pilot; no automatic80epoch launch.


2026-11-03 approved live-target pilot completed: fresh seed 42, two epochs,
3,000 optimizer updates, with the original 80-epoch scheduler horizon. No scale
or numerical guard triggered. Fixed32 bbox K1/K4 AP: epoch 0 = 8.35/5.98,
epoch 1 = 13.15/5.34. Held-out 17-person clean RMS = 0.405/0.492 (original
14.357/246.464); final mean coordinate variance 0.1601. Epoch-1 raw IoU >= 0.5
coverage = 101/133 (K1), 104/133 (K4). Final checkpoint/optimizer finite and at
step 3000, hashes intact, trainer exited. This avoids the observed early
scale/zero-AP failure over two epochs; K4 still trails K1. Pilot artifacts:
`tmp/v40_sigreg/live_target/pilot_20261103_144558`; see
`docs/reports/poseflow_v40_sigreg_live_target.md`. No further training launched.


2026-11-03 15:22 Europe/Stockholm: user authorized the fresh seed42/80epoch live-target
SigReg run after pilot review. Launched `/storage/outputs/pose-graph-diff/results/poseflow_v40_sigreg/20261103_151941` with frozen source/configuration,
unchanged pilot objective/data/optimizer, clean-scale/numerical guard and fixed32
K1/K4 each epoch plus full8000 afterepoch79. Startup finite at step290;
609 source + 5 input hashes verified, 20 dashboard streams preserve history/pins.
This is launch health; learned results are pending. See
`docs/reports/poseflow_v40_sigreg_live_target_run.md`.


## V41 coordinate box diffusion with retained proposals

V41 keeps V40's complete ED-Pose-initialized proposal front end (top200 candidates,
top50 scored reference boxes/features), with live references fixed across reverse
steps. Both train and inference select by score. Proposal Hungarian supervision
is separate from final predicted-box Hungarian matching; the latter chooses the
box and live clean-embedding target together. Shared clean/noisy coordinate
encoding uses three full-image blocks; the reference/timestep-conditioned
predictor has three blocks. State is normalized cxcywh mapped by z=6*(2*b-1),
with uniform t0..999 coordinate Gaussian corruption and coordinate DDIM eta0.
Embeddings are encoded anew each step, not noised or carried. Objective is
proposal +.5clean +.5denoised +10matched-JEPA +.05unique-clean SIGReg. All-slot
classification, matched geometry/JEPA and existing SIGReg RNG/masks remain.
Use `benchmark_poseflow_v41`; see `docs/reports/poseflow_v41.md` for interfaces,
verification and launch provenance. Previous paths/experiments remain intact.


2026-11-03 18:18 Europe/Stockholm: V41 implementation and authorized fresh seed42/80epoch
launch verified at `/storage/outputs/pose-graph-diff/results/poseflow_v41/20261103_181429`. Proposals retained (200candidates/50score-selected
references), coordinate Gaussian state, shared-view encoder, output Hungarian
and live JEPA/SIGReg. 65distinct focused checks and native FP32CUDA strict
restore/global RNG/next-update/K1K4 verification pass;617source+5input hashes
verified. Live step420 finite, cleanRMS0.2945;21dashboard streams
preserve history/pins. Fixed32K1K4 each epoch and full8000 afterepoch79 configured.
First evaluated AP pending; see `docs/reports/poseflow_v41.md`.


## V42 whole-pose coordinate-diffusion baseline

V42 preserves the original DiffusionRegPose coordinate corruption/DDIM, image
proposals, scored person selection, box refinement/detach and original
Hungarian task/auxiliary/proposal/DN losses. At the former expansion boundary,
a whole-pose MLP encodes noisy XY into one person vector. Two pose-refinement
layers use current box references and a learned absolute full-pose reader,
without joint query expansion. Each person gets60 sampling points per head/level
in place of15tokens x4; DN retains4. The native timestep-zero training rule is
unchanged. JEPA remains deferred. Use `benchmark_poseflow_v42` (3E/3D EDPose2/4/6,
256/427,200/50queries,DN50,FP32,batch4accum2,seed42/80epochs).
Fixed32 test-subset pose/box AP runs after each epoch at K1/K4/K8 with paired
Gaussian sources; full8000 afterepoch79 is configured by the launch wrapper.
See `docs/reports/poseflow_v42.md` for contracts, transfer and verification.

2026-11-03 20:35 Europe/Stockholm: V42 fresh80epoch launch verified at `/storage/outputs/pose-graph-diff/results/poseflow_v42/20261103_203019`; live step300, finite. 102focused checks/nativeCUDA restore+nextupdate passed; six K1/K4/K8 pose/box pins verified in dedicated comparison dashboard. Initialization measurements are separate from trained AP, which is pending. See V42 report.


V42 approved repair: the current preset explicitly uses
`pose_readout_mode: box_center_residual`, decoding full-pose residuals against
the incoming current box center. Old configs default to `absolute` for correct
historical restore. No joint expansion or loss/noise change. Initial assignment
quality improves with identical weights. After the completed two-epoch diagnostic,
the user authorized continuation through the original80-epoch schedule. Training
resumes the repaired checkpoint with saved optimizer/scheduler state, without
another pilot cutoff. See `docs/reports/poseflow_v42.md`.

2026-11-03 22:09 Europe/Stockholm: V42 box-center reader pilot completedtwoepochs/3000updates andstopped. Onsame-agefixed32/seed42/256427, K1poseAP0.013->20.094 andboxAP0.490->26.120; K4/K8poseAP16.453/15.229. 18focused/nativeCUDA/restore checks pass,624source+5inputhashesverified. This is subsetevidence; morecalls currentlydegradeAP, finalfulltestunmeasured, JEPAdeferred. Pilotstoppedpendingreviewbeforelongertraining; seeV42report forcomparison/checkpoints andremainingstagegap.

2026-11-03 23:58 CET: V42 full training continuation RUNNING, PID497679, tmux poseflow-v42. Resumed preserved repaired epoch001/step3000 with strict native model/AdamW/scheduler restore verified; live epoch2/step3050, finite loss/gradients. Total80epochs, same frozen source/initialization/data/loss/noise/optimizer. No further pilot cutoff or review gate. Fixed32 K1/K4/K8 after every epoch and full8000 final evaluation retained. Existing TensorBoard33331 indexes V42-repair-box-center/resume; historical streams and sixpins preserved. Pilot endpoint/status/manifest/config remain preserved; continuation.json and continuation.log record the launch.


## V42 targets matched from noisy inputs

`benchmark_poseflow_v42_matched_target` reuses repaired V42 unchanged at inference.
During training, Hungarian matches actual noised/clamped input poses to clean
people using native masked L1/OKS geometry costs once. Complete-person targets
remain fixed for final and all post-selection pose-layer losses; earlier box,
proposal and DN losses retain native correspondence. Unmatched slots receive
classification only. Cropped-out diffusion sources are not supervised targets.
The config records `target_assignment: noisy_input_hungarian`; legacy criterion
and V42 defaults remain unchanged. Fresh seed42/80epochs, same data/optimizer/
noising and fixed32K1/K4/K8/full8000 evaluation, with V42 preserved as control.
See [the matched-target report](reports/poseflow_v42_matched_target.md).

2026-11-04 01:33 Europe/Stockholm: V42-matched-target full80epoch training RUNNING, PID552828, tmux poseflow-v42-matched-target, root /storage/outputs/pose-graph-diff/results/poseflow_v42_matched_target/20261104_012819.628frozen source+5input hashes verified by worker; initialmodelhash exactly matches V42. Actual live epoch0/step300, finite loss/gradients. Current repaired V42 control PID497679 continues at epoch8/step12750. No pilot cutoff, no cancellation, no JEPA.108tests/nativeCUDA model+optimizer+scheduler restore/nextupdate1.28e-7 PASS; legacyoldcriterion36lossfields/24costcases/RNG bitwise exact with matched configuredweights. Fixed32K1/K4/K8 each epoch/full8000 at epoch79; first learnedAP pending epoch0 completion. Dashboard33331 configured42roots and sixsavedpins, indexing verification pending. Retainplanpendingoutcome review; nextsmallstep is same-age AP/assignment comparison, no extra approval gate.


V42 comparison TensorBoard access (2026-11-04): TensorBoard binds only
127.0.0.1:33331. A persistent private Tailscale TCP forward owns the Tailscale
address on port33331 and forwards to that loopback listener, alongside the
existing8501 service. Do not bind TensorBoard to0.0.0.0 on the same port:
that prevents the Tailscale listener from binding (confirmed in daemon logs). Access is available at http://100.122.159.115:33331/ or
http://lynx.tail481642.ts.net:33331/ from connected tailnet devices.
Canonical six-pin links are saved in tmp/v42/dashboard/tailscale_url.txt
and url.txt. Current V42 charts use steps1/4/8 and pose_AP/bbox_AP tags;
older bbox/AP and steps2/3 links do not select the same comparison.
Verification is host/API/browser-side; remote client connectivity is separate.


Current V42 dashboard access: user requested direct binding to `0.0.0.0:33332`
on2026-11-04, replacing the33331loopback/Serve setup above. Canonical link files
now use port33332; the33331Serve route was removed. Existing8501 is preserved.
Use http://100.122.159.115:33332/ or http://lynx.tail481642.ts.net:33332/.


## Corrected V42 box-center training distribution

`benchmark_poseflow_v42_uniform_t` is the corrected whole-pose baseline. Each
image independently samples an integer timestep uniformly from0..999; native
coordinate q_sample corrupts GT/padded coordinate states at that timestep before
the whole-pose encoder. The same actual timestep conditions the model. Existing
output-Hungarian class/box/keypoint/OKS/auxiliary/proposal/DN losses, box-center
reader, 60-point person attention and coordinate DDIM are unchanged. No JEPA or
input-Hungarian targets are used. The new run starts fresh from EDPose with the
same seed42/80epoch/batch4accum2/FP32/clip.1/data/resize recipe.

Historical V42 configs without `training_timestep_policy` default to
`timestep_zero` for reproducible restoration; the corrected config explicitly
sets `uniform`. The target builder's optional `[B]` long `timesteps` argument
validates shape/device/range; omission preserves exact legacy noise/RNG behavior.
Training logs timestep mean/min/max, and the frozen launch wrapper records an
actual-forward timestep histogram. Fixed32 K1/K4/K8 and final8000 evaluation are
retained. Startup checks establish implementation, not learned AP improvement.
The failed matched-target job was cancelled by user instruction and its checkpoint,
source, logs and events are preserved. See the V42 report for live provenance.


2026-11-04: Corrected V42 box-center uniform-t full80epoch training RUNNING, PID666506, tmux poseflow-v42-uniform-t, root /storage/outputs/pose-graph-diff/results/poseflow_v42_uniform_t/20261104_133538. Fresh EDPose/seed42 initialization exact prior box-center hash; no t0 checkpoint resumed. Native coordinate corruption samples per-image integer t0..999 and same actual t conditions model; ordinary output-Hungarian, no JEPA. Live epoch0/step200, finite loss/gradients; observed 1600 images/790 distinct timesteps, range1..998, all10 time bins sampled.107tests/nativeCUDA/strictmodel-AdamW-scheduler restore PASS.631frozen source+5input hashes worker/audit verified. Failed matched-target PID552828 cancelled with interrupted checkpoint preserved; old V42 control PID497679 remains running. TensorBoard0.0.0.0:33332 preserves prior42roots/sixpins and adds V42-box-center-uniform-t (43roots/47streams); scalar history reload ongoing. Fixed32 K1/K4/K8 each epoch/full8000 afterepoch79 retained, no pilot cutoff. Evidence launch_verification.json and local tmp/v42_uniform_t; first learned AP pending. Next follow-up: epoch-end sampled AP and actual-t histogram, without another launch gate.


V42 correspondence audit (2026-11-04): selected image proposals and GT-derived
pose sources are independently ordered, with final output-Hungarian matching.
Training does not establish proposal/source/target identity correspondence.
V40 had explicit proposal-matched source construction with fixed denoising
supervision; V41's final box/JEPA matching is a different guarantee. See
[the code-backed audit](reports/poseflow_v42_correspondence_audit.md) for measured
mismatch, frozen oracle reordering, predecessor differences and limitations.
No architecture or training change follows from this diagnostic alone.

V42 paired-proposal experiment (`benchmark_poseflow_v42_paired_proposal`): match
actual selected image proposal references/class logits to valid complete people
before constructing the coordinate diffusion sources. Matched slots use their
assigned GT pose and keep that target for final and pose-auxiliary losses.
Independent per-image uniform t0..999 is required. Unmatched Gaussian padding
stays classification background; the independent diffusion source bank is not
used for positive slots. Score-top50 selection is identical at train/inference;
V40's forced matched-candidate inclusion is excluded. More valid people than
selected capacity fails explicitly. No inference GT or JEPA; native Gaussian
DDIM, data, optimizer and full80 schedule remain. Correct source/target identity
is an implementation invariant; useful state dependence and AP benefit require
trained comparisons. See [the paired-proposal report](reports/poseflow_v42_paired_proposal.md).


2026-11-04T20:40+01:00: V42 paired-proposal full80 RUNNING, PID758206, tmux poseflow-v42-paired-proposal, root /storage/outputs/pose-graph-diff/results/poseflow_v42_paired_proposal/20261104_203731. Live epoch0/step100, finite loss/gradients;2751 paired positive sources checked with zero source/target mismatch,400 actual timesteps spanning5..995/all10bins. Fresh initial weights match control;129tests/nativeCUDA/strictrestore/nextupdate checks PASS.638source+5input hashes verified. TensorBoard0.0.0.0:33332 adds distinct stream, preserves43previousroots/sixpins; indexing ongoing. Full val/fixed32K1/K4/K8 each epoch and final8000 afterepoch79; no pilot cutoff. Prior controls preserved. Learned state-use/AP effect pending. Report docs/reports/poseflow_v42_paired_proposal.md.


Final launch/dashboard verification: PID758206 is active at epoch0/step200,
finite loss/gradients. Live pairing audit now checks5403 positive sources
across400 microbatches with zero identity mismatches. Actual timestep
range0..999, distinct values808, all10 bins sampled.
TensorBoard fully indexed48 scalar streams/44 roots; `V42-paired-proposal/.`
contains measured training loss through step199. All six saved K1/K4/K8
pose/box AP pins have existing scalar measurements, with all historical roots
preserved. The new run's validation/AP arrive at epoch end. Direct host-side HTTP
to100.122.159.115:33332 passes; a phone-side connectivity check is separate.
Full80 training continues without another pilot gate. Next follow-up is trained
state-dependence/same-age AP assessment. Plan retained pending user outcome review.

## Standalone image-free pose-and-box VAE

`benchmark_pose_vae` trains an independent per-person geometry codec on CrowdPose
train only, with disjoint val and a single final best-checkpoint test evaluation.
No images, pretrained backbone, existence/background supervision, metric-learning
objective or PoseFlow integration are used. Its public interface is explicitly
pixel-space joints and cxcywh boxes plus image WH metadata. Internally joints
are offsets from the supplied box center divided by max(pixel width,height),
with separate image-relative center/log-size placement and observation masks.
The decoder recomposes absolute joints using its predicted box, never GT boxes
or joint extrema. Natural missing joints are unsupervised; synthetically masked
labeled joints remain reconstruction targets. Occluded labeled joints are kept.

The 574,560-parameter residual VAE uses a 32-D Gaussian latent, sampled clean-target
reconstruction, and summed-latent KL warm-up. Seed 42/batch 512/50 epochs, AdamW and
cosine decay are recorded in its own Hydra preset. Private validation draws are
fixed across epochs, independent of checkpointed training augmentation/posterior
RNG. Geometry errors, IoU, latent diagnostics and prior/reconstruction grids are
codec evidence, not detector AP. See docs/reports/pose_vae.md for full contracts.

2026-11-04: First pose VAE 50-epoch/3,500-update run completed; selected epoch49
from disjoint validation, then scored the full test split once. Clean/corruption-
mixture test joint errors .04103/.05333 box-scale units, box IoU .92878/.86186.
Full-val baselines show improved masked-joint completion relative to a training-
mean skeleton, but reduced observed-joint precision versus supplied jittered
coordinates. Final Gaussian-prior grids remain malformed despite 32 active latent
dimensions. This is a trained codec with measured fidelity/generation limits;
no detector-quality or useful unconditional-prior claim follows. See the report.

## Fidelity-first deterministic codec

`benchmark_pose_autoencoder` is a separate clean geometry autoencoder, retaining
width 256/latent 32 and reversible geometry conventions. Clean targets plus flips
train all geometry through the latent; posterior sampling/KL, coordinate noise
and synthetic joint masks are absent. Its OKS-denominator pose loss and per-axis
box-center/log-size/GIoU losses target geometric fidelity. Native codec AP uses
known people, constant scores and no NMS; paired metrics track source identity.
The full-val clean pose and bbox AP must both reach 97.5 before decoder-only latent
robustness training. The encoder and training-derived noise scale remain fixed
in that phase. Acceptance also requires improved three-seed 5%-noise pose and box
AP without losing either clean gate. Test is untouched during this experiment.
This is not detector AP or demonstrated robustness to an actual student. See
`docs/reports/pose_autoencoder.md` for the complete recipe and measured outcome.

2026-11-04: First deterministic control completed 100 epochs/7,000 updates.
Train pose/box AP 93.808/96.954 and validation 92.468/96.608 both miss the clean
97.5 gate. Robustness training was correctly skipped; no candidate is accepted
for integration. Three-seed 5%-latent-noise validation AP is 78.641/86.723.
The training-set shortfall means a purely generalization-based explanation is
insufficient; the cause is not yet isolated. Test remains untouched. Full
provenance, curves and paired error tails are in the autoencoder report.

The unchanged-code lower-rate refinement completed another 100 clean epochs:
train pose/box AP 96.988/98.360, validation 96.049/97.915. Box fidelity passes,
pose fidelity still fails; no robustness phase or integration checkpoint was
accepted. Three-seed 5%-latent-noise validation AP is 80.328/87.460. The user
deferred the proposed relative-joint objective until review of these results;
it has not been implemented. Test remains untouched.

The user subsequently accepted the current clean fidelity and approved an
eight-arm warm-started VAE comparison. Deterministic, sampled/no-KL and two
training-calibrated KL strengths each run with/without synthetic joint masking.
All sampled arms jointly optimize encoder, posterior and decoder with the AE's
geometry objectives; no separate decoder-only robustness phase is used. Exact
mean-output transfer is verified. Mean/sample AP, hidden-joint recovery, latent
noise sensitivity and variance collapse are measured separately. Test remains
untouched. Recipe and current status: `docs/reports/pose_vae_fidelity.md`.

All eight VAE comparison arms completed100epochs and final assessments. The
high-KL unmasked VAE measured97.208/98.383sampled pose/box AP, versus
97.399/98.536for the matched deterministic control. At5%latent noise the pair is
82.743/87.998 versus80.662/87.685: a modest gain with substantial sensitivity
remaining. Its posterior std contracts to .638% of aggregate-mean std; prior
inspection remains poor. Mask training halves hidden-joint median error at30%
hiding (.1462→.0707box scale) while clean sampled pose AP falls to89.761.
These are measured tradeoffs, not a completed robust/generative codec. Test and
detector integration remain untouched. The fidelity preset enables independent
latest-checkpoint saving after an audit found two original last files lagged;
two exact-initialization artifact replays are documented in the report.
Both replays completed with exact selected-model and full validation-history
parity; their final native checkpoints retain all7,000updates.

V42 proposal-to-incoming-pose inference is an opt-in evaluation policy, separate
from paired-proposal GT training. At every call it matches current reference
boxes to enclosing boxes of incoming normalized joints using the configured
native Hungarian matcher with box L1/GIoU. All state values/slots remain; the
class term cancels for a complete same-class assignment. Coordinates are not
moved by matching. No GT or visibility labels enter inference. High-noise joint
boxes are ambiguous; this policy deliberately changes inference association
without asserting training consistency or improvement. Default/native sampling
and active trainers remain unchanged. See the proposal-pose inference report.

The2026-11-05 fixed32 comparison at paired checkpoints3/6 found lower K1/K4/K8 AP with this policy; native sampling remains default. See [the inference comparison](reports/poseflow_v42_proposal_pose_inference.md) for exact replay, permutations and limitations.


V42 noisy-proposal-paired (`benchmark_poseflow_v42_noisy_proposal`, approved
2026-11-05) retains the paired-control clean source bank/noising, then applies
`match_proposals_to_noisy_poses` to actual corrupted states inside the selection
callback. Complete source labels/background provenance follow the inverse
permutation for fixed final/pose-auxiliary losses. Uniform per-image0..999 times
condition both corruption and model. At high noise source identity may be
uninferable; this is not assumed to repair AP. Existing model parameters,
pre-selection losses and historical defaults are preserved.

The new pipeline supplies an optional sampler/description through the V42
evaluation hook; quick_eval_v42 also accepts these explicitly for full evaluation.
Its K1/K4/K8 AP uses matched Gaussian DDIM at every call with consistent encoding/
residual order. Training/validation losses use the new source-following pairing;
no inference GT or visibility enters matching. Seven new/146 regression tests and
native CUDA restore/update/finite-sampling checks pass; learned results pending.
Full80 fresh seed42 and a separate dashboard stream preserve controls. See
[the experiment report](reports/poseflow_v42_noisy_proposal.md).

Periodic validation AP also uses the matching sampler through `_sample_validation_outputs`; the inherited native default is retained for older pipelines. A failing regression exposed the missing hook, and the real validation-step route is covered. The first startup snapshot was superseded before its first epoch; the full experiment restarts fresh with this hook.


2026-11-05T09:02+01:00: V42 noisy-proposal-paired corrected fresh full80 RUNNING, PID869864,
root `/storage/outputs/pose-graph-diff/results/poseflow_v42_noisy_proposal/20261105_085659`.677source+5input hashes verified;146tests/nativeCUDA/legacy evaluator parity PASS.
Live step300, finite loss;8132 source-label checks/zero mismatch and
actual t0..999 coverage. TensorBoard0.0.0.0:33332 fully indexed49streams/45roots,
new measured loss verified, original44roots/sixpins preserved. Validation loss
and matched K1/K4/K8 AP pending first epoch; full8000 after79. No pilot cutoff.
See [the report](reports/poseflow_v42_noisy_proposal.md) for snapshot and audit details.


## V43 frozen-codec LDM implementation

V43 uses the selected unmasked High-KL pose-and-box VAE as a frozen32D first stage,
with the shared V40/V40-EMA inference architecture and fresh initialization.
The codec adapter, full trainval augmentation audit, image-only confidence,
fixed selected slots, latent denoiser and native checkpoint/evaluator paths are
implemented. Sixty-two focused checks and real FP32 CUDA/strict-restore checks pass;
the full80-epoch detector training run launched on2026-11-05 with seed42.
Live TensorBoard label: `V43-frozen-HighKL-VAE-live` on port33332.
Checkpoints are on the home filesystem to reserve sufficient disk capacity;
known-person codec AP remains separate from pending detector AP.
Historical trainval/test and crop augmentation were explicitly selected. See
[the V43 report](reports/poseflow_v43_frozen_vae.md) for codec identity, audit scope
and reconstruction results; these are not learned detector-performance evidence.


2026-11-05: frozen V42 image-confidence diagnostic improves poseAP at both paired
checkpoints while keeping raw geometry/native DDIM unchanged. Epoch31/48000
fixed32 K1/K4/K8 native40.073/13.038/10.207 ->pre-pose scores42.893/35.525/35.274;
epoch3 also improves. Exact raw-selection query indices gather normalized trained
box-classifier logits before pose input; scores remain image/time-conditioned.
638frozen sources/checkpoint, native metrics/weights/RNG and raw geometry verified.
This is score-ranking recovery, not retrained/full-test or improved geometry.
Failed noisy-proposal PID869864 cancelled by user approval, checkpoint/history
preserved. See [the diagnostic report](reports/poseflow_v42_image_confidence.md).


V42 image-only confidence training (`benchmark_poseflow_v42_image_confidence`)
adds an opt-in514-parameter classifier cloned from the box-stage head. It reads
already-normalized selected image/time features BEFORE the source-builder/pose
injection and supplies identical cached logits to final/pose auxiliaries. Those
classification losses use the existing fixed paired50 GT map, independently of
the native200-candidate classifier. Main classification gradients reach the image
frontend and new head; pose modules retain geometry gradients. Paired sources,
uniform0..999 times, native proposal/DDIM/geometry/DN paths remain unchanged.
Legacy mode outputs/state keys/RNG and initial common geometry/tensors are exact;
new strict checkpoint mode adds only head weight/bias. Seven new/153 regressions
and native CUDA restore/update checks pass; fresh full80 launch follows. Trained
state-use/AP benefit is unproven. See
[the training report](reports/poseflow_v42_image_confidence_training.md).


2026-11-05T14:33+01:00: approved V42 paired image-only-confidence full80 RUNNING, PID912787,
root `/storage/outputs/pose-graph-diff/results/poseflow_v42_image_confidence/20261105_142646`.153tests/legacy parity/native CUDA restore checks pass;701source+5input
hashes verified. Live step200, finite loss, actual t0..999 and
5403source-label checks/zero mismatch. New head receives gradients before pose
injection. TensorBoard0.0.0.0:33332 fully indexed52streams/48roots, new loss and
sixpins verified,47prior roots preserved. Validation loss/AP after first epoch,
full8000 after79; no pilot cutoff. Trained performance remains unmeasured.

V43's approved small-noise control compares absolute-x0 with identity-preserving
velocity recovery, equally sampled1/2.5/5% standardized latent noise, using fixed
posterior means and cached16TRAIN/16VAL contexts. Frozen components have prior
trainval exposure; these are conditional recovery metrics, not independent detector
AP. The bounded2000-update/arm trial retains only best/last and gates wider training.
See [the control report](reports/poseflow_v43_small_noise.md).
The completed4000-update small-noise comparison preserves VAL input quality with
velocity, but neither arm meets the foreground recovery gate. The10-epoch extension
remains unstarted. Detailed paired/noise curves and post-gate checks are in the report.

The approved V43 one-image foreground-only 5%-noise diagnostic learns substantial
same-image recovery with fresh noise (about 90% MSE reduction), while fixed noise
memorization fails on unseen draws. The strict fidelity gate remains unmet. All 50
queries are retained, frozen frontend/codec unchanged; no production architecture
change or 10-epoch extension. This is conditional learnability evidence on eight
people, not AP or unseen-image evidence. See the small-noise report for exact
recipe, matched snapshots, error tails and the proposed 16-image scaling control.

Follow-up frozen probes show that the one-image V43 denoiser uses image
conditioning but does not transfer: 16-image VAL paired OKS falls from 0.92045
raw-noisy to 0.72323. The previous automatic scaling proposal is withdrawn.
The next proposed scientific target is explicit image-to-frozen-codec alignment,
with held-out geometry and correct-versus-wrong conditioning as discriminating
checks. This is a proposed training direction, not an implemented architecture
change. See [the thesis report](reports/poseflow_v43_thesis.md).

V43 image alignment is an explicit training task with image-only inference.
Constant internal state/time tokens enter the existing V43 body; clean geometry
is used only for target codes, matching and loss. All geometry is decoded from
the predicted32D code by the frozen VAE. Legacy denoising is the default and task
metadata prevents incompatible checkpoint restores. The separate10epoch preset
uses TRAIN only and VAL monitoring/full evaluation, with best/last retention and
no TEST access. This tests transferable image-to-code prediction; later diffusion
training remains dependent on its results. Existing pretrained frontend and fixed
trainval-statistics exposure are disclosed in the alignment report.

The first V43 alignment run is user-cancelled. Diagnosis confirms proposal
frontend degradation and a conflicting geometry-gradient route through reference
boxes; the zero-token image-only initialization also starts all queries
identically. These confound the alignment hypothesis. Frozen frontend restoration
recovers proposal coverage but not learned pose quality. No replacement run is
authorized/launched; proposed fixes and evidence are in
[the failure report](reports/poseflow_v43_alignment_failure.md).

V44 is the approved native-proposal frozen-codec latent-diffusion baseline. Its
TRAIN-only fixed posterior-mean statistics and single selected-slot assignment
keep targets consistent; native frontend supervision remains separate. The
pretrained High-KL codec and ED-Pose exposure are disclosed. Development uses
TRAIN/VAL only; see [the V44 report](reports/poseflow_v44.md).


## V42 paired-image JEPA denoising

`benchmark_poseflow_v42_jepa` extends paired image-only-confidence V42 with
positive-only final-feature MSE to an EMA pose-coordinate encoder (decay .999),
and .5 native-weighted clean-pose reconstruction through the shared final reader.
The clean encoder has no image input; its reader retains the explicit detached
box-center reference. Direct alignment weight1, native noisy pose/box/DN losses
and uniform coordinate corruption/DDIM remain. Teacher updates only after
optimizer steps and is training-only checkpoint state. Initial inference tensors
match the parent; fresh ED-Pose seed42 full80 training, no trained V42 resume.
Historical validation overlap/fixed32-test monitoring caveats apply. See
[the JEPA report](reports/poseflow_v42_jepa.md) for contracts and evidence.

2026-11-05 22:25 CET: V42 JEPA full80 running at `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v42_jepa/20261105_221931`,
PID990002, verified step150 with finite loss and matching EMA
count. Initial online model matches parent; native CUDA/strict restore and
138 regression/six final JEPA checks pass. Dashboard33332 retains historical
roots/six pins and exposes new JEPA losses; first epoch validation/AP pending.

V44 now has an opt-in velocity comparison against its default x0 prediction.
Both arms warmstart the same saved epoch047 model with a frozen codec and fixed
proposal assignments. They run sequential bounded10epoch continuations with
matched data/seeds/budget; initial function differences are measured explicitly.
The completed comparison failed velocity's AP/recovery gates. A frozen fixed32VAL
diagnostic distinguishes two effects: x0 loses accuracy on generated reverse
states relative to forward-corrupted targets, while velocity is weak even on
forward-corrupted intermediate states. Independent sampler/oracle arithmetic
checks pass. No follow-up training is launched; see
[the comparison report](reports/poseflow_v44_velocity.md) for scope and results.

The approved next V44 comparison tests detached generated-state exposure against
a matched forward-corruption control, retaining the frozen codec, x0 model,
single assignment and original losses. Both arms receive the same mixture of
uniform timesteps and K4 intermediate times; only the state construction differs
on the latter batches. It is bounded to2epochs/2500updates per arm, with no
automatic promotion or extension. See [the rollout report](reports/poseflow_v44_rollout.md).


V45 is a separate image-only progressive prediction family using V44's cached
native top50 proposals and frozen High-KL codec/TRAIN statistics. One new
confidence-ranked person enters each stage; two shared decoder layers refine
all active people bidirectionally. Fixed proposal assignments and normalized
per-person all-stage/all-layer supervision replace latent diffusion. Image-only
confidence and native frontend/DN losses remain. Fresh ED-Pose/full80 TRAIN/VAL
training uses explicit progressive AP TensorBoard tags and best/last retention.
No TEST, depth ordering or trained V44 warmstart. See [V45](reports/poseflow_v45.md).


## V45-task-pose coordinate diffusion with jointly learned geometry codes

The separate task-pose V45 family uses a32D absolute normalized pose encoder and
shared nonlinear reader, independent of image/reference offsets. Time-independent
native image proposals/scores/boxes are cached per trajectory; one selected GT
assignment precedes native coordinate corruption. Two60-point person-attention
layers predict a shared normalized latent residual, initialized to zero.
Both arms train online/EMA-code clean reconstruction on complete augmented poses;
all valid people retain masked denoising supervision. Only the JEPA arm adds
weight1 final-code alignment. Teacher .999 updates after optimizer steps.
Presets benchmark_poseflow_v45_task_pose_control/jepa are independent of the
concurrent progressive-person V45 family. User chose historical trainval/val
overlap and fixed32 TEST monitoring/final8000; disclose this development exposure.
See [the task-pose report](reports/poseflow_v45_task_pose.md) for contracts.

2026-11-07 15:54 CET: V45-task-pose pair verified at `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v45_task_pose/20261107_154258`.
Control PID1935244 is at optimizer/EMA step600; JEPA
queues for fresh full80 after control/final evaluation.143 regression and21
final checks/native CUDA/strict restore passed. Both TensorBoard roots and
original AP pins retained. Details in the task-pose report; learned AP pending.

2026-11-07: User changed V45-task-pose scheduling to parallel. Control continues its existing run; JEPA starts fresh seed42 for80 epochs in tmux `poseflow-v45-task-pose-jepa`. Sequential supervisor disabled to prevent a duplicate launch. Model, data, optimizer and evaluation protocol unchanged.


## V46 canonical ordered generation

V46 uses direct canonical-rank supervision after augmentation:512 normalized
columns, then y/x/box dimensions/masked joints/stable duplicate order. Collisions
preserve every valid person. Sequence positions are independent of native
proposal identities; no Hungarian/native frontend loss is used. Frozen High-KL
posterior-mean codes/statistics and geometry masks remain V44-compatible.
Independent64-row training uses noisy GT prefixes, while image-only sequential
inference re-noises generated clean prefixes and latches EOS. Birth/refinement
objectives have equal task weight. Capacity50; TRAIN/VAL only, full80 and strict
self-contained checkpoints. Generated rank errors and teacher-forced diagnostics
are labeled separately. See [V46](reports/poseflow_v46.md).

2026-11-08: V46 full80 launch verified at `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v46/20261108_181906`, PID2763350.50 focused/legacy checks and full64rows/50people FP32CUDA batch4accum2 restore passed; TensorBoard33332 V46-ordered-sequence/. live. Learned outcomes pending; see V46 report.

V46 has an explicit evaluation-only DDIM eta0 override for existing tokens. New
person latents receive one-time forward noise at the next row's level; identity,
EOS and confidence contracts remain unchanged. The original re-noising default
and checkpoint/training contract remain intact. A paired frozen-checkpoint CLI
compares both on original fixed32VAL with legacy parity and immutable model/codec
checks; this does not silently promote DDIM to the active trainer.

2026-11-09 frozen epoch46 fixed32VAL sampler result: DDIM eta0 did not rescue
pose AP/final refinement (pose AP0–100 .2115->.1043; bbox5.1836->5.6296;
rank1 final latentMSE .7412->.7449).46 checks/original-workspace exact parity
and model/codec invariance pass. Current training/default sampler preserved;
separate comparison TensorBoard streams/report record evidence and coverage.

## V45 task-pose ordered JEPA

The isolated `benchmark_poseflow_v45_ordered` arm uses50 canonical-rank slots:
512 horizontal columns, y, labeled-joint centroid, descending box area, stable
index; firstM sorted people are positive and remaining slots background. Learned
rank queries/native image box stage replace score-selected proposals; inference
never consumes GT count/positions. No Hungarian/DP, encoder candidate or DN
losses. Coordinate diffusion and task-pose JEPA/EMA/reader remain unchanged.
This is a frontend/assignment architectural comparison, not ordering-only.
Historical trainval/overlapVAL/fixed32 TEST/final8000 protocol remains explicit.
See [ordered task-pose](reports/poseflow_v45_ordered.md).

2026-11-09: Ordered task-pose JEPA fresh full80 launch verified at
`/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v45_ordered/20261109_094318`, optimizer/EMA step150.
Native CUDA/strict restore and36 related checks passed; live TensorBoard
`V45-task-pose-ordered-JEPA/.` indexed. First epoch AP pending.

## V47 ordered coordinate generation

V47 retains V46's64 independent ordered rows/capacity50/EOS but corrupts normalized joint+box coordinates at scale6, encodes them through one jointly learned32D person encoder and supervises decoded2D predictions. Coordinate DDIM advances generated prefixes; no latent target or frozen codec/statistics. Native preflight and a two-seed four-TRAIN-image free-running memorization gate precede the approved fresh full80 TRAIN/VAL run. Existing jobs and TEST are preserved. See [V47](reports/poseflow_v47.md); native/learnability verification passed and fresh full80 is active; VAL outcomes pending.

2026-11-09: V47 full80 launch verified at `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v47/20261109_150341`, trainerPID3490732.55 checks, native full64/50people FP32/batch4accum2/restore and two-seed four-image learnability gate passed. Fresh initial hash matches verification; same fixed32VAL, separate TensorBoard stream, existing jobs/artifacts preserved. See V47 report for mixed refinement behavior and measured cost; fullVAL outcomes pending.


## Missing-joint teacher diagnostics

`evaluate_joint_completion.py` evaluates a frozen CrowdPose14 RTMPose checkpoint
with existing person boxes in original-image pixels. Predictions remain separate
from the original annotations; flag0 is unmeasured geometry, and flag1/2
coordinates and flags remain unchanged. Raw scores and independent single-pass
flip consistency are selection heuristics, not correctness or visibility labels.
Primary predictions retain native flip TTA. The released teacher uses trainval,
so VAL recovery metrics are explicitly in-sample GT-box diagnostics.

The 2026-11-09 full 2,000-image VAL audit covers 7,074 valid posed people and
22,112 missing joints; no TEST access or training integration. Missing-joint accuracy requires
reference adjudication. Separate box-only people and fully truncated/out-of-crop
joints need an explicit policy before any full-dataset supplement. See
[completion evidence](reports/joint_completion.md) for metrics, review artifacts
and output provenance.

## V48 clean-prefix ordered coordinate generation

V48 isolates a V47-based birth-only experiment: clean fixed prefixes, one next
person/EOS target per row, no noise, time conditioning or refinement. P+1 rows
train in parallel with one EOS row, capacity50. Existing masked coordinate and
geometry objectives supervise both layers only at PREDICT; effective birth
weights10/.5 replace V47's birth/refinement average. Sequential inference fixes
birth geometry/confidence and immediately removes EOS/capacity-stopped images.
Fresh original-EDPose initialization/full80 seed42 TRAIN/VAL protocol, strict
restore and separate TensorBoard preserve historical jobs/artifacts and TEST.
See [V48](reports/poseflow_v48.md);74 checks/native preflight passed and fresh full80 is active at20261110_095254,PID4145233. Same fixed32VAL, original initialization, separate live TensorBoard AP cards; learned results pending.

### V48-AdaLN parallel variant

The second V48 arm changes only image cross-attention conditioning: valid-prefix
person count0..50 drives a shared learned step embedding and per-layer scale/
shift/residual-gate deltas. Identity initialization preserves baseline common
weights/outputs and subsequent RNG. No GT total-count input,noise,refinement or
teacher-policy change. Same fresh seed42/full80 TRAIN/VAL recipe; existing V48
continues. Strict V48 variant checkpoints and separate TensorBoard run retain
baseline APIs and AP tags. See [V48-AdaLN](reports/poseflow_v48_adaln.md).

2026-11-10: V48-AdaLN native preflight passed (FP32/50people/batch4accum2,peak1.79GiB,exact resumed loss/max delta1.49e-8). Fresh full80 active at `poseflow_v48_adaln/20261110_135143`,PID73156 beside baselinePID4145233. Common initial V48 weights exact;566 source hashes pinned. Same fixed32VAL, separate TensorBoard overlay, learned outcomes pending.


## Token-age diffusion-forcing input builder

The first approved iteration adds only `rows_token_age.build_training_rows`:
50 independent teacher-forced rows, one random source permutation per example,
fixed source identities/missing-joint masks, per-token Gaussian coefficients
1,.1,.05,.025,..., and zero noise for existing people in the final row. The
PREDICT query is pure noise for both person and EOS targets; future/padded inputs
carry no information. All active people have supervision masks, with EOS tails
and explicit rejection above49births. Coordinates retain caller-supplied units.
No model, loss, DDIM sampler or training run is wired yet; historical families
remain unchanged. See [contracts and limitations](reports/token_age_diffusion_rows.md).

## V42 box-to-pose timestep loss

`benchmark_poseflow_v42_timestep_loss` is an isolated corrected V42 box-center
arm. Per-image loss multipliers are .2+1.6*t/(T-1) for all native box geometry
and1.8-1.6*t/(T-1) for pose geometry. Classification, matching costs, model,
corruption and sampling remain native. This means90/10 box/pose emphasis at the
noisy beginning of reverse sampling and10/90 at its clean end, with scale2 to
retain mean native coefficients. Global GT and DN query normalization are
preserved. Fresh seed42/full80; historical trainval/overlapVAL/fixed32 TEST and
final8000 TEST development exposure retained. See [the report](reports/poseflow_v42_timestep_loss.md).

2026-11-10: Fresh full80 V42 timestep-loss launch verified at `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v42_timestep_loss/20261110_222822`.
PID348105,50 optimizer updates; matching/model/sampler unchanged, scheduled
geometry only. TensorBoard stream `V42-box-center-timestep-loss/.`; native CUDA
and strict restore passed. First learned epoch-end AP pending.


## V49 token-age diffusion forcing

V49 integrates the reviewed row builder with a jointly trained coordinate codec,
per-token sigma-conditioned dense decoder and deterministic coordinate DDIM.
50rows/49births, random fixed-source teacher forcing, all-active birth/refinement
objectives and EOS tails replace canonical ordering; training has no rematching.
Fresh original-EDPose/full80 seed42 TRAIN10k/VAL2k, FP32/batch4accum2 and strict
asset-free restore preserve the earlier experimental recipe. Generated errors
use final-set assignment only for diagnostics, never direct randomized-rank GT
comparison. Native AP, reconstruction, generated refinement and cost remain
separate evidence. See [V49 report](reports/poseflow_v49.md).

## V49-balanced objective variant

V49-balanced keeps the exact V49 model, corruption, random teacher forcing and
DDIM sampler. It replaces unbounded squared geometry residuals with Huber
residuals, separates joint/box coordinate means, normalizes four task groups by
fixed shared-decoder gradient medians from32 augmentedTRAIN batches, and balances
each person's first/second/older refinement buckets. Calibrated box:pose weights
are3:1birth/1:1age1/1:3older. FirstEOS has a separate fixed BCE-group weight from
postEOS tails. Fresh original-EDPose initialization and full80 recipe remain;
checkpoint buffers/provenance prevent recalibration on restore. Original jobs,
VAL split/selection protocol and TEST exclusion remain. See
[exact objective and verification](reports/poseflow_v49_balanced.md).

2026-11-11: V49-balanced native calibration/capacity/strict resume passed;70 tests. Fresh full80 active at`/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v49_balanced/20261111_012830`,PID445691,tmuxposeflow-v49-balanced. Model initialization exact originalV49;32TRAIN batches fix scales[2.411630153656006, 2.832282066345215, 2.318622589111328, 2.7152771949768066]. Native peak1.80GiB,nextupdate maxdelta1.19e-7. Same fixed32VAL IDs/initial0AP, separate live TensorBoard stream,72roots/20pins preserve history. Full80/all2000VAL outcomes pending; original V49/V42 jobs preserved.

2026-11-11: User cancelled original V49 and V49-balanced after poor outcomes. Both trainers stopped gracefully; verified best/last checkpoints, optimizer/scheduler state and TensorBoard history retained. Last safe global steps are82500/55000; no full80/final-allVAL completion claimed. V42timestep trainer continues. Proposal-attention identities and time-gated residuals are design-only; preserve the rebalanced objective in the successor. See .plan for pending review.

## V50 proposal-attention successor

User-approved V50 retains V49-balanced objective normalization and introduces sample-local proposal selection before birth plus sigma-gated residual refinement. Original ED-Pose box/frontend supervision is restored; one selected-slot assignment supplies fixed teacher identities. Content/reference identities are detached on the geometry branch; candidate selection keys remain trainable. No cross-image ID vocabulary. The nested native frontend uses explicit backbone optimizer grouping to preserve1e-4 versus2e-4 main rates. Full50rows/49people, seed42/full80, FP32/batch4/accum2 and TRAIN/VAL-only evaluation remain required. See [V50 report](reports/poseflow_v50.md) for contracts and measured verification/launch status. Stopped V49 artifacts and unrelated trainers remain preserved.

2026-11-11 22:03: V50 full80 seed42 launched after native verification;79 checks,100% retained transfer,32TRAIN frozen scales and strict next-update restore. Root`/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v50/20261111_215450`, PID1102983. Initial fixed32VAL0AP is a baseline; full outcome pending. TensorBoard73roots/23pins preserve history. See V50 report for exact measurements/provenance.
