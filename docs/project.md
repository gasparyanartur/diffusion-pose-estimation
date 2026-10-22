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
