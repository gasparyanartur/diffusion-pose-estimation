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

The [CrowdPose viewer](../scripts/data/crowdpose_viewer.py) indexes raw split
annotations by image ID and serves individual records and images over localhost.
The browser draws overlays in source pixel coordinates. Review notes are
stored separately in a local JSONL file keyed by split and image ID.
The manual editor stores named cohorts and their labels through
`scripts/data/manual_annotations.py`. Each write checks the set revision under
a file lock and atomically replaces one custom JSON file. Source annotations
supply the optional reference overlay. The client maps pointer and nudge
input to source pixels.
In custom-set editing, the overlay covers the padded work area. Its canvas
origin and pointer mapping both use the displayed photo’s top-left, including
under zoom. This supports drawing and picking outside-image geometry.
The docked magnifier uses the same pointer mapping and accepts source coordinates
directly from drafts. It crops the decoded image once per animation frame. The
magnifier and placement reticles use local luminance to choose their stroke colors.
Select uses the layer hit test to identify one editable joint or box. Explicit
creation tools save joints with one click and boxes with opposite corners.
The Edit annotations inspector uses hit testing to load the joint or box,
then saves joint placements and visibility changes individually. Joint
coordinate fields commit on Enter or when focus leaves the fields. Box drafts
commit on Apply or Enter. Focus mode zooms around the hit location for the
selected geometry and serializes source-pixel nudges through the same
revision-checked save path.
The Delete button and keyboard shortcut share a save action. Annotation
copy resolves Ground Truth or a custom source on the server for the same
image and appends new person IDs in one revision-checked edit.
The store records up to 20 prior editable snapshots per set. Undo restores
one affected image under the same lock and advances its edit counter.
Autolabel import invalidates history for its seeded image.
The optional live reload endpoint watches the served viewer assets. The
browser waits until writes, drafts, and dialogs clear, then reloads with
the set and image identity saved in session storage.
The bulk picker reuses the source audit index. It stages ordered split/ID
selections in the browser and submits the existing revision-checked queue
add action. Position ranges expand against original split order. Random and
ordered commands select from filtered rows before pagination, skipping queued
and staged IDs. The random command samples without replacement.

The Layers sidebar and the canvas share browser-side box and pose records
keyed by layer origin and person ID. Per-person visibility, hover, and pin
state are local to the current image. The selected-person filter applies the
same visibility rules to drawing and hit testing. The editor draws placement drafts
over the shared layers. For joint drafts, the pose preview substitutes the
draft coordinates for the selected person's slot while rendering the rest of
that person's skeleton from saved fields.
The autolabel window submits selected set members to `/api/autolabel` and polls
the batch by set ID. A configured profile selects a Sapiens2 or YOLO worker.
The background runner loads the selected model once and streams per-image
proposals with their inference provenance. Each
result is committed separately under the annotation lock after checking that
image's edit counter, people and review state. Restarting the server clears job
progress. Imported people, source predictions and human edits persist on disk.

The source audit indexes per-image geometry and visibility statistics on the
first request for each split. The browser filters and groups those rows and
can use the active custom queue as the image list. Per-image audit JSON stores
cohort, completion, and typed tags keyed to the source image, optional source
annotation ID, and optional joint index. Atomic writes take a split lock
and apply a single action to the most recent record.

The standalone `scripts/data/standardize_pose_data.py` command validates and
writes COCO interchange JSON and per-dataset manifests with source hashes and
relative paths to the existing images. The viewer serves the app wiki at
`/wiki`. Its Standards section documents the interchange format independently
of the native CrowdPose editing and evaluation paths.

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
visibility target layout (binary labeled-joint supervision) and timestep-zero
cosine diffusion inputs, while its criterion owns selectable assignment
(reference Hungarian by default), final/auxiliary decoder losses, and the
reference selected two-stage encoder classification, box, and GIoU losses
(`interm_loss_coef=1`, with intermediate box loss enabled). The
CrowdPose person target uses category label `1`, matching the annotation
format and the reference implementation; checkpoints trained with the old
label-`0` target contract must be retrained.
training entry point is
[`scripts/training/train_diffusionregpose.py`](../scripts/training/train_diffusionregpose.py)
with the `pipeline=diffusionregpose` and
`data=crowdpose_diffusionregpose` presets.

Hungarian is the default assignment strategy and preserves external criterion
parity. The optional `matching_strategy=ota` path assigns complete person
groups with DiffusionDet-style Dynamic-K: normalized person-box centers define
spatial candidates, box IoU determines the positive count, and each query can
belong to at most one person. Box-only decoder stages and selected encoder
proposals use class/L1/GIoU matching costs. Pose-capable stages use the existing
class/keypoint/OKS costs while their person boxes still provide OTA candidates
and IoU quality. Repeated target indices supervise the complete class, box,
keypoint, visibility, and area record. Final, auxiliary, and intermediate
outputs rematch independently; denoising outputs retain their constructed
correspondence and bypass both matchers. OTA clean losses normalize by matched
positives, whereas Hungarian retains GT-count normalization.

PoseFlowPipeline is a separate endpoint-flow training path that reuses the
same ED-Pose model, target semantics, task losses, matcher implementation,
denoising correspondence, and CrowdPose post-processing. It defaults to OTA;
the Hungarian override remains available for a matched control. Flow
source-target construction is separate from task matching: each image samples
a continuous float time in [0, 1], builds XY-only state as
(1 - t) * source + t * endpoint, and keeps real people as the only semantic
criterion targets. Source-prior endpoint padding is explicit and does not
create positives. At inference PoseFlow uses flow_sampling_steps forward-Euler
decoder calls; each update subtracts the fixed original source from the
predicted endpoint. It deliberately does not reuse DDIM/noise reconstruction.
Optional renewal restarts weak slots with a paired fresh source/state, while
optional ensemble selects candidates independently within each call, aggregates
those candidates, and applies one cross-step pose NMS.
The standalone evaluator's optional `--flow-seed` resets the flow RNG from
`seed + image_id` before each sample, allowing step-count runs to share their
initial source without changing the stochastic default.
The changed time semantics mean EDPose initialization is permitted, but a
trained diffusion checkpoint is not flow-compatible.

PoseFlow V3 is a distinct checkpoint family. Its person state is two signed
corner points followed by keypoint XY, in the same fixed-source endpoint-flow
coordinates. A 200-slot decoder pool is fully fresh at the first call. Later
calls carry the 50 selected state/source/query-feature/reference tuples and
append 150 fresh source-prior tuples. The first box-attention call uses the
raw state-derived `cxcywh` geometry without sorting or projecting corners;
its spatial reference is `(1 - 0.5t) * image + 0.5t * state`. Image-derived
query content remains present at every time. The selected box references seed
joint references, blended by the same gate. Score-threshold renewal is not a
V3 option because the 50+150 refill is the explicit renewal mechanism. The
matched control retains this complete pool and refill but disables both state
geometry and state-feature conditioning.

PoseFlow V4 retains V3's two-corner-plus-keypoint coordinate state and 50+150
source-paired pool, but changes the proposal stage. Flattened multi-scale
encoder locations become image tokens after explicit spatial/level positional
projection. At each flow call, a learned slot identity, projected timestep, and
fixed-gated (`0.5t`) Fourier encoding of the current coordinate state form 200
queries; ordinary multi-head cross-attention reads the image tokens and an MLP
forms an ephemeral 256-D pose hypothesis. A 32-coordinate head emits two box
corners and 14 joint XY points, then top-50 selection expands each person to a
box token plus 14 joint tokens. Every subsequent decoder layer refines both
box and pose and is supervised, as is the hypothesis prediction, by a separate
pose-aware OTA match. Hypotheses do not persist between Euler calls: only state
and fixed source persist. The V4 control retains tokens, timestep, pool, and
losses while zeroing the coordinate-prior path.

When the V4 training wrapper requests diagnostics, the hypothesis stage records
detached scalar RMS summaries for coordinate context, gated coordinate context,
image tokens, image attention update, pre-attention query, and final hypothesis.
The wrapper also records flow time, selected initial confidence, and
selected initial-to-final box/pose movement. Canonical `train/` and `val/`
tags retain hypothesis/refiner loss components; future runs no longer emit
duplicate debug aliases or the fixed `0.5t` gate mean. V4 and V5 publish equivalent quantities
under common `debug/{train,val}/...` tags; the TensorBoard run label identifies
the model version. Collection performs no additional forward, matching, or
sampling pass; these metrics show activity, not causal use.

The standalone PoseFlow evaluator has opt-in checkpoint probes for V5 and V9.
`--step-probe-dir` records each Euler call's selected predictions, carried/fresh
slot lineage, and CrowdPose AP with the ordinary postprocessor. `--stage-probe-dir`
compares all hypotheses, the selected set, each joint refiner, and final pose
NMS with a fixed GT association and bbox-area OKS. It also evaluates final
geometry with the hypothesis scores and with pose NMS disabled to separate
ranking from suppression. `--feature-probe-dir` performs fixed-seed inference
perturbations of coordinate, hypothesis-time, hypothesis-image, and refiner-image
paths; these are checkpoint sensitivity tests, not retrained ablations.
`--attention-probe-dir` additionally reports V9's learned bias, row-centered
unbiased QK logit scale, and resulting nonedge mass per layer and flow call.

Future V5/V9/V10 training processes record `debug/solution/val_first_batch/...`
only on periodic AP validation epochs. The first validation batch supplies
fixed GT denominators for hypothesis coverage, selected-person refiner quality,
per-step flow turnover, and pre/post-NMS coverage; V9 also records each
refiner layer's nonedge bias, unbiased logit scale, and nonedge attention mass.
The pose-quality summaries use labeled-joint bbox-area OKS with one final-stage
GT association held across refiners; they are diagnostic scalars, not AP.
The development preset enables `solution_feature_diagnostics`, which replays
that same batch with four feature paths disabled under the original sampling
RNG state. These values describe one batch, not the full validation set; AP
continues to use its normal complete validation pass. Existing training
processes do not acquire these tags until restarted with the new code.
The solution summaries skip ensemble mode because its postprocessor suppresses
combined candidates from multiple calls; the logger emits `ensemble_skipped`
instead of comparing those outputs against final-call-only candidates.

PoseFlow V5 preserves V4's hypothesis, two-corner transport state, 50+150
source-paired sampler, and OTA contract, but is a distinct checkpoint family.
Its training source slots are full normalized person states: half are GT-person
resamples transformed by zero-mean center translation with standard deviation
one quarter of image extent and independent `exp(N(0, 0.25))` width/height
scales; the other half interpolate distinct GT person states. The transform is
applied coherently to corners and joints. Inference has no GT pool, so starts
and refills from uniform-center, empirical-mean-size boxes and random joints.
V5's initial and refiner box heads emit corner A, corner B, and center. Only
the corners are converted to the detector's `cxcywh` box/reference; two small
auxiliary losses supervise the direct center and the corner midpoint against
the GT center. They are excluded from matching and inference output.
The V5-only `decoder_loss_mode` selects either the original all-stage objective
or a first-and-final ablation. In the latter, the hypothesis and final clean
outputs, their center losses, final denoising losses, and the corner-order
penalty are optimized; intermediate stage losses are retained for diagnostics.

PoseFlow V11 transports `[cx, cy, width, height, joint XY]` in the existing
scaled flow coordinate space. Source augmentation moves centers and scales sizes
and joints coherently; flow sampling preserves selected-source associations and
50 active plus 150 fresh slots. Sizes are positive at generation and projected
to a minimum of `1e-6` normalized units after Euler updates.

Its coarse hypothesis stage retains learned slot identity, timestep conditioning,
and dense image-token attention, but conditions on only the four noisy box
coordinates. Shared point/size/existence MLPs produce the initial box and logits.
Coarse OTA uses classification, box L1 and GIoU costs without any keypoint tensor;
only box/class losses supervise this stage. Top-score selection expands each of
50 selected latents with 17 learned role embeddings: existence (0), center (1),
size (2), and CrowdPose joints (3–16). Gated Fourier features of each selected
noisy joint condition only its corresponding joint token during training and
direct conditional validation. The V11 sampler passes an explicit forward
control that skips this addition by default (`sampling_joint_conditioning=false`).
Box conditioning and transported flow state remain active. Set the flag to
`true` for historical sampling behavior. Conditioning modules and checkpoint
keys remain unchanged, so existing checkpoints restore strictly. The default
was validated against a joint-branch-zeroing intervention on 100 validation
images; it is an inference mitigation, not proof of a training-distribution fix.
Joint references before
the first decoder layer come from the shared point readout of these tokens.

Each V11 decoder layer gathers image evidence using deformable attention first,
then applies self-attention within each person's tokens, followed by attention
across existence tokens only and a token-wise FFN. Both self-attention operations
use latent Q/K/V, without geometric query-position embeddings or distance bias.
Image sampling uses `[token XY, person width/height]` references; existence and
size tokens use the person center. Learned sampling offsets remain box-scaled.
The point MLP reads the center and joint tokens, the size MLP reads the size token,
and the existence MLP reads token 0. These same three MLPs are shared across all
stages, including coarse predictions from the single latent. Updated references
remain differentiable across refinement layers.

DN examples expand into existence/center/size tokens and share all decoder blocks
and readouts. Image queries are concatenated into one deformable-attention call
per layer; within-person attention processes clean 17-token and DN 3-token groups
separately. Only existence attention crosses persons, with the inherited clean-to-
DN and DN-group isolation masks. DN retains box/class supervision and no pose
loss. Every ordinary refiner receives box/class/keypoint/OKS losses; V11 emits no
corner-order, midpoint or separate center loss.

V11 uses model width 256 and FFN width 2048, keeps the three ResNet/encoder/decoder
stages transferred from EDPose layers 2/4/6, and shares the V5 optimizer builder
and two-cycle schedule. Compatible image-attention, within-person attention and
FFN tensors transfer exactly. Role embeddings, conditioning/readout MLPs and
cross-person existence attention are new parameters. Its own config/checkpoint
namespace leaves V5/V10 unchanged. Validation solution summaries retain flow,
score/NMS and optional RNG-preserving feature interventions; coarse-stage quality
is box IoU rather than pose OKS. Token-role RMS/update RMS, decoded size, reference
saturation and cross-person existence updates are detached and bounded to the
first periodic AP-validation batch under `tokens/step_<s>/layer_<l>/`.

### PoseFlow V19 coordinate refinement with tied memory

V19's diffusion state is `[B,50,32]`: logits of normalized `(cx,cy,w,h)` followed
by fourteen `(x,y)` joints. The parameter-free `CoordinateState` clamps clean
training targets to `[1e-5,1-1e-5]` before logit conversion; sigmoid decodes
geometry. It has no learned codec, normalization, reconstruction or latent loss.
`pred_coordinates`/`coordinate_state` are the explicit output/sampling-state keys;
`pred_latents`/`latent_state` remain compatibility aliases for coordinate logits.

Each call initializes 15 geometry features from coordinate adapters, role features
and time. Every iteration first exchanges information across the 50 hypotheses
through the shared hypothesis self-attention/FFN block. Geometry tokens read image
features with distinct per-layer pretrained deformable attention, residual and norm.
A candidate-local self-attention/residual/norm and FFN operate on box +14 joints
+ own hypothesis: 16 tokens, with no global. This simultaneously reads/writes tied
memory after current image evidence. A copied decoder normalization feeds separate
box `(4)` and joint `(2)` correction heads. Corrections add directly to the current
coordinate logits, supplying the next iteration's references. A copied person-class
row produces confidence; class labels are remapped for the CrowdPose evaluator.
Zero correction heads preserve coordinates exactly, including across iterations.
Learned preservation near clean time remains an empirical training diagnostic.

All box and joint image reads use 4D references and valid-ratio scaling/padding masks.
Joint XY comes from the coordinate state; auxiliary joint widths/heights initialize
from the pretrained joint extents times box size and refine with copied extent heads.
Those extents are call-local, receive no diffusion target, and are excluded from DDIM.
Reference-position encoding is copied from the pretrained decoder. Image attention,
local attention, FFN/norms and heads are distinct per iteration; only the fresh
hypothesis-bank block is shared. Geometry features carry between iterations;
coordinates and hypotheses carry between diffusion calls. Memory resets only for
independent invocations. Fixed50 candidates remain index-tied, not identity-tracked.
There is no V5 200-query proposal stage, refill, DN, extra pre-image FFN or ensemble.

Initialization remaps compatible EDPose encoder/decoder layers 2/4/6, then copies
components into this revised flow. This is not function-preserving EDPose inference.
`initialization_v19.py` verifies copied tensors against the remapped checkpoint,
including the selected classification row, and separately reports exact checkpoint
state/parameter coverage and fresh keys. Time conditioning copied from the source
can still be fresh with respect to EDPose; source-copy fraction is not checkpoint
coverage. Restoration uses the V19 configuration and strict model-state loading.

Training retains the 50/50 single-call versus 2/3-call rollout mixture. Single-call
states use GT-corrupted coordinate logits; rollouts start with Gaussian logits at1
and use independent per-image descending schedules. Images encode once per batch;
there are no state/memory detaches, teacher forcing or added trajectory noise.
Every iteration/call gets independent OTA top5 assignments, matched geometry and
positive-normalized confidence losses. Invisible joints and unmatched geometry are
masked; unmatched confidence is supervised. Total loss averages iterations and calls.
Validation uses the single-call loss. Fixed32 noise25/50/100 and1/2/3-call persistent/
reset grids retain established comparison tags and identify coordinate DDIM provenance.
Training applies DDIM only between supervised calls. The final call uses its clean
prediction directly, including valid single-call examples sampled at time zero.

The optional `detach_decoder_geometry` flag defaults to false. The separate
`benchmark_poseflow_v19_detached` experiment enables it: each intermediate
prediction retains its own supervision graph, then coordinate logits and joint
extents detach before the next decoder layer. Feature/hypothesis recurrence stays
attached. The final prediction is never detached, preserving cross-call coordinate
DDIM and memory gradients. Extent heads have no direct loss and therefore lose their
later-layer training signal in this variant, as in V5's detached reference handoff.
Forward values, parameter count and state-dict keys are unchanged. Training and
strict evaluator construction restore the flag from configuration; quick-test
architecture metadata identifies `V19-detached-geometry` with unchanged metric tags.

`benchmark_poseflow_v19_sampling_detach` instead enables
`detach_sampling_references` (default false). For decoder layers after the first,
it detaches the constructed box/joint references before both positional encoding
and deformable image attention. Coordinates themselves remain attached through
all residual additions; features, memory, initial references and cross-call DDIM
also remain attached. Auxiliary extents are part of the detached references and
their unsupervised heads lose later-layer gradients in this variant too. Both
variants therefore share that extent-gradient boundary. Configurations reject
enabling both flags together. Model weights/forward values stay compatible;
strict evaluator construction restores the gradient flag from the saved config,
and metadata/dashboard use `V19-sampling-detach` with existing comparison tags.

### V20 hypothesis guidance from a frozen 3D pose model

`PoseFlowV20Model` inherits V19 dense-read unchanged and adds one shared linear
readout from the final candidate hypothesis to 17 triples. The readout is used
only for the training auxiliary; inference follows the same image reads, memory
carry, coordinate heads and DDIM transitions. All baseline student trainability
and full geometry-detach boundaries remain intact.

The head emits normalized image-plane XY plus depth, scaled by 30 to match the
released GFPose U3D checkpoint. For XY, projecting back is `(Z_xy/30+1)/2` in
the unpadded image's normalized coordinates. This is equivalent to applying the
upstream 1000-pixel normalization to a virtual square image; it is a source-frame
transfer assumption, not a calibrated metric camera. Twelve shoulder/elbow/wrist/
hip/knee/ankle joints map to Human3.6M indices
`[11,14,12,15,13,16,4,1,5,2,6,3]`. CrowdPose head/neck receive no auxiliary
projection loss. The remaining teacher joints and depth have no fabricated GT.

Only the final layer of the final training call receives auxiliaries, with the
same final-prediction OTA assignment as the task objective. Projected L1 sums
over labeled common XY joints and divides by positive assignments, weight 1.
The prior perturbs each matched pose at an independently sampled teacher time
`tau ~ U(.1,.6)`: `a=exp(-(.1*tau+9.95*tau^2)/2)`,
`sigma=1-exp(-(.1*tau+9.95*tau^2))`, `Z_tau=a*Z+sigma*epsilon`.
The frozen teacher supplies `epsilon_hat=-sigma*score`. The surrogate
`mean(Z * stop_gradient(epsilon_hat-epsilon))`, weight `1`, injects the
score-distillation gradient without a teacher Jacobian. Its scalar value is not
a pose-quality metric. The implied noise-smoothed KL weight is `sigma/a`.
Noise comes from a checkpointed independent CPU generator, so fixed attention
seeding neither repeats it nor changes the sampler RNG. Empty assignments yield
differentiable zero and never load the teacher. Labeled occluded joints remain
supervised; unlabeled coordinates are masked in projection.

`gfpose/` contains an attributed Apache-2.0 adapter for frozen U3D evaluation.
It restores EMA weights strictly, checks the configured SHA256, uses the explicit
null mask and converts raw model output to the score with `-raw/sigma`. Teacher
evaluation is FP32 and its parameters never enter student state or optimization.
Validation/sampling require no teacher file. The `benchmark_poseflow_v20` family
has separate strict checkpoints and inherits dense-read's training recipe.
Useful structural memory must be measured with task AP and intact/reset-memory
controls; an auxiliary pose head alone can be ignored by the detector. Source
framing bias, pose collapse and depth-only corrections remain experimental risks.

### V19 dense hypothesis image read

`PoseFlowV19DenseReadModel` extends the full geometry-detach V19 baseline without
confidence ranking. Once per denoising call, before the existing decoder loop,
all 50 hypothesis queries attend densely to every unpadded encoded image token:

```
E = MHA(LayerNorm(H), F + P, F)
H_dense = H + gamma * (.1 + .9 * (1 - t)) * E
```

`P` retains the encoder's spatial sine positions plus feature-level embeddings;
the encoded-image bundle adds optional position metadata without changing old
state-dict keys or callers. Dense reads require these positions and at least one
valid image token per image. The scalar `gamma` is shared across hypotheses and
calls and starts at zero. Its attention output projection has ordinary nonzero
initialization, allowing a gate gradient on the first update. Dense attention
weights begin receiving objective gradients after the gate moves away from zero.
No FFN or normalization follows the residual, so the initial baseline computation
and baseline-path gradients are preserved. The query-only LayerNorm does not
normalize the retained hypothesis state. Dense attention uses dropout zero.

The time multiplier ranges from .1 at pure noise (`t=1`) to 1 at the clean endpoint.
The learned gate is an unconstrained signed scalar; the multiplier controls its
magnitude by time rather than treating it as a probability. All existing bank
attention, per-candidate local attention, coordinate heads and detach boundaries
remain unchanged. The dense hypothesis still reaches geometry features after the
first local image read; this addition does not directly change that read's sampling
geometry. The updated hypothesis carries through the existing differentiable
rollout and inference interfaces. Images encode once per invocation.

Diagnostics on the first layer of each call record the gate, time strength,
message/update and incoming/outgoing memory RMS, and separately labeled per-head
dense attention entropy. `benchmark_poseflow_v19_dense_read` uses the same rollout
recipe and loss as the full-detach baseline. Its quick-grid metadata identifies
`V19-dense-read`, retaining existing comparison tags. New checkpoint keys live under
`dense_read.*`; `load_baseline_state_dict` accepts exactly a compatible V19 model
state, reports copied state coverage/new keys, and rejects ranking or variant
states. Ordinary evaluator restoration remains strict for the new family.

### PoseFlow V18 pre-image refinement

V18 adds a distinct per-iteration refinement block to V17's 16 box/joint/global
representations, before sampling references are decoded and image evidence is read.
The block applies `u = LayerNorm(x)`, then `LayerNorm(u + Dropout(FFN(u)))`.
Its FFN is Linear → ReLU → Dropout → Linear, with width 256 and FFN width 2048.
This restores V15's pre-image normalization/residual FFN without
its separate memory cross-attention. The persistent hypothesis does not pass
through this block. Shared hypothesis attention, local 17-token attention, memory
carry and geometry-only DDIM remain unchanged. The first iteration still has no
pre-image memory read; local memory affects the next iteration's geometry.

V17's refinement hook defaults to identity and adds no state tensors. The separate
V18 model installs three new blocks and uses its own configuration/checkpoint
namespace and fresh rollout run. Pre/post-refinement RMS and sampling-reference
size/spread diagnostics distinguish the proposed mechanism from learned AP gains.

### PoseFlow V17 tied candidate memory

V17 keeps 15 diffused box/joint tokens per candidate. Its decoder groups those
tokens with a call-local global token and that candidate's persistent hypothesis,
giving 17 tokens per local attention group. Each iteration first applies a shared
self-attention/residual/norm/FFN block across all candidate hypotheses. It then
decodes sampling references from the geometry representations, performs the shared
local deformable image update, and runs a distinct per-iteration self-attention/FFN
over each 17-token group. The final token is returned directly as updated memory;
the other 16 outputs carry into the next iteration. Memory affects subsequent
sampling geometry through those refined representations.

The separate V15 memory reader, global-to-bank writer, QNorm and retention gate are
absent. Cross-candidate communication occurs through hypothesis self-attention;
local groups stay independent. Hypotheses initialize from distinct learned vectors
once per invocation and stay paired to candidate indices. Globals reset per call;
DDIM updates only the 15 geometry latents. Pairwise permutation tests permute both
geometry and its memory; candidate identity does not guarantee GT-person identity.

`benchmark_poseflow_v17` uses the existing 50/50 single-call versus two/three-call
rollout recipe, independent OTA assignments and unchanged loss/optimizer/data
settings. It starts fresh with compatible image initialization in its own checkpoint
namespace. Standard attention residuals and norms replace gated memory dynamics.
Diagnostics distinguish hypothesis-bank self-attention from local attention to/from
the hypothesis; obsolete writer-entropy and retention fields are omitted.

### PoseFlow V15 cross-call rollout training

`PoseFlowV15RolloutPipeline` uses the unchanged V15 model. Training draws one
batch-level branch (rollout probability0.5) and, for rollouts, a uniformly selected
call count from[2,3]. Per-image schedules start at1, sample n-1 distinct indices
from1..timesteps-2, sort descending, normalize bytimesteps-1, and end at0.
The rollout starts with pure Gaussian latents and learned hypothesis slots,
encodes the image once, and invokes differentiable denoise/DDIM transitions.
There is no teacher forcing, intermediate noise or cross-call state detachment.
Globals reset each call; memory remains invocation-local. Gradients from later
calls train earlier predicted latents and memory updates.

Each call receives independent OTA assignments and all three decoder iteration
losses. Denoising losses are averaged equally across calls; codec reconstruction
is evaluated once with the inherited epoch-dependent weight. The single-call
branch and conditional validation use V15's original loss path. Inference and
model-state checkpoints are unchanged; configuration/checkpoint directories and
TensorBoard label V15-rollout distinguish the training experiment.

Diagnostics report epoch branch/call-count frequencies and mean calls. Per-call
schedules, latent changes, memory magnitude/diversity, gate and objectives are
logged on the first rollout batch of each epoch; existing objective-gradient
checks remain enabled. The fixed32 paired persistent/reset evaluation retains
the shared metric tags and provenance. Full-batch three-call CUDA capacity must
be checked with the existing V15 run active before parallel launch.

### PoseFlow V15 local evidence and gated memory

The separate `PoseFlowQueryNormModel` rollout experiment changes the writer to
`E = LayerNorm(MHA(LayerNorm_nonaffine(S), G, G))`. Retention still uses the
original `S`, with no normalization after interpolation. The default V15 writer
uses an identity query transform. Non-affine normalization adds no parameters or
state tensors; the experiment configuration is therefore essential when restoring
a checkpoint. All sharing, local image reads, memory/global lifecycle and DDIM
equations are inherited unchanged. Fresh training uses compatible image components
only and the existing 50/50 single-call/two-or-three-call rollout recipe.

V15 reuses the V14 codec, image bundle, matched role losses and latent DDIM
sampler. Its sampled state remains `[B,50,15,256]`; its decoder representations
contain 16 tokens per candidate including one uncorrupted hidden global token.
All 16 tokens carry between layers; globals reset at the start of each call.
The separate `[B,50,256]` hypothesis bank initializes from distinct learned
vectors once per sampling invocation and carries explicitly between calls.

Each of three iterations reads hypothesis context with residual attention/FFN,
then uses the shared latent readout to estimate box/joint sampling geometry.
Four-level deformable image attention uses positional/role-conditioned queries,
4D box references and 2D joint references with valid ratios and padding masks.
The sampled feature message enters each box/joint representation once through
residual normalization and an FFN. Within-person attention/FFN coordinates all
16 representations, retaining every output. Updated globals provide keys/values
for hypothesis queries. Clean box/joint latents are read out for supervision;
the earlier geometry proposal receives no separate loss.

Memory evidence is `E = LayerNorm(MHA(S,G,G))`, followed by
`S_next = sigmoid(a)*S + (1-sigmoid(a))*E`, with `a=log(9)` initially.
There is no normalization or FFN after this interpolation. Local image updates,
hypothesis updates and latent readout share weights across iterations; hypothesis
reads and within-person coordination have separate per-layer parameters.
Hypotheses are context rather than the only image-information path.

V15 starts a fresh checkpoint family, using V14's image initialization and
training recipe. Reports record retained image tensor/element counts and their
fraction of total state elements. Final memory updates remain differentiable,
but receive no separate memory loss: within-call supervision uses memory through
subsequent decoder iterations. Cross-call training rollouts remain deferred.
Bounded diagnostics report retention, global/memory magnitudes and diversity,
normalized per-head and head-averaged attention entropy, and candidate geometry
spread. Main quick-test tags match previous versions; reset controls remain
under `quick-test-reset/` and extra diagnostics under `debug/quick-test/`.

### PoseFlow V14 evidence memory

V14 keeps the V13 image backbone/encoder and latent cosine-corruption/DDIM
framework, but its state is `[B,50,15,256]`: one encoded 4D box and 14 encoded
2D joints per person, with learned role embeddings and non-affine LayerNorm.
Separate heads decode box center, size and confidence from the box latent;
joint readouts decode XY. Confidence is supervised without an existence latent.
The codec and new decoder modules initialize freshly; compatible image
components retain their transferred weights and initialization provenance.

One shared evidence extractor decodes the incoming noisy geometry before the
first iteration, then uses each iteration's predicted-clean geometry for the
next read. It queries encoded image memory using 4D box and 2D joint references,
valid image ratios and padding masks. Four levels, eight heads and four sampled
points per level/head are used in the default model. Evidence contains image
features plus source representation, geometry embeddings and role embeddings.
Each layer self-attends within a person's evidence and an uncorrupted hidden
global token, with an FFN. Only the resulting global summaries enter the shared
hypothesis updater as keys/values. Hypothesis queries update 50 global memory
slots through residual attention and an FFN; distinct per-layer representation
updates then let all box/joint tokens cross-attend to that memory. A shared
latent head predicts clean states after every iteration. There is no candidate
index embedding, slotwise candidate-memory addition or person correspondence
constraint on hypothesis slots.

`denoise(..., hypothesis_memory=None)` initializes memory from 50 distinct
learned slot embeddings and returns `hypothesis_memory`. Every call initializes
its person-global tokens from one shared learned embedding. Global tokens evolve
within that call and are discarded afterward. The sampler passes returned
hypotheses into the next call and applies DDIM only to box/joint latents.
`persist_hypotheses=False` provides the diagnostic reset control. Encoded image
memory and spatial metadata are cached once per image; neither memory state is
stored as mutable model-global state. Gradients flow through all three decoder
iterations during single-call training; sampling remains under no-grad.

Independent OTA assignments at each layer align geometry and detached clean
latents. Box latent MSE averages matched tokens; joint latent MSE averages
matched labeled joints; their two means have equal weight. Empty masks produce
differentiable zero losses. All unmatched latent geometry is masked, but
unmatched confidence receives focal supervision. Classification and decoded
geometry retain positive-assignment normalization. Clean reconstruction retains
V13's all-slot geometry and slot-mean classification, including synthetic empty
geometry. Because existence is not encoded separately, reconstruction confidence
must be learned from geometry; its usefulness remains an empirical question.
The baseline uses OTA top-k 5, reconstruction decay 1 to .01 over epochs 0-39,
and gradient cap 2.0. Cross-step training rollouts are deferred.

The bounded 32-image quick-test grid runs paired persistent/reset hypotheses
at noise variance fractions .25/.5/1 and 1/2/3 calls with the same initial states.
Artifacts record box coverage at IoU .5/.75, excess good-IoU candidates sharing
the same best target, good/poor-box mean confidence, pose coverage, AP, and
per-call/per-layer latent/memory diagnostics. Raw duplicate fraction uses excess
candidates divided by all raw candidates; it is not a matching or post-NMS
duplicate metric. Global/hypothesis RMS, update RMS, hypothesis slot variance
and attention entropy are bounded detached logs. Random-init smoke tests verify
wiring and evaluator contracts, not learned pose quality.

PoseFlow V13 inherits V12's architecture and corruption/sampling recipe below,
with per-layer set supervision. Each layer independently runs the existing
pose-aware Hungarian matcher (classification, labeled-joint L1 and OKS costs;
no latent cost), using binary class index zero. Matched predictions use that
person's geometry and detached clean latent encoding. Unmatched predictions use
existence zero; only their existence latent token receives denoising MSE. Within
one layer, latent and decoded losses share an assignment. Assignments may change
across layers, without changing the order or identity of sampler states.

Existence latent MSE averages all slots; center, size and joint latent MSE average
matched slots. The four role families have equal weight, with differentiable
zero geometry terms for empty images. Encoded unlabeled joints retain the 0.5
placeholder; decoded joint losses remain masked. Clean reconstruction is still
paired with its source and includes empty-slot random geometry. The total is
mean layerwise latent MSE plus mean weighted layerwise prediction loss plus
clean reconstruction. Weights remain 2/5/2/10/4 for focal/box/GIoU/joint/OKS.
Decoder focal classification sums all slot contributions and divides by matched
person count (clamped to one), restoring the earlier criterion scaling. Clean
reconstruction retains all-slot averaging. The initial V13 launch used slot-mean
decoder classification; its normalization correction is recorded in the run
objective-change artifact and begins in the resumed logging segment.
Final-layer loss charts use unsuffixed names, layers use `_0`, `_1`, `_2`, and
layer averages use `_mean`. In particular `loss_latent` logs the final layer;
`loss_latent_mean` is the term used in the total.

V13 optionally multiplies reconstruction by an epoch-dependent weight. The
default start/end weights are both 1, preserving the objective. The isolated
reconstruction-decay experiment interpolates the logarithm of the weight from
1 at epoch 0 to 0.01 at epoch 39 and holds thereafter; the codec continues
learning. `loss_reconstruction` remains raw and `loss_reconstruction_weighted`
is added to the total once. The schedule uses the absolute restored epoch.
Its training configuration sets the global gradient norm cap to 2.0.

Opt-in `debug/optimization/` probes use `autograd.grad` on the first training
microbatch each epoch to measure latent, prediction and raw/weighted
reconstruction gradients without modifying accumulated parameter gradients.
`on_before_optimizer_step` records accumulated preclip global/component norms,
the expected clipping factor/postclip norm and optimizer learning rates on the
first and every 100th update. Expected postclip values are calculated, not
measurements taken after clipping. Diagnostics are rank-zero and bounded;
they add no forward calls or persistent graphs. Quick-test JSONs also record
the reconstruction weight and configured clipping threshold.

V13 also supports `matching_strategy=ota` through the existing Dynamic-K
matcher. Hungarian remains the default. OTA assigns each selected slot one
person but can assign several slots to the same person; aligned clean latent
targets are repeated accordingly. Classification and geometry normalize by
positive assignments; latent role means use all positive slots for geometry
and all slots for existence. Matching stays independent per decoder layer and
never reorders sampler states. The isolated OTA reconstruction-decay recipe
changes only matching relative to the fresh Hungarian reconstruction-decay
recipe. Quick-test provenance records matcher and candidate top-k.
Per-layer/final/mean loss logging also reports `positive_assignments`,
`target_people` and `assignments_per_person` to show whether Dynamic-K is
actually assigning several slots per person. These are unweighted metrics.

V13 also runs a bounded test-subset diagnostic after validation each epoch,
independently of full-validation AP cadence. A seed-42 manifest persists 32
sampled test image IDs across resumes. Deterministic per-image generators
(seed plus image ID) separately construct target slots and Gaussian noise.
The codec is recomputed at the current weights. Noise variance p=0.25/0.5/1
means sqrt(1-p)*clean + sqrt(p)*noise, with its inverse cosine timestep. Each
condition uses 1/2/3 DDIM calls at evenly spaced descending times to zero.
Image memory is cached once per image; all call counts share starting states.
The last call's readouts are scored, without a further decoder call at zero.

`quick-test/noise_var_{25,50,100}/steps_{1,2,3}/` charts report pose AP/AP50/AP75/AR,
bbox AP, oracle OKS and raw OKS75 coverage. Corrected native postprocessing and
crowdposetools score all GT in the manifest, including overflow beyond 50. JSON
summaries/predictions live under the run's `quick_test/epochNNN/`, with the manifest
in `quick_test/subset.json`. Epoch/global step, seeds, image IDs, transforms and
noise/sampler/evaluator contracts accompany the results. Model modes and RNG
states are restored even on failure. Sanity/repeated epoch hooks are skipped;
only rank zero performs the diagnostic. The 25/50 percent cases are GT-conditioned
recovery checks; the 100 percent case is generation. None is full-test AP.

PoseFlow V12 uses fixed person groups: 50 inputs and 50 outputs, with no
proposal stage, selection, replacement, or auxiliary DN groups. Each group has
17 sampled state tokens (existence, center, size, 14 joints) and one unsampled
global person token. Clean-state MLPs and role embeddings encode target geometry;
non-affine feature LayerNorm fixes each clean token's scale before corruption.
Real targets are randomly assigned to slots and keep that assignment throughout
the pass. Empty slots have existence zero, uniform random joints, and a center
and box enclosing those joints. Overflow selects 50 targets reproducibly and is
logged; evaluation retains all ground truth.

V12 retains the ResNet/four-level, 256-wide, three-encoder/three-decoder recipe,
but its image encoder is independent of diffusion time. Each decoder layer
gathers information with self-attention over the group's 18 tokens, lets global
tokens query image memory through dense cross-attention, communicates between
the 50 global tokens, distributes information with a second within-group
self-attention, then applies the FFN. A shared learned global token is reset
each denoising call; no group-index embeddings are used. Image masks and position
embeddings remain active. Coordinates never define decoder attention references.

Every layer predicts clean latent tokens and decodes binary existence,
center/size boxes, and joints. Clean reconstruction trains the encoder/readouts,
while detached clean latents are used in both corruption and denoising targets.
The objective adds equally weighted latent MSE, decoded task loss, and clean
reconstruction, averaged over decoder layers. Latent MSE balances existence,
center, size, and joints equally. Decoder geometry loss covers real people only;
clean reconstruction covers all slots, with labeled-joint masks on real people.
The usual focal/box/GIoU/joint/OKS weights are retained, without matching.

For normalized time `t`, training uses `z_t = sqrt(alpha(t))*z_clean +
sqrt(1-alpha(t))*epsilon`; the cosine schedule has exact clean/noise endpoints
`alpha(0)=1`, `alpha(1)=0`, with 1,000 training timesteps. Sampling starts from
Gaussian `[B,50,17,256]` tokens and uses deterministic DDIM at times `1-i/K`,
with three calls by default. Only latents enter the next call; readouts never
feed back. Image memory is computed once per sequence. No sampler state
normalization, clipping, renewal, or ensemble is applied. The sole existence
class is mapped to CrowdPose category 1 after corrected native postprocessing.

V12 initializes compatible image modules and decoder self-attention/FFNs from
the selected EDPose layers. New codec, image/person attention, and latent heads
initialize freshly. Initialization reports retained/copied tensor counts and
separate fresh components; this is not whole-model pretrained coverage. The
family has its own checkpoint/config namespace. Bounded first-batch logs keep
clean reconstruction, conditional denoising, Gaussian-start sampling quality,
latent scale/update statistics, existence counts, and overflow separate.

PoseFlow V10 replaces V5's expanded box/joint refiners with one latent per
person at the decoder attention width (256 in the development preset). The
unchanged V5 hypothesis stage creates image-conditioned candidates from learned
slots, timestep and gated noisy-coordinate conditioning. Top-person selection
retains whole-person vectors. The same three-layer geometry MLP and classifier
read the initial hypothesis and every normalized decoder output. Geometry is
absolute: sigmoid outputs give two corners, auxiliary center and joint XY;
there is no addition to previous coordinate logits. Corner-derived cxcywh boxes
supply deformable image-attention sampling references. Latents and references
remain differentiable between layers. Person tokens may attend to other person
tokens; ordinary queries cannot read DN queries, and DN groups retain their
isolation. DN label embeddings receive learned noisy-box conditioning and use
the shared readouts for class/box losses. The coordinate-flow sampler, source
augmentation, independent stage matching and center losses remain V5-compatible.
The original V10 preset uses a single group-warmed cosine decay: main peak LR 1e-4 and backbone
peak LR 1e-5, with 1/10-epoch warmups and separate 1% minima (1e-6/1e-7).
The peak is active at the beginning of the final warmup epoch, following the
existing V6 schedule convention; backbone starts at approximately 1e-6.
The V10 benchmark enables all-stage losses and has a distinct output directory.
The V10-tiny benchmark changes the projected image-feature, hypothesis and
transformer width to 128 and each transformer FFN's inner width to 256, while
retaining eight attention heads and the V10 training contract. EDPose ResNet
weights load unchanged; width-dependent transformer/projection/head weights do
not. V10-tiny therefore retrains those components from random initialization
and writes separate checkpoints. Its reduced width also changes initialization,
so quality differences cannot be attributed solely to compute or capacity.
V5-derived V10 pipelines select the LR schedule from `scheduler_type` through
the common AdamW builder. The original V10 preset retains group-warmed cosine;
the separate V10-tiny V5-LR preset selects V5's two-cycle cosine with zero
warmup. The V5 schedule restarts both main and backbone groups at the same
absolute LR of 3e-5 at epoch 40.
V10 also emits the prediction-based first-AP-validation-batch solution summaries
and optional feature interventions described above. V9 skeleton-bias attention
measurements remain exclusive to V9. Stage losses and ordinary V4-derived
activity summaries remain available. Quality requires a
matched training/evaluation experiment, not just structural tests.

V10 latent summaries use `debug/solution/val_first_batch/latent/step_<s>/layer_<l>/`.
They observe the baseline sampling calls only, not the four feature replays or
DN queries. `person_count` includes all clean selected persons across the batch.
`input_rms` and `update_rms` measure incoming decoder features and their actual
layer update; `update_input_rms_ratio` divides by input RMS clamped to 1e-8.
`person_centered_rms` centers outgoing vectors across persons within each image
(one person yields zero spread). `geometry_logit_saturation_fraction` counts
shared-head logits with absolute value above 5. `box_point_movement_l1` and
`pose_movement_l1` compare normalized decoded coordinates with the preceding
stage for the same selected persons. `tiny_box_fraction` counts persons with
either corner-derived size below 1e-3. All summaries are detached and require
evaluation mode; hooks are removed on exit. `debug/lr/main` and
`debug/lr/backbone` record effective optimizer-group LR each training epoch.
These observations complement fixed-GT quality logs, rather than establish
that latent movement improves predictions.

PoseFlow V9 is a V5-derived checkpoint family using the same ResNet feature
path and EDPose initialization. It changes only expanded joint-query
self-attention: a V9 mask builder uses the 14-joint CrowdPose skeleton to add
a learned `-5` initial logit bias to nonneighbor joint-to-joint pairs within
each person. Each refiner layer has one bias shared across heads and people.
Self and skeleton-neighbor joint pairs, all interactions involving a box token,
and denoising queries keep V5 weights and mask behavior. The existing
`nn.MultiheadAttention` projections remain; this is graph-biased dense
attention, not a sparse graph operator. A bias of `0` recovers V5 attention.
The V9 development benchmark inherits V5's data, optimizer, schedule, and
evaluator settings and uses a distinct run directory.

PoseFlow V6 is a separate V5-derived checkpoint family. It retains V5's source
sampling, coordinate state, OTA losses, center supervision, schedule, and
inference path, while adding an online frozen DINOv3 ConvNeXt-Small backbone.
Stages 2/3/4 (192/384/768 channels at strides 8/16/32) are projected to the
decoder width and kept as three spatial maps. After the existing image
cross-attention in every configured decoder layer, three independent
single-scale deformable-attention modules read those maps using that layer's
current query references. The three readouts and the query state are
concatenated and passed through an MLP; its output is applied as a
zero-initialized gated residual before the decoder FFN. This preserves
per-scale features for the MLP rather than summing scales inside one
multi-scale-attention operation. The DINO path receives the same transformed,
ImageNet-normalized image tensor as PoseFlow and derives per-scale masks and
valid ratios from the batch padding mask. During training it also conditions
denoising queries, preserving their leading query order. V6 initializes all
shared V5/EDPose tensors from the same baseline and treats DINO, projections,
attention modules, MLPs, and gates as newly initialized parameters. V5
checkpoints do not restore V6's new branch and V6 checkpoints are not V5
checkpoints. The extra DINO forward and three per-layer attention reads have
not yet been profiled. V6 trains the main/model parameter group at `1e-4`
with a one-epoch warmup, then cosine decay from epoch 1 through epoch T. Its
ResNet group uses `1e-5`, warms for 10 epochs, and decays independently from
epoch 10 through T; both schedules reach their configured peak at the start
of their stated decay epoch, and neither schedule restarts.

PoseFlow V7 is a distinct V5-derived checkpoint family that keeps V6's
per-decoder-layer DINO residual-fusion design but uses frozen DINOv3
ConvNeXt-Tiny stages 2 and 4. These taps have 192 and 768 channels at strides
8 and 32 for 256-pixel inputs. Stage 2 shares stride 8 with the first selected
ResNet map; stage 4 supplies coarse context. The base transformer uses only
its first two ResNet maps (internal layers 2/3, strides 8/16), so the model
attends to four spatial maps across two branches rather than one shared
four-level pyramid. Each DINO scale has its own single-scale deformable
attention readout; query plus both readouts pass through the fusion MLP before
the zero-initialized residual gate. V7 retains V6's independent cosine
warmups and all-stage losses.

PoseFlow V8 keeps V7's flow, loss, and reduced training setup, but replaces the
ResNet and V7 post-attention DINO fusion branches with a DINO-only feature
path. Frozen ConvNeXt-Tiny supplies all four native stages in fine-to-coarse
order: 96/192/384/768 channels at strides 4/8/16/32 (64x64 through 8x8 for a
256-pixel input). It does not synthesize another pyramid. Each stage has an
independent fine `1x1 Conv + GroupNorm(32)` projection to the configured
transformer width. The coarse adapter bank uses stages 2/3/4 only (192/384/768
channels at strides 8/16/32), while the fine bank uses stages 1/2/3/4. Coarse
features and their three-level metadata feed the transformer encoder and human
query generation; encoded coarse memory remains the source for those
predictions. Fine features bypass the encoder and form the four-level spatial
memory sampled by decoder deformable cross-attention. The encoder's deformable
attention is constructed for three levels and the decoder for four. The paths
share the frozen DINO representation, but use independent projections and
separate masks, spatial shapes, level offsets, and valid ratios. Each downstream
objective updates its own adapter bank. Decoder and loss behavior otherwise
remains unchanged.

V8 initializes compatible transformer and prediction tensors from the selected
EDPose checkpoint. Its ResNet and legacy input projections are intentionally
unused, and both DINO adapter banks start from fresh initialization. V8 has its
own checkpoint family; V7 checkpoints do not load strictly into it.

At the reduced 3-encoder/3-decoder configuration, 256-pixel input, and 200
query/50 proposal setup, measured model parameter counts are:

| Model | Total | Trainable | Frozen |
| --- | ---: | ---: | ---: |
| PoseFlow V5, four ResNet levels | 43,014,736 | 42,792,336 | 222,400 |
| PoseFlow V6, four ResNet levels plus three DINO Small taps | 95,206,675 | 45,529,587 | 49,677,088 |
| PoseFlow V7, two ResNet levels plus two DINO Tiny taps | 67,266,675 | 39,224,147 | 28,042,528 |
| PoseFlow V8, DINO Tiny 3-level coarse / 4-level fine adapters | 42,384,848 | 14,564,720 | 27,820,128 |

Counts include frozen backbone parameters and exclude criterion parameters.
V7 has about 8.3% fewer trainable parameters than V5, but about 56.4% more
total parameters because its frozen DINO backbone contributes 27.8M weights.
V8 has 24,881,827 fewer total parameters and 24,659,427 fewer trainable
parameters than V7; it removes the ResNet and per-decoder DINO fusion path.
Its coarse and fine DINO adapter banks contain 346,368 and 371,712 parameters,
respectively, or 718,080 together. The count is 99,360 lower than the earlier
all-four-level-coarse V8 variant: 25,344 parameters from omitting the stage-1
coarse adapter and 74,016 from constructing three rather than four encoder
attention levels. This also removes stride-4 locations from the encoder while
retaining them in fine decoder memory. These are parameter counts, not measured
runtime or memory results.

PoseFlow can optionally add one shared pose-hypothesis codebook after deformable
image cross-attention in selected decoder layers. Each image-conditioned query
reads from the same learned `C × hidden_dim` token bank, and the same attention
projections and residual gate are reused at every selected depth before that
layer's FFN. `pose_codebook_size=0` omits the module entirely.
`pose_codebook_initial_gate=0` preserves the image-conditioned output exactly,
whereas a finite nonzero initialization activates gradients in the token bank
and attention projections on the first backward pass. Denoising queries remain
an unchanged leading prefix at every selected layer. The initial C=16
experiment conditions only the first pose layer; the separate active C=128
all-layer preset conditions layers 0, 1, and 2 with one shared codebook, so it
also intentionally conditions the box-oriented stage.

For enabled codebooks, PoseFlow records debug-only statistics over clean
queries, excluding the denoising prefix: residual gate, query/attended/residual
RMS, and residual-to-query RMS. It obtains averaged codebook attention weights
only on the first validation batch each epoch, recording normalized entropy and
effective-slot count. These metrics are detached observational summaries and do
not change predictions, losses, matching, or optimizer state.

The DiffusionRegPose data preset follows the external training data and
target semantics: it samples the reference resize/crop policy, normalizes
images with ImageNet statistics, updates pixel targets before converting
keypoints to per-image canonical coordinates and boxes to normalized sigmoid
coordinates, and pads transformed images only to the largest height/width in
each batch. Crop augmentation removes people from supervised targets but keeps
their clipped keypoints in the diffusion-proposal stream, matching external
`keypoints_buquan` behavior. The default model freezes the ResNet stem
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
training epoch and separate test/AP evaluation runs after fitting. Model
construction follows the external same-seed initialization/reset order, and
validation fences CPU/CUDA RNG use while the runner skips Lightning sanity
validation by default. The
published DiffusionRegPose checkpoint is an evaluation/parity target, not the
training initialization. DiffusionRegPose `test_step` currently records test
losses only; full test-split keypoint and bbox AP is produced by
[`scripts/evaluation/evaluate_diffusionregpose.py`](../scripts/evaluation/evaluate_diffusionregpose.py)
so it uses the same sampled pose-NMS output and explicit evaluation resize as
the validation metrics.

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

For architecture-development feedback,
`benchmark_diffusionregpose_development` reduces both dense and query-side
work: three feature levels, two encoder layers, three decoder layers (two box
and one pose), 200 initial queries, 20 groups/proposals, 20 denoising queries,
and a 1024-wide FFN. Its 10% data preset uses batch 16, validates every epoch,
and evaluates AP every 10 epochs. It keeps the same data/target, diffusion,
matching/loss, optimizer, and evaluator boundaries as the paper model. The
resolved model has 31,495,844 unique parameters and is intentionally a separate,
checkpoint-incompatible development architecture. See
[`reports/diffusionregpose-runtime-audit.md`](../reports/diffusionregpose-runtime-audit.md)
for measured timing and memory. Its completed full-data 256-pixel experiment
showed that this is too aggressive for an AP learning gate: only 18.15% of the
retained encoder and 53.29% of the retained decoder elements load from EDPose,
and only one decoder layer operates on pose queries.

`benchmark_diffusionregpose_development_compatible_256` preserves the tensor
shapes of retained EDPose components: four feature levels, a 2048-wide FFN,
six encoder layers, and six decoder layers (two box and four pose). It reduces
runtime through 200 initial queries and 50 person/proposal/DN groups. The model
has 49,527,324 unique parameters and its checkpoint loader enforces 99% minimum
coverage separately across input projections, level embeddings, transformer
encoder, transformer decoder, and prediction heads. Validation sampling uses a
fixed seed so AP checkpoints are directly comparable. A controlled EDPose
initialization evaluation showed that two encoder layers destroy pretrained
bbox AP while the 200/50 query reduction with all six layers preserves it.

The matching 128-pixel development gate uses the same model topology and
training/evaluation lifecycle with 128/192-pixel train and validation resizing.
It exists to measure the lower-resolution speed/AP tradeoff without conflating
resolution changes with another architectural reduction.

The optional `benchmark_diffusionregpose_development_compatible_256_keep_layers`
preset is a reduced transfer model. Its three local encoder and decoder layers
are initialized from one-based EDPose layers 2, 4, and 6 by explicit checkpoint
key remapping. The selected decoder layers become one box stage followed by
two pose-refinement stages; the EDPose pose head used during query expansion is
loaded into a separate expansion head, while the selected pose heads supervise
the two retained refinement layers. This preserves the intended source-layer
initialization but is not exact full-checkpoint inference because the skipped
layers alter intermediate activations.

The matched
`benchmark_diffusionregpose_development_compatible_256_keep_layers_ota` preset
inherits that selected-layer model and changes only clean-query assignment to
pose-aware Dynamic-K OTA. It therefore provides the fast 256/427 comparison
against the completed 3E/3D Hungarian run without changing layer remapping,
data, optimizer, schedule, denoising, or evaluation.

The optional `freeze_backbone` comparison flag freezes all ResNet parameters,
including `layer2`–`layer4`. It changes only gradient participation and
optimizer parameter groups; tensor names, shapes, transformer depth, and
checkpoint loading remain unchanged. The corresponding 256-pixel benchmark
preset is intended to measure the speed and AP impact of using fixed EDPose
image features.

The optional `freeze_backbone_and_transformer` flag extends this comparison to
the full EDPose prefix: the backbone, input projections, label embedding, and
transformer computation are fixed, while the prediction/refinement heads and
encoder proposal heads remain trainable. The heads are shared between the
model and decoder, so the freeze policy is applied by module identity rather
than parameter-name matching. This mode is a diagnostic transfer/runtime
experiment and is not paper-equivalent training.

Its evaluation path is independent of the external repository's runtime
package. CUDA training and inference use a lazily compiled, vendored DINOv3
forward/backward deformable-attention kernel whose scalar bilinear
accumulation matches the checkpoint operator. The imported source is pinned
to revision `6876159a11b4df116f30f667f8c9888617df0751`; its DINOv3 and
Deformable DETR/SenseTime attribution and license are preserved beside the
source. CPU runs and environments where the extension cannot be built retain
the differentiable PyTorch sampler. During training, validation
loss is calculated every epoch and validation AP is calculated every tenth
epoch. `DiffusionRegPoseAPMetric` accumulates the pose-NMS keypoint records,
while `DiffusionRegPoseBBoxAPMetric` accumulates boxes from that same
pose-NMS result. They call
`crowdposetools.COCOeval` with `iou_type="keypoints"` and `iou_type="bbox"`,
respectively. The generic `KeypointAPMetric` remains available to other pose
pipelines.

Native single-call pose postprocessing carries original query IDs through the
class-expanded candidate selection and pose NMS. Surviving candidate indices
are mapped to those query IDs before gathering logits for the final class
top-k. This preserves confidence-to-box/joint association even when candidate
order changes or multiple classes select the same query. Historical scores
using candidate indices directly against raw logits retain the legacy label.
Cross-step ensembles already retain candidate scores with their geometry.

The DiffusionRegPose criterion logs its optimized clean focal classification
loss as `loss_ce`, together with diagnostic positive/negative focal
contributions, matched/unmatched person scores, and matched-query count. Those
diagnostics, plus matches-per-target mean/min/max, are logged under
`debug/{train,val}/`, detached from the graph, and do not change matching or
loss weights.

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
| `scripts/evaluation/evaluate_joint_completion.py` | Frozen RTMPose GT-box joint diagnostics and missing-joint review caches; no dataset mutation |
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

TensorBoard naming is handled separately from the evaluator condition contract.
The optional `scripts.utils.quick_test_logging` bridge maps existing summaries
into an independent dashboard directory while retaining step/time/value data.
It carries no model state and does not modify training or sampling.

### PoseFlow V16 separated box roles

V16 keeps V15's local-evidence/gated-memory recurrence but diffuses 17 tokens
per candidate: confidence, box center, box size, then 14 joints. Its hidden
global token makes 18 decoder representations and still resets on each call.
The codec has separate confidence/center/size encoders and heads; confidence
contains the existence target, center/size contain their corresponding geometry.
All three box-role queries use the assembled 4D box and its coordinate embedding,
with separate role embeddings; joints use 2D references. Globals have no direct
image read. The shared evidence residual, memory gate, per-layer coordination,
latent normalization and DDIM are unchanged.

Matched latent supervision averages confidence/center/size within the box
family, then averages that family with labeled joints. Unmatched candidates
receive confidence classification but no latent or geometry target. Per-role
latent losses and RMS are logged separately. The V16 config uses single-call
training to compare with V15 without confounding rollout training. Its fresh
checkpoint namespace retains compatible image components only; decoder/codec
weights initialize freshly. Bone loss is deferred to the next experiment.

### V15 rollout with joint message dropout

`PoseFlowPathDropoutModel` retains V15's 15-token state and parameter/checkpoint
layout. Training independently drops each joint's local image-attention message
and within-person self-attention receiving message, each with probability .25.
Masks are sampled per image/candidate/joint, decoder iteration and denoising call;
a whole width-256 message is dropped. Surviving messages scale by 1/(1-p).
The p=1 diagnostic boundary zeros messages without division. Box and global
receiving rows are exempt; joint keys/values remain available to global/other
receivers. Latents, hypothesis reads, residual normalization and FFNs remain.
Evaluation uses all paths, with no mask RNG draws or additional parameters.

The inherited attention RNG is forked to a fixed seed. A separate batch-local
mask generator is seeded before that fork, shared across mask draws within the
batch and cleared in a finally block. Masks therefore vary across batches without
changing the existing attention seed. Diagnostics log realized image/self/both
drop fractions per layer/call alongside existing rollout/memory/loss metrics.
`benchmark_poseflow_v15_rollout_path_dropout` keeps the V15 rollout mixture and
training recipe, using a fresh output namespace. This isolates message dropout
from V16's token split; bone loss remains a separate follow-up. Useful memory
must be established by intact/reset/wrong-image memory controls under masked
joint paths; retained latents and pose priors can still provide shortcuts.

### V19 evidence-ranking auxiliary objective

`PoseFlowV19EvidenceRankModel` extends the full-geometry-detach V19 baseline.
Image encoding and each layer's deformable read run once. From identical incoming
features, positions and bank-updated hypotheses, the candidate-local update runs
on three messages `E + sigma * rms(E).detach() * epsilon`, with sigma
`[0, .25, .5]`, per-token channel RMS (floor `1e-6`), and one shared Gaussian
direction per triplet. Only the clean branch's features and hypotheses continue
to later layers/calls; all three branches receive auxiliary gradients. Decoder
dropout is zero. Corruption seeds are drawn before the inherited fixed attention
RNG fork, vary between batches and are reproducible from the run seed.

One shared bias-free linear head maps each updated hypothesis to an unbounded
ordinal score. For each OTA-positive candidate and each pair `i < j`, the hinge
is `relu(m * (sigma_j^2 - sigma_i^2) / sigma_max^2 - C_i + C_j)`.
Average over the three pairs and positive candidates, then equally across
layers and denoising calls. Default `m=1`, loss weight `.1`; no positive assignments
gives differentiable zero. Existing pose losses, independent OTA assignments,
geometry detach and cross-call gradients remain intact. No pose-output quality,
OKS, absolute score or noise-variance regression label supervises the head.

Validation and sampling omit noisy branches and auxiliary losses. The score does
not change inference confidence or detection ranking. It measures a controlled
corruption ordering; success alone establishes neither calibrated uncertainty nor
natural-image confidence. Diagnostics separate level scores, pair gaps/violations,
evidence/hypothesis RMS by layer and call, plus weighted ranking-versus-pose
gradient norms/cosine. The new head adds exactly `hidden_dim` parameters and
requires the separate strict checkpoint/config family.

### V21 fixed-slot hypothesis initialization on the V1 detector

`PoseFlowV21Model` retains the reduced V1 image encoder, one box decoder stage,
then two pose stages, original coordinate-attention fusion, and joint-XY endpoint
Euler sampler. Queries, proposals, groups and persistent hypotheses all number 50.
There is no score-based selection or reordering at expansion: candidate indices
stay aligned across box/joint features, flow state and hypothesis memory. This is
candidate-slot continuity, not tracked ground-truth person identity.

The 50 learned initial hypothesis vectors read the time-conditioned encoder image:
`E = MHA(LN(H), F_t + P, F_t)` and
`H_read = H + gamma * (.1 + .9*t) * E`. The scalar `gamma` starts at zero.
V1 flow time increases from source 0 to endpoint 1, so this multiplier preserves
V20's weak-at-noise/strong-near-clean convention. V1's encoder-output projection
and normalization feed existing class and absolute sigmoid box heads, without
spatial proposal anchors. These initial predictions retain intermediate detection
supervision. Initial box features/references detach as in V1; the separate H
branch remains differentiable.

Decoder attention includes one additional H per candidate. The current preset
uses `hypothesis_update_order: h_then_g`: each layer first updates H from its input
geometry/H using attention, residual normalization and FFN. Geometry then attends
to the fully updated H before its original geometric image cross-attention and
FFN. That exact H is returned for carry; no additional H write follows the read.
Consequently the current prediction loss reaches the final returned H without
an extra network call or rollout training. Existing attention/FFN parameters are
reused; model parameter keys and counts do not change.

Box rows retain cross-person box communication and can read their own H; joint
and H rows stay within their person group. H has no geometric image sampling
reference. Only box/joint/DN features enter prediction heads. Rectangular H/G
query masks preserve DN isolation. Missing `hypothesis_update_order` in older
saved configurations defaults to `parallel`, retaining the original simultaneous
update semantics for strict historical evaluation. Parameter-key compatibility
alone does not distinguish these computation graphs; use the saved configuration.

`hypothesis_memory` is an optional `[B,50,256]` forward input and output. The sampler
carries it separately from its `[B,50,28]` joint state. `persist_hypotheses=False`
resets memory each call with the same source-noise contract. Independent image
invocations initialize memory anew. All 50 slots continue directly; there is no
200-slot bank, gather/scatter, refill or noisy box diffusion.

The separate V21 pipeline/config keeps V1's single-call sources, OTA, two classes,
DN50, summed stage/intermediate/DN losses, learning rates `2e-4/1e-4`, multistep
milestones 31/61 and clip norm `.1`. No 3D teacher, learned codec or reconstruction
loss is included. Reducing initial candidates from V1's 200 to 50 and replacing
spatially anchored proposals are deliberate changes; V21 is not functionally
identical to V1. Rollout training and structural supervision require separate
matched experiments. Strict evaluation restoration uses V21's saved configuration.

V21's separate `quick_eval_v21.py` adapter evaluates native endpoint Euler
sampling rather than V20's coordinate-logit DDIM path. Defaults are 32 fixed test
images, seed 42 plus image ID, and call counts 1/2/3 with persistent/reset H.
Only the pure Gaussian source condition (`noise_var_100`) is implemented.
The pipeline runs this after validation, skipping sanity checks and nonzero ranks,
with RNG/mode/step restoration and exact image coverage checks. Predictions and
provenance are saved under `quick_test/epochNNN/`; standard comparison tags use
`quick-test/` and `quick-test-reset/`. Existing trainers use the same helper via
`scripts.evaluation.watch_poseflow_v21_quick_test`, strict checkpoint restoration,
and TensorBoard logging at the checkpoint's global step. Completed reports and
logging markers avoid re-evaluating completed epochs. The sidecar handles the
current trainer; the hook applies to newly constructed trainers.

### V22 explicit flow-time conditioning

V22 preserves V21's H-derived proposal/class/box heads, rich per-slot H,
H-first decoder order, fixed50 H/XY pairs, single-call training and native Euler
solver. It adds explicit time conditioning at the formerly unconditioned reads:

- FiLM on embedded Q/K in noisy-XY self-attention and XY/reference fusion. Raw
  coordinate values retain their geometric units; no coordinate LayerNorm or
  timestep scaling is applied to XY.
- Per-decoder-layer FiLM on H and geometry attention features, retaining geometric
  positional embeddings and raw attention values. H completes its update before
  geometry reads the same returned state.
- AdaLN through featurewise modulation of existing normalization outputs: dense
  H query, proposal H projection, H/G post-attention and post-FFN normalization,
  and geometry post-image-attention normalization.

The source time MLP is shared by encoder and new conditioners. Physical flow time
stays in0..1; only its embedding input is multiplied by `time_embedding_scale=999`,
matching the inherited1000-step sampler range. A shared width-D bottleneck feeds
zero-initialized scale/shift heads. Identity initialization is exact for the
conditioning operation, while encoder time rescaling is a deliberate additional
change. This makes V22 a new strict checkpoint/config family.

Expanded coordinate references were detached in V21, which would leave new
coordinate conditioners untrainable. V22 preserves the new conditioned joint-XY
gradient into the first pose layer while retaining detached input coordinates,
reference inputs, inherited coordinate-projection weights, box/WH/DN reference
branches, and subsequent reference updates. Frozen inherited projections can
choose slightly different floating-point kernels; isolated identity-coordinate
checks allow sub-micro precision differences. Forward geometry math is unchanged
at identity modulation. No auxiliary loss or rollout is added.

V21's forward gains small overridable hooks for hypothesis preparation,
coordinate fusion and reference detachment, plus explicit timestep keywords.
Its default computations and strict checkpoint keys remain unchanged. V22 records
its embedding scale in held-out quick-test provenance, using the same fixed32
manifest, seed42, K1/K2/K3 and persistent/reset conditions.

V22's corrected preset selects `coordinate_batch_layout: per_image`. Coordinate
fusion transposes[group,batch,joint,2] to[batch,group,joint,2] before flattening.
Reference-coordinate layout was already correct. This correction retains all
FiLM/AdaLN, gradient boundaries, H-derived proposals and fixed slot pairing.
Parameter keys/counts are unchanged. The model/config default for missing saved
fields remains `legacy`, preserving historical V22 semantics; older V1/V21
implementations also retain their historical behavior. Explicit B2/B4 tests with
different times and nonzero FiLM compare batched and individual execution and
verify changing another image cannot change the first image's coordinate output.
B1 and three-call single-image predictions must remain exactly identical.

### V23 randomly supervised first or second call

V23 reuses the corrected V22 model with no model-key/parameter change. Its new
pipeline samples two uniform times per image and sorts them into t1<=t2, then
chooses one supervision branch per batch with probability1/2. The original target
builder accepts explicit t1 while preserving its default behavior for old callers.
The initial state uses the same GT-interpolated training construction as before.

The first branch applies the original criterion at t1. The second runs the model
at t1 in eval/no-grad mode without target/DN inputs, computes endpoint e1, and
forms x2=x1+(t2-t1)*(e1-source). Both x2 and H1 are detached. The only supervised
call is model(image,x2,t2,targets,H1), with the original complete criterion. There
is no warm-up loss, call-loss sum or trajectory backward. Physical coordinate
updates retain the source from target construction and no extra clamp is added.
The existing evaluation context preserves RNG/module modes during the warm-up.

The private CPU schedule generator is stored in the pipeline buffer
`two_step_rng_state` and strictly restored with training checkpoints. It advances
outside the inherited deterministic-forward RNG fork. V23 adds only that pipeline
state/config; evaluation restores the same V22 model. Validation loss stays on
the inherited single-call path and does not advance schedule RNG. Quick-test
artifacts explicitly identify the V23 training protocol while retaining all six
K1/K2/K3 persistent/reset native sampling conditions. Branch and timestep metrics
are logged under `debug/v23/`.

### V24 matched one-call residual h experiment

V24 preserves V1's spatially anchored encoder proposals, 200 initial candidates,
box-stage top-50 selection, original geometry self/image attention and FFNs,
heads, single-call endpoint-flow objective and sampler. Both arms use the same
corrected per-image coordinate flattening; historical families default to legacy.

The `h_enabled` arm adds a separate residual branch after image attention in each
pose decoder layer. Selected box features query encoded image tokens with spatial
positions to initialize one h per person candidate. A shared local writer updates
h from that candidate's box/joint features and an FFN. A separate local reader
produces a geometry message from the updated h and local geometry. Per-stage
scalar gates start at zero; no extra normalization or keys enter the original
geometry attention path. Denoising rows bypass h. h persists between pose layers
within one call and resets for each forward; cross-call recurrence is excluded.
New-module initialization forks RNG, so paired shared tensors and RNG agree.

`PoseFlowV24PipelineConfig` restores each arm strictly and retains the original
OTA/DN/optimizer recipe. `benchmark_poseflow_v24_control` and
`benchmark_poseflow_v24_h` differ only in h enablement and output naming. Primary
comparison is final epoch-79 one-call full-test AP for the seed42 pair. Additional
seed repetitions are deferred; this is an initial pilot. Equal
optimizer updates/data are controlled; the h arm uses additional compute. This
experiment tests this residual h addition, not every difference in V23.

### V25 matched h-to-DINO spatial decoder

`PoseFlowV25Model` retains V24's pose computation and exports final normal-person
h through `decoder_context_outputs`. Historical models return no extra context
outputs. A checkpointed `HFeatureGridDecoder` expands each h over49 learned spatial
positions, then applies shared LayerNorm/MLP to produce384 channels per cell.
No image, coordinate target or geometry-feature bypass enters this decoder.

`PoseFlowV25Pipeline` computes final pose assignments once and passes those exact
pairs through the criterion's optional `final_matches` argument to both losses.
Auxiliary pose-layer assignments retain their original behavior. A plain helper
owns the lazy frozen teacher, keeping its parameters out of Lightning module
registration, optimizers and checkpoints. The pinned local DINOv3 ViT-S/16 map
excludes CLS/register tokens. Image-normalized GT boxes are converted with each
augmented `img_hw`, then pooled with ROIAlign7×7, aligned=True, sampling_ratio2.
Targets are detached; gradients flow through decoded h into the student.

The auxiliary cosine mean uses valid matched candidates and handles empty local
batches. Invalid/clipped-away ROIs are excluded with explicit counts; finite valid
targets are required. Logs separate raw/weighted feature loss, similarity, positive/
skipped assignments, prediction variance and correct-versus-wrong-box similarity.
These diagnostics help detect a generic feature prediction without establishing
person content from loss reduction alone. Normal pose sampling calls neither
teacher nor auxiliary decoder. `benchmark_poseflow_v25_h_dino` preserves V24's
single-call recipe and uses a separate output namespace.

### V26 joint clean encoder and latent denoiser

`PoseFlowV26Model` reuses V24's ResNet, four projection levels, spatial positions,
level embeddings and masks; it has no deformable attention modules. One image
feature bundle is shared by both training branches and all reverse-sampling calls.
The clean branch embeds14 XY joint pairs in resized unpadded pixels, divided
by100, into256 channels. Missing input joints use the image center. Three separate
self-attention/cross-attention/FFN blocks process all50 proposals with timestep
modulation; time0 identifies the clean branch. Non-affine final LayerNorm controls
clean target scale. A separate three-block denoiser predicts detached clean latents
from cosine1000 corruptions. Its final projection is unconstrained. Shared final
MLP heads predict normalized boxes/joints and one existence logit; a native-output
adapter suppresses class0 and retains CrowdPose person class1.

`build_v26_targets` preserves original GT indices, donor image/person identities
and synthetic-pair provenance. Training uses direct source supervision, with
positive-normalized focal/L1/GIoU/joint/OKS losses and no matcher. The objective is
half the sum of clean/recovered task losses plus mean latent MSE (weight1).
`PoseFlowV26PipelineConfig.denoised_task_loss_weight` defaults to0.5 for existing
configs/checkpoints. The latent-only benchmark sets it to0, giving
`0.5*clean_task + latent_MSE`. Its denoised task metrics are calculated without
autograd. The clean encoder and readout train through the clean task; denoiser
parameters train through latent recovery only. Shared image-extraction parameters
can receive both clean-task and latent-MSE gradients. Corruption source/target
remain detached. Model tensors, sampler and strict restoration are unchanged.
Negative geometry is masked; their latent target remains supervised. Metadata and
existence labels are never model inputs. Original batch poses are copied before
augmentation solely to support synthetic-negative fallback, not inference.

A private checkpointed CPU generator owns donor selection, timesteps and noise.
Validation and monitoring do not advance it. Config restoration builds without
pretrained assets; training initialization audits only image-extraction coverage.
The dedicated V26 evaluator reports Gaussian-only K1/K4 keypoint/bbox AP, raw
coverage and duplicate diagnostics, with exact image manifests and RNG/mode
restoration. Native postprocessing is shared with older families. No old model
restoration defaults change.

### V27 per-block image feature levels

`PoseFlowV27Model` reuses V26 latent blocks/readout/sampling but prepares three
separate `(memory, position, padding_mask)` bundles directly from the ResNet
layer4/layer3/layer2 outputs. Their strides are32/16/8 and channels project to256.
The corresponding source level embeddings retain their original level indices.
`_run_latent_stack` pairs each block with one bundle. Spatial tokens stay separate;
there is no concatenate-then-split path. Padding masks and sine positions are
flattened separately for each level.

The clean stack retains input-pixels/100 MLP and final non-affine LayerNorm; the
independent denoiser retains the final unconstrained linear projection. The
inherited training pipeline preserves constructed supervision/RNG contracts and
native prediction dictionaries. `PoseFlowV27PipelineConfig` explicitly selects
three source image levels for audited initialization, disables denoised task
supervision, and builds the distinct model on restoration. Legacy V26 gets no
subset adaptation unless explicitly requested. Existing evaluator loading needs
no initialization assets when restoring a V27 checkpoint.

### V28 clean-branch image stop-gradient

`PoseFlowV28Model` subclasses V26 and overrides only `encode_clean`: it passes
`(memory.detach(), position.detach(), padding_mask)` into inherited clean encoding.
The original live bundle goes into `denoise`. This blocks clean-task gradients to
ResNet, input projections and learned level embeddings without changing forward
values, parameters, initialization RNG consumption or the sampler. Padding masks
remain unchanged. The h0 corruption-source and latent-target detaches remain.

`PoseFlowV28PipelineConfig` builds the distinct model and defaults denoised task
weight0, preserving V26 latent-only's objective. Shared readout heads receive clean
task gradients only. Image extraction receives latent-MSE gradients through direct
denoiser conditioning, not through clean target construction. Historical V26/V27
behavior and checkpoint interpretation stay unchanged; V28 restores strictly.

### V29 OTA supervision over V28 recovered latents

`PoseFlowV29PipelineConfig` builds `PoseFlowV29Pipeline` with unchanged
`PoseFlowV28Model`. A `pipeline_class` class variable in V26 config is the only
shared construction hook; its default retains the historical V26 pipeline.
The V29 matcher adds a sigma buffer to pipeline state, with no model parameters
or initialization RNG changes. Model-only evaluator restore remains strict and
requires no initialization assets or matching at inference.

`build_v29_ota_targets` creates one GT entry per original supervised person,
reuses `DiffusionRegPoseOTAMatcher` and gathers matched clean latent targets from
canonical firstM slots. Unmatched recovered slots choose the nearest detached
negative latent among clean slots originally labeled negative. Slot queries may
change person or background label relative to their corruption source. The target
helper runs under no_grad; h0 target/source remain detached. Uniform all-slot MSE
trains the denoiser and live image extraction. Constructed clean task trains only
the clean encoder/readout via V28's detached image bundle. No recovered pose task
backpropagation is enabled.

V29 logs OTA counts, coverage, multiplicity, reassignment, converted negative
sources, negative-bank utilization and separate positive/negative latent errors.
Denoised confidence/task metrics use OTA labels; construction/clean metrics keep
original source labels. Standard fixed32 Gaussian K1/K4 and final full8000 native
pose/box evaluation remain the deployed-behavior comparison.

### Four-image overfit diagnostic harness

A frozen temporary harness injects a cached constructed-target draw into existing
V28/V29 pipelines, leaving model/loss/gradient implementations unchanged. It
resamples only timesteps/noise with the checkpointed training generator; evaluation
uses independent generators and checks no training RNG advance. It repeats one
native deterministic preprocessed batch4 with constant AdamW learning rates and
accumulation2, then evaluates clean reconstruction, t0/499/999 conditional recovery
and GaussianK1/K4/K16 under3seed bases. Gaussian evaluation forbids clean encoding;
GT is used afterward for metric computation. This intentionally easier capability
check is isolated from full-data training and cannot establish generalization.

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

## V33 no-clipping training variants

The no_clip and no_clip_rescale benchmark configs reuse PoseFlowV33Model and
PoseFlowV32Pipeline with no architectural or checkpoint-schema changes. Both
disable global gradient clipping; rescale uses latent-MSE coefficient10 versus1.
Other objective routes and detached clean source/target contracts stay intact.
Gradient monitoring lives in the frozen launch callback, outside model execution.

## Box-only latent diffusion (V35)

PoseFlowV35Model reuses V26 image and latent stacks, replaces pose input with
pixel-boxcxcywh/100 and simple MLP, and removes joint/DINO/token-decoder modules.
It emits pred_boxes/existence_logits/pred_logits/pred_latents; no fake keypoints.
PoseFlowV35Pipeline constructs positive perturbed duplicates/foreign negatives/
zero fallbacks and uses one BoxOTAMatcher assignment for recovered box and
canonical clean-latent supervision. Explicit person-to-slot maps survive spatial
Hungarian permutation. Only spatial config introduces fixed5x10anchors, Fourier
projection and denoiser-only query/key positions. The clean stack stays anchor
free. LatentImageBlock optional query_position=None preserves legacy output
exactly, verified against frozen V34. Dedicated quick_eval_v35 handles bbox-only
COCOeval with noNMS; generic strict model loader supports the new config schema.

The V35 spatial latent-only config removes the recovered task backward route
while retaining its no-grad monitoring and OTA-based latent assignment. Clean
encoder/sharedheads/image route keeps clean task gradients; denoiser/image route
keeps latentMSE10. Zero recovered weight uses no-grad task computation; positive
weight retains the original forward/backward behavior. No model/schema change.

## V36 MLP-only clean encoder

PoseFlowV36Model subclasses the V35 spatial box model, removes clean_layers,
and implements encode_clean(input_pixels, existence=None) with input MLP and
non-affine LayerNorm only. clean_existence_input selects4 or5 input dimensions;
existence is binary, unscaled and required only for the5-input variant.
PoseFlowV36Pipeline overrides the V35 training-forward hook to pass constructed
positive labels exclusively to clean encoding. Image encoding occurs once for
the denoiser. Clean codes remain detached at corruption and OTA target selection.
Sampling/box evaluation use the existing image-only denoiser API and shared heads.
Historical V35 defaults and state schemas remain unchanged.

V36 config clean_box_input_format selects pixels(default) or normalized.
PoseFlowV36Pipeline chooses constructed input_pixels or input_boxes accordingly;
PoseFlowV36Model applies/100 only in pixel mode. The legacy input_pixels method
argument name is retained for keyword compatibility; its coordinate convention
follows the configured format. No learned tensor/state-schema changes. Strict
restoration uses the saved config, with absent historical fields defaulting to
pixel behavior. Gaussian inference is unchanged and never receives these inputs.

benchmark_poseflow_v36_box_c_normalized combines clean_box_input_format=normalized
with clean_existence_input=true using the existing V36 model/pipeline. There is
no new production model code or checkpoint tensor schema; both input settings
are saved in config and checked on strict evaluation restoration.

V35 spatial_cost_schedule=late_linear enables per-image multipliers in
BoxOTAMatcher.forward; constant(default) retains the historical call/cost path.
assign_box_targets passes the optional multipliers into the one OTA call used
for task and latent targets. The ramp cutoff is stored in config(default500).
No new model or checkpoint tensors. Matcher diagnostics report actual weighted
cost/effective weights; timestep-bin weight means use actual bin image counts.
The schedule is only used during supervised matching, not Gaussian inference.

spatial_cost_schedule=early_linear evaluates the same validated late-linear
helper at the reflected timestep T-1-t. This preserves endpoints and avoids an
off-by-one at t999. The existing per-image matcher integration, weighted-cost
logging and shared assignment serve both schedules. No new model state or draws.

The optional benchmark_poseflow_v35_spatial_early_cost_q200 config sets num_queries/num_proposals/num_group=200 and grid_rows/columns=10/20. Existing configurable-count paths support it without new modules or code branches: Gaussian state is[B,200,256], anchors[200,2], and box outputs[B,200,4]. Historical50/5x10 defaults are unchanged. Restore requires the matching saved grid/config; learned parameter shapes remain unchanged.


## V37 latent OTA with sigmoid metric alignment

PoseFlowV37Pipeline subclasses V35's training wrapper and reuses PoseFlowV35Model
without model-schema or Gaussian-inference changes. latent_v37.py owns the
mean-squared pair distance, detached all-target OTA with real-person coverage
repair, target gathering, and LatentMetricAlignment. The latter adds six pipeline
parameters (three positive softplus scales and three biases), included in AdamW
and strict pipeline checkpoints. Model-only evaluation needs neither alignment
parameters nor training inputs; it restores the unchanged model strictly.

Clean GT/fake metadata follows V35's global spatial permutation. Recovered
assignments select padded clean slots directly, so one mapping supplies detached
latent targets plus box/existence targets. Alignment uses only unordered nonself
clean-clean pairs within each image. It backpropagates through the clean stack
and shared image features; no labels or assignment metadata enter the denoiser.
V35's original OTA and inference paths are unchanged. V37 config saves the
alignment weights/scales, rejects nonzero spatial matching cost, and uses a
separate fresh experiment namespace.


The optional benchmark_poseflow_v37_reweight config changes only two loss
coefficients: alignment_existence_weight1 and latent_loss_weight100. It uses
the same V37 pipeline, six calibration parameters, model state schema and
strict evaluation loader. Original V37 configuration/checkpoints are unchanged.

## V38 clean posterior and mean diffusion

`model_v38.py` subclasses the V35 spatial box model and adds `posterior_mean`
and `posterior_logvar` linear heads after the existing clean LayerNorm. Heads
initialize inside fork_rng, so all shared tensors and global RNG remain identical
to V37-reweight. `encode_clean` keeps `pred_latents=mean` and mean-decoded outputs;
training additionally returns `posterior_logvar`, `posterior_sample`,
`sampled_pred_boxes`, and `sampled_existence_logits`. `training_outputs` requires
explicit `posterior_noise`. Its sampled branch supplies clean reconstruction;
`q_sample(mean.detach(), t, diffusion_noise)` supplies the unchanged denoiser.
The inherited inference path calls neither clean encoding nor posterior sampling.

`training_v38.py` reuses V37's detached matching/coverage helpers, with one
assignment for recovered task and latent targets. It creates no alignment module
or calibration parameters. The config inherits historical alignment fields for
Hydra compatibility but requires alignment_loss_weight=0 and ignores calibration
settings. New posterior settings are saved in the resolved V38 config; the
pipeline state adds `posterior_rng_state` independently of `training_rng_state`.
Strict model restoration includes both heads; strict pipeline restoration includes
both RNG buffers. Historical V37 schemas and source remain unchanged.

V38 logs raw/weighted KL, real/fake posterior standard deviation, mean-code
variance and lower/upper clamp saturation, plus mean-decoded versus sampled clean
box losses. Existing assignment diagnostics remain available. Recovered box L1
by low/middle/high timestep bins uses the same positive-only sum-over-four-box-
coordinates convention as loss_bbox. Epoch aggregation weights each bin by its
actual positive-assignment count and skips empty bins. This avoids conflating
sampled reconstruction or KL progress with denoising geometry progress.


## V39 training wrapper

`latent_fixed_source.py` constructs one positive/person, source/proposal metadata,
foreign/random negatives and a common random permutation. Unlike V35, it neither
duplicates positive sources nor calls spatial Hungarian placement. Input boxes
retain the existing resized-unpadded pixelcxcywh/100 clean-encoder contract.
`positive_latent_mse` masks fake slots before reduction and detaches clean targets.

`PoseFlowV39Pipeline` subclasses the V26 lifecycle directly, avoiding
construction of the V35 OTA matcher. It reuses the unchanged V35 model, shared
box/existence heads, image attention, cosine1000 corruption and Gaussian DDIM.
Both clean/recovered task losses use the same original target table. A
checkpointed CPU generator owns negative construction, permutation, times and
noise; validation uses a deterministic independent seed. Model state shape and
strict native evaluation loading remain V35-compatible; pipeline config records
constructed matching in a separate namespace.

Positive/fake scores and recovery errors are logged by noise bin with actual
positive/fake counts. The frozen run runner adds excess IoU-hit diagnostics to
the existing bbox-only K1/K4 evaluator and compares identical GT-conditioned
noisy states with correct versus batch-shuffled image features. That diagnostic
isolates the denoiser image-feature route and does not measure Gaussian AP.


### V39-denoiser-grad corruption boundary

`PoseFlowV26Model.training_outputs` accepts a keyword-only
`detach_corruption_source=True`. The default reproduces the original detached
corruption exactly; false passes the live clean code to `q_sample`. V39 saves
the setting in its pipeline config and passes it explicitly. Other callers
retain the original default. `positive_latent_mse` continues to detach targets.

Only training gradient routing changes. No parameters/state keys, random draws,
forward predictions, objective coefficients, target assignment or Gaussian
inference operations change. This is a separate source route into input_mlp,
clean_layers and their image conditioning; it does not add target-side gradients.
The resolved config is required to distinguish the two training behaviors when
restoring because their tensor schemas are identical. The frozen subversion
runner records real/fake latent variance within each image and across the
validation batch, plus shared box/existence readout weight norms.


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

The V42 model accepts an optional `pose_input_builder` callback only at the
existing score-top50 selection/pose-injection boundary. It receives selected
class logits `[B,N,C]`, detached current reference boxes `[B,N,4]`, actual
`[B]` timesteps and complete targets. The paired-proposal pipeline uses the
native box/class/GIoU matcher (`include_keypoints=False`), gathers the assigned
clean GT coordinates, then applies native q_sample/clamping. It returns noisy
coordinates `[B,N,2K]` and the fixed query-to-target indices. Final and all
post-selection pose-auxiliary losses share those indices. Earlier box,
encoder-proposal and DN losses use their original correspondence.

The callback runs inside the single existing forward: no repeated encoder or
box-decoder pass, model parameters, model-owned matcher, or persistent GT state
is added. Without a callback, the original forward and RNG behavior are exact.
The shared `build_diffusionregpose_clean_targets` helper retains native
normalized XY/visibility/box/area conversion for both old and new pipelines.
AP sampling calls the unchanged model without GT/callback, including native
score-top50 selection at every DDIM call. See the paired-proposal report.


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

## Standalone pose-and-box VAE

`src/data/pose_geometry.py` reads one person per annotation record, preserving
label masks and original image-size metadata without image decoding. Shared pure
helpers implement isotropic box-relative joint normalization, inverse geometry,
CrowdPose flips and input-only jitter/dropout. The explicit public codec boundary
uses pixel coordinates, separately from canonical detector coordinates.

`src/models/pipelines/pose_vae/` contains the residual width-256/latent-32 geometry
codec, reconstruction/KL functions, Lightning lifecycle and geometry evaluation.
A 46-value masked box+joint input produces Gaussian posterior parameters; separate
box/joint heads jointly decode through the same latent. Absolute joints are
recovered from decoded offsets and the predicted box. No clean box or image
feature is required at inference. Validation owns fixed corruption/posterior
draws and geometry/latent metrics; qualitative exports include fixed prior draws.

`train_pose_vae.py` composes `benchmark_pose_vae` with the shared TrainingRunner,
native checkpoint I/O and best-validation reconstruction selection. It records
resolved configuration, source/annotation hashes and split overlap. The preset
uses disjoint train/val, then tests the best checkpoint once after 50 epochs.
Existing image-conditioned families and their state schemas are unaffected.
See docs/reports/pose_vae.md for normalization, losses and inference examples.

## Fidelity-first geometry autoencoder

`src/models/pipelines/pose_autoencoder/` adds a separate deterministic codec,
native AP evaluator, Lightning lifecycle and gated experiment orchestrator.
It reuses the established geometry normalization and residual-trunk building
block without altering VAE state schemas. Encoder geometry flows through a 32-D
latent; decoder inputs are only latent and image WH. Paired O(B) box metrics avoid
constructing a batch-by-batch GIoU matrix. Native codec AP retains constant scores,
all validation images and deterministic annotation order, without NMS.

The clean 100-epoch stage selects max(min(poseAP,bboxAP)). Only after both metrics
reach 97.5 does the 50-epoch decoder-only stage start, using fixed training-only
latent std. Both stages checkpoint private augmentation/noise RNG together with
native model, optimizer and scheduler state. Final curves compare the same latent
perturbations across three seeds and reject a robust candidate that loses either
clean gate or fails to improve either 5%-noise AP metric. Existing VAE and detector
pipelines are unaffected; train/val-only assessment does not access test data.

Initial full deterministic control outcome (2026-11-04): the 97.5/97.5 clean
validation gate failed (92.468 pose/96.608 box AP), so the orchestrator did not
start robust training and records no accepted integration checkpoint. Decoder-
only training is implemented and smoke-tested, not validated as a full learned
intervention. This preserves the existing VAE and all detector pipelines.

An unchanged-source, lower-rate 100-epoch refinement subsequently reached
96.049 pose/97.915 box validation AP. Pose still fails the clean gate, so the
decoder-only stage remains untested as a full learned intervention. No model,
loss, interface or checkpoint schema changed for this refinement. Proposed
relative-joint supervision remains deferred for user review.

`pose_vae/fidelity.py` and `fidelity_experiment.py` add a separate warm-started
variational comparison without changing original VAE/AE checkpoint schemas.
AE encoder/readout weights transfer into the Gaussian posterior mean and shared
decoder; a new variance head is initialized from training latent scale. Sampled
reconstruction and calibrated KL jointly optimize the entire VAE. Input masking
preserves clean targets. Separate RNG streams, native checkpoints and the shared
training runner retain exact restoration. Full mean/sample AP and held-out-joint
evaluation prevent mean-only accuracy from standing in for stochastic quality.
`benchmark_pose_vae_fidelity` and `train_pose_vae_fidelity.py` expose each arm;
the frozen eight-arm comparison uses train/val only. See the fidelity VAE report.

The completed comparison recovers near-AE sampled fidelity but retains only a
small learned posterior noise scale and modest external-noise improvement.
Synthetic masking improves hidden-joint recovery at a clean-fidelity cost; it
does not produce verified coordinates for naturally unknown annotations.
`benchmark_pose_vae_fidelity` explicitly sets `save_last_every_epoch=true` so the
existing runner callback advances the latest native state even when validation
does not improve. Original selected checkpoints and frozen runtime sources are
preserved; affected latest-state artifacts are replayed separately.

`poseflow/sampling_v42.py` adds a pure proposal/noisy-pose assignment helper and
an optional V42 DDIM sampler. The helper converts `[B,N,2K]` scaled state to
normalized joint min/max boxes, pads degenerate extents symmetrically, and calls
`DiffusionRegPoseMatcher(..., include_keypoints=False)` with identical synthetic
labels on the target side. These labels only make class costs row-constant;
they are neither GT nor foreground predictions. The complete permutation maps
current proposal indices to incoming state indices.

The sampler uses the existing selection-boundary `pose_input_builder` hook
within one model forward per timestep. The reordered state used for pose
encoding is also used for epsilon reconstruction and the DDIM update. It adds
no model parameters or state keys, does not patch the model forward, and has a
native-disabled mode that delegates directly to the original sampler.
`evaluate_v42_proposal_pose_matching.py` compares both policies on saved fixed32
image IDs, with strict checkpoint restore, optional frozen-source validation,
exact native metric replay and separate sampler provenance. The new sampler can
be loaded alongside the immutable frozen model package without modifying it.

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


## V43 frozen pose-and-box codec

`poseflow/codec_v43.py` wraps the frozen High-KL FidelityPoseVAE, preserving its
pixel interfaces and keeping input gradients through the decoder. The wrapper
standardizes posterior samples with fixed trainval population moments including
posterior variance. `audit_v43.py` uses actual historical detector transforms,
current unpadded image metadata and original annotation identities. Its full
12000-image audit and exact original-validation replay are recorded in
[the V43 report](reports/poseflow_v43_frozen_vae.md).

`model_v43.py` reuses the V40/V40-EMA frontend through shared helpers while removing
the old online clean encoder/readout. A32->256 adapter, three reference/time-modulated
image-attention blocks and256->32 projection predict clean standardized codes.
The frozen decoder produces every final box/joint. `latent_v43.py` fixes each
selected image slot to one complete target before noising, supplies zero latent
background targets and balances foreground/background regression. A separate cached
image-only confidence head prevents latent feedback from changing raw scores.
`training_v43.py` preserves global and private RNG, frozen weights/statistics and
strict model/optimizer/scheduler restore; initialization assets are unnecessary
when restoring embedded state. Separate training/evaluation entrypoints preserve
historical families. Native CUDA and62focused checks pass; full80 training launched
on2026-11-05. Runtime checkpoint monitoring uses epoch/max to retain all80 checkpoints
plus last; the model, optimizer and loss recipe are unchanged.


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

V43 optionally supports velocity prediction and a continuous additive-sigma API;
VP coefficients and angular noise conditioning share the exact sigma. Velocity is
converted back to the32D clean code before the frozen VAE produces all geometry.
Default x0 state/output behavior stays compatible, with explicit mode metadata on
new checkpoints. Optional center-loss normalization floors leave targets and native
OKS/AP unchanged and default to legacy0. The cached-control entrypoint freezes image
context/codec and separately checkpoints source identity, optimizer and train/eval
RNG; it is a conditional denoising probe, not a replacement detector evaluation.

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

V44 retains the reduced native image encoder and box decoder through top-50
human selection, replacing joint expansion with32D latent queries. The frontend
runs once without diffusion time; noisy-code/time projections add directly to
selected human features before two retained native deformable decoder layers.
All geometry is decoded by the frozen High-KL codec. Selected Hungarian targets
are assigned once and reused for every latent output; DDIM carries latents with
fixed references and image-only confidence. Native DN/box supervision is retained
before the latent boundary. See [the V44 report](reports/poseflow_v44.md).


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

V44 `prediction_type=v` uses the unchanged latent decoder to predict VP velocity
and reconstructs clean codes as alpha*h_t - sigma*v before frozen-codec geometry
and DDIM. Its regression target is alpha*epsilon - sigma*h0. Default x0 state
keys/output arithmetic remain compatible. Checkpoints carry explicit prediction
semantics; intentional x0-to-v warmstart negates only latent heads and is separate
from restoration. VAE, context caching, proposal identities and assignment policy
are unchanged. See [the comparison report](reports/poseflow_v44_velocity.md).

The frozen K4 trajectory diagnostic reuses one cached context and assignment per
image/checkpoint to compare forward-corrupted targets with generated states at
t999/749/499/249. Independent FP64 DDIM replay, oracle-clean paths and exact
manual-versus-sample parity pass; this diagnostic introduces no model/interface
change. Measured x0 trajectory sensitivity and velocity intermediate-time
weakness are reported separately; neither changes the frozen-codec contract.

The separate V44 rollout training subclass optionally replaces a forward-corrupted
training input with a detached K4-generated intermediate latent. Warm-up calls
use the same cached context/assignment in eval/no-grad mode, preserving module
flags and global RNG; only the last supervised call carries gradients. Its
matched forward arm shares the timestep mixture. Default V44 uses the extracted
legacy input helper unchanged; model keys and public inference are unchanged.
State-policy checkpoint metadata prevents accidental cross-arm restoration.
See [the rollout report](reports/poseflow_v44_rollout.md).


## V45 progressive bidirectional prefix prediction

`human_context.py` shares V44's native cached frontend without arithmetic changes.
V45 appends one proposal feature plus a zero-initialized learned prediction
embedding per stage and reuses two native decoder layers on the growing prefix.
Hidden states carry without detach; fixed detached references drive deformable
image attention. Active-prefix self-attention is bidirectional, future person
tokens absent. Per-layer32D readouts decode through the frozen codec; there is
no corruption/time/DDIM or geometry feedback. One Hungarian assignment fixes
all targets, with each person's predictions averaged over active stages/layers
before foreground/background-balanced latent and positive geometry reductions.
Native frontend/DN/focal terms occur once. Non-reentrant stage checkpointing
preserves full gradients/RNG. Strict V45 restore metadata, separate entrypoint
and native progressive evaluator preserve V44 defaults. See the
[V45 report](reports/poseflow_v45.md) for interfaces, recipe and limitations.


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


## V46 parallel training rows and sequential ordered generation

V46 keeps the original pretrained V44 image encoder without proposal selection
or native decoder/heads. One cached image memory feeds a fresh2-layer dense
pre-LN decoder. Training batches64 independent rows with bidirectional valid-token
self-attention and shared image K/V; no row mixing or hidden-state recurrence.
Each row's ordered GT prefix has a fixed cosine noise level and a separate
PREDICT token. Both layers read clean32D codes and EOS; frozen codec decoding
provides masked geometry loss. Birth/refinement are separately normalized and
equally weighted; EOS BCE balances row classes per image.

Inference executes64 calls, carries clean latent estimates, adds people until
first EOS/capacity50, and independently re-noises the entire prefix for the next
row. Remaining rows refine after EOS. Final geometry comes from row64/layer2;
birth confidence remains fixed. Optional traces are diagnostic; strict checkpoint
metadata rejects incompatible ordering/schedule/objective/sampler policies. See
[the V46 report](reports/poseflow_v46.md).

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

`model_v45_ordered.py` injects learned50 rank queries/reference seeds into the
retained time-independent native box decoder over original image memory. Its
rank features, detached boxes and image confidence are cached for coordinate
DDIM. `training_v45_ordered.py` assigns sorted GT directly to firstM slots and
background to the suffix, using no Hungarian or DP. No native candidate encoder
or DN supervision remains. The two60-offset person layers,32D task codec/shared
reader, complete-pose EMA JEPA and masked pose losses retain their prior contracts.
Ranks are list positions, not fixed spatial cells; GT positional data never enters
inference. See [ordered task-pose](reports/poseflow_v45_ordered.md) for ordering,
checkpoint and evaluation contracts.

2026-11-09: Ordered task-pose JEPA fresh full80 launch verified at
`/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v45_ordered/20261109_094318`, optimizer/EMA step150.
Native CUDA/strict restore and36 related checks passed; live TensorBoard
`V45-task-pose-ordered-JEPA/.` indexed. First epoch AP pending.

## V47 ordered coordinate diffusion

V47 carries x=6(2c-1), where c contains image-normalized jointXY28/boxCXCYWH4. A shared trainable32->256->32 encoder, V46-style row/time-conditioned dense decoder, per-layer32D/EOS heads and shared nonlinear2D reader replace frozen latent diffusion. Unobserved joints use box-center input fill and existing loss masks; decoded sizes use softplus+1e-6. Both-layer coordinate/geometry birth/refinement losses plus balancedEOS train all modules. Image K/V projections cache once per layer/image. Coordinate DDIM updates existing tokens, newborns receive one forward-noise initialization, EOS/capacity latch and all64 rows refine without hidden-state recurrence. Strict isolated V47 APIs/checkpoints and a gated fresh full80 TRAIN/VAL entrypoint preserve historical models. See [V47 report](reports/poseflow_v47.md).

2026-11-09: V47 full80 launch verified at `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v47/20261109_150341`, trainerPID3490732.55 checks, native full64/50people FP32/batch4accum2/restore and two-seed four-image learnability gate passed. Fresh initial hash matches verification; same fixed32VAL, separate TensorBoard stream, existing jobs/artifacts preserved. See V47 report for mixed refinement behavior and measured cost; fullVAL outcomes pending.

## V48 clean-prefix birth-only generation

`rows_v48.py` constructs independent clean GT prefixes with one PREDICT token and
one EOS row per image, padded to batch maxP+1 (at most51). `model_v48.py` reuses
V47's shared learned coordinate codec, cached image attention and retained
frontend. It removes timestep conditioning. Prefix attention is bidirectional;
hidden states are recomputed each step. Only PREDICT readouts train at both
layers (`losses_v48.py`); earlier geometry is never replaced during generation.
Inference exits individually at EOS.5 or50 births; no post-EOS refinement/noise.
Isolated configuration/pipeline/evaluator/strict checkpoints and verified fresh
full80 entrypoint retain V47's data/evaluation recipe. See [V48](reports/poseflow_v48.md).

2026-11-10: V48 native FP32/capacity50/batch4accum2 passed (peak1.75GiB; exact resumed next loss/max state delta2.24e-8). Fresh full80 active at `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v48/20261110_095254`,PID4145233.560 archived source hashes/initial hash match verification;330 retained frontend tensors exact. TensorBoard V48-ordered-birth-only/. AP cards pinned,67 historical roots preserved. Learned outcomes pending.

### V48 generation-step image AdaLN variant

`model_v48_adaln.py` extends V48 through `StepConditionedImageLayer`: one learned
prefix-count embedding and each layer's SiLU+Linear produce scale,shift and gate
deltas for image-query LN/cross-attention residual. Output projection parameters
start at zero; effective scale/gate1,shift0. Image K/V remain cached and shared.
Step comes from valid prefix length, not row placement or GT total person count.
Self-attention,FFN,codec,birth/EOS losses and immediate stopping are inherited.
Strict metadata retains architectureV48 plus variant identity; isolated pipeline/
config and verified fresh full80 entrypoints preserve running baseline source.
See [V48-AdaLN report](reports/poseflow_v48_adaln.md).

2026-11-10: V48-AdaLN native preflight passed (FP32/50people/batch4accum2,peak1.79GiB,exact resumed loss/max delta1.49e-8). Fresh full80 active at `poseflow_v48_adaln/20261110_135143`,PID73156 beside baselinePID4145233. Common initial V48 weights exact;566 source hashes pinned. Same fixed32VAL, separate TensorBoard overlay, learned outcomes pending.


## Token-age diffusion-forcing rows (first iteration)

`rows_token_age.py` permutes valid coordinate sources/masks/identities once and
constructs independent `[B,50,50,32]` noisy rows with per-token noise coefficients.
Earlier people age through .1,.05,.025,...; each birth/EOS query is pure noise.
Both birth and existing-person supervision masks are provided; clean sources
remain separate target metadata. EOS leaves a refinement tail; final-row prefix
inputs are clean, and the reserved final row limits births to49. Padding is zero
and masked. This helper is not connected to a decoder, loss or inference loop.
See [row-builder contract](reports/token_age_diffusion_rows.md) for exact units,
source mapping, endpoint caveats and deferred DDIM/per-token conditioning work.

## V42 timestep-dependent geometry objective

`training_v42_timestep_loss.py` subclasses the V42 pipeline and native criterion,
without changing the model. It passes actual per-image timesteps to a linear
box/pose geometry schedule. Native assignments are cached once per output and
reused for both the unweighted diagnostic and scheduled objective. Classification
and matcher costs stay unchanged; image-specific weights are applied before
global GT reduction, with DN retaining its native query denominator. All final,
auxiliary, encoder proposal and DN geometry terms keep native branch coefficients.
Checkpoint policy metadata and global RNG reject incompatible resumes. See
[the report](reports/poseflow_v42_timestep_loss.md) for exact weights and protocol.

2026-11-10: Fresh full80 V42 timestep-loss launch verified at `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v42_timestep_loss/20261110_222822`.
PID348105,50 optimizer updates; matching/model/sampler unchanged, scheduled
geometry only. TensorBoard stream `V42-box-center-timestep-loss/.`; native CUDA
and strict restore passed. First learned epoch-end AP pending.


## V49 integrated token-age diffusion forcing

`model_v49.py` reuses V47's coordinate encoder/reader and cached image decoder
while encoding both people and noisy prediction queries. A per-token sigma MLP
replaces row-wide conditioning. `training_v49.py` applies fixed source/mask
permutations and reuses the masked V47 birth/refinement objective.
`sampling_v49.py` uses finite ratio-form DDIM at independent token sigmas; EOS
latches births while all50rows complete. No hidden states carry between rows.
`quick_eval_v49.py` separates source-paired teacher diagnostics from generated
final-assignment birth/refinement and native AP. New strict config/checkpoints,
verified-launch trainer and evaluator leave historical families untouched.
See [V49 contracts](reports/poseflow_v49.md).

## V49-balanced loss and calibration

`losses_v49_balanced.py` implements dimensionless robust geometry, explicit
joint/box coordinate means, per-person age-bucket reduction, calibrated
box/pose priorities and firstEOS-aware BCE. `calibration_v49_balanced.py` measures
four shared-decoder gradient medians on TRAIN without updates and restores model
state/RNG. `training_v49_balanced.py` reuses the V49 model/row builder/sampler and
native evaluator while storing frozen scales and validated provenance in strict
variant checkpoints. Isolated configuration, verification and training entrypoints
preserve historical behavior. See [the contract](reports/poseflow_v49_balanced.md).

V49-balanced verification: frozen four-scale buffers/provenance restore with model/AdamW/scheduler/RNG; native next-update loss exact/maxstate1.19e-7. Common originalV49 model hash/330 frontend tensors exact;70 focused/historical checks pass. Fresh full80 active; see variant report for source and calibration hashes.

2026-11-11: Both V49 experiments were cancelled by the user; existing architecture/checkpoints remain preserved. A successor using sample-local proposal attention, detached stored content identities and sigma-gated residual geometry is planned, not implemented. Its gradient-calibration and box-size residual contracts require review before integration.

## V50 local proposal attention and residual geometry

`model_v50.py` wraps the unchanged `NativeHumanContext` helper, retaining only the native box decoder and required image frontend. Dense row decoder/image K/V helpers remain shared with V47. The selector's prefix pooling excludes PREDICT/padding; attention scores cover remaining image-local proposals and EOS. Hard selected feature/reference content is detached for geometry. `training_v50.py` performs one selected-slot assignment, random source permutation and both-layer supervision with `losses_v49_balanced.py`; native frontend losses occur once. Candidate CE is conditional on non-EOS; balanced BCE trains marginal EOS.

Existing coordinates bypass learned reconstruction through a sigma-gated residual; sizes use a positive reference and bounded exponential correction, with exact identity at sigma0. Newborns retain absolute codec decoding. Coordinate DDIM, fixed local identities, EOS latch and50rows/49capacity govern sampling. V50 checkpoints record selector, detach, confidence, residual initialization/parameterization, fixed calibration and frontend contracts; old variants are rejected. See [V50 report](reports/poseflow_v50.md).

2026-11-11 22:03: V50 native verification passed: cached frontend legacy outputs exact, all retained critical components100% pretrained transfer, strict optimizer/scheduler/RNG resumed update maxstate1.64e-07, checkpoint gradient parity, empty/mixed/masked/seeded sampling and full capacity. Unused native time MLPs removed; sigma conditioning uses V50 decoder embeddings. Fresh full80 run active; see V50 report.

V50 gradient-audit clarification (2026-11-12): selector/EOS pooling precedes dense row decoding and shares coordinate/reference projections, so classification updates that geometry representation while leaving row layers/time conditioning untouched. Global clipping couples private residual-head updates to large native frontend gradients. Existing calibration monitors only initial dense decoder parameters; no protected selector boundary or separate clipping budgets are currently implemented. See [audit](reports/poseflow_v50_gradient_audit.md).

## V51 protected gradient ownership

Logging uses experiment-independent metric tags. The comparison event bridge
preserves original summary values/metadata, optimizer steps and wall times while
moving legacy quick-test sampling policies into run labels. It tails appended
events using persisted cursors; source event files and training processes are
unchanged. V50/V51 geometry diagnostics share `diagnostics/` measurement names.

V51 evaluation opts into second-layer birth snapshots from the existing sampler
trace. Native postprocessing/evaluation runs separately on birth and final
geometry; generation, scores and counts are shared. Optional evaluator support
is disabled by default for V50. A checkpoint observer can publish paired
quick-test birth/final AP for immutable runs after verifying final-output parity.

`model_v51.py` specializes three V50 hooks: detached selector prefix, RMS-bounded
time embedding and clean-anchor geometry interpolation. Both anchor/birth
latents decode through the learned coordinate reader. `balance_v51.py` owns
bounded EMA task statistics and disjoint parameter-family clipping;
`training_v51.py` probes encoder/projection/both row layers periodically and
checkpoints weights/counters. Native confidence, selected-slot assignment,
image-memory sharing and image-only DDIM interfaces remain V50-compatible.
[Detailed contracts and acceptance gates](reports/poseflow_v51.md).

2026-11-12 19:54 CET: V51 verification passed and one fresh full80 run is active at`/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v51/20261112_194212`,PID1815487.42 tests; native full capacity/resume and fresh-noise denoising acceptance passed; original jobs preserved. See [V51 report](reports/poseflow_v51.md) for exact provenance and evidence limits.


## V52 joint geometry and time-conditioned deformable reads

`model_v52.py` retains the native proposal frontend and image-local selector, uses direct32-to256 coordinate features and two preLN self/deformable/FFN layers, and predicts clean box plus center-relative joints from256D readouts. Cached image values are shared across rows; per-token sigma drives image-query Ada-LN. Layer1 samples around current noisy boxes (proposal boxes for births), layer2 around detached predicted clean boxes. There is no32D codec or clean-anchor bypass. `training_v52.py` uses frozen TRAIN-only task gradient scales, equal fixed box/pose and birth/refinement priorities and disjoint clipping; V52 restore is strict and asset-independent. Same-trajectory birth/first-EOS/final native AP uses shared tags and distinct run labels. See [V52 report](reports/poseflow_v52.md).

V52 verification:24 checks; cached-attention/native-frontend exact parity; all direct geometry/time-query gradient paths active; native CUDA checkpoint gradients agree, resumed next update maxstate6.05e-8. Fresh full80 run active; source/config/initial-model provenance archived in its namespace. No architecture benefit inferred from verification.


## V53 restored native expansion without refinement

`expanded_pose_v53.py` adapts native person+14joint expansion, postLN decoder layers4/6, reference positional embeddings and box/pose/extent heads with exact empty-prefix native parity. Each joint gets its own deformable image read. Image values are projected once per image/layer; self-attention batches independent teacher rows, preserving native group locality. `model_v53.py` conditions newborns on clean fixed completed-person tokens and image-local proposals. `training_v53.py` supervises only births at both pose layers, one EOS row per image, once-per-batch frontend losses and frozen two-task TRAIN gradient calibration. No codec, sigma/noise, DDIM or inter-generation geometry update is present. Strict restore and isolated configs/entrypoint/evaluator preserve previous families. See [V53 report](reports/poseflow_v53.md).

V53 verified/active: freshseed42/full80 at `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v53/20261113_231953`,PID2842214.22checks; native full49person/50row FP32batch4accum2 preflight, exact retained frontend/pose initialization, gradient/restore parity. Fixed32VAL manifest unchanged; source/config/native inputs archived. Shared TensorBoard stream added; learned outcome pending.

V53 dashboard delivery verified: shared AP cards visibly overlay historical runs;12 comparison AP series/614existing points retained. Live training confirmed; initial AP only, learned outcome pending. See V53 report.

V53 live TensorBoard refresh caught up after initial history indexing;71+train-loss points verified against actual trainer progress.

## V54 training-only empirical prefix geometry

`training_v54.py` subclasses V53's pipeline without changing its model or losses.
`prefix_augmentation_v54.py` decomposes paired errors into box-relative translation,
log-size and local joint displacements, samples full frozen TRAIN residual vectors,
then reconstructs imperfect prefix coordinates before the shared coordinate MLP.
The same coordinates also supply prefix attention references. A person's sampled
error persists across all independent teacher rows in that forward. Targets,
visibility and proposal ownership remain fixed; eval/inference bypass augmentation.
Strict checkpoints embed bank arrays, masks/count/checksum and source provenance.
Training/preflight orchestration has injectable pipeline classes with unchanged
V53 defaults; evaluator measurement tags are shared across both families.

V54 causal safety bound consults only earlier people at the token's introduction;
changing future GT centers cannot change earlier corruptions.26checks/native
FP32 full-capacity verification pass; current fresh80run3262573 uses identical
initial model weights to V53. Earlier failed10-update namespace retained; corrected
run/source/bank contract is authoritative. See V54 report for provenance and limits.

V54 delivery checks confirm636 source/native hashes match its immutable workspace,
strict frozen-bank resume/native gradient parity, and existing shared TensorBoard
AP/loss semantics. All774historical comparison APpoints retained. Current generation
remains V53's deterministic birth-only native keypoint expansion; augmentation is
training-only and never modifies generated previous detections at inference.


## V55 proposal-to-prefix cross-attention

`proposal_context_v55.py` batches one preLN, eight-head256D attention block across independent teacher rows: candidate/EOS queries read individual PERSON prefix keys/values. Empty prefixes use one learned null token. A zero-initialized output projection preserves same-proposal native expansion initially. `model_v55.py` selects corrected features through a narrow V53 selection hook, keeps original detached boxes, and passes corrected features without detaching into existing joint expansion. Private MLPs score candidates/EOS. `training_v55.py` retains V54 bank/augmentation and adds strict V55 contracts. Shared context receives geometry/classification gradients in the geometry clipping family; prefix/raw candidate inputs are detached. Same-forward gradient audit, full-capacity CUDA verification and fresh80 launch precede outcome claims. See V55 report.

V55 EOS query uses unit-scale initialization to match proposal residuals before scoring LayerNorm; initial0.02scale amplified shared EOS correction gradients in a measured frozen-weight audit. Strict contract records this choice.39tests/native full-capacity CUDA verification pass; exact native transfer and267tensor checkpoint-gradient parity verified. Fresh80run3519742 has finite optimizer progress; model39.60Mparameters and2.09GiB native capacity peak. Existing geometry/selector/native clipping policy retained, with shared-context pre/post norms exposed under measurement-only tags.

V55 delivery verified: shared AP cards visibly include V55; all16existing comparison series/870points retained with79previous roots/5pins. Live native API/CSV parity confirmed through step1019 for training, corruption and shared-context gradients; finite trainer progress beyond1160updates. V54 remains active. Full80outcome pending; retain .plan.

2026-11-15 trained-stage audit: classification gradients in proposal_context share
geometry clipping with the private native pose decoder, although they cannot train
the private decoder directly. This couples private pose updates to classification
gradient magnitude. A separate context/private-geometry budget passed exact
gradient-scaling independence and saved-state AdamW replay checks. The approved
V55 version2 implementation keeps native/selector bounds0.1/sqrt3 and separates
shared_context/private_geometry at0.1/sqrt6 each. The combined bound remains0.1;
clipping runs after accumulation and before AdamW. Aggregate geometry pre/post
norm tags remain comparable; private_geometry has separate norm/factor tags, and
there is no single aggregate geometry clip factor. Legacy checkpoints are rejected
under version2. Existing archived trainers retain their original policy.
Preserve V54 prefix augmentation and
existing objectives. See [audit](reports/poseflow_v55_trained_gradient_audit.md).

Fresh V55 version2 launched on user instruction at
`/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v55_split_clipping/20261115_112330`.
It uses the same initialization/architecture/objectives/augmentation as original
V55, with separate shared-context/private-geometry clipping and unchanged native/
selector budgets. The original archived-source trainer continues independently.
Both runs overlay the same TensorBoard measurement tags; only run labels differ.

2026-11-16 frozen transfer audit: retaining pretrained tensors is not functional
pretraining parity. With full six-encoder/two-box frontend fixed, consecutive
pose stages3/4 yield53.93 fixed32VAL pose AP, but skipping to4/6 yields16.08.
With four pose stages fixed, replacing that frontend with encoder2/4/6+box2
drops54.15 to0.73. Later blocks/reference updates depend on the skipped
computation. Native simultaneous versus independent pretrained person groups
retain54.20/54.15poseAP, bounding an intrinsic penalty from independent decoding.
No architecture change implemented. The generic reduced-depth remapper reserves
source pose head0 for expansion; any future retention of source pose stage3
must also initialize its own head0 explicitly (temporary audit mapping did so).
For trained V55, bypassing corrected proposal features with identities/prefixes
fixed changes34.70 to34.17poseAP; masking only the decoder prefix preserves
34.70AP. Joint queries cannot directly attend prefix tokens in the existing
group mask. See [audit](reports/poseflow_v55_pose_bottleneck_audit.md) for controls,
pretraining exposure limits, provenance and geometry/selection decomposition.

Full-depth preservation adapter (2026-11-16): `full_depth_birth.py` takes ownership
of the complete native6encoder/2box/4pose model, reuses existing frontend/expanded
decoder operators, and decodes supplied IDs in independent rows. Native expansion
and first pose head remain tied. Optional encoded-prefix correction has zero
output initialization and retains the direct raw-feature gradient route; no
extra prefix tokens enter native pose self-attention. Image memory/value
projections are cached once. Explicit masks support padded/mixed/empty rows,
and outputs preserve proposal slots and original image query IDs. All four
layer hidden states/references/coordinates match the frozen independent
reference exactly;54.1509pose/39.1008boxAP. This does not preserve grouped
native boxAP45.20, nor establish learned selection/EOS or post-training AP.
25tests and native full50 synthetic gradient/resource checks pass without
changing legacy models or launching training. See [adapter report](reports/full_depth_birth_decoder.md).

V56 integrates the full-depth path with V55 prefix-conditioned selection/EOS.
`model_v56.py` removes direct decoder-prefix tokens, preserves the raw selected
image-feature gradient and tied expansion/first-pose head, and enables prefix
encoder gradients through attention (`detach_prefix=False`, legacy defaultTrue).
Teacher augmentation, one assignment/random order, balanced objectives/split
clipping and immediate EOS remain; newborn supervision averages all4pose layers.
`training_v56.py`/own config/entrypoint add a strict V56 contract and asset-free
restore.22tests and native full-capacity checkpointed-gradient/AdamW/RNG/scheduler
restore checks pass. Full conditional initial parity is exact; fresh80seed42 run
20261116_135212 has actual sequential initial51.38pose/38.06boxAP and finite
optimizer progress. No diffusion, refinement or trained-weight warm start.
Shared TensorBoard tags retained. See [V56 report](reports/poseflow_v56.md).


2026-11-16 current birth-family evaluation: run_v53_evaluation(..., split='test') reads data.test_ann_path; split='val' reproduces historical diagnostics. TEST aliasing VAL is rejected before dataset loading, and existing subset manifests reject split/path changes. V53-V56 presets use official crowdpose_test.json. The shared trainer checks8000 TEST IDs disjoint from TRAIN/VAL, hashes all three annotations, keeps validation loss computation on VAL, selects checkpoints with fixed32 TEST AP, and writes final full_test results. Resumed evaluation transitions use separate checkpoint/event directories; model/optimizer/scheduler/RNG contracts remain unchanged. Comparable metrics retain their tags; split differences are in run labels and report provenance.

2026-11-17 frozen historical evaluation (completed): temporary orchestration lives
under repository-local tmp/all_run_audit, with immutable evaluator revisions,
model source snapshots, job configs, annotation/subset hashes, prediction ledgers
and additive protocol reviews under results/all_run_audit/20261116_203800.
Each completed endpoint covers official TEST exactly once and derives the common
fixed32 measurement from the same predictions. Native model postprocessing is
retained; box-only/pose-only applicability and upstream-proposal box AP are
explicit. Historical reuse requires archived model-definition identity plus
checkpoint, annotation, coverage, image-only inference and evaluator evidence;
script/importer hashes alone are insufficient. Strict trained-weight restoration
from a compatible source replay is qualified separately from exact original-source
identity. No production evaluation or model interface changes are introduced by
this audit. See [results and provenance](reports/all_run_results.md).
All 200 feasible endpoints completed; the 201 declarations retain one documented
zero-prediction source blocker. Three additional older refiner implementations
were already unavailable before queue admission. Temporary compatibility adapters
preserve native V43 image-size arguments, bijectively migrate the sole configured
C16 block's parameter registration, and fill absent training-only V10 defaults
without changing inference arguments or constructing an optimizer. Strict restore,
source limits and original failed attempts remain pinned in additive reviews.

2026-11-17: the [frozen ED-Pose relation audit](reports/person_interference_test_audit.md)
traces 900 box queries through 100 expanded person groups and four pose stages. Blocking
inter-person box-token attention on 32 TEST images leaves mean matched-pose OKS near 0.8774
and shifts joints 0.33 pixels, while materially changing scores, boxes and duplicate competition.
Score/geometry recombination localizes the pose-AP loss predominantly to score-dependent ranking
and suppression. Joint self-attention, deformable image reads, and carried token features have
separate intervention results. “Previous poses” in this simultaneous decoder refers to prior
stage hypotheses. Sequential person-birth and diffusion-call histories are distinct mechanisms.

2026-11-18: the ED-Pose stage follow-up isolated the joint deformable
cross-attention value contribution while preserving the projection bias and box
rows. Removing it from pose stages 1–3 together damaged pose AP much more
than removing it from stage 4 alone (11.82 versus 64.39, baseline 65.00 on
the same 32 TEST images at 800/1333). This is a joint effect for stages 1–3.

2026-11-18. The [GroupPose runner](reports/grouppose_baseline.md) builds from
pinned upstream PyTorch source and reuses the native grouped transformer,
matcher, and keypoint postprocessor with a compatible CUDA operator. It uses
the shared aspect-preserving resize and CrowdPose evaluator. Keypoints remain
in native order and original-image coordinates. The per-image ledger includes
every evaluated image even if its detection list is empty. Pose-only mode
excludes bounding-box metrics and supports AP50/AP75. The reduced preset has
three pose decoder layers and preserves the coarse-keypoint proposal path.

2026-11-18. `scripts/training/train_grouppose.py` uses the pinned GroupPose
dataset transforms at reduced resolution, native Hungarian supervision,
ImageNet ResNet-50 initialization, and batch-4 FP32 accumulation of two
microbatches. It verifies disjoint TRAIN+VAL/TEST and evaluates fixed32 TEST
at initialization and every epoch. The latest and best checkpoints include
model, optimizer, scheduler, and RNG state. See the
[GroupPose baseline report](reports/grouppose_baseline.md).
