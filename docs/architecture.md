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
