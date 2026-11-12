# Feasibility: a POEM-style aligned 2D box-and-pose VAE

Investigated 2026-11-01. Scope: a standalone, image-independent codec that jointly
encodes and reconstructs **boxes, 14-joint poses and existence**, followed later
by a separately trained image-conditioned denoiser. This is a proposed adaptation,
not an implemented model or a performance result. No model, configuration,
checkpoint, optimizer, dashboard or training process was changed.

## Assessment

The standalone codec is a small engineering task. Defining and validating the
neighborhood supervision is the main research risk. Connecting it to the current
latent detector is a separate moderate task because V38 is box-only and currently
ties latent width to attention width. The decoder can mirror POEM's residual MLP;
there is no need for image attention, 3D lifting, cameras, or temporal machinery.
A compact PyTorch implementation of the relevant parts is simpler to maintain in
this repository than importing the TensorFlow training stack. Existing pretrained
weights do not implement the required box/pose/existence reconstruction contract.

## Upstream evidence and what is reusable

Reviewed POEM at revision
`819102cfc28189a4240083bde7a614e57f785f6d`, its most recent POEM-path commit
reported by GitHub at inspection (2026-07-08). The fetched pinned models.py content
exactly matched the master content inspected. Source line numbers below refer to
GitHub source files, including blank lines.

- [models.py](https://github.com/google-research/google-research/blob/819102cfc28189a4240083bde7a614e57f785f6d/poem/core/models.py):
  `fully_connected` (56), `fully_connected_block` (105), `simple_base` (166),
  `simple_model` (208), `_gaussian_embedder` (507). Input FC, two residual blocks
  with two FC layers each, affine BatchNorm/ReLU, configurable dropout and width.
  Gaussian heads predict means and standard deviations; standard deviation uses
  ELU+1. Defaults in the base disable dropout, while train_base explicitly passes
  its dropout_rate default0.3. No coordinate-reconstruction decoder is supplied
  by this Pr-VIPE model.
- [loss_utils.py](https://github.com/google-research/google-research/blob/819102cfc28189a4240083bde7a614e57f785f6d/poem/core/loss_utils.py):
  `compute_keypoint_triplet_losses` (348), `compute_kl_regularization_loss` (579),
  `compute_positive_pairwise_loss` (616). Reusable principles: pair supervision,
  hard/semi-hard negative mining, KL and loss/mining diagnostics. The default
  keypoint distance for negative-label eligibility is Procrustes-aligned MPJPE;
  that label function must be replaced for our task.
- [distance_utils.py](https://github.com/google-research/google-research/blob/819102cfc28189a4240083bde7a614e57f785f6d/poem/core/distance_utils.py):
  learned sigmoid matching probabilities (43), pair distances and diagonal
  Gaussian KL (262). KL sums latent dimensions before the outer mean.
- [train_base.py](https://github.com/google-research/google-research/blob/819102cfc28189a4240083bde7a614e57f785f6d/poem/pr_vipe/train_base.py):
  default alignment uses sampled embeddings, L2-sigmoid matching probability,
  negative-log-mean reduction, triplet-ratio margin log2, positive pair weight.005,
  KL weight.001 and semi-hard negatives. It supports CENTER embeddings too.
  Loss/mining embeddings can differ; the default mining substitutes means for
  samples. Its training labels use 3D joints, even though inference takes 2D.
- [input_generator.py](https://github.com/google-research/google-research/blob/819102cfc28189a4240083bde7a614e57f785f6d/poem/core/input_generator.py):
  `create_model_input` performs person normalization and supports joint jitter,
  dropout and projected 3D augmentation. We should reuse augmentation principles,
  not its person-centered/view-invariant coordinate contract.

Our proposal keeps Gaussian embeddings, metric supervision and residual MLPs. It
adds reconstruction, boxes, existence and masks; uses 2D geometry labels; retains
image position/size; and removes 3D/camera dependencies. It is method adaptation,
not reproduction of published Pr-VIPE or a checkpoint conversion.

## Proposed joint representation and decoder

Per person/slot:

- box b = normalized (cx,cy,w,h), four values;
- joints k = 14 normalized XY pairs, 28 values;
- existence e in {0,1}, one value;
- annotation mask m, 14 bits, supplied to the encoder to distinguish missing
  coordinates from real zero/border coordinates.

Total encoder input:47 values. Reconstruction target:33 values (4 box +28 joint
coordinates +1 existence), with masks supplied as metadata, not reconstructed
as person geometry. Normalize X by image width and Y by image height. Do not
root-center or divide away person size in the actual encoder input.

Proposed initial architecture:

```
[box4, joints28, existence1, annotation_mask14]
  -> FC256 -> two residual [FC256, FC256] blocks
  -> mean32 and logvariance32
  -> z = mean + exp(logvariance/2) * posterior_noise
  -> FC256 -> two residual [FC256, FC256] blocks
  -> box4, joints28, existence_logit1
```

This mirrors POEM's trunk structure, with a newly trained mirrored decoder.
No final normalization on mean or sampled latent. A bounded logvariance head,
such as V38's [-10,2], is a straightforward numerical adaptation of their
positive-standard-deviation head; it is not exact parameterization parity.
Keep reconstruction from samples and report mean decoding separately. Retain
BatchNorm if matching the reference architecture; freeze its running statistics
and disable dropout in the frozen codec stage. A normalization change is a
separate design choice, not required by the adaptation.

Parameter counts calculated from the architecture above, including biases and
five affine BatchNorms in each trunk, excluding running-stat buffers/calibrators:

| Hidden width | Encoder | Decoder | Total |
|---|---:|---:|---:|
|128|81,728|75,809|157,537|
|256|294,464|282,657|577,121|
|1024|4,323,392|4,276,257|8,599,649|

These are architectural calculations, not training/runtime measurements. Width256
and latent32 are a conservative first candidate; 16 is a later compression check
if joint+box reconstruction has enough headroom. POEM's own embedder is not an
encoder/decoder pair; the1024 row is our proposed pair at the reference width.

Decode box and joints with separate output heads from the same latent. Do not
compute the annotation box from joint extrema. Use unconstrained joint outputs
because labeled annotations can fall outside [0,1]. Box sizes should remain
positive, e.g. with softplus; final head choices should follow the augmentation
bounds rather than blindly sigmoid/clamp every coordinate.

## Data audit: material edge cases

Read raw CrowdPose JSON with the repository's existing filter: iscrowd=0 and
num_keypoints>0. No image decoding, fitting, augmentation, or model inference.
Evidence: `tmp/poem_box_pose_audit/data_audit.py` and `data_audit.json`.

| Split | Images | Usable people | Missing-joint fraction |
|---|---:|---:|---:|
|train|10,000|35,659|22.55%|
|val|2,000|7,074|22.33%|
|trainval|12,000|42,733|22.51%|
|test|8,000|29,122|22.91%|

Train has21,727 people with at least one missing joint (60.93%). Median labeled
joint count is12; minimum1. Occluded but annotated joints (visibility1) count as
valid geometry; visibility0 does not. Train has64 labeled joints outside the
image and18,906 outside their annotation box; all kept train boxes are positive,
finite and inside the image. Thus clipping joints or deriving boxes from joints
would alter real targets. Geometry normalization is finite in all inspected rows.

The existing detector config trains on trainval and validates on val: all2,000
val images overlap that training set. For independent codec validation, initially
fit train and evaluate val. Reserve test for final evaluation. The audit only
inspected split counts/annotation properties; no model was tuned on test.

Raw annotations and masks already exist in src/data/datasets.py and transforms.py.
The detector loader converts keypoints to [-1,1] canonical coordinates, whereas
box targets use image-normalized cxcywh. A codec adapter must convert joints back
with (gt_keypoints+1)/2; alternatively read normalized records directly from JSON.
A geometry-only per-person dataset avoids image loading and image augmentation
cost entirely.

## Objective: reconstruction, Gaussian prior and aligned neighborhoods

Proposed structure:

```
L = L_box + L_pose + lambda_exist * BCE(existence_logit,e)
    + beta * KL(q(z|b,k,e,m) || N(0,I))
    + lambda_triplet * L_triplet_ratio
    + lambda_positive * L_positive_pair
```

Box reconstruction uses actual annotation boxes; pose reconstruction uses labeled
joints. Geometry is real-only. Existence, KL and the applicable metric supervision
include backgrounds. L1/MSE plus GIoU/OKS are available locally; task-specific
terms can improve fidelity to AP-sensitive errors. Define reductions explicitly
(per person, valid coordinate, latent dimension) before choosing coefficients.
Do not carry over detector loss coefficients without checking their scales.

Existence is an observed binary input and a Bernoulli reconstruction output, not
a discrete sampled latent coordinate. Gaussian z encodes existence continuously,
so one Euclidean matching metric remains usable. The real dataset supplies e=1;
construct e=0 examples explicitly, including geometry copied from real records
with e=0, to avoid making absence identifiable only by a zero-input shortcut.
False-slot geometry remains unsupervised under the existing confirmed mask. A
compact absent-code region is allowed. Retaining distinct fake geometry would
require an additional, explicitly chosen objective; it is not implied by BCE.

POEM sums KL over dimensions; our V38 averages them. At D32 a POEM-style .001
summed KL has the same numeric weight as .032 dimension-mean KL, before accounting
for the new reconstruction objective. Therefore .001 or .1 are not transferable
coefficients without specifying reduction. Tune using reconstruction and metric
validation, and log each raw/weighted loss, active mean variance, posterior std,
real/fake statistics and clamp saturation.

### Use the same mean-distance for training and matching

Recommended first adaptation:

```
d(i,j) = mean_D((mean_i-mean_j)^2)
p(i,j) = smoothed_sigmoid(b - softplus(a_raw)*d(i,j))
Dmatch(i,j) = -log p(i,j)
L_triplet = max(0,Dmatch(anchor,positive)-Dmatch(anchor,negative)+margin)
L_positive = -log p(anchor,positive)
```

The trainable calibration preserves distance ranking because p is monotone in d.
The matching cost can remain raw mean-squared latent distance. However, V37/V38
Dynamic-K uses exp(-d), which depends on absolute scale, not only ranking. Retain
and inspect its distance/quality/count distributions in the first integration;
using calibrated p for Dynamic-K instead would be an explicit design change.
Removing calibration is therefore harmless for ranking, not automatically for
OTA multiplicities. The paper's default
uses sample-based distribution matching; our deliberate CENTER/mean adaptation
avoids supervising uncertainty distances that the mean-MSE matcher never uses,
and avoids the K-squared sample-pair cost. Posterior variance still learns from
sampled reconstruction and KL. No exact global coordinate-distance regression is
required. Supervision teaches local useful neighbors and near/far separation.

### The nontrivial part is defining meaningful positives and negatives

Replace Procrustes-aligned 3D MPJPE with detached 2D task relevance. A proposed
label/mining rule combines normalized center distance, box-size differences,
articulation distance over jointly annotated joints, and existence mismatch.
Use box-relative articulation *for the pose component of the label* if we want
to avoid counting translation/size once per joint and again through the box.
Actual codec inputs/outputs remain global image-normalized coordinates.

Use near/far thresholds with an ignored uncertain band, not every different
annotation as a negative. Compare geometry, not person identity. Generate nearby
positive partners with small consistent perturbations and mine semi-hard real
negatives. Include pairs that isolate each relevant change: same pose at different
positions; same position at different sizes/poses; e0/e1 with identical geometry.
Require sufficient common annotated joints for a pose-distance label; otherwise
skip that component rather than treat missing coordinates as zeros.

Define one joint notion of matching relevance before mining. Independent
property losses sharing one scalar distance can give contradictory demands
(e.g. same center but different articulation). Their weights/thresholds determine
the compromise and remain an explicit review choice. This is label supervision
for learning a metric, not replacing latent matching with a fixed coordinate cost.

Balance real-real geometry pairs, fake-fake existence pairs and mixed existence
pairs. Most negatives being e-mismatched would permit a good binary classifier
with poor geometry neighborhoods. Measure per-property retrieval and include
hard real-real examples so existence cannot hide failed geometry learning.

## Heavy augmentation without teaching the wrong invariance

- Apply translation/scale/rotation/flips consistently to boxes and joints; a
  rotation transforms all four box corners before taking an axis-aligned box.
  Preserve CrowdPose's actual joint order and left/right swap mapping.
- Keep real poses in the training mixture. Add bounded geometric perturbations;
  uncontrolled joint noise is not evidence of valid pose diversity.
- Add missing-input/dropout robustness while keeping the original labeled target
  mask separate from the input mask. Do not supervise unannotated coordinates.
- Translated/scaled examples get their new coordinates as reconstruction targets.
  They are not automatically identical-latent positives. Nearby perturbations may
  be positives according to the chosen task relevance; large changes may be negatives.
- Decide clipping/cropping/existence rules explicitly. Never silently clamp raw
  out-of-frame labeled joints. Report real validation and fixed-seed augmentation
  stress results separately.

## Staged training and acceptance gates

1. Train this standalone codec from geometry only, with reconstruction, KL and
   alignment. No Hungarian/OTA correspondence is needed for known person records.
2. Evaluate untouched val: box AP and pose AP separately for posterior means and
   sampled codes, plus padded e0 false positives and existence accuracy. Treat >90
   reconstruction AP for both as a target gate, not a predicted result. Verify an
   identity round-trip/evaluator baseline before interpreting codec AP. Report the
   oracle-person inputs and actual coordinate conversion/annotation masks.
3. Validate the *local* metric: neighbor precision, close-versus-far triplet success,
   pairwise existence errors and matching choices on crowded examples. Stratify
   position/size/pose/existence changes; global correlation alone is insufficient.
   Apply controlled latent perturbations and measure decoded box/pose degradation.
4. Freeze encoder, decoder and all normalization buffers in eval mode. Then train
   only the image-conditioned denoiser. Retain recovered geometry gradients through
   the frozen decoder to denoiser outputs; frozen weights do not require no_grad
   around decoding. Means/targets and corruption sources remain detached. This
   differs from removing recovered task loss while jointly moving the codec.

Explicit posterior/augmentation/pair-mining RNG and their checkpoint states should
be separate from detector proposal/diffusion RNG. Early reconstruction-focused
warm-up or KL ramp may help, but schedules are unselected possibilities, not a
confirmed requirement or performance claim.

## Later integration: concrete local changes

| Existing component | Required work | Difficulty |
|---|---|---|
|New standalone geometry codec/MLP|Mirrored residual MLP, posterior sampling and box/joint/existence heads|Low|
|New geometry-only dataset|Read existing JSON, normalize, masks, augmentations and e0 records|Low–moderate|
|New metric training loss|2D pair eligibility, semi-hard mining, balanced groups, same mean metric|Moderate; largest research uncertainty|
|New codec evaluator|Box and pose reconstruction AP, sample/mean split, local-neighbor tests|Moderate|
|model_v26/model_v35 sampler assumptions|Separate latent_dim32 from hidden_dim256; input/output projections; Gaussian/DDIM state32|Moderate|
|training_v35 input draws/targets|Noise width32, preserve box+pose+mask correspondence, explicit existence|Moderate|
|latent_v37.assign_latent_targets|Current gather returns only positive/boxes; new wrapper must gather joints/masks using selected_slots too|Low code volume, important correctness|
|Current V35 evaluator|It is box-only; add pose output/export/evaluation while keeping bbox noNMS contract|Moderate|

Do not shrink the attention model to hidden_dim32 to make tensor shapes fit.
Use lift32->256 inside the denoiser, project256->32 at its output, and keep
sampler/noise/corruption states in32 dimensions. Existing latent OTA distance and
coverage logic can be reused because it is dimension-agnostic. New proposal
metadata must carry boxes, joints and masks through one consistent permutation.

V35's build_v35_targets does not carry joint coordinates/masks and perturbs only
positive duplicate *input* box centers while supervising the original box. That
is a denoising-style clean branch contract. A fixed geometry codec should encode
its exact supervised geometry: box/joint jitter either transforms both labels
consistently or belongs to a separately specified corruption process. Reusing
the old builder unchanged would silently create an endpoint/decoder mismatch.
Likewise V26's old builder uses different proposal construction and cannot simply
replace V35's 50-slot sorting. Add a new isolated builder rather than change old
families. Preserve whether all padded fakes remain candidates and real-person
coverage, as in V37/V38.

The codec can be trained and validated before making any detector changes. A
first reviewable implementation could add isolated geometry_codec.py,
geometry_codec_losses.py and geometry_codec_data.py under the poseflow package,
a scripts/training/train_geometry_codec.py entry point, a dedicated config and
focused tests. A later separately reviewed model version would integrate it.
No new version name or full detector run is selected by this investigation.

## Minimum future verification

- Shape/finite/analytic KL and sampled reconstruction gradients to both posterior
  heads; head bounds; no normalization after mean.
- Real-only box/joint masks; e0 existence/KL/metric participation; visibility1
  remains labeled; empty/minimal-common-joint pairs stay finite and are skipped
  deliberately where pose labels are undefined.
- Matched box/joint normalization and flip/affine inverse round trips; no mutation
  of raw annotations; known tensor order, missing fill and separate target masks.
- Near/far/tie/semi-hard eligibility and class balance; ignored-band pairs do not
  become forced negatives; matching and alignment use the same mean metric.
- Strict codec/config/optimizer/all RNG continuation and frozen BatchNorm/dropout
  behavior; recovered task gradients reach denoiser through frozen decoder while
  codec parameters/buffers remain unchanged.
- Later full integration: one assignment gathers all modalities, latent32/hidden256
  shapes and DDIM transitions, no image/GT in codec reconstruction paths except
  explicit geometry inputs, pose+bbox evaluation contracts and unchanged old-family
  regressions. Native CUDA smoke precedes any full detector launch.

## Open design decisions for implementation review

- Hidden width256/latent32 candidate versus alternatives; normalization/dropout.
- Combined neighborhood definition, property weights/thresholds, minimum common
  joints and real/fake pair proportions.
- Reconstruction reductions/coefficient scales, KL beta and any warm-up schedule.
- Whether inactive geometry should collapse to absence or intentionally retain
  geometry; current confirmed reconstruction mask permits collapse.
- Precise augmentation bounds and treatment of cropped/out-of-frame coordinates.

There is no measured training time, reconstruction AP, latent-neighborhood quality
or downstream AP for this proposed codec. The source audit supports feasibility;
only training and the specified gates can establish usefulness.
