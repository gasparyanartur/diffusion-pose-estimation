# V35 box-only latent detector pair

Approved2026-10-28: isolate object detection before returning to joints. V35-OTA
and V35-spatial use box inputs, no joint/DINO objectives and no17-token readout.

## Model and objectives

ResNet/image projections and two independent three-layer width256 latent stacks
remain from V34. Clean normalized cxcywh inputs become resized-unpadded pixel
cxcywh/100, then MLP4->256->256/GELU. Clean output uses non-affine LayerNorm;
denoiser ends with an unconstrained linear256projection. Shared direct linear
box4/sigmoid and existence1 heads replace the task transformer. No DINO head or
teacher, no joints in inputs, outputs, losses or assignment. Gaussian50x256
DDIM eta0, K4 primary/K1 diagnostic, image features extracted once per sampling
sequence. Clean encoder gets no query-anchor embedding in either variant.

Task = focal existence2 + box L1 5 + GIoU2. Total is
0.5clean_task +0.5OTA_recovered_task +10latent_MSE. Geometry positives only;
normalization is positive count clamped1. The same recovered OTA assignment
selects task box/existence and detached canonical clean-person latent targets.
Unmatched queries use nearest clean negative latent; no-negative-bank cases use
a detached zero background target. Clean corruption source and latent targets
remain detached; clean task gradients reach the image encoder.

OTA explicitly uses classification/boxL1/GIoU costs, not the native pose-cost
branch. Dynamic-K top5, IoU, conflict/coverage repair and candidate penalties100/
10000 remain. Thus V35 changes more than spatial queries relative to V34;
the controlled comparison is between the two V35 runs.

## Proposals and spatial variant

Every valid augmented GT box appears once, independent of joint labeling. Of
R extra slots, floor(R/2) are positive duplicates with XY translation noise
std5 resized pixels; the rest are actual foreign-image boxes transferred in
relative coordinates. Perturbed inputs retain original GT box targets. If a
positive source is unavailable use a foreign negative; if a negative source is
unavailable use a perturbed positive; if neither exists use zero inputs labeled0.
Input centers are not clamped after perturbation. Negative geometry targets are
masked, while actual input geometry/source metadata remain available.

Only V35-spatial globally assigns all50 proposed input centers to fixed5x10
normalized anchors using one Hungarian solve. Originals, duplicates, negatives
and any zero placeholders participate together; every per-slot field moves
with the permutation. Canonical-person lookup is explicit, never first-M.

Only the denoiser receives Fourier1/2/4/8 x/y features -> learned Linear16->256.
The shared uncorrupted embedding enters each block's self-attention Q/K and
image cross-attention Q; values stay content. Anchors also add a squared-distance
cost to OTA. No coordinate-to-anchor task loss or hard regional restriction.
Slot-only permutation equivariance is intentionally replaced by joint
slot-and-anchor equivariance. Global Hungarian minimizes total distance, not
each slot's distance separately; crowded layouts can remain nonlocal.

## Spatial cost calibration

Weight **1** selected without AP tuning on8 deterministic training images,
conditional t0/499/999 and Gaussian K1/K4 outputs. Fresh OTA/spatial variants plus
V34 epoch57 common compatible weights were used for scale context only; launches
remain fresh and import only original EDPose image tensors.

Among candidates sharing the minimum candidate penalty for a GT, median
best-two detection cost gap was0.124 and spatial-cost difference0.131. Their
ratio0.944 selects the smallest positive probe1 from0/1/5/10/25/50/100. Median
native pairwise cost4.06; weighted spatial cost median0.185, p90 0.512. Candidate
penalties remain separate and much larger. Probe1 reduced mean per-image RMS
matched-anchor distance0.468->0.333, preserving full GT assignment coverage.
This is assignment behavior on a cost audit, not learned detection performance.
Detailed records: tmp/v35/cost_calibration.json and cost_calibration.log.

## Verification and execution

61 initial V26–V35 checks and11 final focused tests passed; the focused suite includes additional
canonical-target, box-serialization, attention, padding/batch and config checks.
Historical frozen V34 attention block versus new default-position path is
bitwise equal. CUDA batch4accum2 native Lightning: two optimizer updates for
each variant, finite losses/gradients/parameters, quick box evaluation and
strict teacher/init-free evaluator restoration pass. Common initial model
tensors have identical hashes. OTA41,586,629 parameters; spatial41,590,981.
Peak reserved1.403GiB, allocated1.238GiB or less in smoke. These are wiring checks.

Fresh seed42/80epochs,256/427 resize/augmentations, batch4accum2, FP32/TF32off,
AdamW2e-4/main and1e-4/backbone, wd1e-4, milestones31/61, global clip0.1. Native
trainval training and overlapping val losses remain fit diagnostics. Fixed32
held-out Gaussian monitoring K1/K4 and final epoch79 all8000 held-out K1/K4 use
seed42+imageID. Box COCOeval includes every selected image, scores personclass1,
and uses all50 queries without NMS. No fake joints or pose NMS. Historical V34
bbox AP had pose NMS and additional objectives, so it is a reference rather
than a controlled box-only comparison.

TensorBoard metrics use quick-test/noise_var_100/steps_{1,4}/bbox/AP (and AP50,
AP75, AR), not pose AP. Gradient norms/clipping coefficient, matched counts,
raw/weighted spatial cost, RMS anchor distance and source/fallback statistics
are logged on the shared33329 server. Frozen runs:

- V35-OTA: `/storage/outputs/pose-graph-diff/results/poseflow_v35_ota/20261028_220138`
- V35-spatial: `/storage/outputs/pose-graph-diff/results/poseflow_v35_spatial/20261028_220140`

Each has source/config/input hashes, initialization evidence, cost audit,
launch.log/status.json, seed42 checkpoints and automatic final evaluation worker.
Shared33329 labels are V35-OTA-box and V35-spatial-box; visible progress windows
are V35-ota and V35-spatial in tmux73. Existing jobs are preserved. Startup
passed60finite optimizer updates each; final scalar verification is recorded in
per-run launch_verification.json. First fixed32 AP follows epoch0 validation.

Residual limitations: current clean representations can drift, OTA may change
the corruption source's identity, foreign negatives may accidentally fit real
people, and null background targets are a heuristic. The pair tests the full
spatial bundle (sorting, conditioning and penalty), not its individual parts.

Launch verification completed: **210 finite optimizer updates in each run**.
Actual peak GPU allocated1.466GiB OTA /1.467GiB spatial. Native loss/latent-MSE,
OTA assignment, gradient norms, and spatial raw/weighted-cost/anchor-distance
scalars are populated under V35-OTA-box and V35-spatial-box. Shared Tailscale
33329 returned HTTP200. TensorBoard loads V35 logs first while preserving all
historical labels. Each launch_verification.json pins the observed status and
scalar events. No learned-performance conclusion from this startup evidence.
