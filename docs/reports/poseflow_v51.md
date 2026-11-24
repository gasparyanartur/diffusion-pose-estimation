# V51: protected gradients and clean-anchor refinement

This single successor integrates the user-approved gradient repair after the V50
model/optimizer audit. It is not an ablation and cannot isolate contributions.
Verification and launch evidence are recorded below when available.

## Paired birth and final AP

The V51 evaluator now extracts each actual birth's second-layer geometry from
the existing image-only trace. Birth and final predictions retain the same
sequence identities, person counts and birth scores; each geometry set receives
the same native resize/postprocessing and pose/bbox evaluation independently.
Pose NMS can retain different records when geometry changes. These are birth
snapshots of the refining trajectory, not an independent birth-only sampler.

`metrics` and the existing AP tags retain final-output semantics. `birth_metrics`
and `birth_predictions.json` record the additional evaluation. Both stages log
under the same `quick-test/{metric}` tags in separate birth/final run labels.
V50 uses these same shared tags; birth extraction remains opt-in for V50.

The live run imports an immutable source snapshot. A separate, versioned
checkpoint evaluator follows completed checkpoints without restarting training.
It uses the same fixed32 VAL IDs, seed and data configuration, validates strict
restore and final AP agreement within1e-8, and only then emits the paired scalars.
Checkpoint/source/annotation hashes and parity results accompany each epoch.
Historical epochs whose checkpoints were pruned cannot be reconstructed.
The original best-checkpoint selection remains based on final AP; TEST is unused.

## Forward model

V50's original ED-Pose backbone/projections/three retained image encoders/native
box frontend and sample-local top50 proposal identities remain. Geometry sees
detached proposal content/reference identity; selected-slot assignment is fixed
through a forward pass. Birth/refinement use random source order, image memory
computed once, two shared dense decoder layers, per-token sigma conditioning,
50 independent teacher-forced training rows,49 people plus an EOS row, and
coordinate DDIM/EOS-latched sequential inference. No LDM or hidden recurrence.

The selector now reads a detached geometry prefix. Selector query/key/EOS and
native candidate feature gradients remain trainable, while classification cannot
reshape the coordinate encoder/projection/reference projection. The per-token
additive time MLP is capped at RMS1, with finite derivatives at zero.

Each refinement head predicts a clean latent anchor, decoded through the same
jointly trained coordinate reader as birth predictions. Anchor heads are copied
from corresponding fresh birth heads at initialization and then train independently.
For normalized deattenuated input `b` and clean anchor `a`, `g=min(sigma/.1,1)`:

- XY: `b + g*(a-b)`.
- Positive width/height: `clamp(b,1e-6)*exp(g*log(a/clamp(b,1e-6)))`, with a finite
  exponent clamp. Sigma0 returns the exact input arithmetic, including sizes.

This is a time-gated residual toward a clean prediction. The subtraction supplies
the coordinate path explicitly, avoiding V50's need to learn an enormous inverse
noise-scale gain. At age1 the anchor is the clean prediction; older corrections
shrink with sigma. Its detection/refinement quality remains an empirical question.

The only V50 edits are three default-preserving hooks. V50 historical forward,
frontend, sampling and gradient regression tests remain required.

## Gradient policy

Retain masked coordinate/robust geometry terms, per-person age-balanced reduction,
positive birth/refinement tasks and balanced EOS groups. Four base geometry
contributions preserve birth box:pose3:1, age1 1:1 and older1:3 priorities.

Initial32 TRAIN batches measure task gradient RMS on four shared modules:
coordinate encoder, coordinate projection, decoder layer0 and layer1. Within
each module, task norms are divided by their sum; their geometric mean gives
one score per task, equally weighting modules. Scalar multipliers target shares
[.125,.375,1/3,1/6] for pose birth/box birth/pose refine/box refine, use EMA.95,
are normalized by a common geometric mean and capped[.1,10]. Missing gradients
are skipped. This policy does not promise equal gradients in every module.
Statistics update every100 optimizer steps using extra autograd probes from the
same forward; ordinary microbatches use one backward. No repeated calibration
on restore or validation. All weights/statistics/probe counters are checkpointed.

Clip disjoint native/image, selector, geometry parameter families after accumulated
gradients, each to `.1/sqrt(3)`. Combined norm stays<=.1. This replaces the old
single global clip; it is not followed by a second independent global clip.
The AdamW main/backbone rates remain2e-4/1e-4, wd1e-4, LR drops31/61.
TensorBoard records task gradient EMAs, multipliers, family clip factors, selected
parameter relative updates, time/prefix scales, raw/weighted losses and native AP.

## Verification and run contract

Failing tests precede implementation. Required checks cover protected routes,
bounded/zero time conditioning, analytic coordinate sensitivity, sigma0 identity,
both-layer supervision, masked/empty batches, deterministic image-only sampling,
disjoint clipping, capped statistics, checkpoint corruption/asset-free restore,
checkpointed gradients and unchanged V50 behavior.

Native preflight uses FP32 CUDA,50rows/49people,batch4/accumulation2 and non-square
256x416 capacity inputs alongside existing jobs; measure memory/throughput and
strict resumed next-update agreement. A bounded1000update fixed-TRAIN diagnostic
uses fresh corruption each update and8 independent evaluation draws: pose-refine
and joint coordinate MSE must both beat the noisy bypass by10%. This is a wiring/
learning acceptance test, not AP or unseen-image generalization. No full launch
is permitted if this fails. A first verification found an overly exact CPU/GPU
provenance comparison; numerical tolerance was fixed and corruption testing added.

The authorized full run is fresh seed42/full80, original ED-Pose transfer, fresh
codec/row decoder/heads/AdamW. Preserve V50/V42 trainers/artifacts. Same TRAIN10k,
VAL2k, augmentations, aspect resize256/max427, FP32,batch4/accum2. Initial and
every-epoch fixed32VAL image-only native AP; best bymin(pose_AP,bbox_AP), retain
best/last, selected checkpoint evaluated on all2000VAL after completion. TEST is
unopened. Namespace `poseflow_v51`, TensorBoard `V51-gradient-balanced-residual`,
AP tags `quick-test/{pose_AP,bbox_AP}`; experiment identity belongs in run labels.

## Usage

```bash
venv/bin/python -m scripts.training.verify_poseflow_v51 --output <fresh-verification-directory>
venv/bin/python -m scripts.training.train_poseflow_v51 verification_file=<directory>/verification.json
```

Use an immutable source snapshot for production; verification checks source,
config/input hashes, initial model identity and denoising/native acceptance.
Checkpoints record the V51 contract and reject older families; strict restore
requires no original initialization asset. Live full80 completion and quality
must not be inferred from successful preflight or a launch.


## Verified full80 launch (2026-11-12 19:54 CET)

Fresh run `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v51/20261112_194212`, tmux`poseflow-v51`, PID1815487; active epoch1
(index0), confirmed optimizer step170, finite loss4.988847,
shared-GPU throughput5.71images/s and live peak
2.62GiB. V50PID1102983 and
V42timestepPID348105 remain running. This is one fresh combined run, no pilot
cutoff and no verification-model warm start. Full80/all2000VAL outcomes remain
asynchronous; TEST unopened.

42 focused/historical regressions pass. Native FP32 CUDA50rows/49people,
batch4/accum2,256x416 capacity verification peaks at
1.92GiB; synthetic training throughput
4.41images/s and inference
3.66images/s alongside existing workloads.
Resumed next loss is exactly27.389497756958008; maximum next-state delta
6.8e-08. All retained frontend components have100%
measured initialization coverage. Initial model hash
`2fc3008fc594ddd080078d4af365d774676420d322efbfa19277e8206598f5ab` matches between verification and fresh training;
605 source/config hashes and896 archived source/test/doc files are retained.
Parameters39,146,637; trainable38,924,237.

The fresh-noise learning acceptance used4 augmented TRAIN images,
18people,1000updates and8 separate evaluation noise draws on
those same identities. Pose-refinement task improved76.3%,
joint coordinate MSE62.3% and box-refinement
23.3% against the input bypass. Final pose task
0.00507212 versus0.02138041; joint MSE
1.0888137e-05 versus2.8870267e-05.
This passes the wiring/learning gate; it is not a detection/generalization result.

The same fixed32VAL IDs are preserved. Initial image-only AP is0/0, a pre-training
baseline. Native pose/bbox AP cards are visible and pinned in the existing
TensorBoard33332 dashboard. All73 previous roots and23pins are preserved;
new totals74roots/25pins. Scalar API and live event files verify losses,
adaptive weights, gradient clip factors, relative updates and bounded time
RMS (observed below1). Screenshot `tmp/v51/dashboard.png`; full proof and
source/config/input/initialization manifests are archived under the run root.


2026-11-12 paired AP verification: epoch006 (7 completed epochs), globalstep8750,
fixed32 VAL seed42. Birth pose/bbox AP6.1536/25.7764 versus final5.4534/23.9852.
All57 final native prediction records reproduce the trainer exactly; all final AP
fields have zero difference. Birth also retains57 records after native processing.
Thus refinement reduces pose AP by0.7002 and bbox AP by1.7911 points on this subset.
Only epoch006/last was retained at setup, so earlier epoch birth AP is unavailable.
The independent evaluator PID1943783 follows future saved checkpoints and writes
`/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v51/20261112_194212/diagnostics/birth_ap/tb_logs`, indexed as `V51-birth-final-AP`
on the existing dashboard. All74 prior roots and25 pins remain; four AP pins added.
Training PID1815487 was not restarted.14 focused/historical tests passed;603 original
non-evaluator source hashes unchanged. .plan is retained for outcome review.


2026-11-13 logging correction: experiment-specific AP tags were a presentation
error. The active dashboard now reads live derived event views, with
`quick-test/pose_AP` and `quick-test/bbox_AP` shared by all experiments.
V50 final, V51 final and V51 birth overlay on those same charts; sampler settings
and output stages belong in run labels. Original event files/checkpoints and
trainers are untouched. Legacy raw-event namespaces remain only in preserved
source artifacts. V50/V51 logger code now emits shared quick-test/diagnostic tags;
new checkpoint-evaluator output uses separate birth/final event directories.
The compatibility bridge follows the existing immutable trainers and observer.

All450 source AP series/24,884 captured points reproduce original wall times,
optimizer steps and values exactly, without duplicates. Seven focused tests
cover mapping, append/restart/crash recovery and evaluator regression. Source
roots/config and verification artifacts are in `tmp/tensorboard_comparison/live`.
All75 original roots are preserved as inputs; the duplicate V51 observer final
alias is omitted from the active view because the trainer supplies that curve.
The bridge's initial backfill completed and now tails new events. Original
experiment-specific AP pins were replaced by shared AP pins. The browser visibly
shows the V50/V51 curves together; no new training or inference was run.
