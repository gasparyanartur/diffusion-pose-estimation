# V42 paired proposal sources and labels

Approved 2026-11-04, one full-pose V42 experiment named `v42-paired-proposal`.
This isolates training correspondence from the corrected box-center/uniform-t
V42 control. The existing controls and cancelled noisy-input-matched run remain
preserved. JEPA is deferred.

## Construction and loss contract

At the existing selection boundary, preserve native score-top50 selection from
200 refined image proposals. Hungarian-match the selected class logits and
actual detached incoming reference boxes to valid complete GT person records
using native class/box-L1/GIoU costs, without pose costs. Gather each assigned
person's clean XY into its proposal slot, then apply native coordinate q_sample
with independent uniform per-image integer t0..999. The same actual timestep
conditions the image encoder and corruption. Unmatched slots use native random
padding scaled by6, followed by q_sample/clamping, and remain classification
background without geometry loss. GT people with no labeled joints, including
cropped-out people, are excluded from both positive sources and targets. More
valid people than selected slots raises an explicit capacity error.

The source assignment is also the label assignment for final class/box/keypoint/
OKS and every post-selection pose auxiliary loss. Preselection200-query box,
encoder proposal and DN losses retain native matching. Validation task loss uses
the paired construction without DN. Native Gaussian AP sampling has no GT or
callback and preserves DDIM equations and per-call score-based proposal selection.
Unlike V40, training never forces matched candidates into the top50.

Implementation: shared pure complete-person target helper in native training;
one optional model-forward injection callback; isolated paired-proposal pipeline
and two configs. No parameters or state keys change. The callback runs within
one backbone/encoder/box-decoder forward, preserving feature gradients and
reference detaches. The independently ordered diffusion bank is ignored for
positive sources.

## Verification

The initial test failed at the missing clean-target helper, before implementation.
129 focused/native regression checks pass, covering known swapped proposals,
known-noise q_sample at t0/500/999, matched/background/empty/cropped/batched cases,
capacity errors, shared final/pose-auxiliary assignment, actual mixed nonzero
model/corruption times, task gradients, state keys and strict model restoration.
Tests include existing explicit fixed-label losses when outputs favor different
labels and historical/native evaluator checks.

A separate frozen-before-edit comparison passes 16 target/noise/RNG cases
(explicit/implicit source banks, masks, empty targets, seeded/global draws,
omitted/mixed times). Default V42 forward train/eval, every output field, task
loss and RNG state match exactly. The seeded initial model matches the control
with identical parameters and state keys. Native CUDA checks and launch evidence
are recorded below when complete. These are implementation checks, not learned
AP evidence.

## Training and evaluation

Fresh EDPose initialization with seed42; no old V42 checkpoint resumed. Full80
epochs, FP32, batch4/accum2, clip0.1, resize256/427, unchanged data/augmentation/
optimizer/scheduler. Fixed32 seed42 K1/K4/K8 pose/box AP after each epoch; final
full8000 K1/K4/K8 after epoch79 via frozen launch wrapper. No pilot cutoff.

The hypothesis is that consistent proposal/source/target identity increases
useful low-noise pose dependence and improves same-age Gaussian AP or multicall
behavior. Correspondence tests cannot establish that benefit. Compare equal
optimizer steps against corrected uniform-t V42 and repeat state ablations after
a trained checkpoint is available. V40's confidence trajectory failure remains
a separate unresolved mechanism.

Local verification artifacts: `tmp/v42_paired_proposal/`; frozen live namespace
and TensorBoard verification will be appended after launch.

Native FP32 CUDA verification passed: two accumulated batch4/accum2 optimizer
updates, finite loss/gradients through pose encoder/readers/person attention,
19 paired positives in each of4 microbatches (76 checked sources), zero source/
label mismatches, actual observed times13..981 across16 images. Strict model/
AdamW/scheduler restore and exact restored default predictions pass; controlled
next-update maximum absolute weight delta4.703179e-8. Two-image K1/K4/K8 evaluator
smoke completed with finite outputs; its AP is wiring-only and not a trained
performance result. Peak reserved GPU memory2.15GiB. Initial model SHA256 is
`ef71a0f0543d7109117605ee9e162052d81aaff935c3ea685e7e21db27373628`;
43,551,248 parameters,481 surviving native tensors exact.


Live full80 launch verified 2026-11-04T20:40+01:00: root `/storage/outputs/pose-graph-diff/results/poseflow_v42_paired_proposal/20261104_203731`, PID758206,
tmux `poseflow-v42-paired-proposal`. Fresh seed42/EDPose model verified against
the initial hash by the worker;638 frozen source and5 input hashes verified.
At epoch0/step100, loss51.3206, finite gradients through backbone,
image transformer, pose encoder/readers and person attention. Live audit checked
2751 matched positive sources across200 microbatches,
zero source/target identity mismatches and complete valid-target coverage.
Actual corruption/model timesteps span2..999
with553 distinct values across800 images and all10 time bins
represented. This establishes active, correctly wired training; no learned AP
claim is made before epoch-end evaluation.

TensorBoard keeps `0.0.0.0:33332`, all43 previous roots and six canonical
pose/box K1/K4/K8 pins, adding `V42-paired-proposal` as root44. Prior live
V42 controls remain running. Full validation loss and fixed32 AP populate at
each epoch end; final8000 evaluation follows epoch79. No pilot cutoff.
Evidence: `launch_verification.json`, `status.json`, `manifest.json` in the
run root; local helpers under `tmp/v42_paired_proposal/`.


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


[2026-11-04] Paired V42 multicall diagnosis COMPLETE (read-only)
Frozen paired epoch3/6000/fixed32/seed42/resize256427/native pose NMS, all K1/K4/K8 saved metrics exactly reproduced. PoseAP23.442/9.820/9.108, same-age uniform26.484/23.393/23.878 (see exact comparison in report). Same-final-t/reference/logit controls: original Gaussian input at finalK4/K8 gives21.499/22.151. Carried geometry + controlled scores gives20.892/22.086; controlled geometry + carried scores9.659/9.316. Thus generated-state feedback corrupts confidence ranking, accounting for most AP penalty; this is not an isolated training-route cause. Proposal refs reorder strongly; explicit no-GT spatial remapping of state consistently through model AND DDIM does not rescue K4 (7.143 vs9.820); native replay exact. Training uniform0..999 and source-target pairing verified live; neither proves stable calibration on model-generated background/error states. No production/model/job/dashboard changes. Artifacts tmp/v42_paired_proposal/multistep_audit; report docs/reports/poseflow_v42_paired_multistep_audit.md.
