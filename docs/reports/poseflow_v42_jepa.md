# V42 paired-image JEPA denoising

Approved 2026-11-05. Parent: `benchmark_poseflow_v42_image_confidence`.
New preset: `benchmark_poseflow_v42_jepa`. This is coordinate diffusion with
JEPA supervision inside each denoising call. No latent state is sampled.

The existing noisy coordinate MLP, image-conditioned person decoder, final
normalized person feature and pose reader remain the inference path. Uniform
per-image timesteps 0..999, paired proposal labels, native coordinate DDIM,
60-point box-referenced attention and image-only confidence are retained.

For each positive selected slot, the target is the EMA of the existing pose
encoder and decoder normalization applied to that slot's clean GT coordinates.
The encoder sees normalized XY only. No independent clean image pass or second
matching operation is performed. EMA decay is .999, initialized from the online
modules and updated once after each optimizer step, never at microbatches or
validation. Teacher parameters are frozen and excluded from the optimizer.

The online clean encoding also feeds the existing final pose reader using the
same detached box reference as the noisy prediction. This has an explicit
box-center geometric input but no image-feature bypass. Native visible-joint L1
and OKS losses preserve pose fidelity. The final noisy person features directly
predict the clean target: there is no projection head that can hide alignment
from the reader.

Objective: unchanged baseline + 1.0 mean-positive/channel latent MSE + .5 clean
pose geometry loss (native keypoint/OKS weights). Background has no JEPA or clean
reconstruction target. Empty positive batches return differentiable zero losses.
Intermediate baseline pose losses and all proposal/box/DN losses remain.

Diagnostics report clean/denoised coordinate errors by timestep bin, latent loss,
clean-code diversity/RMS, teacher lag/count and fixed pairing coverage. Low
latent loss is not evidence of better native pose AP. An initialization-only
correct/zero/swapped state diagnostic is a wiring check, not a learned result.

Full training starts fresh from the same ED-Pose/seed42 initialization as the
parent, not from trained V42 weights. Inference parameters and initial tensors
are identical to the parent; only pipeline training state gains the EMA encoder
and count. Checkpoints additionally preserve Python/NumPy/Torch/CUDA RNG; strict
restoration does not guarantee replay of prefetched data-loader batches.

Evaluation retains the historical parent protocol for comparison: validation
losses on CrowdPose val overlap the trainval training set and are fitting
measurements; fixed32 held-out test K1/K4/K8 monitors are development diagnostics,
not final generalization evidence. Final full8000 K1/K4/K8 follows epoch79.
No extra TEST subset is used in preflight. TensorBoard uses the existing
0.0.0.0:33332 server, a separate V42-paired-image-jepa stream and existing six AP
pins. Full80 training has no two-epoch pilot cutoff. Existing runs are preserved.

Hypothesis: a pose-readable clean target makes removing corruption easier in the
whole-person representation. Drift/collapse, image shortcuts, competing task
updates and generated-feedback mismatch remain possible. EMA targets and
reconstruction are safeguards to test, not guarantees of multi-step gains.

Verification and launch evidence will be appended after completion.

## Verification

138 V42/native training regression checks passed, followed by all six final
JEPA checks including RNG checkpoint hooks. Native CUDA batch4/accum2 completed
two finite updates; teacher gradients absent, same-person source labels verified.
Initial online model SHA exactly matches the original paired-image-confidence
build. Strict model/optimizer/scheduler/teacher restoration gives identical
predictions; the next full update differs by at most 2.80e-8. Model-only native
K1/K4/K8 sampling is finite without teacher or GT. Preflight peak reserved GPU
memory was 2.15 GiB. These are execution checks, not trained performance.

Reproduce configuration using:
`python -m scripts.training.train_diffusionregpose --config-name benchmark_poseflow_v42_jepa`.
The archived full-run wrapper also records source/input hashes, pairing/time/EMA
checks and final full8000 evaluation, matching the parent run protocol.


## Verified full80 launch (2026-11-05 22:25 CET)

Run: `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v42_jepa/20261105_221931`. Trainer PID 990002, tmux `poseflow-v42-jepa`.
At verification: epoch 0, 150 optimizer/EMA updates,
finite batch loss 48.1709, zero source-label mismatches and all ten
timestep bins observed. All 734 archived source files
and 5 input/config hashes verified; fresh initial
online model hash equals the parent. Existing three trainers remain alive.

TensorBoard `V42-paired-image-jepa/.` exposes live total, JEPA and clean-geometry
loss scalars. All 52 previous roots and six AP pins preserved;
server remains bound to 0.0.0.0:33332. Direct host-side tailnet HTTP verified;
phone access was not tested. New-run validation/AP begins after its first epoch.
No AP measurements were fabricated to populate empty charts.

Evidence: `tmp/v42_jepa/launch_verification.json`, `preflight.json`,
`regressions.log`, `tests_final.log`; copies and frozen manifest in the run root.
This verifies ongoing execution and logging, not JEPA's learned benefit.
