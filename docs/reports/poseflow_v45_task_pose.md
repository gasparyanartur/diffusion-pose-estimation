# V45-task-pose: jointly learned pose representation and coordinate diffusion

The approved task-defined JEPA model uses the `V45-task-pose` namespace because
another concurrent task owns the distinct progressive-person V45 model. Those
files, jobs and checkpoints are preserved.

## Model contract

The native reduced ED-Pose frontend (ResNet, retained encoder layers 2/4/6,
200 candidates, one box decoder, top50) computes time-independent image context.
Each sampling trajectory caches selected identities, detached proposal boxes,
image features and image-only classifier scores once. The frontend remains
trainable. Native box/proposal/DN objectives are retained. Selected proposal/GT
matching occurs once before coordinate corruption; final/pose-auxiliary losses
reuse that assignment. Early frontend objectives retain native assignments.

Coordinates follow native scale6/cosine1000 corruption and DDIM eta0. One person
query replaces joint expansion; two person-attention layers retain60 deformable
sampling offsets per head/level. The reader never receives image features or
reference boxes. Boxes/scores come from the cached image frontend.

The codec encoder is XY28 -> GELU256 ->32, with non-affine LayerNorm. Its shared
reader is32 -> GELU256 -> GELU256 -> sigmoid28, predicting absolute normalized
image coordinates. The same encoder processes native noisy coordinates converted
back to normalized XY. A learned adapter injects its code into the256D person
trunk with timestep conditioning. One shared residual head at both stages
predicts delta32, with its final linear weights/bias initially zero. Each stage
reads N(E(x_t)+delta). Native auxiliary pose supervision remains. Coordinates,
not latents or hidden features, advance through the sampler.

## Matched objectives

Both arms construct the same teacher, parameters and diagnostics and consume the
same random draws. EMA decay .999 tracks only the clean encoder, updates once
per optimizer step, remains eval/no-grad, and is excluded from AdamW. Both online
and teacher codes are decoded by the same trainable reader:

Lclean = .5 G(R(E(x0)),x0) + .5 G(R(stopgrad(EMA(E)(x0))),x0).

G uses native keypoint L1/OKS weights and normalizes over the selected complete
people. Both arms optimize Lnative + .5 Lclean. The JEPA arm alone adds weight1
mean-person/channel squared distance between final predicted code and detached
teacher code. Reconstruction and JEPA use only fully labeled poses after
augmentation; all valid partially labeled people retain native masked denoising
losses. No missing-joint latent targets are imputed. Empty auxiliary sets return
differentiable zero. No SIGReg, KL, frozen codec or projection head is added.

Nonlinear reconstruction/norm do not guarantee useful Euclidean latent distances.
Diagnostics distinguish online/teacher-code reconstruction, per-joint errors,
pairedOKS/tails, diversity/RMS, target lag and complete-pose coverage. Every five
epochs, fixed monitor examples additionally measure latent perturbations at
0/1/2.5/5% of unit-RMS code scale and correct/zero/swapped coordinate states with
identical image context at t100. These GT-conditioned measurements are separate
from native detector AP and do not prove generation quality.

## Interfaces, initialization and restoration

`prepare_context(images,padding_mask,targets=None)`,
`denoise_coordinates(state,timesteps,context)`,
`sample_coordinates(context,steps=None,generator=None)` and the native
`sample_keypoints` adapter are exposed in model_v45_task_pose.py.

Both arms start fresh from original ED-Pose initialization, seed42; no V42/V44 or
codec checkpoint is reused. Codec, adapters/residual and sampling offsets are
fresh. Initial states/RNG/predictions match exactly across arms. Checkpoints
include coordinate/state/convention/latent-dimension/loss/EMA metadata, teacher
and counter, optimizer/scheduler, and Python/NumPy/Torch/CUDA RNG. Restore is
strict and rejects a different arm. Native evaluator model construction needs
no external initializer/teacher asset after weights are supplied. Prefetched
DataLoader replay is not guaranteed by RNG restoration.

## Training and evaluation

The user explicitly chose historical V42 evaluation: trainval12k training,
overlapping val2k loss as a fit diagnostic, fixed32 TEST development monitoring
every epoch, full8000 TEST K1/K4/K8 after epoch79. This exposes TEST during
development and is not an untouched benchmark. Native pose-NMS and image-ID
paired inference seeds are retained. Compare V45 arms primarily; older V42 has
a different architecture/context computation. Matched calls do not imply equal
compute across model families.

Control then JEPA run sequentially for full80 epochs each, batch4/accumulation2,
FP32, native AdamW2e-4/backbone1e-4, weight decay1e-4, clip.1 and multistep31/61.
There is no two-epoch pilot cutoff. A control failure stops the queue instead of
starting JEPA. Existing runs are preserved. Each arm has a frozen source/config/
input/initialization manifest and persistent tmux-managed progress.

Presets: `benchmark_poseflow_v45_task_pose_control` and
`benchmark_poseflow_v45_task_pose_jepa`. Ordinary single-arm training can use
`python -m scripts.training.train_diffusionregpose --config-name benchmark_poseflow_v45_task_pose_control`.
The archived full-run wrapper, including final full8000 evaluation, is
`python -m scripts.training.train_poseflow_v45_task_pose --root <prepared-arm-root>`;
its root contains manifest.json and seed42/.hydra/config.yaml.

TensorBoard33332 receives V45-task-pose-control and V45-task-pose-JEPA while
retaining every historical root and the original six quick-test pose/box AP pins.
A queued arm has a status notice, not invented AP values. Validation/AP begins
after its first epoch. Launch/verification evidence is appended below.


## Verification and launch (2026-11-07 15:54 CET)

143 focused/historical checks passed, followed by21 final focused/evaluator
checks. Native CUDA used real post-augmentation batch4/accumulation2 and two
optimizer/EMA steps, with complete-pose counts5/9/2/6. Restored predictions are
exact; the matched next optimizer update has identical loss and maximum parameter
difference5.96e-8. Both arms' initial states/RNG match. Native VAL K1/K4/K8
evaluation and complete-pose perturbation/state-use diagnostics are finite.
These are wiring checks, not learned performance.

Pair root: `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v45_task_pose/20261107_154258`. Control is training at epoch0,
optimizer/EMA step600, finite batch loss52.0454;
PID1935244. JEPA is queued and will start fresh after successful control
full80 training and final evaluation. tmux: `poseflow-v45-task-pose`. The persistent
queue fails closed on a control failure; this was exercised during startup.

The first startup hit AF_UNIX path-length limits before any optimizer update.
It was stopped, artifacts/interrupted step0 checkpoint preserved under the pair,
and restarted fresh with a short repository-local TMPDIR. No checkpoint was
resumed and no model/data/optimizer recipe changed. Both arm launch scripts use
the corrected short IPC paths.

All771 frozen source files and five inputs/config hashes per arm verified.
TensorBoard remains0.0.0.0:33332, preserving all58 previous roots/eight existing
pins including the six required quick-test AP cards. Both new roots are visible;
control has real total/clean/raw-JEPA/complete-count loss scalars. Raw JEPA MSE is
a diagnostic in control, whose JEPA coefficient is0. Queued JEPA has a status
notice, no fabricated AP. First-epoch validation/AP remains pending. Host-side
tailnet HTTP verified; phone access not tested.

The AP view includes both task-pose runs alongside the previous comparison;
proof is saved in `tmp/v45_task_pose/dashboard.png`. Evidence and live status:
`tmp/v45_task_pose/launch_verification.json`, `preflight.json`, `final.log`,
`regressions.log`, and the frozen pair manifests/source/config/run logs.

2026-11-07: User changed V45-task-pose scheduling to parallel. Control continues its existing run; JEPA starts fresh seed42 for80 epochs in tmux `poseflow-v45-task-pose-jepa`. Sequential supervisor disabled to prevent a duplicate launch. Model, data, optimizer and evaluation protocol unchanged.
