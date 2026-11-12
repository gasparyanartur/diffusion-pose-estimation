# V44 x0-versus-velocity comparison

The approved question is whether an identity path in latent denoising prevents
V44 from damaging accurate poses and improves pose AP across reverse steps.
This is an end-to-end LDM parameterization comparison. The frozen VAE,
architecture, native proposal path, assignment policy, augmentation and component
weights remain unchanged. There is no decoder-only robustness stage.

## Definition and initialization

The default `prediction_type=x0` remains the historical clean-latent head.
Opt-in `v` heads predict `v = alpha*epsilon - sigma*h0` for the same VP schedule,
where `alpha^2 + sigma^2 = 1`. Every latent decoder output reconstructs
`h0_hat = alpha*h_t - sigma*v_hat` before frozen-codec geometry losses and DDIM.
The identity limit applies as sigma tends to zero; discrete timestep0 still has
nonzero noise. We do not restore the old timestep-zero training bug.

Both arms explicitly warmstart from the same immutable epoch047 source (60,000
updates). The x0 arm copies all state exactly. The v arm negates the weights and
biases of each latent head, retaining all other tensors, codec statistics and
private diffusion RNG exactly. This approximates the learned x0 prediction only
at the high-noise endpoint; it does not preserve the intermediate-time function.
New checkpoint metadata records the mode. Legacy mode-less checkpoints mean x0;
cross-mode restoration is rejected. Warmstart conversion is a separate operation.

Both arms use fresh AdamW and global/data seed42 after construction. Their source
private diffusion RNG is copied. The initial augmented image IDs/geometry are
hashed for eight batches and compared after both arms complete. Loss coefficient10
now applies to balanced foreground/background velocity MSE in v mode; clean-latent
MSE remains logged for comparison. Geometry coefficient0.5 and all native frontend
and slot confidence terms are unchanged, including auxiliary decoder supervision.
Raw x0 and velocity training losses have different units and noise-level weighting.

## Bounded recipe and evaluation

Each arm receives10 epochs /12,500 updates, batch4/accumulation2, FP32, clipping0.1,
AdamW2e-5/backbone1e-5, weight decay1e-4. The existing31/61 milestone schedule has
no drop within this window. Velocity runs first and the matched x0 control follows
sequentially, preserving the existing experiments within GPU memory. The original
V44 full80 run continues independently.

TRAIN/VAL remain disjoint10,000/2,000; TEST stays unopened. Pretraining exposure of
the fixed codec/ED-Pose remains disclosed. Both arms have initial and per-epoch
fixed32VAL K1/K4/K8 AP and fixed paired small-noise reconstruction curves on the
same manifest. Only best/last are kept per arm. Each selected checkpoint receives
full2,000VAL K1/K4 evaluation. No additional training or automatic model promotion
follows the comparison. Per-arm evaluation artifacts use the checkpoint run root,
preventing collisions in the shared source snapshot.

Review positive-noise recovery against directly decoded noisy codes, matched-age
pose AP and multi-step behavior with bbox quality retained. Identity preservation
alone is insufficient. The earlier V43 velocity controls did not establish learned
foreground recovery, so no success is assumed for V44.

## Verification and initial-function differences

Native CUDA preflight completed two accumulated velocity optimizer updates with
the codec/statistics unchanged. Strict optimizer/scheduler/private/global RNG
restoration gave identical next-update loss45.22147750854492 and maximum state
difference1.49e-8. Peak allocated CUDA memory in this check was1,890,453,504 bytes.
A temporary preflight harness initially reused Lightning's optimizer after teardown
had moved its state to CPU; the check was corrected to restore into a fresh GPU
optimizer. The production optimizer implementation did not change.

The new default-x0 model exactly reproduces the original frozen-source outputs
on two images at three timesteps (all decoded geometry, logits and clean latents).
The opt-in mode passes VP inverse/target arithmetic, auxiliary gradient, frozen
codec, explicit head-negation, semantic restore and artifact-routing checks.

Initial fixed32VAL AP, before any comparison updates:

| Mode | K1 pose / box | K4 pose / box | K8 pose / box |
|---|---:|---:|---:|
| x0 |33.55 /46.84 |31.51 /47.49 |27.98 /45.75 |
| v, negated-head initialization |33.63 /46.84 |0.00 /5.06 |0.00 /3.12 |

At5% noise, direct input paired OKS is.92569, x0 output.78911 and initial v output
.85934. Neither mode passes positive-noise recovery at initialization. The severe
initial multi-step v deficit is explicitly retained in the baseline; training must
learn the velocity function. These are initialization diagnostics, not evidence
that velocity training improves or fails pose estimation.

Run/source/config/checkpoint hashes, initial hashes and launch health are recorded
in a fresh comparison namespace after launch. Historical/default V44 behavior and
checkpoints are preserved.

## Launch

Root: `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v44_parameterization/20261106_164359`.
Velocity PID1280237 verified at225 finite updates, epoch0/10; x0 control queued.
All48 focused/regression tests pass.635 snapshot sources, shared checkpoint SHA,
exact initial velocity model identity and TRAIN/VAL hashes are verified. Eight
initial candidate image/geometry fingerprints are saved; control pairing cannot
be claimed until it runs. Original V44 remains active.

TensorBoard33332 exposes `V44-velocity-comparison/.` and `V44-x0-control/.` on
all six existing quick-test AP cards. The queued control shows its measured
step-zero baseline only. Candidate live training scalars are verified. All53
previous roots/pins remain. A duplicate startup event copy was moved outside
the live log directory so TensorBoard follows the trainer file; no trainer log
was modified.

The queue runs velocity then x0, evaluates selected checkpoints on fullVAL and
writes comparison.json. Matched-age differences use the paired per-epoch fixed32
curves. FullVAL compares independently selected best checkpoints after equal
budgets; selected ages may differ and are recorded. No automatic promotion or
further training follows. Learned improvement remains unmeasured at launch.

## Completed comparison (2026-11-07)

Both arms completed10epochs/12,500 updates; queue exit0. Neither promotion nor
additional training was triggered. Full2,000VAL metrics below use independently
selected checkpoints: x0 epoch002/3,750 continuation updates and v
epoch001/2,500 updates. Equal training budgets do not make those selected ages
equal. Matched-age fixed32VAL curves also show the velocity multi-step collapse.

| Mode | K1 pose AP | K1 box AP | K4 pose AP | K4 box AP |
|---|---:|---:|---:|---:|
| x0 |28.14 |44.01 |24.57 |43.11 |
| v |24.71 |40.19 |0.00049 |7.22 |

At5% latent noise, fullVAL paired OKS changes from.91409 to.91316 for velocity,
versus.91374 to.74471 for x0. Velocity preserves near-clean states much better
but fails the5% learned-recovery gate and detector AP gates. Retain x0 as the
reference. This bounded conversion/continuation does not establish that velocity
trained from scratch would fail.

Frozen trajectory diagnosis was approved next: compare forward-corrupted target
latents with self-generated states at identical K4 timesteps, independently
verify DDIM/oracle arithmetic, and keep all image context/assignments fixed
within each checkpoint's branches. No optimizer steps or TEST access.

## Frozen trajectory diagnosis: measured answer

The approved diagnostic completed on the same fixed32VAL images/103 people for
the source epoch047 and both last checkpoints (epoch009/12,500 continuation
updates). This is a paired-person reconstruction diagnostic, not detector AP.
Assignments/context are computed once per image per checkpoint and reused in
every branch. Different trained frontends can select different slots, so slots
and assignments are recorded separately across checkpoints. The same seed1042
plus image ID supplies paired Gaussian draws within each checkpoint. No TEST,
optimizer update or model/statistics mutation occurred.

At each actual K4 timestep, `forward` supplies the correctly corrupted assigned
GT code, while `generated` supplies the state reached by the model's own previous
DDIM steps. The table gives output paired OKS on a0–100 scale:

| Checkpoint / branch | t999 | t749 | t499 | t249 |
|---|---:|---:|---:|---:|
| Source x0 / forward |61.78 |64.06 |67.19 |70.08 |
| Source x0 / generated |61.78 |59.56 |58.80 |57.32 |
| Final x0 / forward |60.56 |63.99 |66.16 |69.49 |
| Final x0 / generated |60.56 |59.18 |58.99 |57.69 |
| Final v / forward |57.80 |30.94 |24.94 |34.17 |
| Final v / generated |57.80 |29.07 |18.11 |13.41 |

There are two measured failures. The x0 model reconstructs progressively better
when given forward-corrupted targets, but progressively worse along its own
trajectory. At t249 the final x0 arm has a11.80 OKS-point gap and foreground
latent MSE rises from.04380 to.21940 (5.01x). Source x0 shows the same pattern
(12.77 points,5.79x MSE), so this is not introduced only by the continuation.
Generated-state foreground MSE exceeds forward-state MSE on all32 images at
each intermediate timestep for both x0 checkpoints. This establishes a
trajectory-distribution sensitivity; it does not prove that rollout training
will fix it or exclude ambiguity/averaging in the first prediction.

Velocity also fails on correctly forward-corrupted inputs: t499 OKS24.94 versus
x0's66.16. Its t249 foreground MSE rises further from.09828 on forward states to
.44089 on generated states, with OKS34.17 versus13.41. Therefore exposure to
generated states alone cannot explain the velocity result. This converted,
short-continuation velocity model did not learn a useful intermediate-time
denoising function. Its near-zero-noise preservation cannot establish such a
function. The diagnostic does not identify which optimization, initialization
or architectural choice caused that weakness.

Direct decoding provides a useful baseline: at t249, the final x0 forward input
has OKS22.03 and its output69.49; the generated input16.55 becomes57.69. Velocity
forward input22.23 becomes34.17; generated input9.44 becomes13.41. Both perform
some denoising here, but velocity is much weaker. These K4 timesteps are not
the1–5% small-noise regime; the separately measured small-noise curves still
apply. Frozen posterior-mean reconstruction is99.05 OKS for these people.

Independent FP64-formula DDIM replay agrees with native FP32 steps to at most
4.77e-7 absolute latent error. Oracle-clean predictions track the expected
forward path within2.39e-7 and return the exact clean code at termination.
Oracle velocity inversion error is at most2.76e-7. Manual learned trajectories
exactly reproduce `sample(steps=4)`. Cached indices/boxes/features/confidence,
all model tensors and codec statistics remain unchanged. These checks rule out
an arithmetic discrepancy in the tested latent update and parameterization;
they are not a proof about every possible sampler or model bug.

Artifacts, scripts, tests, checkpoint hashes, image IDs, assignments, per-channel
state/output statistics, foreground/background losses and paired geometry error
tails are archived under:
`/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v44_parameterization/20261106_164359/diagnostics/trajectory_20261107`.
The two focused helper tests followed an initial failing import, then passed;
CUDA trajectory/oracle/invariance checks passed for all three checkpoints.

The next proposed controlled change is x0 training with exposure to detached
generated intermediate states, retaining fixed assignments and clean targets.
Its purpose would be to reduce the measured generated-versus-forward gap;
architecture and codec changes would confound that test. A bounded matched
control and explicit AP/recovery gates need review before implementation or
training. No follow-up training has been launched.
