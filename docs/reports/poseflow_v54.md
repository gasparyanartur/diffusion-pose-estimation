# V54: empirical geometric augmentation of fixed prefixes

V54 keeps V53's pretrained native person-to-joint expansion and birth-only
sequential generation. The only modeling change is imperfect teacher-forced
prefix geometry during training. Inference remains deterministic: completed
people stay fixed, and each new person is predicted once before EOS or capacity.
The intended benefit is robustness to previous-person prediction errors;
verification and augmentation statistics do not establish an AP improvement.

## Error representation and provenance

A copied V53 best checkpoint (epoch048) supplies paired newborn predictions on
512 unique augmented TRAIN images. Pairing uses the existing GT-to-selected-slot
assignment; no refined-output rematching is involved. No validation/test residuals
are used. The source checkpoint supplies errors only. V54 model weights are
initialized fresh from the original ED-Pose transfer with seed42.

The frozen bank contains1757 full-person residual vectors. Its32 coordinates are
14 joint displacements expressed relative to each predicted/GT box, box-center
translation divided by GT width/height, and log predicted/GT width/height ratios.
A sampled vector first transforms the box and coherently moves its joints; local
joint errors then express departures from that transformed skeleton. Entire
vectors are resampled to retain correlated errors. Missing source joints supply
zero local residual, and missing target joints follow their perturbed box center.

Raw box-center error norm median/p90/p99 in box units is
0.02552/0.10822/0.55156. Absolute log-size error quantiles are
0.02403/0.13257/0.46058. About54.3% of raw vectors exceed at least one cap,
including joint caps. Consequently the applied augmentation deliberately excludes
part of the raw error tail; it does not reproduce the complete error distribution.

## Training policy and invariants

Each person independently stays clean with probability0.25; otherwise its sampled
residual is scaled uniformly between0.5 and1. Center translation is bounded to
0.15 box units, log-size change to ±log1.25, and each local joint displacement
to0.25 box units. Center movement is additionally limited to25% of the nearest
earlier GT-center distance in the shuffled prefix. Future GT centers never enter
this bound; a later person coincident with an earlier center permits no translation. Supplied
coordinates are clipped to image range with positive extents. These geometric
bounds reduce identity ambiguity but cannot guarantee correct person identities
in every crowd.

A person receives one perturbation per forward, reused in every teacher row.
The coordinate MLP and prefix attention references both see those same coordinates.
GT targets, visibility, proposal identities, used-candidate masks, random target
order and EOS labels remain unchanged. Padding and future-person geometry supply
no information. Only newborns receive geometry losses, at both pose layers.
Validation reconstruction diagnostics use clean prefixes. Free-running detection
always uses the model's fixed previous predictions, without augmentation noise.

There is no Gaussian feature prior, KL penalty, codec, refinement objective,
hidden-state recurrence, rollout redesign or extra sampling step. Existing
pose/box task definitions, TRAIN-only frozen gradient normalization and disjoint
clipping remain. The empirical source bank does not model missing people,
wrong proposal selection or later changes in the V54 error distribution.

## Interfaces, restore and recipe

The unchanged V53 model provides prepare_context, forward_rows, predict and
sample_keypoints. V54 augmentation is in training_v54.py and
prefix_augmentation_v54.py. Bank arrays, joint masks, count, checksum and
TRAIN/source provenance are checkpoint state. Restore needs neither the source
V53 checkpoint nor the residual-bank file nor original initialization assets.
Strict contracts reject changed augmentation policies and corrupted bank state.

The run retains fresh seed42/full80, FP32, batch4/accumulation2, AdamW2e-4 with
backbone1e-4, weight decay1e-4, drops31/61, disjoint total clipping bound0.1,
resize256/max427 and existing augmentation. TRAIN10000/VAL2000 and fixed32 VAL
quick-test are unchanged; TEST remains unopened. Retain best and last, select by
min(pose_AP,bbox_AP), then evaluate selected checkpoint on all2000VAL.

Same measurements use the same TensorBoard tags, including quick-test/pose_AP,
quick-test/bbox_AP and all shared birth losses, gradients and cost metrics.
Only the run label changes. Prefix corruption fraction, clipping fraction and
coordinate MSE describe the added augmentation independently of architecture.

## Verification and outcome

Failing tests preceded the helper and pipeline. Focused checks cover geometric
reconstruction, correlated row reuse, masks, bounded movement/coincident centers,
empty images, deterministic generators, immutable targets and identities, exact
clean-mode V53 loss parity, clean evaluation RNG, frozen-bank gradients and
asset-independent strict restore. Legacy V53/V52/V50 checks are retained.
Native full-capacity FP32/batch4/accum2 checkpoint-gradient and next-update restore
verification must pass before launch. Measurements and launch identity are added
below after completion. Learned quality, robustness and cost are separate results.

Native preflight passed:25 focused/regression checks,50rows/49people, FP32,
batch4/accumulation2,256x416 input. Peak allocation2,201,673,728bytes (~2.05GiB);
0.868s/update (9.22synthetic images/s) alongside V53. Full-capacity inference
3.77images/s in this preflight. All255 gradient tensors pass checkpoint parity
(maxabs1.91e-6). Resumed next-update loss17.294039726257324 is identical;
max state delta9.50e-8. These cost measurements are synthetic/shared-GPU results.
The initial model hash matches fresh V53, so the residual source is not a warm start.

Launch audit caught and corrected a future-target route in the safety bound: it
initially consulted all GT centers. Only the new V54 trainer was stopped at its
first reported10updates; its namespace20261114_112037 is preserved. A regression
now requires changing future GT centers to leave earlier noisy prefixes identical.
The corrected bound consults only earlier known people and remains fixed across
rows. A fresh full verification and namespace supersede the initial attempt.

Corrected authoritative verification is `tmp/v54/verification_causal/verification.json`:
26checks pass; peak allocation unchanged,0.607s/update,13.18synthetic images/s,
4.34full-capacity inference images/s. Checkpoint-gradient maxabs1.79e-6;
next loss17.29407787322998 matches resume exactly (maxstate6.55e-8).
Frozen pose/box gradient scales are2.3932216/1.0361296. Original transfer matches
all retained399frontend/94pose tensors exactly. Initial model SHA256 is
`d03d00d283159d9717e3efc2cb61c01976aa849830ede4055f335d8e5d9dd1aa`,
identical to fresh V53. Fresh run root is
`/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v54/20261114_112502`;
636 executable/native source hashes and912 archived files, bank and copied source
checkpoint are retained. Live training/dashboard verification follows.

Corrected trainer3262573 is active; finite optimizer progress confirmed through
step260. First18logged batches average72.0% perturbed people,30.4% of perturbed
vectors bounded, and normalized-coordinate MSE0.000786. These measurements
confirm augmentation wiring, not detection improvement. V53 trainer2842214 remains
active. TensorBoard retains78previous roots and5stored pins; V54 shares existing
AP charts. Full legacy history/live scalar parity is verified separately below.

Delivery complete: trainer3262573 confirmed through1180updates with finite loss,
while V53 remains active at epoch65. TensorBoard's initial history scan finished:
all14comparison APseries/774previous points are preserved exactly, alongside
78previous roots and5stored pins. New training/corruption scalars refresh through
step1169. V54 is visibly overlaid on the existing bbox/pose AP cards; initial
AP is only the step0baseline. Screenshot/AX/API/history proofs and final docs are
archived under the run. The full80run and learned detection benefit remain pending.
