# V49-balanced: robust, calibrated token-age objectives

This is a separate V49 run with the same model, fresh original ED-Pose frontend
initialization, coordinate codec, 50 independent training rows, random source
permutation, per-token sigma schedule, 49-person capacity and sequential DDIM
sampler. It implements the normalization changes approved on 2026-11-11.
The original V49 run and artifacts are preserved. It is a combined intervention;
an improvement cannot be attributed to one loss change without later controls.

## Geometry and coordinate objectives

Joint-coordinate MSE averages only labeled XY channels. Box-coordinate MSE
averages its four CXCYWH channels independently, so missing joints no longer
change the implicit box/pose ratio. The normalized 2D targets and original masks
are unchanged. Both decoder layers are supervised through the trainable codec.

For each joint, let `e` be the existing OKS exponent, squared pixel displacement
divided by `8*sigma_joint^2*.53*GT_box_area`. Replace `e` with `rho(sqrt(e))`,
where `rho(r)=r^2` for `|r|<=1` and `2|r|-1` otherwise. Apply the same transition
to the box-relative center residual and log width/height ratio. GIoU is retained.
The robust residual is quadratic near the target and linear for large errors;
person-size dependence remains. This is not the bounded `1-OKS` objective.

Unnormalized per-person tasks are:

- `pose = 10*joint_coordinate_MSE + .5*robust_pose_geometry`.
- `box = 10*box_coordinate_MSE + .5*(robust_center + robust_log_size + 1-GIoU)`.

Average both layers, then rows within each person's refinement-age bucket,
then that person's present buckets, then people. The three refinement buckets
are age 1, age 2 and age >=3. Age is row index minus sequence rank; the terminal
zero-sigma override must not change it. Birth is averaged separately per person.
Each person therefore keeps equal total birth/refinement supervision despite
different introduction rows or available tail lengths.

## Fixed TRAIN gradient calibration and priorities

At fresh initialization, use 32 seeded augmented TRAIN batches of four images.
There are no optimizer updates. For each unprioritized birth-pose, birth-box,
age-balanced refinement-pose and refinement-box objective, measure the L2 norm
of its gradients over the shared two-layer decoder. Its fixed scale is the
midpoint median across batches, with a positive `1e-8` floor; an entirely
inactive or nonfinite task rejects calibration. Divide each task by its scale.
Do not substitute the earlier epoch-5 audit constants or use VAL for calibration.

Calibration seed is 42049. Model parameters/buffers and global RNG are preserved.
Norm samples, image IDs, configuration, annotation/model hashes and resulting
scales are archived. Four scales/readiness/sample count are frozen checkpoint
buffers; full provenance is embedded in each checkpoint. Restores need neither
the initialization asset nor the external calibration report.

After calibration, box/pose loss weights are `.75/.25` at birth, `.5/.5` at age 1
and `.25/.75` at age >=2. Apply priorities inside age buckets, before averaging.
These are loss weights, not a guarantee of those exact gradient ratios after
learning. The overall objective is `.5*(birth + refinement) + EOS`, with unit
overall task scale. Preserve the original AdamW recipe and clipping.

EOS BCE is averaged across both layers. Within an image, person-target rows,
the first EOS row and later EOS rows receive weights `.5/.25/.25`, renormalized
over present groups; then average images. This prevents long easy EOS tails
from diluting the first stop decision. Empty images remain EOS-only. Teacher
first-EOS probability/accuracy metrics use the second layer, as does sampling.

## Verification, usage and interpretation

Failing-first unit checks cover robust derivatives/local arithmetic, missing
joints, empty/mixed images, age normalization, EOS tail-length invariance,
both-layer gradients, frozen calibration/model/RNG, strict restore and corrupted
provenance, checkpointed gradient parity and identical original model/RNG
initialization. Historical V49 rows/model/evaluator tests remain unchanged.

```bash
venv/bin/python -m scripts.training.verify_poseflow_v49_balanced --output tmp/v49_balanced/verification
venv/bin/python -m scripts.training.train_poseflow_v49_balanced \
  verification_file=tmp/v49_balanced/verification/verification.json
```

Full training is guarded by matching source/config/input/initialization hashes,
calibration provenance, full-capacity native FP32 preflight, resumed next-update
agreement and a two-image native evaluator smoke. Native verification uses all
50 rows, 49 people plus EOS, non-square 256x416 images, batch 4 and accumulation 2.
Resource failure must be reported before changing that recipe.

Fresh seed42/full80 uses TRAIN10k/VAL2k, original augmentation/aspect resize,
AdamW2e-4/backbone1e-4, weight decay1e-4, clipping.1 and LR drops31/61. TEST stays
unopened; the unused eager test slot points to VAL. Evaluate the same fixed32
VAL images before training and each epoch, select best by `min(pose_AP,bbox_AP)`,
keep best/last and evaluate selected weights on all 2,000 VAL after80. AP tags remain
`quick-test/token_age_diffusion/rows_50/{pose_AP,bbox_AP}` under a distinct
`V49-balanced-token-age` TensorBoard run. Raw tasks, actual weighted contributions,
age buckets, calibration scales and first-EOS metrics are logged separately.

Report free-running detection AP, source-paired teacher reconstruction,
generated refinement with fixed final assignment, count/duplicates/identity
switches and computational cost separately. Calibration or finite preflight is
not learned performance. Randomized pure-noise birth ambiguity and the decoder
reconstruction floor remain unresolved architectural risks.




## Verified launch — 2026-11-11

All 70 focused/historical tests pass. Native FP32 / 50 rows / 49 people / batch 4 / accumulation 2
verification passed at 256x416: peak allocated 1,934,926,336bytes (1.80 GiB), warmed
verification update 7.74 images/s and capacity inference 4.40 images/s on the shared
GPU. These measurements include concurrent jobs; inference had two resident
verification models and is not a controlled speed comparison with the original.
Next-update loss was exactly 8.616812229156494 before/after restore; maximum
state delta 1.19e-7; optimizer/scheduler/RNG restore passed. All codec parameter
gradients were positive. Measured retained frontend coverage is 100% across 330
tensors. Common initial model SHA is `a6b2745f0742f049d811837e61359da1dd266a5e3dfca3f6bb414589c868be37`.

32 augmented TRAIN batches produced frozen scales in task order
(pose_birth,box_birth,pose_refine,box_refine):
`[2.411630153656006, 2.832282066345215, 2.318622589111328, 2.7152771949768066]`. Model hash before/after calibration is identical.
These are fresh-initialization scales, not the earlier trained-checkpoint audit
ratios. All task priorities are applied after calibration.

Fresh full80 run: `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v49_balanced/20261111_012830`, tmux `poseflow-v49-balanced`, PID 445691.
Live epoch index 0 / optimizer step 120, finite loss. Same fixed32 VAL
IDs match original V49 exactly; initial native pose/bbox AP are 0/0 (wiring only).
586 source hashes / 861 archived source-test-document files and initialization,
calibration/input/config hashes are preserved with verification and launch proof.
Original V49 PID 377259 and V42 timestep PID 348105 remain active; other artifacts
were preserved. Existing TensorBoard 33332 adds `V49-balanced-token-age/.` with
live component/scales/AP data, preserving 71 prior roots and 20 pins. Full 80 epochs and
selected all 2,000 VAL outcomes remain asynchronous; no learned improvement claimed.


## User cancellation — 2026-11-11

The user cancelled this experiment after reviewing its poor detection/refinement
results. The trainer exited gracefully after SIGINT. Best and last checkpoints
remain intact and were loaded and SHA-verified with optimizer/scheduler state.
The last completed checkpoint is at global step 55000; subsequent partial-epoch
updates are not claimed saved. Per-run `cancelled_by_user.json` records the exact
stop and preserved assets. No final 80-epoch/all-2000-VAL completion is claimed.
A proposal-attention identity and time-gated residual successor is being planned;
no successor code or training run has been started.
