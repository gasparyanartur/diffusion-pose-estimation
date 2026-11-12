# V44 x0 detached-rollout comparison

The approved question is whether training on generated intermediate states
reduces the measured gap between forward-corrupted targets and the model's own
K4 trajectory. This is a bounded end-to-end training comparison, with a frozen
High-KL codec and unchanged x0 model architecture. It is not a new VAE or a
codec-decoder fine-tune.

## Matched construction

Both arms load the same original x0 epoch047 checkpoint, SHA256
`dbea133795c5d85f282a613f010f1ff16ec3ac4e5f009467379ddb78690f8cfa`.
The saved model and private diffusion RNG are copied strictly; fresh optimizer
and global/data seed42 follow construction. Initial model hashes are identical.
The source's codec/ED-Pose pretraining exposure remains unchanged and disclosed
in the parent V44 report. TRAIN/VAL are disjoint10,000/2,000; TEST is unused.

On half of batches, both arms use the existing independent uniform timesteps.
On the other half, both choose one shared intermediate K4 timestep uniformly
from749/499/249. The `forward` control receives a forward-corrupted clean target.
The `rollout` candidate starts from the same Gaussian draw and advances from999
to that chosen timestep through its current denoiser and deterministic DDIM.
Only the latent state is carried. Warm-ups use evaluation mode and no gradients;
all original module flags and global RNG state are restored afterward. The final
supervised denoising call runs in the normal training mode.

Image context and selected-slot matching happen once before state construction.
The same features, boxes, confidence and assignments serve all calls. Warm-ups
have no clean-code injection, rematching, decoder-to-encoder loop or gradient
through previous predictions. Final supervision retains all original frontend,
slot confidence, balanced foreground/background clean-code and decoded geometry
terms, including decoder auxiliaries. Image frontend and denoiser train normally;
codec weights, statistics and evaluation mode stay frozen.

Both arms use the same branch/time/noise draws and preserve the normal global
RNG stream through extra warm-ups. First eight augmented image/geometry batches
are hashed and compared after both runs. Frontends diverge with optimization,
so their selected slots and assignments need not remain equal across arms.

## Recipe and evidence

Each arm is capped at2epochs/2,500 optimizer updates: batch4, accumulation2,
FP32, AdamW2e-5/backbone1e-5, weight decay1e-4, clip0.1, no LR drop inside the
window. Rollout runs first and forward control follows sequentially. This
matches optimizer updates and data exposure, not compute cost. Existing runs
are preserved. Best/last checkpoints only, in a fresh source/config snapshot.

Baseline and each epoch log fixed32VAL K1/K4/K8 pose/box AP and the existing
fixed1/2.5/5/10% latent-noise paired reconstruction curves under `quick-test/`.
Each independently selected best checkpoint receives full2,000VAL K1/K4 AP.
Their selected ages can differ. Both final last checkpoints receive the same
fixed32VAL t249 forward/generated diagnostic so its comparison is matched-age.
Those paired OKS/IoU/latent errors are not detector AP.

The summary records a passing candidate only when all gates pass:

- K4 pose AP improves on fullVAL and matched-age fixed32VAL; box AP is retained.
- K1 pose AP is within1 absolute AP point of control; fullVAL K1 box AP is retained.
- Selected-checkpoint1% and5% output paired OKS are no worse than control.
- The last-checkpoint forward/generated OKS gap shrinks and generated-state OKS improves.

A smaller gap caused only by degrading forward-corrupted performance is
insufficient. No automatic promotion, longer run or additional experiment
follows. A failed bounded continuation would not prove that every rollout
training recipe fails.

## Verification

Failing-first tests cover default input/RNG parity, matched branch/time/noise,
no target injection, exact detached warm-up arithmetic, mode/RNG restoration,
single context/matcher reuse, empty images, auxiliary/final gradients,
frozen codec, strict policy metadata and acceptance gates. All36 focused and
related regression checks pass. Existing non-square/missing-joint/capacity
checks remain in the regression set.

Native CUDA completed two accumulated updates with forced trajectory exposure.
A strict optimizer/scheduler/private/global RNG restoration reproduced next loss
18.634130477905273 exactly and next state within2.71e-8. Codec tensors/statistics
were unchanged. Peak allocated CUDA memory was1,882,209,792 bytes. The final
trajectory evaluator passed a2VAL-image/13-person CUDA wiring check. These are
finite/restoration checks, not learned-performance evidence.

Training configuration and state policy live in a separate subclass/entrypoint.
Default V44 arithmetic and model state keys remain unchanged. Restoration
requires matching state-policy metadata; intentional original-x0 warmstart is
separate from restoration.

## Usage

Use `scripts/training/train_poseflow_v44_rollout.py` with `arm=rollout` or
`arm=forward`, an explicit source checkpoint and its SHA256. The queue script
runs both verified arms from a snapshotted run namespace and writes
`comparison.json`. Source/config/annotation/initial-weight hashes, optimizer
recipe, preflight results, first-batch fingerprints and measured baseline are
saved with the run. See the run namespace recorded in `.plan` for live status.
