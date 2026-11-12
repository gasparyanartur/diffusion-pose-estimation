# V42 paired training with image-only confidence

Concrete plan approved2026-11-05 after the frozen existing-head diagnostic.
Config `benchmark_poseflow_v42_image_confidence`, run name
`v42-paired-image-confidence`; separate from failed noisy-state matching.

## Model and supervision

An opt-in `confidence_mode=image_only` adds a separate514-parameter linear head,
cloned from the final box-stage classifier after native EDPose transfer. Gather
already decoder-normalized IMAGE features using the actual raw-logit native top50
indices. Compute the new head once before the pose-input callback, corruption
and noisy-pose feature injection. Final and post-selection pose-auxiliary logits
use that same tensor. Image/time features are trainable; pose coordinates and
pose-stage features never enter this classifier. Confidence can still change
with timestep and current image-proposal selection.

Working paired-proposal supervision remains: Hungarian image references->clean
GT, construct sources, uniform independent per-image t0..999 corruption, fixed
source/GT labels for final and pose auxiliaries. Background slots stay negative.
A separate classifier prevents the native200-candidate box labels and the fixed
selected50 labels from sharing one scoring head. Classification coefficients,
auxiliary multiplicities, geometry/DN/encoder/proposal losses remain unchanged.
Both final and pose-aux classification terms now train the SAME image-only head.
Classification gradients reach the image frontend; no main-classification route
passes through the pose encoder/readers/person attention. Those modules retain
geometry gradients; native DN paths are preserved.

No noisy-state proposal rematching, proposal caching, VAE or JEPA. Whole-pose
representation, reader, attention reads, box references, score-top50 selection,
coordinate diffusion and DDIM are unchanged. Default `pose_conditioned` mode
has historical outputs/state keys/RNG exactly. New saved mode/config adds only
`image_confidence_head.weight/bias`; strict restoration requires that mode.
Head copying consumes no RNG and all old initial tensors match the paired control.

## Verification

Seven new checks pass after intended missing-module red;153 focused V42/native
regressions pass. Checks cover actual normalized feature/index alignment, B>1,
same-image/time score invariance to changed noisy poses, initial native K1/K4/K8
raw geometry parity, fixed sources/labels/times, empty people, DN/aux behavior,
classification-versus-geometry gradient routes, config dispatch and strict restore.
Local seven-case legacy-source comparison proves exact train/eval outputs,
native sampling and RNG/state keys.

Real FP32 CUDA preflight:43551762 parameters,514 new;481 surviving native tensors
and all common initial model tensors exact. Two accumulated optimizer updates,
finite geometry/classifier gradients, exact prediction restore, strict AdamW/
scheduler restore and controlled next-update difference1.19e-7 pass. Changed noisy
input leaves actual CUDA confidence exact at fixed images/times. Finite validation
loss and two-image K1/K4/K8 image-only-score sampling pass. These are wiring checks,
not learned AP. Original common initial SHA is
`ef71a0f0543d7109117605ee9e162052d81aaff935c3ea685e7e21db27373628`;
new model SHA is
`09c88bd96755dbb1c25f7fbf8a520496bd1294d7718402be813f797f8d1541ac`.

## Run and interpretation

Fresh EDPose seed42, FULL80 epochs, batch4/accum2/clip.1, same data/resize256427,
optimizer, scheduler and native sampling as paired control. No pilot cutoff or
classifier-only warmup. Bounded native checkpoint retention fits checked storage;
no historical checkpoints are deleted. Every epoch: full validation loss and
fixed32 K1/K4/K8 pose/box AP; full8000 evaluation after79. All native inference
paths automatically return the new model's image-only logits. Evaluation reports
persist confidence_mode and source features. TensorBoard gets a distinct stream
on existing0.0.0.0:33332 with every current root and six pins preserved.

Frozen rescoring improved paired epoch31 AP40.073/13.038/10.207 ->
42.893/35.525/35.274, with unchanged raw geometry. Same-age uniform-t control
43.163/42.422/41.622 remains stronger. Training this head changes classification
gradient routing and can therefore change learned geometry; better state use or
trained AP is unproven. Compare same-age checkpoints and retain the uniform-t
control; do not interpret finite launch or frozen score gains as trained results.

Artifacts: `tmp/v42_image_confidence/` tests/parity/preflight, frozen launch
helpers/run pointer and dashboard checks. Run manifest/status and launch
verification will record actual snapshot/job progress. No control or V43
production source is modified by this iteration.

## Live launch

Fresh full80 run started2026-11-05T14:27+01:00, PID912787, tmux
`poseflow-v42-image-confidence`; root `/storage/outputs/pose-graph-diff/results/poseflow_v42_image_confidence/20261105_142646`.
Worker verifies701 frozen sources/5inputs and exact initialization. Initial live
epoch0/step100 has finite loss and correct
paired source/time/pre-pose confidence audits. All existing TensorBoard roots/pins
are preserved; measured new scalar indexing verification is pending. No trained
AP yet; validation loss/AP follow the first complete epoch.

Final launch verification (2026-11-05T14:33+01:00): full80 PID912787 active at
epoch0/step200, finite loss28.8679.
All701 frozen source hashes/5 input hashes match. Actual t0..999 (808 unique over
1600 images) and all10bins verified;5403positive sources/zero target mismatches,
confidence computed once before pose/source injection and new head gradients
nonzero/finite. Dashboard52scalar streams/48roots fully indexed, measured
`V42-paired-image-confidence/.` training loss through step149;
all47previous roots/sixpins preserved. Host-side direct Tailscale-address HTTP
succeeds. Learned val/AP pending first epoch; no further pilot gate. Evidence:
run manifest/status/launch_verification/dashboard_verification and local
`tmp/v42_image_confidence/`.
