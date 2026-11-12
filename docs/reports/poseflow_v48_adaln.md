# V48-AdaLN: generation-step-conditioned image attention

This is a second V48 run, as requested, not a new version family. The baseline
V48 trainer/checkpoints/source snapshot remain intact. User approved a parallel
run on2026-11-10.

## Controlled architectural change

V48's clean fixed prefixes,512-column ordering, learned shared coordinate codec,
P+1 parallel training rows, birth-only10*coordinate+.5*geometry+balancedEOS,
both-layer supervision and immediate EOS/capacity50 stopping are reused exactly.
There is no noise, refinement, diffusion sampler or teacher-policy change.

The only new path is generation-step modulation in each layer's image
cross-attention. Step is the number of valid prefix people,0..50, derived from
the valid person/PREDICT mask, independently for each row. It is not the padded
row index or GT total person count. One learned256D step embedding is shared
across layers and across the tokens in a row; each image-attention layer has its
own SiLU+Linear256->768 producing scale,shift and residual-gate deltas:

q=(1+gamma(step))*existing_LN(h)+beta(step)

h'=h+(1+delta_gate(step))*CrossAttention(q,imageK,imageV)

Only the image-query normalization and cross-attention residual are modulated.
The encoder,image K/V,self-attention and FFN retain their V48 behavior. Image
memory and projected K/V are cached once per image/layer, shared by independent
training rows and sequential generation steps. The model learns the gate pattern;
stronger early image reliance is a hypothesis, not a forced monotonic schedule.

Modulation output matrices/biases start at zero: scale/gate1 and shift0. New-module
construction preserves the base initializer's RNG state. All common V48 weights,
initial outputs and post-initialization RNG therefore match. Modulation receives
gradients on the first update; the step embedding's initial gradient is zero by
construction and becomes active after modulation changes. No trained baseline
checkpoint supplies weights; initialization is original ED-Pose plus fresh
V48 codec/decoder, with identity modulation added.

## Implementation and contracts

`model_v48_adaln.py` subclasses V48 and reuses cached attention. The isolated
`training_v48_adaln.py` config/pipeline retains the V48 public interfaces, losses,
image-only sampler and evaluator. Configuration is
`benchmark_poseflow_v48_adaln`; entrypoints are
`scripts.training.verify_poseflow_v48_adaln` and
`scripts.training.train_poseflow_v48_adaln`.

Checkpoints retain architectureV48 with strict
`variant=image_cross_attention_generation_step_adaln` and detailed conditioning/
initialization metadata. A baseline checkpoint and variant checkpoint reject
one another. Model,optimizer,scheduler and global RNG restore without original
initialization assets. Existing V48 source modules are unchanged.

## Verification and run protocol

81 focused and historical checks pass. New failing-test-first checks cover exact
initial common-weight/output/RNG parity, common-gradient parity, conditioning
influence and prefix-derived step IDs under independent-row execution, checkpoint
training parity, modulation and delayed embedding gradients, strict asset-free
variant restore, deterministic/batch-independent generation, immutable births,
EOS/capacity and unchanged recipe configuration.

Native preflight is FP32,50people+EOS,batch4/accum2,256x416 images alongside the
baseline. It checks the native initial common-weight hash against the original
V48 manifest, retained frontend transfer coverage, finite/nonzero learned-codec
and AdaLN gradients, restored next-update/scheduler agreement, forced50-birth
sampling, resource measurements and native two-VAL-image evaluator smoke.
A resource failure must be reported before changing the recipe.

Fresh seed42/full80 uses the same V48 TRAIN10k/VAL2k split, augmentation, resize,
FP32,batch4/accum2,AdamW2e-4/backbone1e-4,wd1e-4,clip.1,drops31/61. Same fixed32VAL
before/per epoch, best=min(poseAP,bboxAP),last retained and selected full2000VAL
after training. TEST remains unopened. Source/config/input and initial hashes,
initialization proof, preflight and test evidence accompany the frozen run.

TensorBoard adds a separate `V48-AdaLN-ordered-birth-only` run using the same
`quick-test/ordered_birth_only/{pose_AP,bbox_AP}` tags for baseline overlays.
Existing roots and pins remain. Teacher-forced/generated birth errors,count/EOS
steps,throughput and memory retain V48 labels and arithmetic.

## Status and interpretation

Native preflight passed and the parallel full80 run is active at
`/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v48_adaln/20261110_135143`.
TrainerPID73156,tmux`poseflow-v48-adaln`; launch verification reached epoch0/step170,
finite loss12.07970,LR2e-4,shared-GPU16.39images/s and peak1,982,794,752bytes.
BaselinePID4145233 continued at epoch23/step29760 during this check.

Full-capacity training peak was1,924,361,728bytes (1.79GiB). First-update28.66s
includes cold CUDA compilation/startup and is not steady throughput. Forced50-birth
inference measured7.19images/s alongside baseline, with two resident verification
models; it is not an isolated deployment benchmark or a controlled cost comparison.
Resumed next-update loss627.4702758789062 matched exactly; maximum state delta
1.49e-8 and matching scheduler. Both layers' modulation weights/biases and shared
step embedding have finite/nonzero gradients after the second update.

The common V48 initial-model hash is exactly
`0935436bdbaea1b5a03f424028147e219e10803f8f4281d6d2f191b004c349e9`,
matching the baseline's original fresh initialization. All330 retained frontend
tensors have exact100% ED-Pose coverage. The full variant initial hash is
`a2c668dfae6e7c65280798d7945f70a21255bb6d74f9c88fc22040e376f34470`.
All566 archived source/config/script hashes and four launch input hashes match;
verification additionally pins the baseline manifest. Same fixed32VAL IDs,
TRAIN/VAL split/recipe and TEST protection are verified. No native verification
updates were carried into training.

TensorBoard33332 adds `V48-AdaLN-ordered-birth-only/.` beside
`V48-ordered-birth-only/.`, using the same pinned AP tags. All68 previous roots
and18 pins are preserved. Native initial AP is0/0 before training; epoch
outcomes/final selected2000VAL remain asynchronous. No accuracy benefit is inferred
from wiring/parity tests. Compare free-running AP at matched image exposure,
rank-wise birth errors and computational cost separately. Fixed32VAL is noisy;
selected fullVAL evaluation is needed for the final quality report. Gates are
feature-channel multipliers, not a calibrated measure of image importance.

2026-11-10 final V48-AdaLN delivery check: both V48 trainers active, baselinePID4145233 preserved. Comparison browser shows both checked runs and pinned AP cards; live training scalars indexed (11 points through step109). Screenshot tmp/v48_adaln/dashboard.png. Setup/verification/archive/parallel launch complete; full80 outcomes pending.
