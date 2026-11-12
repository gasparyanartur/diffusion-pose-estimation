# V45 task-pose ordered JEPA

This arm replaces proposal identities with 50 learned canonical-rank queries.
After augmentation, GT people sort by 512 normalized horizontal columns, box
center y, labeled-joint centroid x/y, descending box area, and stable input index.
Slot i receives person i; slots M..49 are background. Exact duplicate people are
retained. Overflow is rejected. No Hungarian, dynamic programming, GT count or
GT positional encoding is used during inference.

`OrderedTaskPoseModel` reuses the original ED-Pose backbone, retained three-layer
encoder and one native box-decoder stage. Learned rank features and reference
seeds replace native candidate selection. Image position encodings and rank
embeddings identify image locations and output ranks respectively; rank is not
a fixed spatial cell. Box features, detached boxes, confidence and rank IDs are
computed once and cached through K1/K4/K8. Box predictions are directly trained
at the rank frontend stage. Image confidence predicts person existence per rank.
The model does not use the old 200 candidate outputs, encoder proposal losses,
or DN queries/losses. Unused native candidate/DN weights are frozen where
applicable. Historical implementations and their jobs are unchanged.

The task-pose codec (28->256->32 with non-affine normalization), shared absolute
reader (32->256->256->28 sigmoid), zero-initialized latent residual, two native
person-attention layers with 60 offsets/head/level, scale6/1000-step coordinate
cosine diffusion and coordinate DDIM remain the V45 task-pose JEPA formulation.
No hidden state is carried between calls. GT poses are corrupted in their fixed
rank slots; background sources are noise as in the existing task-pose arm.

Both pose stages retain native masked L1/OKS. The rank frontend receives native
box/classification losses. Clean reconstruction trains the reader on online and
EMA teacher codes, weighted .5. Final complete-person JEPA MSE has weight1.
Teacher decay .999 updates after optimizer steps; partial poses retain masked
geometry losses. Empty complete-pose batches return differentiable zero aux
losses. Explicit rank targets replace every set-assignment call. Classification
supervises positive prefix and background suffix at training; inference predicts
all50 slots without knowing M or masking by GT count.

This changes frontend learning as well as target correspondence: any AP change
cannot be attributed to ordering alone. Column/rank boundaries are discontinuous
and crowded people may exchange rank under augmentation. Tie-breaks preserve
capacity; they do not guarantee stable identities under localization errors.

## Usage

Hydra preset: `benchmark_poseflow_v45_ordered`. The dedicated durable entrypoint
is `scripts/training/train_poseflow_v45_ordered.py --root <prepared-run-root>`.
The run manifest pins source/input hashes, initialization hash and configuration.
Fresh original ED-Pose initialization, seed42, FP32, batch4/accum2, full80 epochs,
optimizer/augmentation/scheduler match historical task-pose runs. No trained
checkpoint warmstart or short pilot cutoff. Checkpoints record rank capacity,
ordering, absolute-coordinate/latent/objective/EMA contracts and RNG; strict
restore rejects incompatible contracts. Inference requires model weights only.

The intentionally retained historical V42 protocol trains on trainval, measures
overlapping VAL loss, monitors fixed32 TEST development AP for K1/K4/K8, then
runs final full8000 TEST. This is development exposure, not untouched-test evidence.
Compare matched training ages and report single-call AP separately from sampling
degradation. Wiring/gradient/restore checks are not learned-performance evidence.

Implementation verification and launch details are recorded in `.plan`, `.log`
and the archived run's `preflight.json` and `manifest.json`.

## Launch

Fresh full80 root: `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v45_ordered/20261109_094318`. tmux `poseflow-v45-ordered`. Initial
model SHA256 `299c06c8bb009536017378c4094f0daa3f92e7a9a2e52060a30f3ca9ac9abef5`.
36 related checks and8 final focused checks passed; native CUDA/strict restore
next-update max difference1.19e-7. Real optimizer progress verified; first learned
AP follows the first completed epoch. TensorBoard label
`V45-task-pose-ordered-JEPA/.`, existing comparison/pins retained.
