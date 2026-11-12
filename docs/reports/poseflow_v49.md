# V49: token-age diffusion forcing

## Experiment contract

V49 completes the reviewed token-age design using V47's shared trainable
coordinate encoder/reader, the original pretrained ED-Pose image frontend with
three retained encoder layers, and two fresh dense decoder layers (width 256,
eight heads, FFN 2048, GELU, pre-normalization, zero decoder dropout). All valid
person and prediction-query coordinates pass through the same encoder. Each
token receives its own sinusoidal Gaussian-coefficient embedding; query tokens
also receive the shared learned prediction embedding. Active-prefix attention
is bidirectional. Image memory and per-layer image keys/values are cached once.
Transformer hidden states do not carry between rows. No proposal selection,
Hungarian/OTA training matching, frozen latent codec or KL objective is used.

This adapts per-token-noise sequence modeling to person-set construction; it is
not a reproduction of the causal architecture or decision-making experiments
in [Diffusion Forcing](https://arxiv.org/abs/2407.01392).

## Targets, noise and objective

Each example receives one random permutation shared across its 50 independent
teacher-forced training rows. Source identities, masks and coordinates stay
aligned. Coordinates contain 14 joint XY pairs and box CXCYWH, normalized by
resized image size. Missing input joints use their own box center; original
supervision masks remain intact. Corruption operates on `x=6*(2*c-1)`.

For person age zero, `sigma=1`; subsequent ages use `.1,.05,.025,...`. Each
input is `sqrt(1-sigma**2)*x0 + sigma*epsilon`. Gaussian draws are independent
across tokens and training rows. The final row gives existing people zero input
noise. The next-person/EOS query is always pure noise, including EOS tails;
its input type never reveals whether the target is a person or EOS. Padding is
zero and masked. The reserved final row allows at most 49 births; excess counts
raise rather than truncate. Empty images have EOS supervision only.

Both decoder layers predict a latent decoded to normalized clean coordinates,
plus EOS logits. Existing V47 coordinate/geometry arithmetic and masks are
reused. The objective is

`10*(coordinate_birth + coordinate_refine)/2`
`+ .5*(geometry_birth + geometry_refine)/2 + balanced_EOS`.

Birth averages both layers then people. Refinement first averages subsequent
rows and layers per person, then people. EOS balances person-target and
EOS-target groups per image, then images. There is no additional noise-level
weighting or age-dependent box/pose emphasis in this experiment. The final
clean-GT reconstruction endpoint is not evidence of successful generated-state
refinement.

## Sampling and restore

`prepare_context`, `forward_rows`, `predict` and image-only `sample_keypoints`
retain the previous family interface structure, with `forward_rows` accepting
`noise_sigmas[B,R,Q]` instead of one time per row. Only the new model uses the
new interface; historical families are unchanged.

Inference executes all 50 rows. Each row draws a fresh noisy query and predicts
clean coordinates for existing people and a possible birth. The first EOS
decision at probability >=.5 latches; capacity 49 also stops births. Later rows
still refine existing people. Scores remain `1-P(EOS)` at birth.

Coordinate states transition using deterministic DDIM, without independently
re-noising an existing person:

`x_next = alpha_next*x0_hat + (sigma_next/sigma_current)*(x-alpha_current*x0_hat)`

where `alpha=sqrt(1-sigma**2)`. This ratio form avoids dividing a residual by a
tiny sigma. Terminal zero noise returns the clean estimate directly. A birth's
actual pure-noise query becomes its initial state and transitions to sigma .1
before the next row. Final detections use row 50/layer 2. Optional traces contain
row/layer/rank, sigma, clean predictions, EOS and birth/stop decisions; ordinary
inference retains only final outputs. Per-image generators make evaluation noise
independent of batching.

The strict V49 checkpoint records architecture, source ordering, rows/capacity,
coordinate scale, corruption, per-token conditioning, losses, sampling, EOS and
initialization. Model, optimizer, scheduler and Python/NumPy/Torch/CUDA RNG restore
without initialization assets. Training activation checkpointing preserves RNG;
inference disables it.

## Evaluation and training

Fresh seed 42, 80 epochs, TRAIN 10,000 / VAL 2,000, original augmentation and
aspect-preserving 256/427 resize, FP32, batch 4, accumulation 2, AdamW main rate
2e-4/backbone 1e-4, weight decay 1e-4, clipping .1, LR drops at 31/61. TEST remains
unopened; the eagerly loaded unused test slot points to VAL. Original ED-Pose
pretraining exposure is inherited. No trained V47/V48 checkpoint initializes V49.

The same fixed 32 VAL images are evaluated before training and each epoch with
native pose/bbox AP and postprocessing. Best is selected by minimum pose/bbox AP;
last is retained and selected best is evaluated on all 2,000 VAL images after 80
epochs. Full training is gated on source/config/input hashes, initial-state hash,
initialization coverage, native full-capacity preflight and evaluator verification.

TensorBoard AP tags are `quick-test/token_age_diffusion/rows_50/{pose_AP,bbox_AP}`.
Birth/refinement/EOS losses, count error, EOS row, duplicate/coverage/identity
switch diagnostics, throughput and memory are separate scalars. Random-order
generated diagnostics pair final outputs to GT once for measurement and reuse
that correspondence at birth and later ages; this assignment never affects
training or sampling. Nearest-GT duplicate counts and identity switches are
diagnostic heuristics, not native AP or definitive identity labels. Teacher-forced
source reconstruction is labelled separately from generated trajectories.

## Verification and launch status

67 focused/historical checks passed after recorded failing tests. They cover
source/mask permutation, leakage exclusion, per-token noise, all-active loss
reductions, row output/gradient parity, bidirectional influence, both-layer and
codec gradients, checkpointed parity, DDIM endpoints, EOS latching/tails,
capacity, seeded batch independence, strict restore and deterministic evaluation.
Native resource/restore/evaluator verification and full launch are recorded below
when complete. Passing preflight or initial AP does not establish learned gains.

```bash
venv/bin/python -m scripts.training.verify_poseflow_v49 --output /path/to/fresh/verification
venv/bin/python -m scripts.training.train_poseflow_v49 \
  verification_file=/path/to/fresh/verification/verification.json
```

Use an immutable matching source snapshot for verification and training; the
trainer rejects changed source/config/input hashes. Scratch belongs under local
`tmp/v49/`. Source/config/input manifests and measured costs live in the fresh
output namespace; all existing trainers and artifacts are preserved.


Native verification (2026-11-10) passed at `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v49/20261110_232439`:

- FP32,50rows,49people,batch4/accumulation2,256x416; peak allocated training
  memory 1,943,252,480bytes (1.81GiB).
- Resumed next-update loss 617.1036682128906 exactly matched; maximum
  model-state delta 5.96e-08; scheduler state matched.
- Every coordinate encoder/reader parameter had a finite nonzero gradient.
- Forced49-birth image-only inference 4.03images/sec
  alongside existing jobs and two resident verifier models. Cold first update
  33.22s includes startup; it is not steady training throughput.
- Measured retained frontend coverage100%;330retained tensors exactly transferred.
- Two-image native evaluator smoke passed. Fresh trainerPID377259 launched in
  tmux `poseflow-v49`, initial state hash `a6b2745f0742f049d811837e61359da1dd266a5e3dfca3f6bb414589c868be37`
  exactly matches verification. 579source/config/script hashes pinned.

The initial fixed32 evaluation precedes optimizer updates. Live-training and
TensorBoard delivery verification are recorded separately once scalars flush.


Launch delivery verified: trainerPID377259, epoch0, optimizer
step120, finite loss4.044580, LR0.0002,
shared-GPU throughput7.06images/sec and allocated
peak2.46GiB. Initial fixed32VAL
AP0/0 is a wiring baseline, not learned performance. TensorBoard33332 exposes
real training scalars and both AP tags under `V49-token-age-diffusion/.`; both
cards are visibly pinned.71roots preserve70prior roots;20saved pins preserve all
18prior pins. Screenshot: `tmp/v49/dashboard.png`.

Implementation, verification, archival and fresh full80 launch are complete.
Learned epoch-end AP, selected best checkpoint and fullVAL results remain
asynchronous. Existing trainerPIDs4145233,73156,348105 were active at delivery.


## User cancellation — 2026-11-11

The user cancelled this experiment after reviewing its poor detection/refinement
results. The trainer exited gracefully after SIGINT. Best and last checkpoints
remain intact and were loaded and SHA-verified with optimizer/scheduler state.
The last completed checkpoint is at global step 82500; subsequent partial-epoch
updates are not claimed saved. Per-run `cancelled_by_user.json` records the exact
stop and preserved assets. No final 80-epoch/all-2000-VAL completion is claimed.
A proposal-attention identity and time-gated residual successor is being planned;
no successor code or training run has been started.
