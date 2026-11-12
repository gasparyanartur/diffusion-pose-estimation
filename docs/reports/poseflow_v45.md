# V45: progressive person prediction with bidirectional prefix attention

V45 replaces V44 latent diffusion with image-only progressive person prediction.
The native image encoder, one box-decoder layer and confidence-ranked top-50
selection run once. The two retained person-decoder layers share weights across
50 growing-prefix stages. Each stage appends that proposal's image feature plus
a shared zero-initialized learned prediction token. All active tokens attend
bidirectionally, including earlier tokens attending to newly introduced people.
Future person tokens are absent; the full image memory remains available.

Raw decoder hidden states carry across stages with no detach, reset or repeated
feature injection. Each shared layer has its own 256-to-32 head, followed by the
frozen High-KL codec for all geometry. There is no noisy state, time conditioning,
DDIM, teacher forcing, latent feedback, target rematching or learned stopping.
Proposal i (one-based) receives 2*(51-i) decoder-layer updates. Because older
predictions can be revised, this is not a strict autoregressive factorization.

## Training contract

Fresh ED-Pose transfer follows V44's exact native cached frontend; latent heads
are fresh and the prediction embedding starts at zero. No trained V44 state is
imported. The SHA-pinned codec and TRAIN-only posterior-mean statistics are those
of V44. Frozen codec decoding remains differentiable with respect to predicted
codes; the codec weights/statistics stay frozen/eval.

One selected-slot Hungarian assignment fixes every clean standardized code,
positive mask and geometry target for the entire forward pass. Unmatched slots
have zero-code targets and no geometry loss. Every active person's prediction
is supervised after both layers at every stage, weighted 1/[2*(51-i)]. After
summing these normalized per-person errors, positive/background latent means
are balanced with weight10 and positive geometry has weight0.5. Native frontend,
encoder and DN supervision plus weight2 selected focal classification are added
once. Selected confidence remains image-only. All50 proposals, including
background, enter the active window. Empty images keep background/frontend
losses; missing joints retain native masks and capacity overflow is rejected.

Activation checkpointing covers each shared two-layer stage using non-reentrant
checkpointing with RNG preservation. Gradients span the entire growing sequence.
The per-person geometry helper preserves the previous geometry-loss mean API.
V44's frontend was moved verbatim into a shared mixin; its outputs and default
behavior are covered by exact parity/regression checks.

## Recipe and evaluation

The approved run is seed42/full80epochs, FP32, batch4/accumulation2, AdamW2e-4
(backbone1e-4), weight decay1e-4, clip0.1 and LR drops31/61. Native256/427 resize,
TRAIN flip/crop, disjoint TRAIN10000/VAL2000; TEST stays unopened. All existing
runs are preserved. Codec and ED-Pose pretraining may have prior trainval
exposure, so VAL is not an independent pretrained-detector test.

Initial and per-epoch fixed32VAL use the same seed42 selection as V44. Final
stage50/layer2 geometry and image-only confidence use native CrowdPose pose NMS
and evaluator. The best checkpoint maximizes min(poseAP,bboxAP); best/last are
retained and selected best receives full2000VAL after training. No diffusion
K-step curves or noise tags are generated. GT is used only after image-only
prediction for metrics and fixed-proposal birth/final diagnostics.

TensorBoard exposes `quick-test/progressive/stages_50/{pose_AP,bbox_AP}` and
rank-grouped birth/final latent and geometry errors. `train/total_step` (and epoch reductions), component
losses, gradient norm, measured images/second and peak allocated CUDA bytes are
logged. Initial step-zero AP is an untrained baseline. Existing57 dashboard
roots and six pinned historical AP tags are preserved; two progressive AP
cards are added to the saved dashboard URL.

## Verification and provenance

All 53 focused/related tests pass. Failing-first tests cover growth/shared layers/state carry, future-token
exclusion, later-active-person influence on earlier predictions, single context
and assignment, equal accumulated weights and single-prediction loss equivalence,
geometry mean parity, codec invariance, checkpoint gradient parity, deterministic
image-only evaluation, empty/mixed/missing-joint/non-square cases and strict
contract/capacity handling. Related V43/V44/codec regressions are retained.

Native full50 FP32CUDA batch4/accum2 completed two optimizer updates. Training
peak allocation was 1594383872 bytes (~1.49GiB); the complete restoration probe
peaked at2456589312bytes. Restored next-update losses both39.7752685546875;
maximum next-state difference5.96e-8. Codec unchanged. Three-image native AP
wiring check passed. The16image training preflight including startup/validation
took36.57seconds; this is not a steady-state throughput estimate. These are
implementation checks, not learned-performance evidence.

The isolated training API exposes prepare_context, predict(context,image_wh,
collect_trace=False), and image-only sample_keypoints. Optional traces include
stage/layer/original proposal identities and predictions; final-only inference
omits traces. A strict V45 contract and embedded model/codec/optimizer/scheduler/
RNG state allow native restoration without initialization assets. Sources,
resolved config, input SHA256, initial model hash and preflight evidence are
snapshotted before launch. Run provenance and live verification follow below.

## Usage

```bash
PYTHONPATH=. venv/bin/python scripts/training/train_poseflow_v45.py pipeline.latent_statistics_file="$PWD/tmp/v44/train_statistics.pt"
```

V44 statistics must already exist and pass codec, annotation and TRAIN metadata
checks. Use a fresh Hydra run directory; never restore a V44 checkpoint as V45.

## Limitations

The decoder processes25.5times the token-layer visits of one V44 denoising call;
this is not a compute-matched comparison. Confidence ranking determines who
receives extra refinement, and background tokens can influence active people.
Birth/final errors need not improve monotonically. AP, refinement and runtime
must be reported separately; isolating progressive introduction requires a
subsequent matched control. No automatic extension or architecture promotion.


## Verified full80 launch (2026-11-07T15:41+01:00)

Run: `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v45/20261107_153347`. Fresh source snapshot657files plus5input hashes verified;
initial model hash exactly matches native preflight. Live PID1926106,
epoch0/80, verified optimizer step90 with finite loss. All50
stages and the original recipe are retained. Fixed32VAL image IDs equal V44's.
No pilot cutoff; full2000VAL selected-checkpoint evaluation follows training.

TensorBoard33332 exposes `V45-progressive-prefix/.` with measured progressive
pose/bbox AP and live `train/total_step`, throughput and peak allocations. All57
historical root paths and six saved AP pins are preserved; the saved dashboard
URL adds the two progressive AP cards. InitialAP is a measured untrained baseline,
not learned performance. The direct host address responds successfully.
`launch_verification.json` records source/input/initialization/split/scalar evidence.
The previous V42 JEPA process completed its own80epochs normally; it was not
interrupted by this work.
