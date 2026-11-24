# V52: joint clean geometry with large person features

Status: verified and launched2026-11-13. Fresh80epoch training is active; detection/refinement outcomes pending.

V52 keeps sample-local proposal selection and sequential person generation from
V50/V51, and replaces the learned32D coordinate codec and dense image reads.
A person has a shared256D feature. Each of two independently parameterized
layers predicts clean geometry directly:14 joint offsets relative to a predicted
box center, sigmoid box center, and softplus positive width/height. Absolute
joints equal center plus offset; offsets are unbounded. There is no dedicated
box/pose latent partition, latent regression loss, or predicted diffusion delta.
The V51 clean-anchor interpolation/bypass is removed: even clean inputs are
supervised predictions.

## Decoder and image reads

Each layer has independent-row bidirectional self-attention, multiscale
box-referenced deformable image attention, and an FFN. Width256,8heads, FFN2048,
GELU, pre-LayerNorm, zero dropout;4samples per head/feature level. All50rows are
parallel during training. Image encoding and each layer's projected image values
are computed once per image. Row queries flatten only for image reads, without
cross-row self-attention or duplicated image encoding.

The first layer uses the deattenuated noisy current box for existing people
(sampling size clamped to[1e-4,2]), and selected native proposal boxes for births.
Layer2 samples around layer1's detached clean predicted box. Hidden feature
gradients remain connected between layers. No hidden state carries between rows.
Per-token sigma conditions features and the image-read query through adaptive
LayerNorm (Ada-LN: learned time-dependent normalization scale and shift).
Small nonzero attention query weights open that gradient path at initialization
while retaining the native radial sampling-offset biases. Sampling coordinates
include per-level image valid ratios. Padded memory values are zeroed once.

## Objectives and sampling

The existing coordinate-space DDIM eta0 token-age schedule,50rows,49people plus
EOS capacity, random teacher permutation and one selected-slot assignment per
forward are retained. Image-only native proposal confidence, used-proposal
exclusion, EOS latching and post-EOS refinement are retained. There is no new
box-specific diffusion policy or box/joint update gate.

Pose and box objectives retain masked coordinate MSE and robust decoded geometry
terms. Both layers are supervised; birth/refinement each have half the geometry
budget, and box/pose each have half within those groups. Refinement keeps existing
per-person present-age-bucket averaging. Each task contributes0.25 times its loss
divided by its frozen calibration scale. These are fixed priorities at all times;
there is no task-age schedule or online adaptive balancing. Scales are median
shared two-layer decoder task gradient norms over32 augmented TRAIN batches at
fresh initialization; all weights and RNG are preserved during calibration.
EOS, conditional proposal-selection and native frontend/DN terms remain.
Selector prefix features are detached. Native/selector/geometry families retain
disjoint clipping at0.1/sqrt(3) after accumulation.

## Evaluation and checkpoints

Same fixed32VAL IDs, original native postprocessing and per-image seeded generation
before training and after each epoch. Birth, first-EOS and final geometry come
from the same trajectory with the same identities and birth scores. First-EOS
includes the stopping row's refinement; capacity stops fall back to final geometry.
AP is published under exactly `quick-test/pose_AP` and `quick-test/bbox_AP`;
only run labels distinguish experiments and snapshots. Teacher-forced errors are
separate diagnostics. Checkpoint selection uses final min(pose_AP,bbox_AP).

Strict V52 checkpoints contain architecture/sampling/loss/calibration contracts,
initialization provenance, model/frozen scales, global RNG; the training checkpoint
also restores optimizer/scheduler. Restore creates the architecture without
external initialization assets. Prior-family checkpoints are rejected.

Fresh seed42/full80 training uses original ED-Pose frontend initialization,
TRAIN10000/VAL2000, FP32, batch4/accumulation2, resize256/max427, existing
augmentation, AdamW2e-4/main and1e-4/backbone, decay1e-4 and LR drops31/61.
Best and last are retained; best is evaluated on full2000VAL after training.
TEST remains unopened. Existing runs/artifacts are preserved.

## Verification and interpretation

Failing-test-first model/training iterations cover joint readout, actual
center coupling, current-box/time query influence, row isolation, padding,
checkpointed gradients, frozen calibration/RNG, strict restore, empty/masked
people, frontend/cached-attention parity, EOS/capacity and per-image seeded output.
Native preflight must exercise50rows/49people at batch4/accum2 on non-square
images, native CUDA deformable forward/backward, gradient parity, resumed next
update and full-capacity inference. Resource failures must be reported before
changing recipe. Source/config/input hashes and initial model coverage/hash
are archived with an immutable source snapshot.

This is a combined architecture change. Passing verification establishes wiring
and reproducibility, not detection quality or beneficial refinement. Report AP,
paired refinement behavior and computational cost separately.

## Verified launch

- Run root: `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v52/20261113_135134`; trainerPID2442843; tmux`poseflow-v52`.
-24 focused/model/training/legacy/evaluator/TensorBoard tests passed. Native CUDA checkpoint gradient parity covered243 parameter tensors (max absolute delta1.91e-6).
- Full50row/49person FP32 batch4/accum2,256x416 synthetic preflight:0.808s/optimizer update,9.90 images/sec,2,079,212,544 peak allocated bytes. Full50row forced-capacity inference4.68 images/sec. Shared RTX3090 alongsideV51; synthetic capacity measurements, not epoch timing.
- Restored next-update loss14.47447156906128 exactly matches uninterrupted; maximum model/buffer delta6.05e-8; scheduler agrees. All retained frontend blocks have100% exact pretrained tensor coverage.
-39,362,573 total /39,140,173 trainable parameters. Initial model SHA256`583aaac2144979ccfbd7d146b9dac7354e6aacb041dfe3b999eccb8475bb751d`.614 executable/config source hashes and904 archived source/doc files.
- Fixed TRAIN calibration scales pose_birth/box_birth/pose_refine/box_refine:12.93449,3.42448,12.62062,3.27432. Frozen after32 augmented TRAIN batches; model hash unchanged.
- Initial fixed32VAL final pose/bbox AP0/0.00611124; birth0/0.00660660; firstEOS0/0.00611124. These are fresh-model baselines, not learned results.
- Confirmed active training at optimizer step140, epoch index0, finite loss5.31139. Early measured training throughput7.57 images/sec; includes existing concurrent workload.
- TensorBoard retains74 existing roots and all5 pins, adding final/birth/first_eos run labels with identical metric names. Live scalar API and browser verify birth/firstEOS/final alongside V50/V51 on the same two AP cards. Screenshot: tmp/v52/dashboard.jpg.
- V51 trainer1815487, birth observer1943783 and historical comparison bridge1976347 preserved. V50 had already completed; its checkpoints/events retained. No TEST evaluation or new experiment sweep.

Dashboard delivery note: the existing Python reader remains active on33332; prioritizing complete new streams allows the primary comparison to load while older history indexes. A loopback-only fast-reader test was rejected because its restart handling omitted36 cached history points (duplicate-step or out-of-order records). No source event was changed; temporary reader servers were stopped. All77 configured roots and saved pins are retained, and initial V52 snapshot values agree with native evaluation.
