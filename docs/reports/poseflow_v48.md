# V48: clean-prefix person generation, stopping at EOS

User-approved on 2026-11-10. V48 is an isolated V47-based experiment: jointly
trained coordinate codec, clean fixed prefixes, no refinement or diffusion.

## Architecture and objectives

After augmentation, valid people use V46/V47's512-column canonical ordering,
including deterministic collision tie-breaks. A person has32 image-normalized
coordinates:14 jointXY pairs and boxCXCYWH. Unobserved input joints use the box
center; coordinate and geometry objectives retain their original masks.

The shared learned V47 person encoder maps32->256 GELU->32 nonaffineLayerNorm.
The shared reader maps32->256 GELU->256 GELU->32, with softplus+1e-6 box sizes.
Its input is clean normalized coordinates, exactly V47's clean endpoint after
inverting x=6(2c-1). There is no Gaussian corruption, timestep embedding, latent
loss, frozen VAE or auxiliary autoencoding loss.

The retained original ED-Pose image frontend uses encoder source layers2/4/6 and
V47's freezing policy. A fresh two-layer256-wide,8-head,2048-FFN GELU preLN,
zero-dropout decoder uses shared image K/V cached once per image/layer. Learned
rank and PREDICT embeddings identify sequence positions. Valid tokens attend
bidirectionally within each independent prefix; rows never attend to each other.
Hidden states are recomputed from coordinates at each generation step.

An image withP people hasP birth-target rows and one EOS-target row. Rows train
in parallel, padded only to the largestP+1 in the batch, up to51. In each row,
clean GT prefix people precede a separate PREDICT embedding. Only PREDICT
readouts receive direct supervision, after both decoder layers. Safe dummy
PREDICT tokens in inactive batch rows avoid all-masked attention; those rows
carry no loss. Original targets remain separate from inputs.

The objective is10*coordinate_birth+.5*geometry_birth+balancedEOS. Birth losses
average both layers and valid people. EOS separately averages person-target
rows and the single stop row per image, weights present groups equally, then
averages images. Empty images receive one EOS target and zero geometry/coordinate
loss. Removing V47's task average doubles effective birth coefficients relative
to V47; total loss values are not directly comparable.

## Generation and interfaces

`model_v48.py` exposes `prepare_context(images,padding_mask=None)`,
`forward_rows(context,prefix_coordinates,token_types,padding_mask)`,
`predict(context,image_wh,collect_trace=False)` and an image-only
`sample_keypoints` wrapper. Only the second layer's current PREDICT coordinates
are appended. Previous birth coordinates and confidence remain fixed exactly.
Confidence is1-P(EOS) at birth. Each image stops immediately at P(EOS)>=.5 or
50 births and leaves subsequent decoder batches. There are no post-EOS rows,
re-noising, DDIM updates or recurrent hidden states. EOS at capacity is not
queried during inference; a50-person training image includes its EOS row.

Outputs retain native geometry/confidence plus valid masks, counts, sequence
ranks, EOS step, generation step counts and stop reason1=EOS/2=capacity. Optional
traces contain step,layer,image/rank,coordinates,EOS probability and decisions.
Normal inference retains final outputs only. Strict V48 checkpoints reject
changes to the architecture, order, objectives, clean-prefix policy or sampler;
restore model/optimizer/scheduler/global RNG without reading initialization assets.

## Verification and run protocol

Tests were written failing first for each iteration.74 focused/historical tests
pass, including clean rows/one EOS, masks/empty/capacity, equal birth weighting,
EOS group balance, batched/independent output and gradient parity, bidirectional
influence, excluded future/padding, checkpointed gradient parity, both-layer
codec gradients, exact legacy frontend parity, immutable births, mixed-image
stopping, deterministic evaluation, strict restore and preset construction.

Native preflight uses FP32,50 people plusEOS, batch4/accum2 and256x416 images.
It measures training/forced50-birth inference, checks optimizer/scheduler/RNG
restore and next-update agreement, and exercises native VAL postprocessing and
AP evaluation. Resource failure must be reported before changing the recipe.
Verification and full training use an immutable source snapshot; fresh training
must match source/config/input hashes and initial model hash. Native verification
updates are discarded before the training run. Existing jobs/artifacts remain.

The run uses seed42/full80, original TRAIN10k/VAL2k, V47 augmentation/resize,
FP32,batch4/accum2,AdamW2e-4/backbone1e-4,wd1e-4,clip.1,drops31/61. TEST stays
unopened. Same fixed32VAL before/per epoch; best=min(poseAP,bboxAP),last retained;
selected checkpoint evaluated on all2000VAL after training. TensorBoard uses
`quick-test/ordered_birth_only/{pose_AP,bbox_AP}` and separate teacher-forced
versus generated birth errors by rank, count/EOS steps, throughput and memory.
Native bbox AP retains box-only annotations absent from valid-pose supervision,
as in V47; evaluator labels are not filtered to raise reported AP.

## Outcome status and interpretation

Native verification passed and a fresh full80 run is active at
`/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v48/20261110_095254`.
TrainerPID4145233,tmux`poseflow-v48`; launch verification reached epoch0/step200
with finite loss6.90969,LR2e-4,shared-GPU11.41images/s and peak1,995,543,552bytes.
The initial model hash is
`0935436bdbaea1b5a03f424028147e219e10803f8f4281d6d2f191b004c349e9`,
matching verification exactly.560 source/config/script hashes and4 launch inputs
are archived; the fixed32VAL IDs match V47. All330 retained frontend tensors
match original ED-Pose initialization exactly (100% element coverage).

Native full-capacity training peak was1,877,937,664bytes (1.75GiB).
First-update timing28.43seconds includes cold CUDA startup/compilation and is not
steady throughput. Forced50-birth inference measured5.04images/s alongside V47,
with two resident verification models; it is not an isolated deployment benchmark.
Resumed next-update loss627.9840698242188 matched exactly, with maximum state
delta2.24e-8; scheduler state agrees. All10 learned codec tensors have finite,
nonzero gradients. `verification/` contains measurements, restore checkpoint,
gradient coverage, initializer proof and two-VAL-image evaluator smoke.

TensorBoard33332 indexes `V48-ordered-birth-only/.`; pose/bbox AP cards are visible
and pinned. All67 prior roots/16 server pins were retained, with2 V48 pins added;
the browser's existing4 pins were also retained. Initial fixed32 AP is0/0 before
training; subsequent epoch outcomes remain asynchronous. V47PID3490732 remains
running; historical artifacts and TEST were preserved. Screenshot:
`tmp/v48/dashboard.png`.

No detection improvement is claimed from unit/native wiring tests. This experiment removes prefix noise and refinement together,
changes effective birth weighting, and drops timestep conditioning. Compare
free-running AP at matched image exposure, teacher/generated birth errors and
computational cost separately. Exposure mismatch remains between GT training
prefixes and generated inference prefixes. No scheduled sampling or curriculum
is included.

2026-11-10 final dashboard check: historical-root indexing completed; live V48 train/total_step advanced from step19 to649 (65points), trainer continuing epoch0/step650. Initial AP cards pinned/visible; all setup checks complete.
