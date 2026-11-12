# V46: ordered person generation with parallel denoising rows

Status: implemented, native verification passed, fresh full80 training launched and TensorBoard verified. Learned epoch-end results pending.

## Architecture and targets

V46 retains the original ED-Pose backbone, input projections and encoder layers
2/4/6 used by V44. It discards the proposal/box decoder and all native detection,
DN and Hungarian objectives. A fresh two-layer pre-LN dense decoder (256 channels,
eight heads, FFN2048, GELU, dropout0) reads the cached image memory.

Valid augmented GT people are sorted by clamped floor(512*center_x/image_width),
then vertical center, horizontal center, box dimensions and masked joint
coordinates, with stable input order for exact duplicates. Column collisions
preserve all people. All geometry/masks/identities receive the same permutation.
Targets are standardized posterior means of the frozen High-KL32D pose-and-box
codec, with the verified TRAIN-only posterior-mean statistics. The codec SHA is
`7168bffc7452f83e5e7c70b062d55ff867d2cb7d1eb0932cd1339cdc9e0ee951`.
Codec decoding remains differentiable; parameters/statistics remain frozen.

Each of64 independent training rows contains min(row_index,P) noisy clean-GT
prefix people, a separate learned PREDICT token and padding to64 positions.
The prediction target is the next canonical-rank person, or EOS. Rows after EOS
retain the full GT prefix, refinement and EOS supervision. Capacity is50, with
an error for larger valid GT counts. Empty images train EOS on every row.

Noise depends only on row: u=(63-row_index)/63 and
alpha_bar=cos²((u+.008)/1.008*pi/2)/cos²(.008/1.008*pi/2). Endpoints are exactly
pure noise and clean. Gaussian draws are independent across rows/tokens;
PREDICT/padding never receives a clean target code. Position and sinusoidal
time embeddings identify rank and row. Self-attention is bidirectional within
rows; image attention shares K/V across all row queries. No row attends to
another row. A single RNG-preserving checkpoint wraps the batched two-layer
training decoder; inference uses no activation checkpointing.

## Loss and sampling contracts

Both layers predict clean codes and EOS logits. Birth predictions are averaged
over layers then people. Refinement predictions are first averaged over all
subsequent rows/layers for each person, then people. Birth and refinement get
equal task weight:10*mean(latent_birth,latent_refine)+.5*mean(geometry_birth,
geometry_refine)+EOS. Geometry preserves the existing per-person OKS-exponent,
box-center/size/GIoU arithmetic and missing-joint masks. EOS BCE balances person
and EOS row groups within each image, then averages images/layers. Padding/EOS
has no latent/geometry objective; there is no background-code regression.

Inference starts with only PREDICT and executes64 sequential rows. The second
layer's clean estimates replace the prefix; P(EOS)<.5 appends a new person until
capacity50. First EOS latches stopping; subsequent rows still refine and compute
EOS but cannot append people. Each next row independently corrupts the entire
predicted clean prefix at its own noise level. This is whole-prefix re-noising,
not DDIM or hidden-state recurrence. Birth confidence1-P(EOS) stays fixed.

Final detections use row64/layer2, valid slots only, with native CrowdPose pose
NMS and pose/bbox AP. Sequence ranks replace native proposal identities. `stop_reason` codes are0 active,1 EOS,2 capacity; `eos_row=0` means no EOS decision was observed.
`prepare_context`, `forward_rows`, `predict` and image-only `sample_keypoints`
are exposed. Optional traces record row/layer/rank/latent/EOS/birth/stop metadata.
Checkpoints include architecture/noise/order/objective/sampling/init/codec
contracts, frozen codec/statistics, model, optimizer, scheduler and RNG. Restoring
constructs architecture without opening ED-Pose, codec or statistics assets.

## Recipe, diagnostics and limitations

Fresh original ED-Pose initialization and fresh decoder; no trained V44/V45
warmstart. Seed42/full80, TRAIN10000/VAL2000, no TEST. Keep V44 augmentation,
resize256/max427, FP32,batch4/accum2, AdamW2e-4/backbone1e-4,wd1e-4,clip.1,
LR milestones31/61. Pretrained detector/codec exposure remains a caveat.

Initial and per-epoch evaluation use the same fixed32 VAL images as V44. Keep
best=min(pose_AP,bbox_AP) and last; evaluate selected best on full2000VAL after
training. Per-image seeds make evaluation independent of batch partitioning.

TensorBoard AP: `quick-test/ordered_sequence/rows_64/{pose_AP,bbox_AP}`.
`ordered_sequence/teacher_forced/` reconstructs from GT prefixes;
`ordered_sequence/generated/` reports direct rank errors only for emitted ranks.
These diagnostics have different coverage and are not interchangeable with AP.
Log counts, EOS row, capacity stops, separate birth/refinement/EOS losses, peak
memory and throughput. Do not require monotonic refinement improvement.

The dataset's valid annotation count peaks at13 in TRAIN and VAL, so most64-row
training examples are refinement rows. Equal task weighting prevents their count
from overwhelming birth supervision. Teacher-forced GT-prefix training differs
from generated-prefix inference; the experiment measures this design as approved.
Dense64-row training still has substantial compute and memory cost despite
parallel rows. Report full-capacity failure before changing the requested recipe.
No improvement can be attributed specifically to ordering or progressive
introduction without a subsequent matched control.

## Verification and run evidence

Focused CPU checks and regression evidence are archived under repository-local
`tmp/v46/`. Full64-row/50-person/batch4/accum2 native FP32 CUDA verification,
resource measurements and resumed-next-update agreement must pass before launch.
The rejected progressive V45 epoch13/step17500 checkpoint was preserved before
stopping only PID1926106; task-pose control/JEPA jobs were preserved.

Native verification passed with50 focused/legacy tests. Full64rows/50 synthetic
people/batch4/accum2/FP32 ran alongside both task-pose jobs on RTX3090. Peak
training allocation3,027,100,160 bytes (2.82GiB). Warm microbatches took
0.417–0.449 seconds (about9.3 images/second); the first1.33-second call includes
setup cost. Overall first16-image measurement was5.90 images/second. These are
synthetic-geometry wiring/resource measurements, not full-epoch throughput.

Restored next accumulated-update losses were exactly269.588134765625 and
269.5621337890625; maximum parameter/state difference1.4901161e-8. Frozen codec
hash remained unchanged. Initial model SHA:
`c534d9e32d2fe7cb09d1cb7e493cc321c25215bf2d3038165468031b003d3500`.
The preflight was repeated with initialization before data iteration, matching
the training entrypoint RNG order. Native3VAL smoke AP0 is an untrained wiring
check, not learned performance.

Fresh full80 run: `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v46/20261108_181906` (tmux `poseflow-v46`, PID2763350).
Frozen workspace sources: 670 files verified; all5 input hashes
verified. Initial model hash exactly matches the corrected preflight. TRAIN/VAL
10000/2000 are disjoint; fixed32 IDs match V44. Baseline fixed32 pose/bbox AP0
reflects fresh EOS-only predictions before updates, not a trained result.

Live verification: epoch0/80, optimizer step320, finite reported loss
6.6448, measured training throughput9.20 images/second including current
training-loop overhead. Full-epoch throughput remains pending. TensorBoard33332
run `V46-ordered-sequence/.` exposes baseline AP and live separate birth/refine/EOS
losses. API points and visible browser AP cards verified. All60 previous roots
and8 previous saved pins preserved;61 total roots. The indexer initially scanned
the new event stream before baseline values were flushed; restarting the same
all-root configuration after flush made the cards available immediately.

Best/last retention and selected-checkpoint full2000VAL evaluation after epoch80
are configured automatically. Outcome review, epoch AP/refinement behavior and
final fullVAL remain pending; retain the approved `.plan` entry.


## Frozen DDIM sampler comparison

V46 supports an evaluation-only override `sampler="ddim"` on `predict` and
`sample_keypoints`. The default remains `"renoise"`; training/checkpoint contract
and the running full80 workspace are unchanged.

For existing active tokens, use deterministic DDIM eta0:
`epsilon_hat=(z_t-a_t*z0_hat)/b_t`,
`z_s=a_s*z0_hat+b_s*epsilon_hat`. Newly generated tokens have no incoming noisy
state, so forward-noise their predicted clean code once at the next-row level.
Draw the same per-image/row noise tape in both arms; DDIM ignores fresh draws for
existing tokens. Preserve EOS latching, capacity,64 rows, birth confidence and
native postprocessing. The terminal clean row returns its clean prediction.

Reproduce a paired fixed32 VAL comparison in a fresh output directory:

```bash
venv/bin/python scripts/evaluation/compare_poseflow_v46_samplers.py \
  --run-root /path/to/original/v46/run \
  --checkpoint /path/to/completed/last.ckpt \
  --output /path/to/fresh/sampler_comparison
```

The CLI copies/hashes the checkpoint before use, restores without initialization
assets, verifies exact legacy output/RNG parity against the original frozen model
on3VAL images, reuses original fixed32 IDs/annotation paths, evaluates both arms,
and verifies model/codec hashes unchanged. It saves source/input/codec manifests,
AP/count/EOS metrics, rank-wise birth/final errors with coverage, first-person
per-row trajectories and separate TensorBoard streams. This is a frozen subset
diagnostic; it neither changes the training sampler nor establishes fullVAL gains.

DiffusionDet's Table4c motivates testing DDIM: direct prediction feedback degraded
with iteration, DDIM alone improved slightly, and DDIM plus box renewal improved
more. This comparison changes only DDIM; it adds no renewal or DDPM arm.
Source: [DiffusionDet, Sec4.4/Table4c](https://arxiv.org/pdf/2211.09788).

Frozen comparison completed; measured results and scope are recorded below.


### Frozen comparison outcome (2026-11-09)

Checkpoint: completed epoch46 (zero-based45), optimizer step57500,
SHA256 `a5b649ece7eef15e2d2b2bc4f476b37c5863160e447d8d8243067396a1678e0c`. Same original fixed32VAL IDs, seed42,
64 rows, capacity50, FP32, birth-noise tape and native pose NMS/AP in both arms.
Diagnostic namespace: `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v46/20261108_181906/diagnostics/ddim_20261109_085306`.

| Metric | Independent re-noising | DDIM eta0 |
| --- | ---: | ---: |
| Pose AP (0–100) | 0.2115 | 0.1043 |
| Bbox AP (0–100) | 5.1836 | 5.6296 |
| Count MAE | 2.75000 | 2.90625 |
| Rank1 birth latent MSE | 0.322582 | 0.322582 |
| Rank1 final latent MSE | 0.741180 | 0.744930 |
| Rank1 birth geometry loss | 9.904446 | 9.904446 |
| Rank1 final geometry loss | 20.455314 | 23.607868 |
| Evaluation images/sec | 1.241 | 0.845 |

AP values in saved JSON/TensorBoard remain fractions; the table multiplies by100.
Rank1 has the same32-image coverage in both arms; higher ranks carry their own
coverage counts. Birth rank1 is exactly identical, and original sampler output,
full latent traces and RNG state match the original frozen workspace on3VAL
images exactly. Model and codec hashes were unchanged across both evaluations.
46 focused/legacy regression tests pass, including oracle DDIM, newborn noise,
EOS, terminal/noise schedule boundaries, RNG parity and evaluator metadata.

DDIM reduced some intermediate errors and removed much of the noisy variation,
but its first-person trajectory still drifts from about.323 at birth to.745 at
row64. It did not rescue pose AP or final refinement on this frozen subset;
bbox AP improved modestly. This does not establish a result for other DDIM
settings, other checkpoints, or fullVAL, and does not identify the remaining
failure mechanism. No sampler promotion, retraining or active-job modification
was performed.

Timings include trace collection and a shared GPU, with sequential arm execution;
these are diagnostic timings rather than controlled deployment benchmarks. Both
arms peaked at268363776 allocated bytes in this process.

`comparison.json`, per-arm prediction/metric JSON, first-person64-row trajectories,
`trajectory.png`, source/input/codec manifests and `verification.json` are archived
in the diagnostic namespace. TensorBoard comparison labels:
`V46-sampler-renoise-epoch046/.` and `V46-sampler-DDIM-epoch046/.`, sharing
`quick-test/sampler_comparison/rows_64/{pose_AP,bbox_AP}` and
`trajectory/rank_01/{latent_mse,geometry}`. Scalar API values/64-point traces
verified;63roots preserve all61 previous roots and12pins preserve10 previous
pins. Browser verification passed: both sampler runs selected, both AP cards and both64-row trajectory overlays visible; comparison tab retained. The run filter intentionally permits TensorBoard's `/.` suffix.
