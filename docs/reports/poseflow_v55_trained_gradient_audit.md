# V55 trained-checkpoint gradient and optimizer audit

2026-11-15. Diagnostic only: no production code, trainer, objective, augmentation,
checkpoint policy or TensorBoard history was changed.

The audit found a specific optimization coupling: classification gradients in the
shared proposal-to-prefix attention block consume almost all of the clipping
budget also assigned to the private pose decoder. The decoder is connected and
updates, but its updates depend unnecessarily on classification-gradient size.
Launch-time finite-gradient checks did not establish balance after training.

## Evidence and reproducibility

Run: `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v55/20261114_184631`.
Frozen snapshots include complete AdamW moments and step counters:

| Completed epochs | Global step | Snapshot | SHA256 |
| --- | ---: | --- | --- |
| 15 | 18750 | `epoch015.ckpt`, source `epoch014.ckpt` | `5f59a19bea5caebae78f10def7533cd4d0a11fc2d4ec258b52e76f6b447be286` |
| 19 | 23750 | `latest.ckpt`, source `last.ckpt` at audit start | `51c69c8786cd4b6d5a10d6a8a622657376ab6b34fc65b3427542a01adfccf31b` |

Artifacts and temporary scripts: `tmp/v55_gradient_audit/`, including `audit.py`,
`validate.py`, `sample_manifest.json`, `snapshots.json`, `epoch015.json`,
`latest.json`, `validation.json` and `summary.json`. The filename `latest` denotes
the frozen audit-start snapshot, not the checkpoint currently training.

Both ages use exactly the same 32 augmented TRAIN images, seed 4205515, eight
batch-4 microbatches and four accumulation-2 update windows. Image IDs, counts and
resize are recorded. All five weighted objectives are differentiated from the
same forward. Their summed gradients agree with actual total backward to maximum
relative L2 error 1.65e-6. Every trainable parameter has unique optimizer ownership.
FP32 and the native CUDA attention backend are verified. All gradients and updates
are finite. All 644 pinned source/config/native file hashes and both snapshot
hashes were rechecked unchanged at completion.

## What is imbalanced

Gradient clipping limits the joint L2 norm of a parameter group. The current
policy assigns equal bounds `0.1/sqrt(3)` to frontend, selector and geometry.
Geometry includes both the shared attention block and the private pose decoder.
Consequently, larger classification gradients in attention reduce every pose
gradient in that same group, even though classification cannot train the private
pose parameters directly.

At epoch 19, median shared-attention gradient norms over four update windows are:

| Weighted objective | Gradient L2 norm |
| --- | ---: |
| Pose birth | 0.0831 |
| Box birth | 0.0246 |
| Proposal selection | 1.3492 |
| EOS | 1.1180 |

The attention block accounts for a median 97.96% and 98.48% of squared geometry
gradient norm at epochs 15 and 19 respectively. A completion-time check of the
latest 100 paired live CSV records through step 74609 gives 98.78%, so this
coupling persists much later. This live check measures norms; optimizer replays
below remain restricted to epochs 15 and 19.

Task-gradient directional similarity (cosine: +1 aligned, 0 orthogonal, -1
opposed) does not show consistently opposing pose/selection gradients. Their
shared-attention cosines range from -0.058 to +0.157 across eight windows.
Dominant magnitude, rather than proven systematic cancellation, supports the
clipping correction. These measurements do not justify silently removing the
classification route or adding another adaptive loss-weighting algorithm.

## Candidate correction and actual AdamW updates

Preserve frontend and selector bounds at `0.1/sqrt(3)`. Split the existing geometry
budget into shared attention and private geometry, each `0.1/sqrt(6)`. The squared
bounds still sum to `0.1^2`. This changes gradient clipping only; architecture,
task weights/calibration, augmentation, learning rates and AdamW remain fixed.

Each replay restores the exact original weights and a deep copy of the saved
optimizer state, then takes one actual AdamW step. Ratios below are median
candidate/current parameter-update L2 norms, not ratios inferred from clipping:

| Parameter block | Epoch 15 | Epoch 19 |
| --- | ---: | ---: |
| Pose layer 1 | 2.14x | 2.00x |
| Pose layer 2 | 2.38x | 2.09x |
| Joint heads | 2.02x | 1.97x |
| Image projections | 2.34x | 2.28x |
| Shared attention | 0.95x | 0.95x |

Every checked private pose-layer/joint-head/image-projection update increases;
full per-window ranges and per-task update alignment are retained in JSON.
Current AdamW updates already point downhill for the weighted pose objective in
both pose layers and joint heads in all eight windows. There is no evidence that
these parameters are entirely disconnected or never learning.

The production CUDA clipping path agrees with the diagnostic calculation to
relative L2 error 4.40e-8. Multiplying attention gradients by ten changes private
geometry clipped gradients under the current policy (maximum absolute difference
0.005826); under the candidate policy the difference is exactly zero. This
independence is the concrete protection the correction establishes.

Unclipped and classification-to-context removal controls are archived. The latter
also changes the shared clipping factor; it does not isolate gradient routing
from clipping and is not evidence for deploying a detached context branch.

## Local loss check, not detection AP

Using the epoch-19 snapshot and its saved optimizer, one update was evaluated on
the optimized eight TRAIN images and a separate eight-image TRAIN pair, with
identical RNG. Frontend updates are bitwise identical across policies.

| Weighted loss | Before | Current clipping | Separate attention budget |
| --- | ---: | ---: | ---: |
| Optimized pose | 0.662934 | 0.647041 | 0.641574 |
| Separate TRAIN pose | 0.107550 | 0.106201 | 0.105914 |
| Optimized box | 0.216715 | 0.192156 | 0.183305 |
| Separate TRAIN box | 0.035672 | 0.035369 | 0.035380 |

The separate-pair pose advantage is modest (0.27% lower than the current-policy
post-update loss). Separate-pair box loss is slightly worse than current clipping,
although both improve from baseline; EOS does not improve on that pair. This is
a teacher-forced local optimization check, not free-running AP, a new training
experiment, or a promise of long-term gains.

## Other checks and limits

- Both retained pose layers and supervised joint/box heads receive gradients.
  Six tensors in the first extent head are consistently unused: its outputs only
  update next-layer references, which are detached. This expected route differs
  from a disconnected supervised joint head; no pruning was performed.
- Classification gradients respect the protected raw-input/private-pose
  boundaries. Geometry gradients reach shared attention. No correction-scale
  explosion was found: raw proposal RMS 0.631–0.637, correction RMS 0.325–0.370,
  corrected RMS 0.601–0.680, EOS residual RMS 1.031–1.059.
- Empty images, partially missing joints, and synthetic batch-4 non-square
  49-person/50-row capacity all have finite forward/backward results. Existing
  target filtering excludes annotations with no labeled joints; this convention
  was checked separately and left unchanged.
- Trained checkpointed/non-checkpointed gradients agree to maximum absolute
  difference 1.42e-7. Original weights restore exactly after offline updates.
- User reports prefix augmentation strongly benefits box AP and slightly benefits
  pose AP. Preserve V54's empirical TRAIN-only augmentation in any clipping fix.
  This audit retains augmentation throughout; it neither isolates its causal
  contribution nor establishes why box gains are larger.

V54 was complete and V55 remained active at completion (epoch index 59, about
74550 updates). No trainer was stopped or restarted. No finite audit guarantees
future gradients or competitive AP. The recommended next iteration is the narrow
clipping correction, with a reviewed strict policy/checkpoint contract; no
production correction was implemented during the diagnostic.

## Approved implementation

Following user approval, the separate budgets are implemented in
`training_v55.py` with strict checkpoint contract version2. An explicit-budget
helper in `balance_v51.py` validates disjoint ownership and the combined norm
bound. Legacy equal-budget callers keep their original defaults and arithmetic.
Existing V55 version1 checkpoints fail restore under version2, including when
only the gradient policy is incompatible. Archived trainers and artifacts are
unchanged; no checkpoint conversion, continuation or new training was launched.

The model, empirical prefix augmentation, frozen calibration, loss definitions
and AdamW configuration are unchanged. Shared `native`, `selector` and
`shared_context` measurement tags remain. `private_geometry` identifies the
new separate group; aggregate `geometry/preclip` and `geometry/postclip` still
measure the union of context and private geometry. No aggregate clip-factor tag
is emitted because that union now has two factors.

The failing-test-first iteration recorded three failures before implementation.
All32 focused and regression tests then passed, covering exact private-geometry
independence from context scaling, accumulation inputs, bounds, empty/absent
gradients, nonfinite rejection, duplicate ownership, checkpoint rejection,
checkpointed gradients and V53/V54 regression behavior. Native CUDA verification
is recorded separately under `tmp/v55_clipping/native_verification/`.

Native verification passed FP32, 50 rows/49 people, batch4/accumulation2 and
256x416 images. Actual combined clipped norm: 0.09999998. All267 gradient tensors
passed checkpointed/direct parity (maximum absolute difference1.67e-6). Restored
next-update loss was exactly17.28714370727539, maximum model-state difference
1.19e-7, with scheduler agreement and RNG restoration. Peak allocation2.09GiB;
one optimizer update0.6769s (11.82 training images/s), capacity inference4.09
images/s on the shared GPU. These timings are preflight measurements, not a
controlled speed comparison.

The initial model hash matches the original V55 exactly. Only the clipping helper,
V55 pipeline and V55 verifier changed among644 pinned production sources.
`tmp/v55_clipping/implementation_proof.json` records these checks. The existing
archived-source trainer was still progressing beyond77250steps and retains its
original policy. Applying the new policy to training needs a separately reviewed
continuation or fresh-run plan; this approved iteration implements and verifies
the correction without launching a new run.

## Authorized fresh training run

The user subsequently instructed launch. A fresh seed42/full80 version2 run
started on 2026-11-15 at11:23CET under
`/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v55_split_clipping/20261115_112330`,
tmux `poseflow-v55-split-clipping`, trainer PID4115804. It uses original ED-Pose
initialization rather than a trained version1 checkpoint. Initial weights,
frozen calibration scales, empirical augmentation bank and fixed32VAL IDs match
the original V55 run; only the verified clipping policy differs.

The archived workspace contains925 files, including all644 source/config/native
files verified before launch. Native preflight, initialization coverage, input
hashes, bank provenance, calibration and strict contract are archived with the
run. Runtime proof in `launch_verification.json` confirms FP32/batch4/accum2,
80epochs, TRAIN10000/VAL2000 with no overlap, no resume, finite optimizer progress
beyond180updates, and all four actual clipping bounds. Original V55 PID3519742
continues independently on its archived version1 source, which was rechecked
unchanged.

The run retains best/last checkpoint selection by the smaller fixed32VAL pose/box
AP and selected-checkpoint full2000VAL evaluation after training. TEST remains
unopened. Initial fixed32VAL AP is initialization-only evidence: pose0.00063649,
box0.02890581. Full80 quality remains pending.

TensorBoard uses run label `V55-split-clipping` and the existing measurement tags,
including `quick-test/pose_AP` and `quick-test/bbox_AP`. Dashboard33332 retains
all80 previous roots and5pins. Initial new AP, loss, augmentation and separate
clipping scalars were verified against source JSON/CSV. Historical reader catchup
and visible chart proof are tracked separately in the run's delivery artifacts.

Launch delivery passed: all18 prior comparison series and1100 recorded points
are preserved exactly in TensorBoard, with80 previous roots and5pins retained.
Both shared AP cards visibly include `V55-split-clipping` beside the historical
runs. New loss/augmentation/clipping scalars were checked against source CSV
through step1249 after reader catchup. The fresh run completed its first epoch
and advanced beyond1400updates; original V55 remains active. Dashboard cold-start
indexing was slow, and comparison roots were moved ahead of unrelated diagnostic
streams without changing any root paths, tags, pins or event files. Proofs:
`dashboard_history_parity.json`, `launch_verification.json` and delivery screenshot
in the fresh run namespace. Full80 training and selected fullVAL outcomes remain
pending; launch verification is not an AP benefit claim.
