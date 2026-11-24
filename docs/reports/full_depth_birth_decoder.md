# Full-depth birth decoder: preservation iteration

Implemented after approval on 2026-11-16. This is an isolated decoder adapter
and verification tool, not a new trainer or a learned sequential detector.

## Result

The adapter reproduces the frozen independent-person pretrained reference:

| Fixed32 VAL measurement | Reference | Adapter |
| --- | ---: | ---: |
| Pose AP | 54.1509 | 54.1509 |
| Pose AP50 | 70.3645 | 70.3645 |
| Pose AP75 | 61.7620 | 61.7620 |
| Pose AR | 58.7379 | 58.7379 |
| Box AP | 39.1008 | 39.1008 |

Every pose-layer input hidden state, input reference and output hidden state,
and every coordinate readout, matches exactly in native CUDA FP32. Image memory,
proposal features, detached boxes, logits and original query IDs match exactly.
All50 proposals are supplied explicitly, with native proposal objectness,
resize256/max427 and the same native postprocessing. No GT sampling inputs,
learned sequence selection, EOS or optimization are involved in these AP results.

The grouped native reference scores54.1956pose/45.2040box. Independent groups
preserve pose closely but lose about6.10 box AP relative to grouped decoding.
This iteration preserves the independent reference, not the complete grouped
detector. Missing interaction between person queries is a remaining difference.
Pretrained upstream annotation exposure is uncertified; these VAL results are
transfer-preservation diagnostics, not a clean held-out benchmark claim.

## Implementation and interface

`src/models/pipelines/pose_estimators/poseflow/full_depth_birth.py` provides
`FullDepthBirthDecoder(native=...)`. It takes ownership of a full native model:
six image encoder blocks, two box decoder blocks and four consecutive pose
blocks3/4/5/6. Sparse-depth inputs are rejected. Construct the native model with
the complete checkpoint loaded directly; the sparse-layer remapper is unused.

The adapter reuses `ProposalFrontend`, `NativeHumanContext`,
`ExpandedPoseDecoder` and `ProposalContext` without editing their legacy code.
The original model's expansion head and first pose head share parameters; the
new adapter preserves that tie when the native model has no separate expansion
head. This matters for training gradients, not just initial numerical parity.

- `prepare_context(images, padding_mask=None)` computes the full image/box
  frontend and four image value projections once.
- `decode_proposals(context, proposal_indices, valid_mask=None, prefix=None,
  prefix_valid=None, collect_trace=False)` decodes supplied image-local IDs
  `[B,R]` in independent rows. Repeated IDs are legal. Explicit `valid_mask`
  supports mixed counts and padded IDs; invalid active IDs are rejected.
- `prefix` contains encoded features `[B,R,Q,256]` at native width256, with a
  boolean `prefix_valid` mask. It is not clean coordinate targets. Omitted
  prefix bypasses correction; an explicitly empty prefix uses the learned null
  key. No direct prefix tokens are inserted into the pretrained pose layers.
- Correction inputs are detached, as in the proposal-context block; its
  differentiable residual is added to the original selected feature. The direct
  native feature-gradient route remains intact. The output projection starts
  at zero, so any valid prefix gives identical initial pose predictions.
- Output includes final normalized `pred_coordinates [B,R,32]`, native-format
  `pred_keypoints`, `pred_boxes`, person confidence/logits, masks/counts,
  proposal IDs and original query IDs. Background-class logits are suppressed
  consistently with existing birth output formatting. Invalid rows have zero
  coordinates/confidence and IDs-1. Optional traces contain four layer readouts;
  normal calls contain no trace.

There is no selector, EOS head, sampling loop, loss calibration, optimizer or
trainer checkpoint contract in this adapter. Existing model/training families
and metric tags are unchanged. Masks for missing target joints remain the
responsibility of downstream objectives; this adapter consumes no GT labels.

## Verification

Failing-test-first evidence is retained in
`tmp/full_depth_birth/failing_tests.log` (missing adapter) and
`failing_confidence.log` (birth confidence formatting). Final focused and legacy
checks: **25 passed**, including six new tests plus V44/V53/V55 regressions.
They cover full frontend parity and one encoding, every pose readout, native
head sharing, zero-correction invariance, finite correction/native gradients,
opened correction influence, padded NaN exclusion, independent row output and
gradient parity, repeated IDs, empty outputs, mixed counts, non-square image
padding, invalid input rejection, and strict state-dict restore/rejection.

`scripts/training/verify_full_depth_birth.py` requires native CUDA, checks the
original asset SHA256, compares against both a live frozen reference and the
archived fixed32 audit metrics, and rejects any mismatch. It pins11 source
hashes before execution and verifies them unchanged afterward. All model state
values also remain unchanged through the gradient/resource diagnostic.

Authoritative result: `tmp/full_depth_birth/native_final/verification.json`.
Pretrained SHA256:
`0f50dae126a73ad2a15c41906c9ba6d8efa3421ef6cc2fbdc878238bd26b8f93`.
Initial/final adapter state SHA256:
`bd3420378650391b7d3af42c97db3787ee502c1b0f98425d988116c5fc27df1b`.
No TEST evaluation or optimizer updates were performed.

## Resource measurements

RTX3090, FP32, synthetic non-square256x416, batch4, all50 proposals,
accumulation2, all four layer outputs and growing prefix masks. The synthetic
loss is masked coordinate MSE for gradient verification; it is not a trainer
preflight for the complete native/DN/selector/EOS training objectives.

- Peak allocated memory: **2,275,435,520 bytes / 2.12 GiB**.
- Two forward/backward microbatches: **0.360s**, 22.24 images/s; no optimizer step.
- Cached image frontend for four images: **0.0367s**.
- Batched50-person decoding for four images: **0.0155s**.
- Sequential50 supplied-proposal calls for four images: **0.842s**, about
  **4.55 images/s** including the cached frontend.
- Sequential and batched coordinates agree within2e-5. All four deformable
  sampling-offset paths, the tied expansion head and the zero-initialized
  correction projection have finite nonzero gradients.

These are short shared-host measurements, not controlled throughput benchmarks.
Actual EOS counts and the complete training recipe will change computational
cost. No batch, precision, layer count or gradient policy was reduced to fit.

## Remaining reviewed work

The approved preservation iteration is complete. A subsequent integration must
specify the new family/config and strict checkpoint contract, learned proposal
selection/EOS, unchanged prefix augmentation and training objectives, calibration
for the restored architecture, full trainer preflight and a concrete launch
namespace. No training launch is included here. Preserving initial conditional
geometry does not guarantee final sequential AP or prevent training drift.

See [the preceding audit](poseflow_v55_pose_bottleneck_audit.md) for the causal
controls motivating full-depth transfer and the independent/grouped distinction.
