# V56: full-depth sequential person births

User-authorized integration and launch, 2026-11-16. Restores the working native
six-encoder/two-box/four-pose transfer rather than sparse2/4/6 initialization.

The verified full-depth birth adapter supplies the native frontend and pose
decoder, including the shared expansion/first-pose head. V55 proposal-to-prefix
attention selects the next image-local proposal and supplies a zero-initialized
feature correction. Previous people remain fixed geometry; no prefix tokens
are inserted directly into pose layers. No diffusion or iterative refinement.

One necessary backward-route correction accompanies integration: the prefix
coordinate/reference encoder now trains through prefix attention. Its earlier
direct decoder route is absent, so retaining the old detach would leave this
encoder untrained. `ProposalContext(detach_prefix=True)` remains the legacy
default; V56 setsFalse. Image candidates and proposal identities remain detached
on classification paths. The raw selected image-feature gradient is retained
for geometry. Classification does not train image or pose-decoder parameters.

Teacher forcing, one selected-slot Hungarian match per forward, random person
order, empirical TRAIN prefix augmentation and masks remain. All four pose
readouts supervise newborn geometry, averaged across layers and people. Retain
the existing balanced pose/box objectives, fresh32TRAIN calibration and four
disjoint clipping budgets. Capacity49 plus EOS,50teacher rows. Inference stops
per image at first EOS>=0.5 or capacity; confidence is native objectness times
birth nonEOS. No stopping-policy change, refinement tail or inference noise.

## Verification

22focused/legacy tests pass after failing tests preceded implementation. They
cover initial fixed-ID native parity, future/padding exclusion, one context and
assignment, live prefix/context gradients, four pose heads, disjoint ownership,
checkpoint gradient parity, empty/missing-joint cases, capacity, deterministic
EOS/capacity sampling and strict asset-independent restore.

Native FP32 CUDA verification:

- All32 fixed VAL images/all50 supplied proposal IDs: **54.1509pose AP,
  39.1008box AP**, exact hidden/reference/coordinate parity against the frozen
  independent-person reference. This is conditional transfer preservation,
  not learned sequential/EOS AP; native grouped boxAP remains45.20.
- Full actual frontend/DN/geometry/selection/EOS objectives, non-square256x416,
  batch4, accumulation2,49people+EOS/50rows: **2.10GiB peak**,0.588s/update,
  13.60images/s; full-capacity inference3.80images/s. Short shared-host
  measurements, not controlled throughput benchmarks. No recipe reduction.
- 407gradient tensors pass checkpointed versus direct parity, maxabs7.75e-7.
  Restored next AdamW update loss34.530521392822266 agrees exactly;
  maxstate8.20e-8 and scheduler/RNG agree. A single synthetic warmup opens the
  zero correction before post-warmup gradient-route checks; launch restores
  every initial model weight.
- Initial model SHA256:
  `db2ec8b7f3b7de8b13c65a0f3b5f97952e37236d9d1a1ea7f211108011454cf8`.
  All652executable/config/native sources pinned and unchanged during verification.
- Fresh pose/box calibration scales0.3293851/0.2767948 from32augmented TRAIN
  batches, no weight updates. Frozen V54/V55 TRAIN residual bank retained.
- Original ED-Pose SHA256:
  `0f50dae126a73ad2a15c41906c9ba6d8efa3421ef6cc2fbdc878238bd26b8f93`.
  All retained frontend/pose tensors match the direct full-native asset transfer.

Evidence: repository-local `tmp/v56/native_verification/verification.json`,
conditional_parity/calibration/gradient_routes/preflight reports and
`tmp/v56/iteration1_final.log`. Initialization coverage proves tensor loading;
the independent conditional control separately proves function preservation.

## Training and delivery contract

Fresh seed42/full80, original TRAIN10k/VAL2k and resize/augmentation. FP32,
batch4/accumulation2, AdamW main2e-4/backbone1e-4, decay1e-4, LR drops31/61,
combined clipping0.1. No trained V55 warm start. Own strict V56 contract records
all native layers, head sharing, prefix gradient routes, objectives, calibration,
bank, initializer and sampler. Restore includes model/optimizer/scheduler/RNG
without external initialization or bank assets. Old checkpoint families rejected.

Evaluate actual image-only sequential predictions on the same fixed32VAL before
training and every epoch. Keep best by min(poseAP,bboxAP), last, then evaluate
selected checkpoint on all2000VAL after completion. TEST remains unopened.
Pretrained upstream annotation exposure is uncertified; preserve that caveat.

Use established TensorBoard measurement tags, including `quick-test/pose_AP`
and `quick-test/bbox_AP`; only run label `V56-full-depth-prefix` differs. Preserve
all previous runs/pins/events and verify scalar overlays/live values. Fresh
namespace, archived source/config/tests/native kernels/input hashes, exact
initialization and preflight are required before launch. Launch/live/dashboard
proof and run identity will be recorded below.

## Fresh launch

Started at `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v56/20261116_135212`,
PID1001527, isolated tmux`poseflow-v56`, from938archived files/652pinned executable
sources. Initial model matches the verified hash exactly; calibration, frozen
bank and fixed32 IDs verified. Model48,617,995parameters,48,395,595trainable.
Prior source/checkpoint artifacts remain unchanged. Finite real optimizer
progress is confirmed beyond200updates at roughly11images/s; live peak allocation
about2.81GiB. The complete80-epoch outcome is pending.

Actual free-running initial fixed32VAL: **51.3848pose AP / 38.0594box AP**,
poseAR55.7282. Counts are poor before learning (MAE45.78125, typically capacity
emission); this is separate from the supplied-all50 conditional control.
The exact initial model/bank/calibration is also retained as
`initial_reference.ckpt` for evaluation, alongside its hash and baseline metrics
in`initial_reference.json`; this file is not a trainer-resume checkpoint.

Dashboard adds only run label`V56-full-depth-prefix` to common AP/loss/gradient
tags, retaining81prior roots and5saved pins. Initial AP cards and live source
scalar parity are verified. All20previous comparison series/1294recorded AP
points match after history indexing. Screenshot`tmp/v56/dashboard.png` and
`dashboard_history_parity.json` verify common cards and preserved values.
The initial cold scan of the remaining historical streams can temporarily lag
new runtime points; source parity was checked through the indexed step29 while
the actual trainer had progressed beyond780updates. This is dashboard ingestion
lag, not a stalled trainer. Existing events/roots/pins are preserved.


2026-11-16 TEST correction: original run's quick-test/final endpoint was VAL, potentially included in ED-Pose pretraining. Retained those artifacts without relabelling them. User-requested TEST continuation is `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v56/20261116_135212/test_continuation_20261116_202836` in tmux `poseflow-v56-test`, restored from preserved epoch025/global_step32500 with optimizer/scheduler/RNG. Original last/best checkpoint SHA256s are in transition.json. A370-update uncheckpointed tail is replayed; an initial startup logging metadata attempt is archived separately and superseded. The model, optimizer, losses, augmentation and80epoch budget are unchanged. Separate checkpoint directory resets old VAL selection scores; new quick-test uses32 official TEST images, final selected model all8000 TEST. Native frozen checkpoint smoke: poseAP32.2895/boxAP36.2987 (0–100), not a full-TEST or initial-model result. Nine focused tests pass, including resume epoch/step logging. TensorBoard TEST continuation keeps common tags and old VAL history.

[2026-11-16T21:02:02.225158+01:00] V56 TEST continuation CANCELLED by user; stoppedPID1256288, last logged epoch27/step34360; saved checkpoints hash-verified unchanged, all logs/evaluation/TensorBoard artifacts retained. Root /home/gasparyanartur/outputs/pose-graph-diff/poseflow_v56/20261116_135212/test_continuation_20261116_202836. Not an80epoch completion; fullTEST endpoint not performed.
