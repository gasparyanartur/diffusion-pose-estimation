# V55: proposal-to-prefix attention before joint expansion

V55 replaces V54's mean-pooled next-person selector with candidate-specific
cross-attention over individual previous people. It also uses the selected
contextualized feature to initialize native person/joint queries. Previously
emitted people remain fixed. There is no diffusion or pose refinement.

## Model and selection

One pre-normalized, width256/eight-head attention block processes all50 candidate
queries and a learned EOS query. Keys/values are previous-person features made
from geometry, detached proposal identity and box reference. Teacher rows batch
independently. A learned null key/value is valid only when the prefix is empty;
PREDICT/padding/future people are never valid keys. Prefix order has no learned
position embedding, so selection is permutation-invariant for the same set.

Each query receives an additive attention correction. Its output projection is
zero-initialized, so initial native expansion is exactly preserved conditional
on selecting the same proposal. The learned EOS query starts at unit scale, matching proposal feature residuals
before scoring LayerNorm. Private candidate/EOS MLP scores are fresh;
free-running selection is not initially equivalent to V54. Candidate attention
weights index previous people, not selection probabilities. Used proposals are
masked before candidate selection; the marginal EOS logit subtracts the
log-sum-exp of available candidate scores from the EOS score.

The selected corrected feature enters native expansion without another output
normalization. Original proposal boxes and sample-local identities are retained.
The model keeps pretrained native encoder2/4/6, box layer2 and pose layers4/6,
with one person plus14joint queries at each birth. Image memory/projections are
cached once per image; no image encoding is repeated per teacher row.

## Gradients and training

Raw candidate features and prefix features are detached at the new attention
block's inputs. Its output remains differentiable: pose, box, candidate selection
and EOS objectives all train this shared block. Classification gradients do not
enter the prefix coordinate encoder or native frontend through this route.
Native frontend/DN objectives and the downstream pose image/prefix routes retain
their existing ownership. Zero output initialization initially suppresses Q/K/V
projection gradients, but the output projection learns immediately and opens
that route; this is covered by an update test.

The shared block belongs to the geometry clipping family. Private score heads
belong to selector; native frontend retains its own family. All three disjoint
families retain bound0.1/sqrt(3). Shared-context pre/post-clipping norms are logged
as `train/gradients/shared_context/preclip` and `.../postclip`; geometry family
norms include both geometry and classification contributions to this block.
Pose/box calibration retains32 augmented TRAIN batches and the median norm on
shared native pose layers. It is recomputed for V55, then frozen. No additional
adaptive gradient balancing is introduced.

V54's frozen1757-person TRAIN residual bank, causal prefix augmentation bounds,
clean probability0.25 and unchanged identity/mask policy are retained. Teacher
rows still use one Hungarian selected-slot assignment and random GT order per
forward; the newborn proposal is teacher-forced. Inference hard-selects a
proposal, emits its second-layer geometry once, and stops at first EOS>=0.5 or
49people. Confidence is native objectness times birth non-EOS probability.

## Verification and run contract

The approved run is fresh seed42,80epochs, original ED-Pose initialization,
FP32, physicalbatch4/accum2, resize256/max427, existing TRAIN10k/VAL2k split,
AdamW2e-4/backbone1e-4, weight decay1e-4 and LR drops31/61. Quick-test uses the
same fixed32VAL images before training and after every epoch. Best/last are
retained; best uses min(pose_AP,bbox_AP), then full2000VAL evaluation. TEST stays
unopened. V55 strict checkpoints embed architecture, bank, calibration and RNG
contracts and restore without initialization assets.

Primitive/model/training tests and native full-capacity FP32 CUDA preflight are
required before launch. Same-forward per-objective gradient norms/cosines are
saved separately from AP results. Shared tags `quick-test/pose_AP` and
`quick-test/bbox_AP` permit direct overlays; only the run label differs. Existing
jobs and event/checkpoint history are preserved. Resource/launch measurements
and final run location will be recorded after verification.

This combined change tests contextualized selection plus birth initialization.
Any AP gain cannot isolate those two contributions. Ranking noise and separate
proposal/pose image branches remain deferred. Learned outcome is pending.

## Prelaunch scale audit

The first native gradient audit exposed an initialization issue: a0.02-scale EOS
query received the same attention residual used for unit-scale proposals, then
passed through scoring LayerNorm. EOS dominated the shared output-projection
gradient. At fixed weights/batch/RNG on a one-update checkpoint, scaling only
the EOS base query by50 reduced its shared-context gradient norm24.05 to1.944;
pose/box/selection losses were bitwise unchanged. EOS loss itself changed, so
this is a gradient-route diagnostic, not objective or performance equivalence.

A failing regression checks unit-scale initialization and the reduction of EOS
correction-gradient amplification. Fresh V55 uses a unit-scale EOS query,
recorded in the strict contract; objective weights and clipping are unchanged.
Initial measurements in tmp/v55/verification_native are superseded by repeated
verification in tmp/v55/verification_final. No trainer used the first preflight.

## Verified launch

Authoritative preflight: `tmp/v55/verification_final/verification.json`. All39
focused and legacy regression tests pass. The extracted V53 default selector
also matches its frozen archived implementation exactly on all four outputs
and99 gradient tensors in the mixed-row CPU fixture. Native frontend399 and
pose-decoder94 retained tensors match original ED-Pose exactly.

Full-capacity FP32 CUDA verification (50rows/49people/batch4/accum2,256x416)
measured2,244,299,776bytes peak allocation (2.09GiB),0.6711seconds per accumulated
update and3.760images/s full-capacity inference alongside V54. Checkpointed vs
direct267 gradient tensors differ by at most1.85e-6. Resumed next-update loss
17.28717041015625 matches exactly; maximum state difference5.96e-8. These are
synthetic capacity and restore checks, not AP results or isolated speed claims.

The corrected post-two-update TRAIN diagnostic measures shared-context norms
0.3943 for normalized pose,0.3349 for normalized box,0.5185 for selection and
1.8456 for EOS. They remain unequal; no claim of equal gradient magnitude or
causal AP improvement follows. The combined geometry-family clip factor is
0.02509 on that batch. Per-objective norms/cosines are saved in
`verification/gradient_routes.json`.

Fresh seed42/full80 run:
`/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v55/20261114_184631`.
Trainer PID3519742, tmux `poseflow-v55`. Parameters39,603,451 total,39,381,051
trainable. Initial model SHA256:
`d4319444b4bd15731c243c4e206a1ec0d169c84b2ec57cf4190a33fed7897e64`.
All644 executable/config/native source hashes match both checkout and immutable
workspace (923 archived files). Inputs, bank/source checkpoint, source hashes,
initialization coverage, calibration and native preflight are archived with the
run. Optimizer progress confirmed beyond130updates with finite losses, roughly
10.3images/s alongside V54; learned epoch AP remains pending. V53 completed
normally and V54 was preserved. Dashboard delivery proof is being finalized.

## Dashboard and delivery

V55-proposal-prefix-attention uses the existing shared quick-test pose/bbox
charts. All16 prior comparison series/870points are preserved exactly, together
with79previous configured roots and5stored pins. The existing browser tab now
includes V55 in its run filter; both AP cards visibly include the new run.
Screenshot and accessibility proof: `tmp/v55/dashboard.png` and
`tmp/v55/dashboard_ax.txt`. Initial AP points are pre-training measurements.

After the initial full-history indexing pass, native TensorBoard API values for
loss, empirical corruption and shared-context pre/post gradients through step1019
match the trainer CSV exactly. Proof is archived as
`dashboard_history_parity.json` and `dashboard_live_refresh.json` at the run root.
Finite optimizer progress is confirmed beyond1160updates; V54 remains active.
Full80training and learned detection-quality outcome are pending. The `.plan`
entry is retained for outcome review. No further architecture change is implied.
