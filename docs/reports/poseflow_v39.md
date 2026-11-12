# V39: one-to-one source reconstruction

Status: implemented, verified and running; first-epoch AP pending.

## Training contract

Start from `benchmark_poseflow_v35` (V35-OTA), retaining its box-only nonspatial
model, image initialization,50 proposals/256 channels, three clean and three
denoiser blocks, cosine1000 corruption, deterministic DDIM, FP32, batch4 with
accumulation2, clipping0.1 and seed42/80epoch recipe. No V37 alignment or V38
posterior components are used.

Exactly one slot represents each valid real person. Spare slots use foreign
boxes (different image IDs), otherwise random valid image-relative boxes. All
slots/metadata are randomly permuted together. Clean and recovered tasks retain
original source targets; neither OTA nor Hungarian executes. Class labels do
not enter model inputs.

`L = 0.5 Lclean + 0.5 Lrecovered + 10 Llatent_positive`.

Task losses retain focal classification2, normalized box L1 5 and GIoU2.
Classification covers all slots; box geometry covers real slots only. Latent
MSE averages over original positive slots and256 channels and compares to
detached clean same-slot codes. No direct fake latent/geometry reconstruction
exists. All-fake batches retain classification and connected zero reconstruction.
Cross-slot attention can propagate indirect gradients through fake states.

## Verification

17 focused checks passed (six fixed-source and eleven V35 regressions). They
cover unique-person targets/provenance, permutation consistency, no matcher
execution, empty/full/invalid/no-donor cases, permanent pairing, fake masks,
clean-target detach, native RNG continuation, noise-bin count aggregation and
finite K1/K4 inference. Initial test collection had a fixture import-path error
which was corrected; it was not an intended-red objective test.

Native CUDA verified two FP32 batch4/accum2 optimizer updates, finite parameters
and gradients, exact initial model/global RNG equality with V35-OTA,282/282
EDPose image tensors imported identically, strict model/pipeline/optimizer
restore and next proposal/timestep/noise draws. Model has41,586,629 parameters;
peak allocated1.101GiB/reserved1.344GiB. Two-image K1/K4 smoke produced100boxes
per condition. These checks establish wiring, not learned performance.

## Monitoring and limitations

Per-epoch fixed32 Gaussian K1/K4 bbox AP and coverage use the existing noNMS
evaluator with all50 predictions. Excess IoU50 hits count extra predictions
choosing the same nearest GT bbox, reported raw and at score0.5; this is a
diagnostic definition, not AP matching. Final epoch79 full8000 K1/K4 is separate.

Training logs real/fake scores, recall/false acceptance at0.5, positive latent
MSE and bbox error by noise bin with actual class counts. Each epoch also saves
a first-validation-batch diagnostic holding source latents/noise fixed and
shuffling image features only in the denoiser, at t0/500/999. Both class BCE
and positive geometry are reported; this is GT-conditioned evidence, not
Gaussian detection. Focal scores are not claimed calibrated probabilities.

Foreign/random negatives can overlap real people; overlap frequency is logged.
At high noise, source identity/class can be ambiguous. One-to-one supervision
does not guarantee unique generated detections. Fake states still participate
in DDIM despite having no latent target, so this partial objective is not an
established full-scene score-matching likelihood. Compare same-age Gaussian
AP/coverage/duplicates and original-source recovery; lower total loss alone
cannot establish success.

## Cancellation and launch

V37-reweight `/storage/outputs/pose-graph-diff/results/poseflow_v37_reweight/20261031_232257`
and V38 `/storage/outputs/pose-graph-diff/results/poseflow_v38_vae/20261101_001343`
were stopped via SIGINT at the user's request. Their status now records
`cancelled_by_user`; `cancellation.json` preserves the previous status and
checkpoint lists. Original V37 had already completed. Historical runs remain
on the shared port33330 dashboard. Fresh seed42/80epoch run: `/storage/outputs/pose-graph-diff/results/poseflow_v39/20261101_193746`.
Frozen source/config/input hashes and exact initialization are in `manifest.json`;
CUDA/restore evidence is in `preflight.json`. Training runs in tmux session
`poseflow-v39`, with a separate `progress` window. `status.json` and
`launch.log` provide persistent progress. The shared dashboard is
[TensorBoard](http://lynx.tail481642.ts.net:33330/), label `V39`; all13
previous entries and AP pins are retained. Final8000 K1/K4 runs automatically
after epoch79, with same-age fixed32 comparisons to the saved V35-OTA baseline.
First epoch AP is pending; early finite losses are verification only.

Live verification:500 finite updates,511 frozen source hashes and5 input hashes
verified. TensorBoard serves44 new scalar tags with14 configured runs, prior
history/AP pins preserved. Cancellation checks confirm both earlier jobs exited
and all recorded checkpoints remain. Evidence: `live_verification.json` in the
run root. First-epoch AP remains pending.

The canonical experiment name is V39. The V39 run path is an alias to the
original frozen launch directory; the active trainer and original provenance
remain intact. `experiment_identity.json` records this naming correction.
Legacy config/class names remain compatible with the launch checkpoint files.
