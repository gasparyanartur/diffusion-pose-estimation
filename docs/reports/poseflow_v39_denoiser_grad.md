# V39-denoiser-grad

Status: implemented, verified and running; first-epoch AP pending.

## Change

The V39 clean encoder now receives gradients through the corruption input:
`h0=E(box,I); ht=a_t*h0+sigma_t*noise; recovered=D(ht,t,I)`.

`detach_corruption_source=false` is the only behavior change from V39. The
latent objective remains positive-only MSE against `stopgrad(h0)`; there are no
target-side gradients. The total remains0.5clean_task+0.5recovered_task+10latentMSE,
with focal2/L1 5/GIoU2. One real slot/person, source pairing, fake classification
only,50proposals/256channels, image initialization, optimizer/data, FP32,
batch4/accum2, clipping0.1 and K1/K4 DDIM are unchanged. Seed42/80epochs starts
fresh alongside V39, not from its trained weights.

## Verification

Four intended-red tests exposed the missing configuration/keyword and absent
encoder gradients; two existing-property checks already passed. After the
implementation,40 focused checks pass across V26/V35/V39 and the new variant.
They verify exact initial state/global RNG/forward losses/Gaussian inference,
independent recovered latent and decoded gradient routes, analytical corruption
Jacobian at t0/t999, detached latent targets/fake masks, empty cases and strict
restoration with identical next draws.

Native FP32 CUDA verified two batch4/accum2 finite optimizer updates, two-image
K1/K4 evaluation with100detections each, exact original V39 initial model hash
and282/282 identical EDPose image tensors, strict model/pipeline/optimizer
restore and next draws. State schema and parameter count41,586,629 are unchanged.
Peak allocated1.294GiB/reserved1.375GiB in the smoke.

A frozen native forward isolates raw clean-encoder gradient norms: detached
source gives latent0/decoded0; live source gives latent1.5323/decoded24.0956.
Forward losses are exactly equal. These are unweighted preclip diagnostic
norms, not claims about optimizer impact or learned performance.

## Monitoring

Retain per-epoch fixed32 Gaussian K1/K4 bbox AP, coverage, excess IoU hits,
noise-bin real/fake scores and source recovery; final epoch79 full8000 K1/K4 is
separate. A first-validation-batch feature-shuffle diagnostic additionally logs
clean/recovered real/fake variance within each image and across the batch,
number of eligible groups and box/existence head weight norms. Within-image
spread requires at least two slots of the relevant class. Validation overlaps
trainval; feature/representation probes are conditional diagnostics, not AP.

Decoded losses constrain collapse but do not guarantee useful separation;
head weights can compensate for contracted person differences. The source
gradient is attenuated by a_t at high noise. Compare matched-age recovery and
Gaussian AP alongside these representation measures; do not equate decreasing
latent loss with detector improvement. No extra regularizer is introduced.

## Run provenance

Fresh seed42/80epoch run: `/storage/outputs/pose-graph-diff/results/poseflow_v39_denoiser_grad/20261101_212530`.

Source/config/input hashes, exact initial model hash and gradient-boundary
settings are frozen in `manifest.json`. `preflight.json` and
`gradient_routes.json` preserve native verification. `status.json`/`launch.log`
record progress; tmux session `poseflow-v39-denoiser-grad` includes a `progress`
window. Shared [TensorBoard](http://lynx.tail481642.ts.net:33330/) label
`V39-denoiser-grad` preserves all previous runs and AP pins.

The V39 baseline continues at
`/storage/outputs/pose-graph-diff/results/poseflow_v39/20261101_193746`.
A completed epoch12 baseline checkpoint and config were copied into
`baseline_reference/` with a checksum, preserving a control for later
matched-age representation diagnostics. This checkpoint was not used for
variant initialization. Fixed32 results remain available by epoch; automatic
final8000 K1/K4 and same-age comparison are configured. No performance claim
from startup or finite updates.

Live verification passed at update260: source/config hashes verified,44
new TensorBoard scalar tags,15 configured runs with prior history/AP pins
preserved. Baseline V39 continues without restart. Evidence is saved in
`live_verification.json` in the run root. First-epoch AP pending.
