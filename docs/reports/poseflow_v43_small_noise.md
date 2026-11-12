# V43 small-noise control

The approved experiment isolates conditional latent recovery on16 TRAIN and16 VAL
images. Both arms use the same epoch011/18000 parent (SHA256
`ca99d2eb200e1939773c73935b3efab7c617a43e2ce6d8d083cdc6ea184bb6c9`),
frozen image features/reference slots, fixed posterior-mean targets and noise draws.
The x0 arm retains its absolute head; the v arm intentionally starts a new zero
velocity head with the same denoiser body. This compares these complete recovery
recipes; it does not isolate head initialization from parameterization.

The frozen parent/codec/statistics have prior trainval exposure. VAL here is held
out from these new optimizer updates, not an independent detector benchmark.
Known clean source codes and pre-noise target assignments are supplied for this
conditional denoising test. It is not Gaussian-start detection or an AP result.

## Parameterization and contracts

For additive sigma, a=1/sqrt(1+sigma²), b=sigma*a, state=a*u0+b*epsilon, and
v_target=a*epsilon-b*u0. The v arm recovers u0_hat=a*state-b*v_hat. Both coefficients
and angular noise conditioning come from the same continuous sigma. Sigma0 yields
exact identity; positive-noise performance must beat direct noisy decoding to count
as learned recovery. Zero is evaluated but not used as an unobservable random-v
training target. Every pose/box still comes exclusively from the frozen VAE decoder.

`PoseFlowV43Model.denoise_at_sigma(y, sigma, encoded_image, references=..., image_wh=...)`
accepts additive noisy standardized latents y[B,N,32] and sigma[B]. Existing integer
timestep/default-x0 behavior and state keys are preserved. Prediction type is
recorded and checked during restoration. Original checkpoints without that metadata
are explicitly x0; their absolute head cannot silently be loaded as velocity.

`geometry_loss(..., center_min_side_px=1)` floors only the width/height normalizing
center errors. Its default0 preserves all earlier AE/VAE/V43 losses. Annotation
geometry, mask handling, native OKS denominator, size/GIoU and metrics are unchanged.
The first cached trial is latent-only; the optional floor is implemented/tested for
a later geometry-loss stage and is not claimed to have trained this trial.

## Recipe and gate

Each arm gets at most2000 AdamW updates, lr2e-4, wd1e-4, clip1, batch4, seed42.
Noise1/2.5/5% is equally sampled per image. Training RNG is separate from fixed
paired evaluation seeds42/43/44; zero and10% are additional evaluation levels.
No random crop/flip or image-front-end optimization occurs in this cached test.
Both optimizers update the latent input, time conditioning, reference-box embedding,
attention/FFN blocks and output head. Source labels/masks and slot identity are fixed.

Only best TRAIN mean positive latent error and last checkpoints are retained.
Model, frozen codec/statistics, optimizer, cache identity, mode and private/global
RNG are embedded. Evaluation restores without original initialization assets.
A SHA-bound cache supplies the exact fixed geometry and image context.

Tiny-set success requires finite gradients and immutable frozen tensors; zero-noise
pixel difference<=1e-4; positive-noise training MSE<=1e-5 at every level and paired
OKS/IoU no worse than clean-code decoding by more than.005. VAL results are reported
separately, and best selection never uses VAL. Passing an identity gate alone is
not success. A10-epoch extension or wider noise curriculum requires separate review.

## Verification and artifacts

Focused TDD checks cover velocity inversion/zero endpoint gradients, continuous
metadata and decoder gradients, mean-source identity/RNG, thin-box/default parity,
mode restore rejection, padded cache packing and next-update replay. A real FP32
CUDA smoke uses2updates/arm on2/2images; v checkpoint re-evaluation matches exactly.
The restored native next update differs by at most2.8645e-8 in parameters, with
identical draws and frozen tensors. Original epoch011 K1/K4/K8 outputs replay
bitwise on2images/6trajectories. Regression counts and final run results are recorded
below when available. Local evidence lives under tmp/v43/small_noise; run snapshots
include source/config/input/cache hashes and fixed split identities.

The earlier frozen diagnosis and approved staged proposal are preserved under the
original run's diagnostics/small_noise_plan. These motivate the trial; they are not
results from the new optimizer run.

## Completed bounded trial

Run: `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v43_small_noise/20261105_152101`. Both arms completed2000updates/4000total,
using16TRAIN images/54people and16VAL images/46people. Each noise level uses three
fixed paired draws. All611 frozen source files, two annotation files, parent and
cache hashes verify;21logged draw traces and final private RNG states match between
arms. Frozen image/codec tensors are bitwise unchanged. The process exited0.
67 regression checks pass. Dashboard V43-small-noise-control/x0 and /v retains all
52previous streams and sixAP pins, giving54streams/49roots.

The table is the FINAL matched2000-update checkpoint on the16VAL cache, not AP:

| Noise | Raw noisy mean OKS | Absolute x0 OKS | Velocity OKS | Raw/velocity latent MSE |
|---|---:|---:|---:|---:|
| 1% | 0.97888 | 0.61048 | 0.97891 | 0.00009906/0.00009981 |
| 2.5% | 0.96511 | 0.60850 | 0.96529 | 0.00061911/0.00062128 |
| 5% | 0.92045 | 0.60366 | 0.92191 | 0.00247645/0.00246888 |

Velocity gives exact zero-noise pixel parity and avoids the absolute head's large
validation reconstruction error. At5% noise the validation OKS gain is only about
.0015, and latent MSE reduction is about.3%; at1/2.5% MSE is slightly worse than
raw noise. This is input preservation with negligible learned foreground recovery,
not a successful denoising result. Per-seed VAL metrics are retained in
`post_control_audit.json`; no single-seed selection was used.

BOTH tiny-set gates FAIL. Final TRAIN positive MSE at1/2.5/5% is
.0003725/.0004690/.0008450 for x0 and
.0001032/.0006412/.0025322 for velocity, exceeding the<=1e-5 gate at every level.
The v clean-identity gate passes; x0 fails it. The x0 arm reduces TRAIN error and
improves TRAIN geometry but overfits; VAL OKS stays near.61.

Best TRAIN checkpoints are x0 step2000 and v step1500. Both also fail the gate.
Final2000 comparisons above avoid mixing training ages. Four checkpoint files
(best/last per arm) are about204MiB each, about816MiB total.

## Checks after the failed learning gate

Re-encoding every cached positive target reproduces its stored mean code exactly
(maxdifference0). Foreground and background gradients reach the velocity head,
with finite nonzero norms at1/2.5/5%. On a four-image frozen probe, foreground
velocity loss stays1.05–1.07 while background loss is.028–.044. Foreground head
gradient norms(.87–.90) exceed background(.38–.52) on this probe. Thus a detached
foreground route, mismatched target code, or demonstrated foreground gradient
starvation is not supported. This does not establish the unique optimization
cause; image-conditioned noise prediction remains underfit in the bounded budget.

The10-epoch stage was NOT launched. The next reviewable diagnostic is a single-image,
foreground-only5%-noise overfit control, followed by a fixed-noise versus fresh-noise
comparison if needed. This removes competing targets and varying noise levels to
separate basic learnability from noise-target variance; it does not assume that
background gradients caused the failure. No extra optimizer updates or architecture
changes were made after the4000-update comparison.

Evidence: `verification.json`, `results.json`, `manifest.json`, `cache.pt`, per-step
evaluation JSON, best/last checkpoints, `dashboard_verification.json`, and
`post_control_audit.json` in the run namespace. The original full V43 run and both
other active trainers remain untouched.

## Approved single-image foreground diagnostic — completed

Run: `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v43_foreground_probe/20261105_160859`. User approved the proposed learnability probe with “Go ahead”. Both fresh- and fixed-noise arms completed 2,000 updates each, exit0. No production code, architecture, codec, original trainer or 10-epoch stage changed.

First existing TRAIN cache image 101403, eight people, all 50 queries retained in self-attention/forward. Only foreground velocity outputs contribute to regression. The same epoch 011 parent body/frozen frontend/codec and zero-initialized v head are used as in the prior control. Noise magnitude is fixed at 5%; fresh arm draws new independent Gaussian noise for each of four repeated image copies per update. Fixed arm repeats the first four draws. AdamW at 2e-4, weight decay 1e-4, gradient clipping 1, seed 42, private CPU noise seed 10042; no geometry loss or background regression.

“Unseen” below means three paired noise draws (seeds 42/43/44) on this SAME training image, not unseen people/images. This is one training seed. Frozen components retain disclosed prior trainval exposure. These are paired reconstruction metrics, never detector AP.

| Final 2,000-update result on unseen 5% noise | Latent MSE | Paired OKS | Paired box IoU |
|---|---:|---:|---:|
| Clean codec | 0 | 0.98987 | 0.98255 |
| Raw noisy | 0.00247242 | 0.88975 | 0.93233 |
| Fresh-noise training | 0.00024923 | 0.97643 | 0.95283 |
| Fixed-noise training | 0.00345076 | 0.88103 | 0.92716 |

Fresh-noise training reduces latent MSE by 89.92% on the three unseen draws. Their individual MSE values are 0.00020279 / 0.00028398 / 0.00026090. The 95th-percentile labeled-joint error improves from 12.048 px to 4.141 px; clean codec is 4.113 px. Thus the current path can learn substantial actual noise removal in this restricted setup. This does not identify which change from the 16-image control caused the gain: image count, foreground-only loss and fixed noise magnitude changed together.

The strict gate still FAILS for fresh noise: latent MSE 0.00024923 exceeds 1e-5, paired OKS 0.97643 is below codec 0.98987 − 0.005, and paired IoU 0.95283 is below codec 0.98255 − 0.005. Exactzero parity remains 0 px. Training only at 5% does not establish robust recovery across magnitudes: 1% geometry is essentially raw-noisy quality and slightly worse for boxes.

| Noise | Raw OKS | Fresh-arm OKS | Raw box IoU | Fresh-arm box IoU | Fresh-arm MSE |
|---|---:|---:|---:|---:|---:|
| 0% | 0.98987 | 0.98987 | 0.98255 | 0.98255 | 0.00000000 |
| 1% | 0.98474 | 0.98470 | 0.97752 | 0.97717 | 0.00008073 |
| 2.5% | 0.96144 | 0.97248 | 0.96248 | 0.96567 | 0.00027557 |
| 5% | 0.88975 | 0.97643 | 0.93233 | 0.95283 | 0.00024923 |
| 10% | 0.69748 | 0.90599 | 0.87202 | 0.91150 | 0.00203043 |

The fixed-noise arm can memorize its four training perturbations to latent MSE 1.13e-14 and codec-level geometry at the final checkpoint; its fixed-draw gate passes. It worsens unseen-noise MSE/geometry, so fixed-draw memorization is insufficient. There is a transient training-draw error spike around step 1500, visible in the saved curve; final and best snapshots are separately preserved. Best selection is step 2000 for fresh (same-image unseen-draw MSE), step 1250 for fixed (training-draw MSE). The table uses matched FINAL 2,000 snapshots.

All 406 snapshot files, probe source, parent/cache and annotations verify. Both best checkpoints reproduce saved unseen-draw metrics exactly; last-checkpoint model/optimizer/private RNG restoration and frozen frontend/codec states pass. Three focused red-to-green tests cover loss masking, fresh/fixed RNG and the gate. The two-update CUDA smoke for each arm pass. Original three trainer PIDs remain alive.

TensorBoard now has 56 scalar streams / 50 roots, preserving all 54 prior streams and six AP pins. New streams are `V43-foreground-probe/fresh` and `/fixed`; both show loss through step 2000. Conditional metrics use their own labels. Native direct-address HTTP verified.

![Single-image recovery curves](/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v43_foreground_probe/20261105_160859/learning_curves.png)

Artifacts include `recipe_audit.json` (actual diagnostic recipe; inherited checkpoint `settings.control` serves evaluator compatibility), `manifest.json`, `launch_manifest.json`, `verification.json`, `dashboard_verification.json`, per-step JSON, `results.json`, four best/last checkpoints, frozen source, and `learning_curves.png`. A first import failed before any initialization/updates because the snapshot lacked its project-root pyproject marker; the marker was added and launch corrected, with evidence retained.

Next proposed controlled step, requiring review: repeat the foreground-only, fixed 5%-magnitude fresh-noise setup with the existing 16 TRAIN / 16 VAL cache, same parent initialization and 2,000-update budget. This changes image count alone relative to the successful partial-learning probe, measures unseen-image recovery under the disclosed frozen-component exposure, and tests whether the gain survives a larger target set. It does not isolate all differences against the original mixed-noise/balanced-loss control. No new training or architecture change has been launched.

## Follow-up: transfer and conditioning thesis

The automatic 16-image foreground-only scaling proposal above is superseded by
[the thesis-driven analysis](poseflow_v43_thesis.md). Frozen probes show that the
one-image model uses image conditioning but damages other images: VAL paired OKS
0.92045 raw -> 0.72323 denoised. The gain is image-specific; it does not establish
a transferable conditional pose prior. No further training was launched.
