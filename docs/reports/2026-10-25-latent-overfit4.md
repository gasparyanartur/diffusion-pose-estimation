# Four-image overfit capability checks: V28 and V29

Launched2026-10-25. These are bounded memorization diagnostics, evaluating the
same four training images repeatedly. They are not held-out or full-dataset AP.

Root: `/storage/outputs/pose-graph-diff/diagnostics/latent_overfit4/20261025_224813`. Frozen source/harness: `/storage/outputs/pose-graph-diff/diagnostics/latent_overfit4/20261025_224813/workspace`.
Model configs: `/storage/outputs/pose-graph-diff/diagnostics/latent_overfit4/20261025_224813/v28_config.yaml`, `/storage/outputs/pose-graph-diff/diagnostics/latent_overfit4/20261025_224813/v29_config.yaml`.
Existing full-data V28/V29 jobs remain running with their original frozen sources.

## Controlled inputs

| Image ID | Supervised people |
|---:|---:|
| 113269 | 1 |
| 106326 | 2 |
| 111895 | 4 |
| 103487 | 8 |

There are15 GT people in total. Images come from CrowdPose trainval, selected
at seed42 with1/2/4/8people, at least8 labeled joints/person and resized person
height>=40pixels. The precise input batch, original/resized dimensions, masks,
annotations and50slot proposals perimage are frozen in `fixed_batch.pt` and
SHA256-checked at startup. Proposals/donors are constructed once at seed42 and
held fixed. GT coordinates are used only for training and explicitly labeled
clean/conditional evaluations; Gaussian sampling receives image features and noise.

![Training inputs](/storage/outputs/pose-graph-diff/diagnostics/latent_overfit4/20261025_224813/inputs.png)

## Two experiments

- `Overfit4-V28`: existing V28 clean-image stopgrad, fixed-correspondence latent
  targets,0.5clean-task + all-slot latentMSE1.
- `Overfit4-V29`: existing V29 OTA person targets/nearest-clean-negative targets,
  same latent-only coefficients and V28 model/gradient routing.

Both start fresh from original EDPose image initialization and fresh seed42 latent
modules. Both full initial model hashes equal
`268bdfe42eadf545bcddc5d1223af05fb8a0fd152389d7e34eff0a122e73abe2`.
The clean encoder and readout remain jointly trained and h0 is recomputed on every
microbatch. The latent values themselves are not held fixed.

## Deliberate simplifications

Native deterministic256/427 resize and normalization, no augmentation, the same
cached batch4 repeated with accumulation2. Constant original AdamW learning rates
2e-4/main and1e-4/backbone, decay1e-4, clipping0.1, FP32/TF32off. No epoch scheduler
steps in the bounded5000optimizer-update diagnostic. Timesteps remain uniformly
sampled0..999 independently perimage, and Gaussian latent noise is freshly drawn
on every microbatch. Both families share the same private draw sequence, starting
after the common fixed proposal construction. Existing implementations of losses,
detach routes, latent normalization, heads and samplers remain unchanged; the
cached-target draw adapter is diagnostic-only and does not edit production code.

## What gets measured

Initial, every250updates and final evaluation on the SAME4 images:

1. Clean reconstruction from the supplied GT-derived proposal inputs.
2. Conditional recovery at t0/499/999 with fixed separate evaluation noise.
3. Gaussian-start generation at1/4/16 DDIM calls under3 evaluation seed bases
   42/10042/20042, each plusimageID; seed-specific values and mean are both logged.

Each condition reports native pose/box AP after corrected pose NMS, raw OKS50/75
person coverage before any score filtering/NMS, score>=0.5 coverage, accepted and
repeated predictions, mean existence and latent slot variance. Clean/conditional
also report original-person OKS; conditional source-latent MSE and OTA-target MSE
are separately labeled. OTA can change person identity, so own-source OKS alone
must not be used as generated-set quality. Standard conditional confidence/OTA
training diagnostics and all task components are logged separately.

Training loss validation is replaced by direct repeated training/evaluation on
these exact inputs. No test-set generalization or training-set memorization is
inferred from conditional loss alone. Check whether accurate distinct people
appear in Gaussian samples, including their confidence and postprocessing fate.
An unsuccessful5000update endpoint is a failure of this bounded test, not proof
that the architecture can never memorize the images.

## Verification and artifacts

Both real-GPU two-update smokes passed, with nonzero finite backbone/projection,
clean encoder, denoiser and head gradients; complete same-image evaluation; strict
model/pipeline/optimizer restoration and exact next timestep/noise draws.
Peak reserved memory including smoke restoration was1.434GiB perprocess. Startup
checks fixed-cache/source/config/annotation/initialization hashes. Fixed-target
objects persist while successive noise draws differ and rewind exactly.
The Gaussian evaluation forbids calling the clean encoder, and cached-featureK4
sampling matches the native sampler bitwise. Evaluation preserves training RNG.

The smoke found a missing project marker and a diagnostic import-order issue;
both were corrected before launching long runs. Their earlier logs are preserved
under `tmp/latent_overfit4`; passed frozen-source smoke results are in `smoke_v28`
and `smoke_v29` at the diagnostic root.

Checkpoints contain model/pipeline state, optimizer and private draw state:
initial, latest every250updates, named every1000updates, final5000. Predictions
and metrics are under eachfamily `eval/stepXXXXX`; `latest_metrics.json` and
`status.json` provide progress. No checkpoint is selected by monitoring scores.

## Access

Shared TensorBoard: http://lynx.tail481642.ts.net:33329/ .
Labels: **Overfit4-V28**, **Overfit4-V29**. No additional server.
Primary curves: `overfit4/Gaussian_mean/K4/pose_AP`,
`overfit4/Gaussian_mean/K4/raw_coverage_oks75`,
`overfit4/clean/pose_AP`, `overfit4/conditional/t0/pose_AP`.
All AP scalars use0..1; multiply by100 for AP points.

Separate tmux sessions `overfit4-v28` and `overfit4-v29`, persistent
`v28/console.log` / `v29/console.log`, and a shared visible overfit progress viewer.
Runtime launch/headroom/full-data job preservation are recorded in `launch.json`;
`manifest.json`, `preflight.json`, input/config/source hashes preserve provenance.

Launch verified: v28 PID2858267 update240/5000, v29 PID2858273 update130/5000. Both full-data training process trees remain active. Initial3seedK1/K4/K16 evaluations completed; native parity, no clean encoder access and training RNG preservation passed. Actual loss/clean/conditional/Gaussian scalar data verified on shared TensorBoard, local/TailscaleHTTP200. Learned fit outcome remains pending.

## Final results — 2026-10-26

Both runs completed 5,000 updates. These are memorization results on the same
four training images (15 people), one training seed; Gaussian results average
three inference noise seeds. They are not held-out AP.

| Pose AP (0–100) | V28 fixed correspondence | V29 OTA |
|---|---:|---:|
| Clean latent decode | 93.07 | 93.07 |
| Conditional recovery, t=0 | 71.41 | 18.68 |
| Conditional recovery, t=499 | 89.12 | 45.51 |
| Conditional recovery, t=999 | 5.71 | 31.25 |
| Gaussian start, 1 call | 7.17 | 36.94 |
| Gaussian start, 4 calls | 57.87 | 32.40 |
| Gaussian start, 16 calls | 76.03 | 17.49 |

V28 demonstrates small-data Gaussian-generation capability and substantial
benefit from iterative sampling. V29 improves one-call generation but loses
quality with more calls. Raw OKS75 coverage, before confidence filtering/NMS,
changes from 20.0% to 91.1% to 95.6% for V28 and from 71.1% to 62.2% to 40.0%
for V29 at 1/4/16 calls. Thus V29's decline includes losing geometrically correct
person predictions, rather than merely changing scores.

Both clean branches cover all 15 people at OKS75 and have approximately 0.99875
own-person OKS. Near-clean recovery still degrades the set: V28 retains 14/15
people at OKS75; V29 retains 6/15. This set metric remains meaningful under OTA
reassignment, unlike source-slot identity accuracy. V28's better recovery at
t=499 than t=0 also exposes imperfect low-noise behavior.

This does not prove assignment is the sole cause: OTA also changes target
multiplicity and unmatched-slot targets. Fixed proposals, absent augmentation,
and constant learning rate limit comparison with full-data training.

Full-data V28 separately completed epoch79 and all 8,000 held-out images:
1-call pose/box AP 0.036/0.544; 4-call pose/box AP 0.773/4.277. Its small-data
capability therefore does not establish useful full-data performance. Full-data
V29 remains active (epoch56 at inspection); no final evaluation is available.

Final overfit metric files: `<root>/v28/latest_metrics.json` and
`<root>/v29/latest_metrics.json`, with the root specified above. Strict final
checkpoint restoration and next proposal/noise draw reproduction passed.
