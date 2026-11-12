# V35-spatial-box: denoising stage and noise-level audit

Frozen epoch-74 checkpoint, fixed32 held-out CrowdPose images (133 people), seed 42+imageID for Gaussian starts, native resize 256/427, FP32 with matmul/cuDNN TF32 disabled. All 50 predictions, person label 1, native bbox evaluator, no NMS. This is a subset diagnostic, not the full 8,000-image result.

Checkpoint: `/storage/outputs/pose-graph-diff/diagnostics/v35_iteration_audit_20261031_002218/constant/epoch074.ckpt`
Raw results and prediction records: `/storage/outputs/pose-graph-diff/diagnostics/v35_iteration_audit_20261031_002218`

## Native Gaussian-start trajectories

Each row decodes the predicted clean latent from that call, not its noisy DDIM state. High numeric t means high noise, early in reverse diffusion. Native K1 and K4 endpoints reproduce logged AP exactly.

| Calls | Stage | t | AP | People covered at IoU >= .5, any score | Covered at score >= .5 |
|---|---:|---:|---:|---:|---:|
| 4 | 1 | 999 | 22.46 | 102/133 | 71/133 |
| 4 | 2 | 749 | 18.03 | 94/133 | 57/133 |
| 4 | 3 | 499 | 17.69 | 86/133 | 59/133 |
| 4 | 4 | 249 | 18.16 | 87/133 | 61/133 |
| 8 | 1 | 999 | 22.46 | 102/133 | 71/133 |
| 8 | 2 | 874 | 19.16 | 99/133 | 53/133 |
| 8 | 3 | 749 | 16.53 | 86/133 | 52/133 |
| 8 | 4 | 624 | 16.21 | 84/133 | 52/133 |
| 8 | 5 | 499 | 16.00 | 77/133 | 52/133 |
| 8 | 6 | 374 | 16.76 | 76/133 | 54/133 |
| 8 | 7 | 249 | 17.31 | 75/133 | 53/133 |
| 8 | 8 | 124 | 17.38 | 78/133 | 56/133 |

K4 loses 4.44 AP at the first repeat (t749), another .33 at t499, and recovers .47 at t249. K8 loses 6.47 by t499, then recovers 1.38 over its remaining calls. Both raw geometric coverage and confident coverage fall; the decline cannot be explained solely by changing confidence thresholds. Stage deltas describe these trajectories, not isolated causal contributions of each call.

## One-call conditional denoising at known noise levels

Clean inputs include every person, positive perturbations and foreign negatives, constructed deterministically in groups of four. F is extracted independently per image at native B1 sizing and reused. The same Gaussian epsilon is used across noise levels for each image. No training OTA reassignment is applied to these set-level evaluation predictions.

| t | Gaussian noise coefficient sqrt(1-alpha_bar) | One-call AP | Raw coverage | Confident coverage |
|---:|---:|---:|---:|---:|
| 0 | 0.0064 | 32.57 | 125/133 | 93/133 |
| 99 | 0.1671 | 32.29 | 124/133 | 91/133 |
| 249 | 0.3911 | 30.51 | 122/133 | 90/133 |
| 499 | 0.7114 | 27.82 | 113/133 | 85/133 |
| 749 | 0.9251 | 25.65 | 110/133 | 79/133 |
| 899 | 0.9879 | 23.03 | 109/133 | 79/133 |
| 999 | 1.0000 | 22.32 | 105/133 | 71/133 |

An exact clean input to D at t0 gives 32.67 AP; q(h0,0) gives 32.57, since native t0 still has small nonzero corruption. Conditional AP drops 10.25 from nearly clean to maximum noise. Low-noise denoising works better when its input comes from the actual clean-encoding distribution than the lower-noise stages of Gaussian-start trajectories. This supports a generated-state mismatch or error-amplification hypothesis, but does not establish its cause.

Clean readout alone has 11.62 AP despite covering 129/133 people (128 confidently): it produces 861 score>=.5 predictions. D(h0,F,0) reduces that to 266 and achieves 32.67 AP. Duplicate positives make clean readout AP a poor pure reconstruction score; use coverage alongside it. Do not interpret the conditional AP improvement as proof of latent identity preservation: clean and denoised task assignments differ, and the saved same-source latent MSE is not the remapped training MSE.

## Changing the training timestep distribution

No distribution was changed and no training run was launched by this audit. More high-t samples without importance weights allocate more optimization to image-conditioned generation from weak latent signal; more low-t samples allocate more to preservation/refinement of nearly clean inputs. The latter are still corruptions of true clean latents, not necessarily the model-generated states that fail here. Consequently, a low-t bias is not automatically a cure for rollout damage.

Uniform training minimizes (1/T) sum_t E[L_t]. Sampling q(t) without weighting instead minimizes sum_t q(t)E[L_t], changing all timestep-dependent denoiser losses and their aggregate balance with the clean task. Multiplying each sampled timestep-dependent loss by 1/(T*q(t)) preserves the uniform objective in expectation, while changing estimator variance and finite-training behavior. Keep q(t)>0 on all timesteps for this interpretation. Training timestep frequencies and inference call spacing are distinct controls.

Temporary probe sources: `tmp/v35_iteration_audit/probe.py` and `placement.py`. No production model, training config, live job or TensorBoard stream was modified.

## Frozen-model inference call placement control

These custom spacings directly vary where repeated calls are spent. Each starts from the same per-image Gaussian state and uses the same native deterministic DDIM update; the final call always returns its predicted clean latent. They do not change training or prove that retraining with biased timesteps helps.

| Denoiser timesteps | Calls | Final AP |
|---|---:|---:|
| 999 | 1 | 22.46 |
| 999, 749, 499, 249 (native) | 4 | 18.16 |
| 999, 749 | 2 | 18.03 |
| 999, 499 | 2 | 19.86 |
| 999, 249 | 2 | 21.79 |
| 999, 249, 124, 0 | 4 | 20.35 |
| 999, 874, 749, 624 | 4 | 16.21 |

Four calls concentrated at lower noise (999,249,124,0) give 20.35 AP, versus 16.21 with repeats concentrated at high noise (999,874,749,624). Both remain below the one-call 22.46 baseline. This is a same-checkpoint/same-seed placement diagnostic on fixed32, not a selected replacement sampling protocol. A late schedule also changes step sizes and intermediate-state distributions, so the result cannot isolate timestep sensitivity from trajectory effects.
