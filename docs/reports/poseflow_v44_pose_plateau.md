# V44 pose plateau: frozen-checkpoint diagnosis

2026-11-06. No production code, optimizer updates, training policy or active-run changes.

## Evidence

Current fixed32 VAL epoch47 detector AP (not full validation): K1 pose33.55/box46.84, K4 pose31.51/box47.49, K8 pose27.98/box45.75.

A frozen copy of epoch047 was evaluated on the first16 images of the existing fixed32 VAL manifest (63 matched people), seed1042 plus image ID. The selected proposal assignment remains fixed. These are conditional paired metrics, not detector AP.

| Condition | Paired OKS | Paired box IoU | Median / p95 joint error, px |
|---|---:|---:|---:|
| Codec posterior mean | .9900 | .9876 | .97 / 4.65 |
| 1% noisy input, direct decode | .9868 | .9810 | 1.21 / 4.84 |
| 1% denoised output | .7663 | .8666 | 5.29 / 19.52 |
| Same output, GT box and predicted relative joints | .7978 | 1 | 5.04 / 20.11 |
| Same output, predicted box and GT relative joints | .8925 | .8666 | 2.78 / 7.47 |
| 5% noisy input, direct decode | .9174 | .9350 | 3.50 / 9.69 |
| 5% denoised output | .7642 | .8684 | 5.49 / 20.53 |

At5%, oracle-box OKS .7922 versus oracle-articulation .8935. Box replacement preserves predicted relative joint coordinates, rather than translating absolute points only. These interventions isolate output error components; they do not prove a training cause.

The denoiser responds to additional small latent perturbations: median output/input latent-change norm ratio .454 at1% and .479 at5%. Thus complete latent-input ignorance is not supported. Suppression of perturbations alone is not failure; the biased, inaccurate recovered geometry is.

## Crop audit

512 seed1042-selected TRAIN images, deterministic historical augmentation seed42+imageID: 1,692 augmented person views versus1,793 uncropped. Frozen posterior-mean codec geometry loss median .0181, p99 .404, maximum46.14 after augmentation, versus uncropped median .0162, maximum .328. Worst augmented boxes have minimum sides1.40 and1.85px. Rare tiny crops create substantial error tails even with the correct latent. TRAIN epoch47 geometry loss mean87.62 versus median sparsely logged batch loss1.46 suggests investigating outliers; sparse logs and this512-image audit do not establish their gradient contribution or causal role. Do not infer this uniquely explains the plateau.

## Interpretation and next decision

On these VAL people the frozen codec represents accurate articulation. The LDM substantially degrades near-clean codes, and the relative joint layout is a major part of that error. Modifying the VAE residual MLP is not supported by this evidence. Whole-pose feature attention and decoded-pose supervision may still limit Gaussian-start accuracy; low-noise preservation alone cannot establish improved detection.

Proposed next controlled comparison for review: retain the VAE, proposal/assignment contracts, native frontend and decoder; compare current absolute-x0 prediction against standard velocity prediction, with clean reconstruction alpha*h_t - sigma*v_hat and target alpha*epsilon - sigma*h0. Use the same seeds, batches, update budget, codec, evaluation draws and component weights except the required latent-target parameterization. Record initial-function differences; checkpoints must declare head semantics. At small noise the clean estimate has an identity path rather than requiring wholesale latent reconstruction.

Acceptance requires learned positive-noise recovery beyond direct noisy decoding, preserved box AP and better matched-age pose AP/multistep behavior, not only zero-noise identity. Earlier V43 velocity controls preserved inputs but failed foreground recovery; this is not evidence velocity will solve V44's Gaussian-start pose accuracy. No new comparison is implemented or launched. Separately audit tiny-crop gradient contributions before changing augmentation or loss normalization; do not silently bundle that change with parameterization.

## Artifacts

Full conditions, checkpoint SHA256, exact image IDs, per-condition tails and crop examples are in results.json/crop_rows.json under the run diagnostics/pose_plateau directory. A frozen checkpoint copy remains at tmp/v44/pose_diagnosis/frozen_best.ckpt. The source is the running snapshot, not an altered implementation.
