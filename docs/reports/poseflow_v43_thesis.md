# V43 thesis: transferable image-conditioned latent denoising

## Question and working thesis

The scientific question is whether the frozen pose-box VAE gives a useful state space for an image-conditioned denoiser: on people excluded from denoiser optimization, does correct image conditioning allow the model to improve corrupted geometry beyond raw-noisy decoding, while preserving nearly clean inputs?

The working explanation for the failure is inadequate learned alignment between image/proposal features and clean VAE geometry. The one-image improvement may be an image-specific memorized conditional pose prior. Identity-preserving parameterization fixes destructive behavior at zero noise; it does not itself teach a transferable conditional distribution. This is a falsifiable explanation, not a proven unique optimization cause.

For squared latent error, the target function is the conditional mean E[z_clean | z_noisy, image, proposal]. A training set containing one image supplies only eight clean person codes. New Gaussian draws around these eight fixed codes test recovery around those anchors; they do not test learning the pose distribution across images. Exact recovery is also not generally guaranteed when different clean poses can explain the same noisy code and image evidence. Thus the 1e-5 latent MSE gate is an engineering diagnostic, not a theoretical requirement for a good score model or a substitute for geometry/AP.

## Predictions declared before probing

1. A transferable conditional denoiser should improve held-out-image geometry and benefit from correct versus wrong conditioning.
2. If one-image gains are local, they should fail to transfer to other images.
3. If the image route is unused, replacing image memory/features should leave the trained-image recovery unchanged. This subhypothesis can be rejected independently of generalization.

## Frozen test and measured answer

No optimizer updates. Same source targets, noisy latent tensors, image dimensions, reference boxes and all 50 queries. Three paired noise draws at sigma 0.05. Swap complete cached memory/position/padding and/or actual selected appearance features from deterministic donor images; do not change target identity or geometry. The 16 VAL images contain 46 people; the single trained image contains eight. Neither is a native AP evaluation. Frozen parent/codec have prior trainval exposure, so this is not independent detector testing.

| Checkpoint / population | Raw OKS | Correct conditioning OKS | Wrong memory + features OKS | Correct box IoU |
|---|---:|---:|---:|---:|
| One-image model / trained image | 0.88975 | 0.97643 | 0.88506 | 0.95283 |
| One-image model / other 15 TRAIN images | 0.93127 | 0.73129 | 0.73547 | 0.89594 |
| One-image model / 16 VAL images | 0.92045 | 0.72323 | 0.72538 | 0.90112 |
| Original 16-image v control / 16 VAL images | 0.92045 | 0.92191 | 0.92161 | 0.94628 |

On the trained image, replacing only memory gives OKS 0.94596; replacing only selected appearance features gives 0.90664. Replacing both gives 0.88506. Therefore the single-image model DOES use image conditioning. An image-independent memorization explanation is rejected for this probe.

Its improvement nevertheless fails to transfer. VAL latent MSE is 0.01463562 versus raw 0.00247645, about 5.91 times worse. VAL box IoU is 0.90112 versus raw 0.94597. The broader control has only a 0.00030 OKS difference between correct and swapped conditioning and about 0.3% latent MSE reduction versus raw. These are small measured effects in this bounded run; no significance/general-population claim is made.

This supports image-specific conditional overfitting and the absence of a useful transferable correction in the tested checkpoints. It does NOT prove an intrinsically unsuitable VAE, a disconnected image route, lack of model capacity, or that background loss caused failure. Nor does it uniquely separate insufficient training from architecture or objective mismatch. The previously proposed 16-image foreground-only optimization does not isolate this key scientific question and is withdrawn as the automatic next step.

## Strategy and decision that follow

Make image-to-VAE alignment an explicit learning target, then test whether it supplies useful conditioning for denoising. The proposed next implementation should predict a clean pose-box latent from image/proposal features WITHOUT a noisy latent input, supervised with clean codec means and decoded labeled-joint/box geometry. This directly tests what geometric information the image branch can learn. It must use the real TRAIN population and a held-out-image validation protocol, rather than another one-image memorization gate. All output geometry still comes from the frozen VAE.

If that image-only prediction generalizes, use its features/estimate to condition the identity-preserving denoiser and test whether it improves noisy geometry over both raw decoding and wrong-image conditioning at 1%, 2.5% and 5%. This proposal is a curriculum/alignment hypothesis; image-only prediction need not reconstruct every latent exactly or reach detector AP 97.5 by itself. It must demonstrate measurable held-out geometry information before being used to justify additional diffusion training.

Decision tree: poor held-out image-only prediction means repair image-to-code representation/conditioning first; useful image-only prediction with no denoising gain points to fusion/denoising optimization; correct-image denoising gains that disappear under conditioning swaps justify extending diffusion training. Keep train/validation accuracy and paired error tails to distinguish underfit from overfit. Final acceptance remains native pose/box AP and geometry preservation, not loss alone. New training/production changes require a concrete implementation plan and review; none launched here.

## Evidence and limitations

`conditioning_results.json` contains per-condition geometry metrics, latent errors, correction/output RMS, donor IDs, noise hashes, checkpoints and cache SHA256. Both model states are unchanged. Native single-image metrics replay exactly; original 16-image VAL metrics replay within 2e-6 for FP32 batch-shape differences. `replay_check.json` records verification. Swapping features creates counterfactual combinations with original reference boxes, so it tests dependence and cannot alone prove a unique training cause. There is one training seed and three evaluation-noise seeds, and a small fixed validation population.
