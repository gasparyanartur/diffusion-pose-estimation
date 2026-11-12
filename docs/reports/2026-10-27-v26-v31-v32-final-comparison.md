# V26-latent, V31 and V32: final comparison

Checked 2026-10-27. All three completed epoch079,120000 optimizer updates and the full8,000-image held-out evaluation. This compares original V26-latent, not V26-latent-only.

Exact image-ID lists, annotation SHA256, resize256/427, seed42+imageID, FP32, TF32off, deterministic latent DDIM eta0, corrected native pose NMS and evaluator identities match. Training resolved configs differ only in family/namespace and explicit0.5 recovered-task coefficient (already implicit in original V26). Underlying model_v26.py is byte-identical across frozen snapshots. One training seed; these differences do not establish statistically reliable gains.

| Model | Pose AP K1 | Pose AP K4 | Box AP K1 | Box AP K4 |
|---|---:|---:|---:|---:|
| V26-latent | 0.63 | 7.33 | 3.39 | 14.71 |
| V31 | 0.64 | 7.95 | 3.37 | 15.17 |
| V32 | 0.49 | 7.73 | 2.66 | 15.60 |

## What changed

- V26-latent: live image features in clean and denoised branches;0.5 clean task+0.5 recovered task+latent MSE. h0 detached before corruption/as MSE target.
- V31: same objective/architecture with clean image features detached. Clean task does not train image extraction; recovered task/MSE do. Both h0 detaches retained.
- V32: live clean image gradients restored and DINO supervision on both clean/recovered latents. Added0.5*(DINO_clean+DINO_recovered), shared7x7x384 prediction head, frozen DINOv3 teacher. Relative to V31 this combines two changes; relative to original V26 it adds DINO on otherwise matching core architecture/recipe. No teacher/head inference execution.

## Findings

V31 leads four-call pose AP at7.95, +0.61 points over V26. V32 reaches7.73, -0.22 versus V31 and +0.39 versus V26. V32 leads four-call box AP at15.60, +0.43 versus V31 and +0.89 versus V26. These are small changes on a poor baseline, not a breakthrough. V24 control reference remains54.91 pose AP at one call, with different architecture/inference compute.

All remain weak at one call (0.49–0.64 pose AP). Four calls substantially improve all three. These final evaluations establish only K1/K4 behavior, not a longer-call optimum.

| Model | K4 coverage at score>=0.5, OKS>=0.5 | Pre-NMS duplicates | Post-NMS detections |
|---|---:|---:|---:|
| V26-latent | 52.28% (15225/29122) | 210542 | 63823 |
| V31 | 52.81% (15378/29122) | 232736 | 62378 |
| V32 | 51.40% (14968/29122) | 192414 | 62358 |

Coverage is measured before NMS and includes an existence-score threshold; it is not AP, raw score-free recall or final detection recall. V32 emits fewer confident duplicates, but does not improve this coverage or pose AP over V31. Duplicate and false-positive behavior remains substantial.

## Final conditional validation losses

These use GT-constructed clean/noisy inputs and are not Gaussian-start detection AP.

| Model | Clean task | Recovered task | Latent MSE |
|---|---:|---:|---:|
| V26-latent | 1.699 | 10.237 | 0.0992 |
| V31 | 2.068 | 10.302 | 0.1028 |
| V32 | 1.676 | 10.255 | 0.0989 |

The recovered task loss remains approximately10.25 in all three runs, while clean task loss is1.68–2.07. Adding DINO did not close the conditional task-loss gap. Latent MSE is also very similar; it does not establish equal representations or equal detection ability.

## Did V32 learn the DINO target?

Final validation cosine: clean 0.8035, recovered 0.7942. Correct-minus-wrong-person cosine: clean 0.0606, recovered 0.0472. The auxiliary target is being learned and there is positive person discrimination, but high absolute cosine alone overstates specificity. The wrong-person gap is evaluated only where multiple valid people are available; it cannot simply be subtracted from the overall cosine to recover a global wrong-person score.

DINO loss/cosine were measured during conditional validation, not Gaussian-start generation. They do not show that Gaussian-start latents recover the right visual features. No evidence here supports claiming representation collapse, image-attention failure, or assignment ambiguity as the sole cause.

## Evidence

- V26-latent: [20261024_194251](/storage/outputs/pose-graph-diff/results/poseflow_v26_latent_diffusion/20261024_194251/seed42/full_test/metrics.json), final checkpoint `seed42/checkpoints/epoch079.ckpt`; checkpoint/config hashes stored in the metric JSON.
- V31: [20261026_142350](/storage/outputs/pose-graph-diff/results/poseflow_v31_denoised_task/20261026_142350/seed42/full_test/metrics.json), final checkpoint `seed42/checkpoints/epoch079.ckpt`; checkpoint/config hashes stored in the metric JSON.
- V32: [20261026_150316](/storage/outputs/pose-graph-diff/results/poseflow_v32_dual_dino/20261026_150316/seed42/full_test/metrics.json), final checkpoint `seed42/checkpoints/epoch079.ckpt`; checkpoint/config hashes stored in the metric JSON.

Machine-readable comparison and final validation scalars: `tmp/latent_final_comparison/comparison.json`. No model or training changes were made for this comparison.
