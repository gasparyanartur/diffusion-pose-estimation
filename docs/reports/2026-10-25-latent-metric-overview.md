# V26/V27 reported-metric overview

Snapshot2026-10-25. V26 pair compared at step73499 (49 completed epochs); V27 comparison uses step26999 (18 completed epochs) for all3 models. Data extracted directly from shared TensorBoard event files, with metric definitions verified in training_v26.py, latent_v26.py and quick_eval_v26.py. No training/model/dashboard changes.

## Main interpretation

The clean representation/readout learns the supplied task. Removing the recovered-latent task loss reduces numerical latent error but worsens decoded conditional pose quality and Gaussian-start detection. It is partial learning, not no learning: latent-only K4 monitor AP rises0.47 at12epochs to2.63 at49epochs, but original V26 rises4.49 to9.67 over those same ages.

Lower clean loss and lower latent MSE do not show successful generation. Clean branches receive GT coordinates; conditional losses start from corrupted GT-derived latents; Gaussian-start quick-test sampling receives images and noise only. Validation losses overlap trainval, so clean results demonstrate fitting rather than held-out detection.

![Trends](/home/gasparyanartur/dev/pose-graph-diff/tmp/v26_metric_overview/overview.png)

## Same-age V26 pair: all validation and quick-test tags

| Metric | Original latent | Latent-only |
|---|---:|---:|
| `val/loss` | 6.29229 | 0.848116 |
| `val/clean/loss_ce` | 0.0133764 | 0.0130121 |
| `val/clean/loss_bbox` | 0.0876516 | 0.0701597 |
| `val/clean/loss_giou` | 0.184689 | 0.157754 |
| `val/clean/loss_keypoints` | 0.103147 | 0.0849587 |
| `val/clean/loss_oks` | 0.0225762 | 0.0177845 |
| `val/clean/total` | 1.95617 | 1.61306 |
| `val/denoised/loss_ce` | 0.057078 | 0.133368 |
| `val/denoised/loss_bbox` | 0.174059 | 0.202068 |
| `val/denoised/loss_giou` | 0.336156 | 0.396034 |
| `val/denoised/loss_keypoints` | 0.770911 | 1.07746 |
| `val/denoised/loss_oks` | 0.264423 | 0.362657 |
| `val/denoised/total` | 10.4236 | 14.2943 |
| `val/loss_latent` | 0.102423 | 0.041588 |
| `debug/v26/val/unique_people` | 3.537 | 3.537 |
| `debug/v26/val/positive_proposals` | 26.555 | 26.555 |
| `debug/v26/val/duplicate_proposals` | 23.018 | 23.018 |
| `debug/v26/val/synthetic_fraction` | 0 | 0 |
| `debug/v26/val/clean_latent_variance` | 0.230614 | 0.134729 |
| `debug/v26/val/clean_latent_rms` | 1 | 1 |
| `debug/v26/val/time_mean` | 502.968 | 502.968 |
| `debug/v26/val/clean/positive_score` | 0.854433 | 0.850656 |
| `debug/v26/val/clean/negative_score` | 0.0647603 | 0.057586 |
| `debug/v26/val/denoised/positive_score` | 0.683601 | 0.738934 |
| `debug/v26/val/denoised/negative_score` | 0.191963 | 0.206556 |
| `debug/v26/val/latent_mse/low` | 0.023311 | 0.0035194 |
| `debug/v26/val/latent_mse/middle` | 0.0626653 | 0.0209035 |
| `debug/v26/val/latent_mse/high` | 0.218659 | 0.0990343 |
| `debug/v26/val/clean_task` | 1.95617 | 1.61306 |
| `debug/v26/val/denoised_task` | 10.4236 | 14.2943 |
| `debug/v26/val/latent_mse` | 0.102423 | 0.041588 |
| `quick-test/noise_var_100/steps_1/AP` | 0.00998038 | 0.00122192 |
| `quick-test/noise_var_100/steps_1/AP_E` | 0.0613 | 0.0011 |
| `quick-test/noise_var_100/steps_1/AP_M` | 0.0103 | 0.0011 |
| `quick-test/noise_var_100/steps_1/AP_H` | 0.0209 | 0.0158 |
| `quick-test/noise_var_100/steps_1/bbox/AP` | 0.0398585 | 0.0133794 |
| `quick-test/noise_var_100/steps_1/bbox/AP50` | 0.117708 | 0.0749971 |
| `quick-test/noise_var_100/steps_1/bbox/AP75` | 0.0257457 | 0.000677654 |
| `quick-test/noise_var_100/steps_1/bbox/AR` | 0.109494 | 0.0424051 |
| `quick-test/noise_var_100/steps_1/gt_people` | 133 | 133 |
| `quick-test/noise_var_100/steps_1/covered_people` | 0 | 17 |
| `quick-test/noise_var_100/steps_1/duplicate_predictions` | 0 | 262 |
| `quick-test/noise_var_100/steps_1/accepted_predictions` | 1 | 880 |
| `quick-test/noise_var_100/steps_1/coverage_oks50_score50` | 0 | 0.12782 |
| `quick-test/noise_var_100/steps_4/AP` | 0.096715 | 0.0263244 |
| `quick-test/noise_var_100/steps_4/AP_E` | 0.1957 | 0.084 |
| `quick-test/noise_var_100/steps_4/AP_M` | 0.0777 | 0.0111 |
| `quick-test/noise_var_100/steps_4/AP_H` | 0.1616 | 0.0526 |
| `quick-test/noise_var_100/steps_4/bbox/AP` | 0.141359 | 0.0773299 |
| `quick-test/noise_var_100/steps_4/bbox/AP50` | 0.347528 | 0.182916 |
| `quick-test/noise_var_100/steps_4/bbox/AP75` | 0.0907031 | 0.0604904 |
| `quick-test/noise_var_100/steps_4/bbox/AR` | 0.232911 | 0.218987 |
| `quick-test/noise_var_100/steps_4/gt_people` | 133 | 133 |
| `quick-test/noise_var_100/steps_4/covered_people` | 69 | 43 |
| `quick-test/noise_var_100/steps_4/duplicate_predictions` | 807 | 330 |
| `quick-test/noise_var_100/steps_4/accepted_predictions` | 1103 | 669 |
| `quick-test/noise_var_100/steps_4/coverage_oks50_score50` | 0.518797 | 0.323308 |

## Reading the metric families

| Family | What it measures | Insight / limitation |
|---|---|---|
| clean loss_ce/bbox/giou/keypoints/oks, clean_task | Decode latent constructed from supplied GT joint coordinates | All components slightly better for latent-only; useful clean pose information can be learned. Does not test discovering people from noise. |
| denoised loss_ce/bbox/giou/keypoints/oks, denoised_task | Decode latent recovered from corrupted clean targets, averaged over sampled times | Every component worse in latent-only at matched age. Its joint L1 is~40% worse and OKS loss~37% worse. This is a task-fidelity failure despite lower latent MSE. |
| latent_mse, low/middle/high | Squared error to current detached clean latents; time thirds0–333,334–666,667–999 | Latent-only0.0416 vs0.1024, but target spaces are learned and differ. High-time error0.0990 vs low0.00352 within latent-only: recovery remains harder at high noise. |
| clean_latent_variance | Channelwise variance over batch and proposal axes of CLEAN targets | Lower in latent-only0.1347 vs0.2306. Targets share more common structure numerically; this can affect MSE scale. It is not denoised-slot variance and not proof of constant-output collapse. |
| clean_latent_rms | Overall clean target magnitude | Exactly~1 from non-affine LayerNorm by construction; it is not a success metric. |
| clean positive/negative scores | Mean existence confidence on constructed positive/negative GT-source slots | Both separate positives/negatives well (~0.85/~0.06), consistent with clean fitting. |
| denoised positive/negative scores | Same average scores on recovered conditional latents, across noise times | Latent-only positive mean is higher but negative mean also higher. Means hide mistakes and do not establish ranking/calibration; focal existence loss is actually worse0.133 vs0.057. These are not Gaussian-start scores. |
| quick-test AP | Keypoint detection quality after native postprocessing, score ranking, stricter matching and multiple OKS thresholds | Primary Gaussian-start diagnostic. At4calls9.67 original vs2.63 latent-only. Fixed32 is noisy; final full-test is a different endpoint. |
| AP_E/AP_M/AP_H | Pose AP on easy/medium/hard crowd subsets | At4calls latent-only is lower in all3 at matched age. These are small groups within32images; do not diagnose a single crowd-density failure from them. |
| bbox/AP, bbox/AP50, bbox/AP75, bbox/AR | Box precision across thresholds, loose/strict threshold AP, average recall | At4calls latent-only7.73 boxAP vs14.14 original. BoxAR is closer21.90 vs23.29, while poseAP has a much larger gap: coarse localization alone is insufficient. Box recall and pose metrics are different tasks. |
| covered_people / coverage_oks50_score50 | Pre-NMS GT people having any raw pose with OKS>=0.5 AND existence>=0.5; count and fraction | They are the same observation in different units, not independent evidence. No one-to-one matching or false-positive penalty; one prediction can cover multiple overlapping GT people. |
| accepted_predictions | Raw50-slots/image with existence>=0.5, summed over32images | Confidence/count behavior, not correctness. One-call latent-only accepts880/1600 versus1 original yet covers17/133 and has worse AP. |
| duplicate_predictions | Extra accepted, OKS>=0.5 predictions whose best GT is a person already represented | Original807 vs latent-only330 at4calls. Fewer duplicates coincides with fewer accepted and covered people; it is not independently improved set prediction. Not a count of all false positives. |
| gt_people | Fixed subset GT denominator |133 in both; check dataset/manifest consistency, not learned behavior. |
| unique_people, positive_proposals, duplicate_proposals | Construction counts BEFORE the model | Same~3.54 distinct people/image and26.56 positive slots,23.02 duplicated positive slots. These are training/validation input metadata, not generated coverage. |
| synthetic_fraction | Fraction of constructed slots using synthetic fallback |0 on measured validation; foreign donors available there. No evidence fallback drives validation failure. |
| time_mean | Mean sampled diffusion timestep |~503 matched across runs, consistent with intended uniform schedule; mean alone cannot prove uniformity. |
| total train/val loss | Weighted optimized objective | Not comparable between original and latent-only because a positive task term was removed. Loss terms duplicated under val and debug tags are not independent signals. Training shares these definitions and should not be treated as held-out quality. |

## One call versus four calls

At49epochs, original one/four-call poseAP1.00/9.67, latent-only0.12/2.63. Four calls help both. The one-call covered-people advantage17vs0 for latent-only is substantially confounded by score thresholds (880vs1 accepted raw slots). At4calls the original has BOTH higher coverage69vs43 and AP.

The separate pinned epoch046 larger-call sweep shows latent-only K4->K128 boxAP7.52->11.53 while poseAP1.90->1.65. Thus extra calls can improve boxes but do not automatically fix articulated poses. Those are47epoch results and must not be substituted for the49epoch scalar snapshot.

## V27 at its matched age

| Metric | V26 original | V26 latent-only | V27 |
|---|---:|---:|---:|
| debug/v26/val/clean_task | 4.05602 | 4.02053 | 3.74253 |
| debug/v26/val/denoised_task | 12.1555 | 15.11 | 15.3824 |
| debug/v26/val/latent_mse | 0.139833 | 0.0569527 | 0.0554128 |
| quick-test/noise_var_100/steps_4/AP | 0.0452058 | 0.0128706 | 0.00525584 |
| quick-test/noise_var_100/steps_4/bbox/AP | 0.0901315 | 0.06184 | 0.0273541 |
| quick-test/noise_var_100/steps_4/covered_people | 49 | 39 | 46 |

V27 at18epochs covers46 people versus39 for latent-only but has worse K4 poseAP0.53 vs1.29. This is another direct example of coverage improving without overall pose-detection quality improving. Scale scheduling has not rescued this objective at this snapshot; no final V27 result is claimed.

## What the metrics support

1. Clean task representations are learning; blanket no-learning or complete clean-latent collapse is unsupported.
2. The main disconnect is preserving task-useful information through denoising and turning it into accurate generated people. Small latent error is not enough.
3. Score threshold effects explain some apparently favorable coverage curves, but geometry losses and larger-call raw coverage show this is not only a scoring problem.
4. Multiple calls help relative to one call; adding many calls chiefly helps coarse boxes rather than articulated poses in the measured latent-only sweep.
5. Repeated person predictions and missing distinct people coexist. Lower duplicates alone is not success.
6. Attention audit shows person-localized, head/scale-specific reads; it does not establish causal image-feature usefulness or isolate the failing module.

## Limits and next decisive measurement

All tabulated AP is fixed32 monitoring, not full8000 test. Loss validation overlaps training. Current target spaces drift and differ. The loss ablation simultaneously changes denoiser task gradients, shared-head adaptation to recovered latents, and image-feature task gradients through the denoising branch; metrics do not identify which route causes the regression.

A useful next diagnostic is one matched checkpoint comparison of clean decoding, low-noise recovered decoding, and Gaussian generation, each with raw OKS50/75 coverage and score/NMS effects separated. This would locate which stage loses accurate distinct people. It is not another architecture proposal, and no new experiment was launched here.

Raw evidence: `/home/gasparyanartur/dev/pose-graph-diff/tmp/v26_metric_overview/series.json` and `snapshots.json`; metric definitions in `training_v26.py`, `latent_v26.py`, `quick_eval_v26.py`. Prior pinned diagnostic reports retain their own epochs and limits.
