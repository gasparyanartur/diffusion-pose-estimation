# What actually improves with repeated pose inference?

## Experiment-history review, ten research insights, and a two-week closeout plan

**Date:** 10 November 2026. **Purpose:** decide what can become a defensible research result within approximately two weeks. **Status:** evidence review and proposed experiments; no new training or inference was performed for this report.

## Executive assessment

There is enough evidence to finish this project as a focused empirical study. There is much less evidence that another architecture iteration will produce a competitive new pose estimator within two weeks. Those are different possible endpoints, and they imply different next actions.

The most credible central question is:

> When an image-conditioned pose estimator is run repeatedly, does its evolving state improve localization, improve selection among existing poses, or simply disturb an image-driven prediction?

This question fits the actual experiment history. The project repeatedly produced models that reconstruct poses accurately, react to images, depend on their carried state, or improve an auxiliary objective, yet fail to improve native pose detection with additional calls. Several investigations went beyond observing a bad AP curve: they held weights and geometry fixed, changed a specific score or state route, and identified which part of the computation caused the measured deterioration.

The strongest existing finding is **confidence drift under generated-state feedback**. In V40 and paired V42, changing only the score source recovers much of the AP lost during repeated inference. This is a controlled result about those checkpoints. It is not yet a broadly validated scoring method: the retrained image-only-confidence V42 remains weaker than the ordinary uniform-t control, and substantial geometric failure survives in other models.

The strongest training intervention is **V19's coordinate-gradient boundary**. Full geometry detachment dramatically outperforms stopping sampling-reference gradients alone, despite identical forward values at fixed weights. The backward decomposition is unusually well investigated. Its remaining weaknesses are the small monitored cohort, one training seed, and the fact that iterative-reference detachment already has close precedent in detection transformers.

The most promising recently completed positive ablation is **V45 task-pose JEPA**. It improves four-call full-test pose AP by 3.11 points over its matched control. However, it reduces one-call AP slightly, and its multicall results remain worse than its own one-call result. It supports a narrow claim about reduced repeated-call damage, pending replication.

My recommendation is to make confidence/state interventions the core of a bounded closeout study. Use the full-test step curves and memory controls to explain why the question matters. Treat the gradient and JEPA results as alternative focused follow-ups, rather than trying to make all ten insights into one paper. The other experiments should provide context and limits, not ten simultaneous new research programmes.

### Reading guide

- [Evidence definitions and evaluation caveats](#1-what-was-reviewed-and-how-to-read-the-evidence)
- [The experiment history and common step-response table](#2-the-history-as-a-research-trajectory)
- [How state, geometry, and scores interact](#3-the-analytical-framework)
- [The ten insights, with mechanisms and proposed tests](#4-ten-concrete-insights)
- [Other attempted fixes and why they do not yet supply a closeout](#5-what-the-remaining-experiments-rule-in-or-out)
- [Ranked options, fourteen-day schedule, and completion criteria](#6-which-angle-should-we-choose)
- [Prior work and novelty limits](#7-related-work-and-claim-boundaries)
- [Branch-by-branch inventory](#appendix-a-coverage-of-the-experimental-branches)

## 1. What was reviewed and how to read the evidence

The review covered the original reproduction and runtime reports, the selected-layer Hungarian/OTA comparison, the PoseFlow chronology, the experiment diary, result-bearing reports under `docs/reports/`, historical diagnostic reports under local `tmp/`, and saved metric JSONs under both output roots. The first review indexed 79 quick-result groups and 98 AP-summary records. These are inventory counts, not independent experiments: aliases, repeated evaluations, controls, and duplicate summaries occur in the history.

The report distinguishes four levels of evidence:

| Evidence type | What it establishes | What it cannot establish by itself |
|---|---|---|
| Full-set native detector evaluation | Performance of a specified checkpoint and inference protocol on that image set | Training-seed robustness, untouched-test generalization, or the cause of a difference |
| Frozen-checkpoint intervention | The effect of changing a specified computation at those weights and inputs | Why training learned that behaviour, or what a retrained alternative will achieve |
| Conditional recovery or codec reconstruction | Recovery when GT-derived geometry, correspondence, or boxes are supplied | Image-only detection from noise or correct person discovery |
| Tiny-set overfit or implementation smoke | Memorization in that setup, or correct wiring/finite execution | Transfer to unseen images or useful learned pose quality |

Unless explicitly stated otherwise, AP numbers in this report are multiplied by 100. Thus a stored value of `0.3553` is reported as 35.53 AP. **K** denotes sampling calls, not decoder layers. A three-layer decoder called four times performs more computation than one three-layer call; different model families also have different cost per call.

**Pose AP** concerns keypoint detections. **Box AP** concerns person boxes and must not be substituted for pose AP. **Paired OKS** measures the geometry of predictions attached to known people and does not include the same ranking, matching, and detection failures as native AP. Most paired OKS values are shown on the 0–1 scale; selected tables explicitly use OKS points, 0–100.

### 1.1 Evaluation exposure and historical compatibility

Many historical runs train on CrowdPose trainval, 12,000 images, while reporting validation losses on the contained 2,000-image VAL split. Those validation losses measure fitting. Fixed 32 TEST monitoring and repeated full 8,000 TEST evaluations also informed development. The historical TEST results therefore cannot be described as a newly untouched confirmation set, even when TEST was excluded from weight optimization.

V44 and the newer ordered experiments use disjoint 10,000-image TRAIN and 2,000-image VAL for their own optimization. Their source initialization and, where applicable, codec may have earlier trainval exposure. This makes VAL useful for studying the new optimization, but not necessarily an independent test of every pretrained component. Do not combine the two split protocols into a single leaderboard.

The project also corrected a confidence-to-query association bug in pose postprocessing. Older legacy-scoring AP and later corrected AP are not interchangeable. The historical paper-scale curves below retain their original evaluation contract and are labelled accordingly. The later reduced-model comparison uses corrected score association consistently. A final closeout evaluation should use one frozen evaluator and replay its baseline before interpreting interventions.

There is an additional box-evaluation issue in the ordered models: the native box metric includes some box-only people excluded from pose-sequence supervision. A perfect list of posed people therefore does not guarantee 100 box AP. This does not explain their near-zero pose AP, but it matters for oracle gates and box ceilings.

### 1.2 Terms used in the report

| Term | Meaning here |
|---|---|
| Geometry | Predicted joint positions and/or person-box coordinates, before considering how confidently the model ranks them |
| Confidence / score / logit | The model's ranking signal for a predicted person; a logit is the value before a sigmoid converts it to a probability-like score |
| Diffusion state | The coordinates or learned code that the sampler transports between calls; it is distinct from the final decoded pose |
| Hidden state / h / memory | Additional carried feature information in the recurrent variants; it is not automatically an identity tracker or a probabilistic pose state |
| Latent code / codec | A learned numeric representation of pose geometry, and the encoder/decoder mapping between that representation and coordinates |
| Native AP | Detection evaluation using the actual sampler, predicted scores, and normal postprocessing, rather than supplying GT identities or geometry |
| Conditional recovery | Predicting from GT-derived, possibly corrupted inputs; useful diagnostically, but easier and different from image-only detection |
| NMS | Postprocessing that suppresses overlapping/duplicate pose predictions; changing scores can change its survivors |
| Hungarian / OTA | Different target-assignment strategies; Hungarian uses one-to-one matching, while this project's Dynamic-K OTA can supervise multiple predictions per person |
| EMA target | A slowly updated copy of an encoder used to supply training targets; its target quality and stability still require measurement |
| DN queries | Extra training queries with known perturbed-GT correspondence; their presence alone does not establish successful diffusion sampling at inference |

## 2. The history as a research trajectory

### 2.1 A functioning reproduction and a meaningful reduced baseline

The supplied DiffusionRegPose ResNet- 50 checkpoint evaluated at approximately 72.68 pose AP in the native implementation. This established a working released-checkpoint inference/evaluation path. Training reproduction remained below that level: the external-code 80-epoch run reached approximately 68.80, and the final native full-model standalone S 1 evaluation was approximately 67.38 under its historical scoring contract. This is a successful inference reproduction with an unresolved training gap, not a reproduced 72.7-AP training result.

The first compact model was much faster but learned very poorly. Preserving ED-Pose-compatible computation and explicitly transferring selected layers produced a useful development model. In the original matched reduced Hungarian/OTA comparison, full-test pose AP improved from 51.57 to 54.90 under the then-current scoring path. In the later common corrected-evaluator sweep, the corresponding K1 results are 51.80 and 54.96. The latter values should be used together; the former pair documents the original controlled experiment.

This is already a real positive result for pose-aware many-to-one supervision in the reduced setting. It is not strong novelty by itself, because changing assignment/cardinality is an established approach, and this project has not established its generality across training seeds or paper-scale training. It is nevertheless important: there is a capable baseline to compare against, and later failures are not all attributable to a broken dataset or evaluator.

Sources: [original chronology](/home/gasparyanartur/dev/pose-graph-diff/reports/diffusionregpose-to-poseflow-chronology.md), [matched assignment report](/home/gasparyanartur/dev/pose-graph-diff/reports/diffusionregpose-ota-comparison.md), [corrected reduced-model sweep](/home/gasparyanartur/dev/pose-graph-diff/tmp/reduced_step_experiments/summary.md).

### 2.2 Iteration rarely turned into sustained refinement

The corrected reduced-model sweep is one of the clearest summaries of the earlier project. All entries below use full 8,000 TEST, resize 256/427, per-image seed 42, FP 32/TF 32 off, and the corrected native postprocessor. The models and training recipes differ; compare each model's curve before attributing cross-model differences to any one component.

| Model | Checkpoint epoch | K1 pose AP | K2 | K3 | K4 | K5 | K8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reduced DiffusionRegPose, Hungarian | 79 | 51.80 | 51.86 | 51.76 | 51.75 | 51.65 | 51.57 |
| Reduced DiffusionRegPose, OTA | 79 | 54.96 | 54.88 | 54.93 | 54.84 | 54.91 | 54.94 |
| PoseFlow V1 | 79 | 55.11 | 55.10 | 55.10 | 55.10 | 55.10 | 55.10 |
| V3 | 79 | 56.95 | 54.84 | 39.04 | 29.84 | 28.01 | 28.05 |
| V5 | 79 | 55.95 | 56.07 | 56.08 | 56.06 | 56.01 | 56.00 |
| V6, DINO-conditioned | 79 | 56.71 | 56.84 | 56.84 | 56.89 | 56.88 | 56.84 |
| V19 full geometry detach | 79 | 54.13 | 55.01 | 55.25 | 55.30 | 55.32 | 55.31 |
| V19 dense read | 67 | 53.66 | 54.45 | 54.72 | 54.75 | 54.79 | 54.73 |
| V20, 3D-guided training | 44 | 48.99 | 49.95 | 50.23 | 50.22 | 50.01 | 49.57 |

Dense-read V19 and V20 were stopped at different ages. Their rows are available-checkpoint results, not a completed matched test of 3D guidance. V6's modest improvement also does not isolate DINO from all recipe differences relative to V5.

The paper-scale history prevents an overly broad conclusion that diffusion never helps. The native full-resolution DiffusionRegPose final checkpoint improved from approximately 67.38 at K1 to 68.77 at K2 in its historical evaluation; five paired inference seeds all showed a positive K2 change, with mean gain 1.28 AP points. That tests sampling-seed variation for one trained checkpoint, not five training runs. Paper-scale V5 was nearly flat around 68.13–68.17 over K1–K8. These are separate settings from the corrected reduced table.

The question that follows is narrower than whether diffusion or flow is intrinsically useful: **what computation accounts for the observed gain, plateau, or collapse in each trained estimator?**

### 2.3 Latent modelling added capability but exposed additional failure modes

V26–V39 explored learned latent pose/box targets, task-gradient routes, DINO supervision, matching, normalization, and variational constraints. Some versions reconstructed clean GT inputs well while Gaussian-start generation stayed weak. A matched completed comparison gave:

| Model, epoch 79/full 8,000 TEST | K1 pose AP | K4 pose AP | K1 box AP | K4 box AP |
|---|---:|---:|---:|---:|
| Original V26 latent | 0.63 | 7.33 | 3.39 | 14.71 |
| V31, recovered task gradients restored | 0.64 | 7.95 | 3.37 | 15.17 |
| V32, dual DINO supervision | 0.49 | 7.73 | 2.66 | 15.60 |

The DINO target was being learned in V32: validation cosine was approximately 0.80, with positive correct-versus-wrong-person discrimination. This did not close the gap between clean reconstruction and image-only generation. It is evidence against treating auxiliary feature agreement as sufficient evidence of successful pose inference.

V40 restored explicit image-proposal conditioning and reached 48.00 full-test K1 box AP, but fell to 25.83 at K4. V42 restored a reasonable whole-pose coordinate baseline, while its pairing, confidence, and JEPA controls revealed that several plausible changes could reduce multicall quality. V43/V44 separated codec quality from denoiser quality. V45–V48 then explored progressive or explicitly ordered person generation. These later branches make the failure mechanisms easier to distinguish, but have not supplied a competitive new endpoint.

Sources: [completed V26/V31/V32 comparison](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/2026-10-27-v26-v31-v32-final-comparison.md), [V40 state audit](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v40_state_audit.md). Later endpoints are given in Insight 9 and Appendix A.

## 3. The analytical framework

A useful simplified description of one call is:

\[
(\hat y_t,c_t)=f_\theta(I,s_t,t,h_t),\qquad
(s_{t'},h_{t'})=T(s_t,\hat y_t,h_t),\quad t'<t.
\]

Here \(I\) is the image, \(s_t\) the sampled coordinate/latent state, \(h_t\) an optional carried hidden representation, \(\hat y_t\) decoded geometry, and \(c_t\) detection scores. Each architecture uses a different subset of these arguments. The final metric is approximately:

\[
\operatorname{AP}\bigl(\operatorname{Postprocess}(\hat y_t,c_t)\bigr).
\]

Consequently, a change in AP can reflect different effects:

1. Better or worse raw geometry and person coverage.
2. Better or worse confidence ordering for that geometry.
3. Changed NMS survivors caused by geometry or scores.
4. Changed person correspondence or candidate selection.

These effects are not additively separable in general. A score swap is a controlled counterfactual, not a universal percentage decomposition of a total AP gap. Likewise, dependence on a state does not establish that the state contains useful corrective information.

The history supports at least three empirically distinct regimes:

| Regime | Example | Discriminating observation |
|---|---|---|
| State contributes very little | Native reduced DiffusionRegPose; an early uniform-t V42 checkpoint | Large state changes cause tiny prediction changes |
| State changes ranking badly while useful geometry remains | V40 and paired V42 | Holding geometry fixed and replacing scores recovers AP |
| State/denoiser damages the geometry itself | V26 latent-only, V44, ordered generated trajectories | Raw/paired geometry worsens, including with confidence effects excluded |

A coherent report should establish these regimes and their boundaries. It should not force them into a claim that one bug explains every unsuccessful experiment.

## 4. Ten concrete insights

### Insight 1. Generated-state feedback can destroy confidence ranking before it destroys useful geometry

**Phenomenon.** The model produces enough useful geometric candidates to support substantially better AP than its native scores obtain. Repeated calls alter the ranking of those candidates so strongly that measured detection quality collapses.

**Evidence.** The frozen V40 epoch 79 audit uses 32 TEST images, fixed image/proposal slots, seed 42 plus image ID, resize 256/427, all 50 boxes, and no NMS. It reproduces the saved native metrics exactly.

| V40 condition | Fixed 32 box AP |
|---|---:|
| Native K1 | 43.29 |
| Native K4 | 20.62 |
| K4 geometry with K1 scores | 43.42 |
| K1 geometry with K4 scores | 20.36 |
| Zero denoiser input at every K4 call | 42.82 |

Across K1–K4, average box-center movement is only 1.14 resized pixels, while average absolute probability change is 0.621. Queries above score 0.5 increase from 2.66 to 49.75 per image; there are about 4.16 GT people per image. The geometry/score swaps directly establish a large score-induced AP penalty. The count inflation describes the trajectory, but is not itself a calibrated false-positive measure.

Paired V42 replicates the phenomenon for pose estimation. At epoch 31, replacing native pose-conditioned confidence with an already-trained image-based pre-pose score raises fixed32 K4 pose AP from 13.04 to 35.53 and K8 from 10.21 to 35.27. Raw boxes/joints are exactly preserved. The normalized classifier scores are gathered using the actual selected query indices; no GT or learned score fit is used.

**Explanation.** A plausible training mechanism is that pose-like generated states become misleading evidence of existence. Background slots have geometry predictions too, and the sampler feeds those predictions back. A classifier can then become confident in its own generated outputs. This explains the observed pattern, but the frozen intervention does not isolate why the classifier learned it.

**What weakens the angle.** The rescored paired V42 still trails the same-age ordinary uniform-t control: at epoch 31, K4 is 35.53 versus 42.42. The subsequently completed image-only-confidence training run also trails uniform-t on full TEST, 39.38 versus 50.87 at K4. Therefore the score intervention repairs a failed variant; it has not established a superior estimator. V40 is box-only, so it cannot alone establish full-pose behaviour.

**Potential angle.** A study of feedback-induced ranking failure, with a frozen, non-oracle scoring intervention and an explicit strong-control comparison. This is the highest-confidence two-week research option because the core causal intervention already exists in two architectures.

**Smallest useful follow-up.** Extend V40's four score/geometry conditions and V42's native/pre-pose conditions to the full fixed evaluation set, with paired image/noise draws. Retain raw coverage, pose/box AP, AR, NMS counts, and query provenance. Use the uniform-t model as an essential performance reference. If the ranking effect does not generalize beyond fixed32, report that limit and stop proposing it as a general repair.

Sources: [V40 audit](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v40_state_audit.md), [paired V42 multicall audit](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v42_paired_multistep_audit.md), [V42 score intervention](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v42_image_confidence.md).

### Insight 2. Reasonable pose AP can coexist with almost unused diffusion-state variation

**Phenomenon.** An architecture can contain a diffusion-state route without learning a substantial state-dependent geometric update in the tested regime.

**Evidence.** The native reduced DiffusionRegPose study uses the final selected-layer OTA checkpoint, 32 TEST images, five paired initial-state seeds, native trajectory states, and exact replay checks. Input differences of 48.6–96.0 pixel-equivalent units cause only 0.0116–0.0156 pixels of final joint movement. All 336,000 observed incoming first-pose joint references have negative X and Y; clamping maps them to the same boundary coordinate. Positional routes still retain some reference information, so this is weak dependence rather than exact independence.

The first-pose joint cross-attention output equals its projection bias in all 480 observed calls. Replacing just that output with the bias changes nothing. Removing the stage's joint self-attention/FFN updates, however, drives AP to zero. Removing the final pose stage's joint image values reduces final-call AP from 50.42 to 18.45. The stage is doing necessary computation even though its first joint image-value route is inactive.

An independent early uniform-t V42 probe is consistent with image-dominated prediction: epoch 11 fixed32 K1 AP is 32.83 with native coordinates, 32.77 with zero coordinates, and 32.74 with another Gaussian seed. These are checkpoint-specific measurements; late t0-only V42 learned substantial input-pose dependence and then failed on Gaussian inputs, showing that the behaviour changes with recipe and age.

**Explanation.** Strong image/proposal features can solve much of the task while the injected state route contributes little. A clamp can additionally erase a particular geometric path. This is an implementation/learned-computation explanation for the tested model, not evidence that diffusion is generally redundant.

**What weakens the angle.** The decisive route measurements are from one reduced checkpoint. A full-resolution released model may behave differently. Conversely, a small observed AP change after zeroing a state does not establish that every local derivative or every meaningful perturbation is small. Abrupt out-of-distribution interventions must be paired with native seeded trajectory comparisons.

**Potential angle.** A state-use audit for iterative pose prediction: identify whether extra calls exploit the evolving state or repeatedly solve the same image-conditioned regression problem. Its value comes from exact intervention/replay and validation on credible released or full-size checkpoints.

**Smallest useful follow-up.** Replicate native-seed sensitivity, fixed-image state replacement, clamp/reference statistics, and stage-route ablations on the supplied checkpoint and the final native full-resolution checkpoint. Report physical coordinate movement and raw pose coverage alongside AP. Preserve the first-stage feature computation when removing the inactive image-value route.

Source: [integrated image/state investigation, Sections 4 and 5](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/2026-10-24-image-information-report.md). Historical timestep/source caveats are also recorded in [V42 collapse audit](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v42_box_center_collapse.md).

### Insight 3. The backward boundary between geometric refiners has a large learned effect

**Phenomenon.** Stopping only coordinate sampling references is insufficient to recover the benefit of stopping the full inter-layer geometric state. The remaining coordinate-residual gradient route materially changes optimization.

**Evidence.** Three V19 recipes share architecture, 42,008,567 parameters, source initialization, data/optimizer/schedule, batch 4/accumulation 2, clip 2, and the single-call/rollout mixture. At epoch 24/37,500 updates on the same fixed32 TEST cohort:

| Gradient boundary | K1 pose AP | K3 pose AP | K1 box AP |
|---|---:|---:|---:|
| Original | 22.55 | 20.75 | 29.78 |
| Sampling-reference detach | 28.79 | 27.13 | 36.14 |
| Full inter-layer geometry detach | 42.75 | 46.14 | 41.07 |

At fixed weights, switching the full/sampling boundary leaves all stage outputs and losses exactly unchanged. With the sampling route stopped, consider logit updates \(z_1=z_0+d_1\), \(z_2=z_1+d_2\), \(z_3=z_2+d_3\). If \(g_i\) is the stage-i geometry-loss derivative with respect to \(z_i\), the direct residual route gives early heads terms proportional to \(g_1+g_2+g_3\), then \(g_2+g_3\), then \(g_3\). Full geometry detach removes those direct future-coordinate terms while retaining feature-level and cross-call learning.

The first box head's total/own gradient-norm ratio is 2.87 with sampling-only detach versus 1.00 with full detach; the first joint head is 1.87 versus 1.00. In frozen full-image probes, geometry-gradient norms rise 1.62–2.78 times while confidence gradients change by only about 0.01–0.02%. Fixing final assignments across stage losses still leaves opposing coordinate-gradient components, so OTA identity changes are not the main explanation for the observed conflict.

**Explanation.** Early coordinate heads are asked to correct errors both before and after later heads have applied their own corrections. Those requests can disagree. A simple scalar example is an early prediction below the target and a final prediction above it: the early loss asks for an increase, while the final residual route asks for a decrease. This changes the gradient field and the effective balance of geometry/confidence training even though scalar loss weights and forward computation match.

**What weakens the angle.** Opposing gradients are not inherently wrong; they are legitimate derivatives of a composed objective. The comparison does not isolate directional conflict from implicit weighting, clipping, AdamW history, or rollout effects. Clip multipliers cannot be interpreted as equal changes in AdamW parameter steps. The learned AP comparison is one seed and fixed32; the frozen diagnostic checkpoints have different ages, appropriately used only for within-checkpoint comparisons.

**Potential angle.** A pose-specific empirical study of coordinate-level versus feature-level supervision through iterative refiners. This is the largest clean training effect in the reviewed history.

**Smallest useful follow-up.** Repeat the three declared boundaries at matched training ages on one additional training seed, with a frozen full-set evaluation. Reuse the exact-forward gradient probe. Do not add clipping or loss-weight changes to the initial replication. If a new explanatory control is affordable afterward, normalize accumulated coordinate contributions separately from stopping them; this would test whether magnitude alone explains the effect.

**Novelty boundary.** Reference detachment is established in Deformable DETR, while DINO deliberately restores a controlled adjacent-layer gradient route through its look-forward-twice update. Our result cannot be framed as discovering that detachment helps in general, nor as disproving DINO. Its possible contribution is locating which backward paths matter in this pose/rollout setting and measuring why a sampling-only stop is insufficient. See [DINO, Section 3.5](https://arxiv.org/html/2203.03605v4#S3.SS5).

Source: [original gradient investigation](/home/gasparyanartur/dev/pose-graph-diff/tmp/v19_sampling_analysis/findings.md).

### Insight 4. Memory dependence is substantially larger than the net gain from iteration

**Phenomenon.** Resetting a recurrent model's state can be disastrous even when carrying that state adds little accuracy beyond the first call.

**Evidence.** Final V19 full-detach on full 8,000 TEST gives 54.13 pose AP at K1,55.25 at K3 with persistent memory, and 45.29 at K3 with reset memory. The persistent/reset gap is 9.96 points; the K1-to- K3 improvement is only 1.12 points. Its best tested pose AP is 55.32 at K5, only 0.08 above K3. V5 remains ahead across matched call counts, reaching 56.08 at K3.

The same distinction recurs in other variants. V19 dense-read epoch 67 has 54.72 at K3 persistent versus 42.07 reset; V20 epoch 44 has 50.23 versus 40.27. Neither establishes that the added memory/dense read/3D objective creates a better endpoint than the stronger image-conditioned reference. Earlier V15 writer investigations also found nearly identical slot messages and little effect from replacing the carried slot bank with its mean in specific checkpoints; tying memory to candidates changed the contract rather than proving useful retrieval automatically.

**Explanation.** A model trained to expect carried state can use it to maintain consistency or score scale. Resetting is then a distribution shift. Such dependence is real, but it is a different comparison from training an equally capable model without the memory route. The large reset penalty cannot be advertised as a ten-point improvement over a memory-free baseline.

**What weakens the angle.** Different states may carry geometric and scoring information jointly, and a small net AP change could hide beneficial corrections offset by other errors. The current evidence supports a plateau and a strong reset effect; it does not uniquely identify what every memory dimension represents.

**Potential angle.** Establish the difference between maintaining a stable recurrent trajectory and accumulating new useful evidence. This strengthens the interpretation of the confidence/state study more than it justifies another memory architecture.

**Smallest useful follow-up.** On one final V19 checkpoint, combine persistent/reset/mean/shuffled memory with fixed-geometry score comparisons and score-free one-to-one coverage. Separate each perturbation from coordinate-state changes. If a state changes ranking but not best achievable geometric coverage, describe it as a scoring/consistency role rather than geometric recovery. A matched trained no-memory arm would be required for a causal claim about the value of learning with memory.

Sources: [full V19/V5 comparison](/home/gasparyanartur/dev/pose-graph-diff/tmp/v19_v5_full_steps/findings.md), [reduced family/memory sweep](/home/gasparyanartur/dev/pose-graph-diff/tmp/reduced_step_experiments/summary.md), [V19 deficit diagnosis](/home/gasparyanartur/dev/pose-graph-diff/tmp/v19_v5_deficit_audit/findings.md).

### Insight 5. Lower latent MSE can hide a severe loss of geometric accuracy

**Phenomenon.** Improving average error in a learned representation can make detection worse when the error in task-relevant directions is not controlled.

**Evidence.** The matched V35 spatial experiment changes the recovered decoded-task coefficient from 0.5 to 0 while preserving the remaining recipe and initialization. At epoch 11/12 completed epochs:

| V35 condition | Clean box L 1 | Recovered box L 1 | Latent MSE | Fixed 32 K1 box AP | K4 box AP |
|---|---:|---:|---:|---:|---:|
| Decoded supervision retained | 0.0420 | 0.1006 | 0.02499 | 15.26 | 14.03 |
| Latent-only denoiser | 0.0423 | 0.2554 | 0.02266 | 3.98 | 3.10 |

Clean codes retain almost the same reconstruction accuracy. The denoiser's latent MSE is lower, but its decoded box error more than doubles and K4 AP loses 10.93 points. Score-free IoU 0.5 coverage falls from 87/133 people to 38/133 at K4, establishing a geometric failure beyond changed confidence thresholds.

A frozen analysis provides a concrete reason to distrust the scalar loss. The linear box/existence readout uses at most five latent directions before its nonlinear output functions. Correcting error in that readout subspace restores assigned-target decoding; correcting the complementary subspace leaves box outputs essentially unchanged. At high noise, about 67% of the examined latent-only checkpoint's error lies in the readout-relevant subspace. It would be wrong to claim that most of its error is harmless: the probe shows that the remaining error is strongly concentrated where it matters.

V26 latent-only independently demonstrates the weakness of a diversity-based explanation. Its K64 generated latent variance recovers to approximately the clean reference, but fixed32 pose AP stays around 0.62 and raw person coverage is worse than at K4. Spread has returned without accurate person recovery.

**Explanation.** A simple analytical observation helps. Let a code be reparameterized as \(z'=Az\), with decoder \(R'(z')=R(A^{-1}z')\). Decoded outputs remain identical, but Euclidean error becomes \(\|A\Delta z\|^2\). Its numerical meaning depends on representation coordinates. This identity does not prove the cause of a particular failed run; it explains why a raw latent MSE has no automatic geometric calibration.

Locally, decoded-coordinate squared error behaves approximately like \(\Delta z^T J_R^T WJ_R\Delta z\), for a stated coordinate weighting \(W\). This is a Taylor approximation, not an exact expression for native AP, focal loss, or all OKS behaviour. It suggests measuring decoder sensitivity rather than choosing larger MSE coefficients blindly.

**What weakens the angle.** Removing the decoded objective changes several training routes together, including denoiser learning and readout adaptation. The probes do not isolate every route. Decoder-aware losses are also an established general idea; an unqualified novelty claim would be weak.

**Potential angle.** A practical diagnostic result: identify which latent-error measurements predict downstream pose failure across checkpoints and noise levels. If Euclidean MSE systematically ranks failed models above better geometric models while decoded or sensitivity-weighted errors recover the correct ordering, that is an interpretable finding with reusable evaluation code.

**Smallest useful follow-up.** Use frozen V26/V35 and fixed-codec V43/V44 checkpoints. Measure latent MSE, latent variance, paired geometry, and decoder-weighted errors on the same examples. Test correlation and ranking on images/checkpoints excluded from any choice of metric weights. Begin with evaluation; a new loss/training recipe is a separate possible experiment, not required to establish the diagnostic.

Sources: [decoded-loss audit](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/2026-10-29-decoded-loss-and-v36-audit.md), [V26 latent-only failure audit](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/2026-10-25-v26-latent-only-failure-audit.md).

### Insight 6. Accurate representation, stable reconstruction, and useful denoising are separate capabilities

**Phenomenon.** A codec can encode correct articulation accurately while a learned denoiser maps almost-correct codes to substantially worse poses. An identity-preserving parameterization can avoid some destruction without learning a transferable correction.

**Evidence.** V44 epoch 47 was probed on 16 fixed VAL images/63 matched people with a frozen codec and fixed assignment:

| Conditional readout | Paired OKS, 0–1 | Paired box IoU |
|---|---:|---:|
| Codec posterior mean | 0.9900 | 0.9876 |
| Direct decode after 1% latent noise | 0.9868 | 0.9810 |
| Denoised output at 1% | 0.7663 | 0.8666 |
| Denoised output, replace box with GT | 0.7978 | 1.0000 |
| Denoised output, replace relative articulation with GT | 0.8925 | 0.8666 |
| Direct decode after 5% latent noise | 0.9174 | 0.9350 |
| Denoised output at 5% | 0.7642 | 0.8684 |

The codec can represent these poses; a better codec decoder is not the first explanation supported by this probe. Articulation contributes strongly to the recovered error. The denoiser also responds to changed latent inputs, so complete input ignorance is not supported in this V44 diagnostic.

The eight-arm fidelity experiment had already established high clean codec fidelity. The unmasked High-KL VAE achieved approximately 97.21 sampled pose AP and 98.38 box AP on full VAL, with posterior-mean pose AP 97.25. That is GT-conditioned reconstruction, not a detector result. Its apparent robustness advantage depends on how perturbation size is normalized: at 5% of each model's own latent standard deviation it improves over the deterministic control, but the same dominance does not hold under a shared initial-reference perturbation scale.

The V43 small-noise controls separate stability from learning. On 16 VAL images, velocity prediction approximately retains raw-noisy OKS at 1%, 2.5%, and 5%; absolute-x 0 prediction reduces it to about 0.61. However, the velocity model's positive-noise improvement is tiny and fails the registered learning gate. A one-image foreground model later improves same-image unseen-noise OKS from 0.8898 to 0.9764, demonstrating learnability in that restricted setup. On 16 otherVAL images it instead reduces OKS from 0.9205 to 0.7232. It uses image conditioning but does not transfer usefully.

**Explanation.** Three distinct questions must be answered: can the state represent an accurate pose; does the denoising transformation preserve already accurate information; and does correct image conditioning support a correction that generalizes to new people/images? A clean reconstruction score answers only the first. An identity skip can answer part of the second. Neither supplies the third automatically.

**What weakens the angle.** The frozen codec has its own data exposure and crop-related error tails. In a 512 TRAIN-image crop audit, rare tiny boxes cause very large codec losses. Those are real confounds, but their causal gradient contribution was not measured. Similarly, qualitative malformed standard-normal VAE samples do not prove that a separately trained diffusion prior cannot model the standardized latent distribution. Codec prior quality and downstream diffusion learning must remain distinct.

**Potential angle.** A staged empirical test for pose latent diffusion, with raw-noisy decoding and identity preservation as mandatory baselines. The result would explain why very high reconstruction AP and successful tiny-set denoising failed to predict useful detection.

**Smallest useful follow-up.** Freeze one codec and compare identity, x0, and velocity outputs at matched positive noise on a fixed held-out population. Require a correct-versus-wrong-image conditioning advantage and retained low-noise geometry. If these tests fail, do not launch another Gaussian-start full detector under the assumption that the codec alone makes the state space suitable.

Sources: [V44 plateau](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v44_pose_plateau.md), [codec fidelity comparison](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/pose_vae_fidelity.md), [small-noise trials](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v43_small_noise.md), [conditioning/transfer diagnosis](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v43_thesis.md).

### Insight 7. Generated-state exposure improves trajectory recovery at a measurable preservation cost

**Phenomenon.** The denoiser performs differently on forward-corrupted GT states and states produced by its own sampler, even at the same timestep. Training on generated states reduces that mismatch but does not preserve all other capabilities.

**Evidence.** The frozen V44 trajectory diagnosis uses one cached context/assignment per image and independently checked DDIM arithmetic. For the final x0 continuation at t249, forward-corrupted input produces 69.49 OKS points, while generated input produces 57.69. Foreground latent MSE rises from 0.0438 to 0.2194. The source checkpoint shows a similar gap, and generated-state MSE exceeds forward-state MSE on all 32 images at that time. Independent sampler replay and oracle checks pass; no arithmetic discrepancy was found.

The follow-up starts both arms from the same V44 epoch 47 model. Half the batches use ordinary uniform-time corruption; half use a matched K4 intermediate time. The control receives forward-corrupted GT codes. The candidate receives a detached state obtained through its current denoiser's no-gradient warm-up. Context, assignment, time/noise mixture, optimizer-update budget, and first augmented batches are matched. Codec and inference sampler remain fixed.

| Selected-checkpoint full 2,000 VAL result | Forward control | Rollout exposure | Change |
|---|---:|---:|---:|
| K1 pose AP | 28.08 | 27.65 | −0.43 |
| K4 pose AP | 24.95 | 28.36 | +3.41 |
| K1 box AP | 43.80 | 43.06 | −0.74 |
| K4 box AP | 43.45 | 44.63 | +1.19 |

Both selected checkpoints are epoch 0 after 1,250 continuation updates; each arm's full budget is 2,500 updates. Matched-age last-checkpoint fixed32 trajectories show:

| t249 output quality, OKS points | Forward control | Rollout exposure |
|---|---:|---:|
| Forward-corrupted state | 70.37 | 64.18 |
| Generated state | 55.72 | 59.89 |
| Forward/generated gap | 14.66 | 4.29 |

The generated-state improvement is real. The smaller gap is partly obtained by degrading the forward-corrupted branch, however. Small-noise preservation worsens, and the final matched-age fixed32 K4 box AP is lower. The candidate fails the combined full VAL, matched-age, and small-noise gates; only the trajectory gate passes.

**Explanation.** Forward-corruption training does not expose the model to the same errors, identities, and background geometry produced by its own predictions. Adding exposure changes the fitted input distribution. A finite model/optimization budget can trade performance across those distributions. This comparison supports that tradeoff for this continuation, not an impossibility theorem about scheduled sampling or rollout training.

**What weakens the angle.** It is a short, single-seed continuation from an already trained model. Equal optimizer steps are not equal compute because warm-ups add calls. Different loss weighting, exposure frequency, or longer training could change the outcome. Those possibilities do not justify calling the current combined gate a success.

**Potential angle.** A measured tradeoff between generated-trajectory robustness and near-clean/forward-distribution fidelity. A successful next intervention would need to improve the former while explicitly protecting the latter.

**Smallest useful follow-up.** Only if this becomes the chosen closeout: one matched continuation retaining a separately normalized forward/low-noise preservation objective alongside generated-state supervision. Predeclare an allowable K1/box tolerance and positive-noise geometry tolerance; accept only simultaneous retention and generated-state gain. Do not expand to another architecture when the gate fails.

Sources: [trajectory diagnosis](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v44_velocity.md), [rollout protocol](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v44_rollout.md), [completed measured comparison](/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v44_rollout/20261107_030445/comparison.json).

### Insight 8. A hidden-arm distinction is accessible in image features but unused by the native coordinate output

**Phenomenon.** The image representation can preserve task-relevant distinctions even when the native pose reader fails to localize the corresponding occluded joint.

**Evidence.** Pinned same-age V24 control/h models were tested on 512 new synthetic hidden-arm families. The visible version provides a competence control: person detection succeeds in 1,024/1,024 hidden-condition images, and visible-wrist localization succeeds in 1,023/1,024 control images and 1,024/1,024 h images. With the occluder present, both models localize 0/1,024 wrists correctly.

A newly supervised frozen-feature reader can nevertheless distinguish the paired hidden-arm alternatives:

| Reader input, all 512 test families | Control, both alternatives correct | h model |
|---|---:|---:|
| Projected backbone map | 36.7% | 83.4% |
| Encoded image map | 86.3% | 96.9% |
| Encoded image map, changed appearance | 83.0% | 98.2% |

The all-family map analysis was added after the registered token-based reader failed its 90% coverage admission criterion: only 156/512 families retained the required token correspondences. It is useful evidence, but must be labelled post hoc rather than presented as the clean registered primary endpoint. The registered restricted-cohort results are retained in the source report.

The final matched detector pair has only a small full-test effect: control 54.914 versus h55.295 pose AP, a 0.381 point advantage for one training seed. Accessible hidden-arm information and aggregate detector AP therefore change by very different amounts.

**Explanation.** The native estimator may not have learned a readout or objective that converts the available visual distinction into the required hidden coordinate. This is a stronger hypothesis than simply saying the image has no evidence. However, binary short/long-arm classification does not establish that complete accurate wrist coordinates are present in a linearly decodable form.

**What weakens the angle.** Synthetic construction can introduce shortcuts. The attempted ownership task failed native-control admission and cannot support an ownership result. The depth task is ceiling/specificity limited: pretrained features already achieve 100% ordering and outside-arm pixels support 97.5% pair ordering. These findings prevent a claim of newly acquired 3D reasoning or reliable person ownership. The hidden-arm result also needs an actual coordinate readout before being described as a recovered pose capability.

**Potential angle.** A representation-to-output bottleneck study for occluded joints. This could be a focused result about accessible evidence, reader choice, and the failure of native regression under controlled occlusion.

**Smallest useful follow-up.** Freeze the same estimators and train a small coordinate or coordinate-residual reader on separately constructed training families. Keep family-level held-out splits, visible controls, changed appearance, and outside-arm controls. Compare native coordinates, native-output-only reader, image-feature reader, and image-feature-plus-h reader. A consistent held-out coordinate gain beyond those controls is a positive result; a failed coordinate reader narrows the existing claim to class-distinction accessibility.

Source: [integrated information study, Sections 3, 5, 7 and 9](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/2026-10-24-image-information-report.md).

### Insight 9. Task-pose JEPA reduces repeated-call damage, but the benefit is formulation-dependent

**Phenomenon.** A reconstructable task-defined target can provide a useful auxiliary signal in one coordinate-diffusion architecture even though related latent/JEPA additions fail elsewhere.

**Evidence.** The completed V45 task-pose pair uses identical original initialization/common inference tensors, same teacher construction and reconstruction objective, same data/optimizer recipe, epoch 79/120,000 updates, and the same 8,000 TEST images, seed 42, resize 256/427 and native coordinate DDIM. The JEPA coefficient is the intended training difference.

| Full TEST metric | Control | JEPA | JEPA minus control |
|---|---:|---:|---:|
| K1 pose AP | 42.77 | 42.03 | −0.74 |
| K4 pose AP | 32.87 | 35.98 | +3.11 |
| K8 pose AP | 30.17 | 33.15 | +2.98 |
| K1 box AP | 50.89 | 51.15 | +0.26 |
| K4 box AP | 50.93 | 51.19 | +0.26 |
| K8 box AP | 51.02 | 51.24 | +0.22 |

K1-to- K4 pose deterioration changes from 9.90 points in control to 6.05 in JEPA. JEPA reduces the deterioration by 3.85 points, which is different from its 3.11 point cross-arm K4 gain because the K1 baselines differ.

The target encoder maps complete clean XY poses into 32 dimensions; a shared reader reconstructs absolute joint coordinates. Both arms train online/EMA clean reconstruction, and only JEPA adds final-code alignment. Coordinates advance through the sampler. This is coordinate diffusion with latent supervision, not latent diffusion. The auxiliary terms use complete augmented poses; partially labelled people still receive masked native supervision.

V42 JEPA has the opposite observed endpoint relative to its parent: K4 pose AP falls 39.38→34.32 and K8 falls 31.58→23.24. That comparison adds both latent alignment and clean reconstruction, whereas the V45 pair already shares reconstruction in both arms. It is therefore evidence that the V42 combined intervention failed, not a controlled proof that architecture alone determines the sign of a JEPA effect.

**Explanation.** The V45 target is explicitly constrained to retain pose information through reconstruction, while the reader and gradient routes differ from previous formulations. It may regularize a representation used by the iterative update and reduce articulation drift. That is a hypothesis requiring geometry/state probes. The AP table alone does not identify whether improvements arise through decoded geometry, score/NMS interactions, or optimization changes.

**What weakens the angle.** There is one training seed. Both JEPA and control remain worse at K4/K8 than their own K1. The best JEPA result does not approach the stronger reduced baselines. Historical TEST development exposure is substantial. Consequently, the existing result is a positive auxiliary-loss ablation, not evidence that we have achieved useful accuracy scaling with test-time computation.

**Potential angle.** A narrowly framed positive result: reconstructable pose-target alignment reduces accumulated error under repeated coordinate prediction. It is a credible second-seed replication candidate if a positive training result is the desired endpoint.

**Smallest useful follow-up.** Repeat the matched pair with one new training seed and identical fixed evaluation draws. Diagnose pose errors with scores fixed within a stated matched-slot protocol, and compare correct/zero/swapped incoming states and low-noise preservation. Keep the no-JEPA arm's teacher/reconstruction/RNG construction intact. Accept only a repeated positive K4 effect with stated K1 tolerance and a verified geometric interpretation; do not tune new JEPA weights on TEST.

Sources: [task-pose contract](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v45_task_pose.md), [control final metrics](/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v45_task_pose/20261107_154258/control/seed42/full_test/metrics.json), [JEPA final metrics](/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v45_task_pose/20261107_154258/jepa/seed42/full_test/metrics.json), [V42 JEPA final metrics](/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v42_jepa/20261105_221931/seed42/full_test/metrics.json).

### Insight 10. Ordered generation fails at both the initial birth mapping and subsequent self-generated refinement

**Phenomenon.** The recent ordered models do not merely accumulate errors from imperfect prefixes. Their first prediction, which receives no generated prefix, is already weak; subsequent generated refinement can make it substantially worse.

**Evidence.** V46's final epoch 79 fixed32 VAL diagnostics track the same first canonical-rank target:

| V46 rank 1 condition | Latent MSE | Geometry loss |
|---|---:|---:|
| First birth | 0.3344 | 11.40 |
| Final teacher-forced row | 0.0108 | 0.30 |
| Final generated row | 0.7919 | 35.07 |

The teacher-forced final row receives GT-derived state under its registered policy. It therefore measures conditional recovery, not a successful rollout. Both the initial image-to-person prediction and the generated trajectory are poor. An evaluation-only DDIM replacement for independent re-noising failed to rescue the earlier epoch 46 checkpoint, with fixed32 pose AP 0.2115→0.1043.

V47 moves to a jointly learned coordinate codec and coordinate DDIM. The four-image training gate reaches 98.80 pose AP and 91.73 box AP under both inference seeds 42/43, with exact counts. These are two inference seeds on one memorized training setup, not two independently trained models. After full training, its selected epoch 25 checkpoint obtains 0.0362 pose AP on all 2,000 VAL images. V46's selected epoch 76 checkpoint obtains 0.7297. These are selected-checkpoint full VAL results, not final-epoch results, and the architectures differ.

V48 removes noise/time/refinement and keeps clean fixed prefixes and birth-only supervision. As refreshed for this report, baseline epoch 68 fixed32 VAL pose AP is 1.027 and box AP 7.983; AdaLN epoch 44 is 0.884/7.216. These are interim, unequal-age snapshots. At matched epoch 39, baseline/AdaLN pose AP is 0.989/0.828: there is no demonstrated rescue on this small cohort, but no completed full-run comparison is claimed.

V48 baseline epoch 68 rank 1 coordinate MSE is 0.02308 with either teacher or generated evaluation because the first prefix is empty. Rank 2 teacher-prefix MSE is 0.01383 versus 0.02236 with generated prefix. This identifies an initial prediction problem plus an additional prefix-dependent error. It does not establish their respective contribution to total AP, since emitted populations and NMS also matter.

**Explanation.** Canonical order removes a set-matching ambiguity but creates a particular conditional prediction problem: which person is next, and how is that person's full articulation inferred from the image and previous outputs? A deterministic coordinate reader may average uncertain alternatives, or the image-to-rank mapping may be inadequately learned. Those are hypotheses. Neither the low AP nor the first-rank loss proves averaging without inspecting predictions and appropriate single-person controls.

**What weakens the angle.** The dense row decoder, absence/change of native proposal paths, fresh decoder components, loss allocation, coordinate representation, and supervision differ from stronger baselines. A four-image overfit gate does not isolate which of those changes causes failed generalization. Box-only annotations complicate box ceilings. Current V48 runs are still active; their outcomes must be archived before declaring final failure.

**Potential angle.** A diagnostic study separating birth, conditional refinement, prefix exposure, and free-running generalization. It is a useful endpoint for the ordered branch, but currently a high-risk basis for a new competitive method within two weeks.

**Smallest useful follow-up.** Evaluate first births on single-person images versus crowded images; compare canonical-rank error with best-person geometric error; retain teacher-prefix and generated-prefix evaluation on common eligible ranks. Preserve birth-time outputs separately from refined outputs. If first-person localization is poor even on unambiguous single-person images, prioritize the image-to-pose mapping rather than another sampler or rank-conditioning change. No new large ordered architecture should follow automatically.

Sources: [V46 final fixed32 metrics](/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v46/20261108_181906/seed42/quick_test/epoch079/metrics.json), [V46 selected full VAL](/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v46/20261108_181906/seed42/final_results.json), [V47 selected full VAL](/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v47/20261109_150341/seed42/final_results.json), [V47 gate/protocol](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v47.md), [V48 epoch 68](/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v48/20261110_095254/seed42/quick_test/epoch068/metrics.json), [AdaLN epoch 44](/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v48_adaln/20261110_135143/seed42/quick_test/epoch044/metrics.json).

## 5. What the remaining experiments rule in or out

The ten insights are selected research directions, not an assertion that the rest of the history is irrelevant. Several other investigations constrain the closeout decision.

### 5.1 Assignment consistency is a real issue, but it has not supplied a sufficient repair

V42's frozen proposal/source/target audit finds only 22 of 133 known-source inputs assigned back to the same supervised person on fixed32 near-clean inputs. Another 32 target a different person and 79 are unmatched/background. A seeded augmented TRAIN probe gives similar proportions. The corresponding proposal bank covers most people somewhere, but its same-index source/proposal alignment is much poorer.

This demonstrates a correspondence mismatch. It does not establish that per-source identity must always be preserved in a valid permutation-invariant set model. The scientific problem is whether the supervised recovery contract matches what we claim the state represents and what the sampler actually carries.

The project already tried several repairs. A frozen GT-assisted input permutation improves source/proposal alignment without improving AP; inference spatial remapping also fails; source-preserving V39 remains a poor box detector; paired V42 learns but develops severe confidence drift. It would therefore be misleading to recommend simply matching the slots again as an untried two-week solution.

Sources: [correspondence audit](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v42_correspondence_audit.md), [matching review](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v42_matching_review.md), [proposal-to-pose inference comparison](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v42_proposal_pose_inference.md).

### 5.2 SIGReg and live targets exposed stability problems rather than a validated latent regularizer

The original V40-SIGReg run's held-out clean-code RMS grows 0.849→14.357→246.464→2136.947 over initialization/epochs 0/1/2; fixed32 box AP becomes zero. The gradient investigation identifies a possible outward latent-space update with a live corruption source and detached target, but the parameter-Jacobian probe does not prove target detachment alone caused the full AdamW trajectory.

A live-target correction passes a two-epoch bounded pilot with clean RMS about 0.492 and nonzero AP. The subsequent longer run still crosses the scale guard during epoch 13. This history does not support another default regularizer weight sweep. A well-posed separate stability study would need to isolate target gradients, normalization, and regularization scale. That is more scope than the primary closeout needs.

Sources: [failure audit](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v40_sigreg_failure.md), [live-target pilot](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v40_sigreg_live_target.md), [longer-run record](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v40_sigreg_live_target_run.md).

### 5.3 Annotation completion is useful work, but currently lacks a validated target population

The RTMPose teacher recovers existing labels well on GT boxes: PCK@0.05 is 90.28% over 76,924 labelled VAL joints. However, its training includes TRAIN + VAL and the evaluated boxes are ground truth. This is an in-sample conditional measurement.

For 22,112 missing VAL joints, accuracy is unknown. Only 28.19% agree within 2% of box size across independent original/flip passes; only 8.25% meet both score ≥0.6 and that agreement threshold. Missing predictions concentrate at crop edges and include wrong-person/truncation failures. Agreement and plausible appearance cannot replace missing reference coordinates.

There could be a useful small annotation-quality result after human adjudication, but that would change the project focus and require new reference work. Blanket pseudo-label completion is not an evidence-backed shortcut to a strong estimator within two weeks. Existing visibility 1/2 labels must remain preserved, and proposed labels must remain separate from official GT.

Source: [missing-joint completion report](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/joint_completion.md).

### 5.4 Depth ordering and 3D guidance remain limited by target definition and evidence

Visible-only mean depth is undefined for 41.27% of TRAIN people and 40.88% of VAL people because they have no flag 2 joints. This is a property of the annotation flags, not proof those people are physically invisible. The frozen 32 VAL-image DA3 probe shows 11/75 pair-order reversals under flip, mostly near ties. It measures teacher consistency, not true depth accuracy.

V20 adds a frozen 3D-pose score-distillation signal, but its available full-test checkpoint is younger than the dense-read reference and does not establish an isolated improvement. Earlier depth/point-cloud exploration also found insufficient visible geometry for the intended skeletal prior. A new monocular 3D representation would introduce target, camera, and association questions rather than resolve the current 2D failure quickly.

Sources: [depth-order audit](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/pose_depth_order.md), [V20 diary](/home/gasparyanartur/dev/pose-graph-diff/docs/experiments.md:541), [depth/segmentation feasibility](/home/gasparyanartur/dev/pose-graph-diff/reports/3d-prior/depth-segmentation-compatibility.md).

## 6. Which angle should we choose?

### 6.1 Rank the options by the evidence still missing

| Closeout option | What already exists | Essential missing evidence | Two-week assessment |
|---|---|---|---|
| **A. State use and confidence drift** | Exact frozen interventions in two architectures; broad step curves; strong controls | Full-set intervention replication; released/full-size state audit; uncertainty and limits | Best chance of a complete empirical result with mostly evaluation work |
| **B. Coordinate-gradient boundary** | Very large matched-age effect; exact-forward backward decomposition | Another training seed; full-set learned comparison; careful positioning against DETR precedent | Best training-focused option if compute and retained checkpoints permit |
| **C. Task-pose JEPA** | Completed matched full TEST pair with +3.11 K4 AP | Another training seed; geometric explanation; declared K1 tolerance | Plausible narrow positive ablation; weaker endpoint performance |
| **D. Hidden-joint readout bottleneck** | Strong visible controls; native failure; accessible hidden-arm distinction | Actual coordinate-reader result with held-out families/shortcut controls | Plausible focused representation study; more benchmark-construction risk |
| **E. Preservation-constrained rollout** | Measured exposure gap and short-continuation tradeoff | A repair passing all retention/generation gates | Higher risk: the current trial already fails the combined gate |
| **F. Ordered generation redesign** | Valid overfit/gate infrastructure; documented birth/trajectory failures | A transferable image-to-person mapping and a useful free-running model | Poor choice for a two-week closeout; too many possible changes remain |

This ranking is a judgement about the distance to a finished study. It is not a numerical publication probability. A's value is explanatory; B/C are more direct candidates if the required endpoint is a replicated training gain. None currently supports claiming a new state-of-the-art estimator.

### 6.2 The recommended central claim

The strongest claim we could aim to establish is:

> In the studied image-conditioned iterative pose estimators, changes in detector AP across repeated calls can arise from distinct state-use, confidence-ranking, and geometric-recovery mechanisms. Controlled frozen interventions distinguish them, and a non-oracle change of score source recovers substantial ranking losses in specific failed trajectories.

Every qualification matters. “Studied” limits the scope. “Can” avoids a universal explanation. “Specific failed trajectories” acknowledges that fixing a failed variant does not outperform the strongest baseline. “Controlled” requires exact native replay and unchanged geometry. If replication supports only two regimes rather than all three on credible checkpoints, the final claim should shrink accordingly.

A useful completed package would contain a small set of final figures and tables:

1. A common-evaluator full-set AP-versus-call table, with architecture/compute differences disclosed.
2. Paired state variation versus raw joint/box movement, including a released/full-size model.
3. Fixed-geometry score swaps and score-fixed geometry swaps on V40 and paired V42.
4. Raw coverage/paired geometric errors and AP/AR showing when failure includes geometry.
5. A compact record of failed apparent repairs: image-only-confidence retraining, reference remapping, and identity-preserving parameterization.

The negative controls are important. Without them, the story risks overstating a score fix; with them, it becomes an account of which mechanism an intervention repairs and which mechanism it leaves unresolved.

### 6.3 A concrete fourteen-day schedule

No experiment below is launched by this report. The schedule is a proposed scope for review and depends on measured evaluation throughput.

| Days | Work | Concrete deliverable | Decision/stop condition |
|---|---|---|---|
| 1–2 | Freeze checkpoint/config/source identities, evaluation split, image IDs, seeds, postprocessing, and primary contrasts. Replay baseline predictions. | Audited protocol and baseline-parity records | Resolve any parity/score-association mismatch before interpreting interventions |
| 3–4 | Full-set V40 native K1/K4 and geometry/score hybrids; paired V42 native/pre-pose scores; uniform-t performance reference | Full-set ranking-intervention table with raw-output identity checks | If the effect is restricted tofixed 32, narrow the claim; do not invent a new classifier to rescue it |
| 5–6 | Native seeded state-use probes and selected route ablations on reduced plus released/full-size checkpoints | State-sensitivity and route-dependence results | If the released/full-size case differs, present that as a boundary rather than generalizing the reduced failure |
| 7 | Freeze interpretation and choose whether the core mechanism has replicated | One-page claim/evidence/counterevidence decision | End architecture exploration; no new branch after this point |
| 8–9 | Complete the predeclared confidence/state controls, raw geometry/coverage, and seed sensitivity | Final controlled comparisons | At most one bounded extra inference policy; retain negative results and strong baseline |
| 10–11 | Image-cluster uncertainty, crop/scale/crowding slices, runtime measured under a stated workload | Robustness/uncertainty tables and cost context | Do not select the headline checkpoint or policy using these final slices |
| 12–14 | Write the study, prepare figures and reproducibility instructions, verify artifacts, and close outstanding outcome records | Complete report/paper draft and reproducible evidence package | Finish at the registered scope even if the repair is negative |

For the primary option, default to no new full estimator training. A new training run adds scheduling and interpretation risk without being necessary to test the central frozen-computation claim. If option B or C is chosen instead, replace the state/score extension block with its matched second-seed training comparison from the beginning; do not attempt all three routes sequentially.

### 6.4 Evaluation details that make the result interpretable

**Use paired inputs.** Match image IDs, image-dependent initial noise, time grids, and frozen weights. Separate changing the incoming state from changing the timestep or rebuilding proposals. If slot identity changes between calls, do not swap tensors by row index without an explicit validated correspondence. V40's cached-reference experiment and V42's same-final-call counterfactual offer different valid contracts.

**Preserve pre-NMS records.** Keep query IDs, raw boxes/joints, logits, selected indices, and per-step state. When scores change, NMS survivors may change despite identical raw geometry. Report that effect rather than saying the final detection lists are identical.

**Avoid oracle headline methods.** GT ranking, GT boxes, and target-preserving permutations can diagnose a failure. They are not deployable improvements. Existing-image-head scores are a practical intervention; an extra Gaussian-state forward used to isolate feedback also incurs additional compute and is primarily diagnostic.

**Use the right uncertainty unit.** For sample uncertainty, resample whole images and recompute official AP for each resampled dataset. Duplicated images must preserve multiplicity, for example through consistently remapped annotation/prediction IDs. Averaging per-image AP is not aggregate CrowdPose AP. Inference-noise replicates should be paired across conditions and reported separately from training-seed replication. A confidence interval over one fixed, development-exposed dataset does not remove selection bias.

**Keep geometry and detection metrics together.** Include raw score-free coverage, one-to-one geometric matching where appropriate, paired OKS/IoU for known assignments, AP/AR, and score/NMS statistics. Any-overlap coverage can count duplicates favourably, so distinguish it from unique-person recall. Report crowding and person-scale slices without treating correlated joints or repeated calls as independent samples.

**Keep the strong control visible.** A+22 AP repair on a failed branch is scientifically informative, but the reader must also see the stronger ordinary model. This is essential to distinguish a mechanism result from a competitive-method claim.

### 6.5 What counts as finishing?

A minimum finished result is a reproducible, bounded empirical study: exact baselines, validated interventions, full-set effect estimates where claimed, uncertainty, counterexamples, and a clear statement of which hypotheses were confirmed or rejected. This can be completed even if there is no new trained architecture.

A stronger finish adds a repair that improves a meaningful baseline while preserving geometry, single-call performance, and cost under a declared tolerance. The current score interventions and JEPA ablation justify investigating that possibility, but do not guarantee it.

A competitive new method would additionally need convincing performance against appropriately matched current baselines, more than one training seed for the central effect, and a defensible novelty argument. The present record does not establish that endpoint. A two-week deadline should not depend on achieving it.

## 7. Related work and claim boundaries

This is targeted positioning, not an exhaustive novelty search.

**Iterative gains and ensembling are already distinguished in prior work.** DiffusionDet's Appendix B.2 explicitly compares last-step predictions with multi-step ensembles: its reported final-only AP is 45.80/45.74/45.46 at one/three/five steps, while ensembling gives 46.62/46.79 at three/five. Therefore “more calls do not necessarily improve the last estimate” is not by itself a new observation. Our potential contribution is a pose-specific intervention study identifying score/state/geometry mechanisms. [DiffusionDet, Appendix B.2 and Table 8](https://arxiv.org/html/2211.09788#A2.SS2).

**Iterative-reference gradient design is established.** Deformable DETR detaches updated references; DINO's look-forward-twice scheme deliberately permits a controlled adjacent-layer supervision route. A finding that unrestricted coordinate-residual gradients hurt V19 should be positioned against that design space, not presented as a universal benefit of stopping gradients. The relevant distinction is the precise backward path, pose supervision, and rollout setting. [Deformable DETR official source](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py), [DINO, Section 3.5](https://arxiv.org/html/2203.03605v4#S3.SS5).

**Separating person scoring from keypoint queries is not new architecture novelty.** GroupPose explicitly includes an instance query for scoring alongside keypoint queries. Our confidence result must concern measured generated-state feedback, an intervention's effect, and its failure boundaries; simply introducing another person-confidence head is not sufficient novelty. [GroupPose](https://arxiv.org/abs/2308.07313).

**DiffusionRegPose is the direct methodological reference.** Its stated approach samples image-conditioned poses through noisy candidate interaction and progressive refinement. A local audit can test which of those computations influence a given released or retrained model. It cannot infer the training provenance of supplied weights merely from a checked-in training-code quirk. In particular, a source path that forces timestep 0 does not prove that the released checkpoint was trained with that setting. [DiffusionRegPose, CVPR 2024](https://openaccess.thecvf.com/content/CVPR 2024/html/Tan_DiffusionRegPose_Enhancing_Multi-Person_Pose_Estimation_using_a_Diffusion-Based_End-to-End_Regression_CVPR_2024_paper.html).

Before external submission, the selected angle still needs a focused search for direct precedent in iterative detection/pose confidence drift, inference-state ablations, and geometry-sensitive latent diagnostics. The current report supplies evidence for a possible contribution; it does not establish a “first” claim or a publication venue fit.

## Appendix A. Coverage of the experimental branches

This inventory records what the branch contributes to the closeout decision. It is not a single comparable leaderboard. The detailed snapshot inventory remains available as [latest quick-result records](/home/gasparyanartur/dev/pose-graph-diff/tmp/closeout_review/latest_quick.json) and [extracted AP summaries](/home/gasparyanartur/dev/pose-graph-diff/tmp/closeout_review/aggregate_AP.json). Those inventories were assembled during the first review; the refreshed V48 snapshots are linked in Insight 10.

| Branch or investigation | Result/status relevant to closeout | Interpretation/use |
|---|---|---|
| Initial simple backbone/graph probes | Frozen DINO stage- 4 simple-head AP about 0.8 versus trainable ResNet- 18/34 about 4.9/5.0; feature/freeze contracts differ | Does not establish DINO's intrinsic localization ceiling |
| External/native DiffusionRegPose reproduction | Supplied-model inference reproduced; own training remains below supplied weights | Credible checkpoints and evaluator; unresolved training gap |
| Compact/runtime presets | Faster training setting; early compact model learned almost nothing | Runtime smokes and capacity reductions are unreliable quality proxies |
| Selected-layer Hungarian/OTA | Controlled positive full-set assignment result; corrected evaluator confirms the direction | Useful baseline experiment; limited novelty and seed coverage |
| PoseFlow V1/codebook/V3/V4/V5 | V1/V5 largely flat step curves; V3 strongly deteriorates; some branches stopped | Motivates the state-use question; no single cause established across models |
| V6/V7/V8 DINO image paths | V6 corrected full-test AP 56.71 at K1, best 56.89; backbone/fusion contracts differ in V7/V8 | Stronger features alone did not produce sustained refinement |
| V9/V10/V11/V12 token/readout families | Structured attention, tiny models, coordinate-conditioning and latent-state studies; scoring/inference fixes complicate old metrics | Preserve protocols/probes; no common final causal comparison established here |
| V13/matching/reconstruction decay | Full runs and combined interventions; latest fixed32 ordinary/OTA K1 about 17.40/35.06 | Matching/optimization matter; combined recipes and small cohorts limit attribution |
| V14 evidence bank | Stopped after epoch 49; latest fixed32 K1 about 1.29 | Evidence bottleneck diagnosed; no positive final endpoint |
| V15 baseline/rollout | Baseline full-test K1 about 52.49; rollout 128-image subset K1/K3 about 54.69/55.03 | Rollout reduces carry failure but does not establish large iterative gains |
| V15 dropout/query normalization | Dropout 128-image K1/K3 45.50/43.46 versus rollout 54.69/55.03; query normalization changes memory activity | Simple attempts to force memory use did not yield a robust quality improvement |
| V16/V17/V18 | V16 stopped early; V17/V18 completed, with modest subset K1-to- K3 changes | Different memory contracts; no strong standalone superiority result |
| V19 original/full/sampling detach | Large matched-age full-detach benefit; exact gradient routes verified | Insight 3; strongest optimization follow-up |
| V19 ranking/dense read/V20 | Ranking stopped at epoch 44; dense read at 67; V20 at 44; full-test curves/reset controls retained | No completed same-age positive evidence for ranking/dense/3D additions |
| V21/V22/V23 | Fixed-H/H-first/time/selected-call branches; carried H can strongly hurt V21; V23 completes without clear iteration gain | Loss-connected state and time inputs do not guarantee useful recurrence |
| V24 matched h/control | Full-test K1 gain 0.381 AP; K2/K3/K5 nearly flat; synthetic studies complete | Insight 8; accessibility and native use separate |
| V25 explicit h-to-DINO | Stopped after epoch 50; auxiliary target learning is not a final AP comparison | No final detector gain established |
| V26 original/latent-only/V27 | Original full-test K4 pose AP 7.33; latent-only/scale variants poor; near-clean/spread probes retained | Insights 5/6; latent loss/diversity are insufficient evidence |
| V28/V29/V30 | Clean-image stop-gradient, OTA, and head-tolerance controls; V28 full-test K4 pose AP 0.77; others stopped | Some gradient routes isolated; no successful latent-only endpoint |
| V31/V32 | Completed full-test K4 pose AP 7.95/7.73; DINO target learned | Auxiliary alignment did not close the generation gap |
| V33/no-clip/rescale/V34 | V33 full-test K1/K4 pose AP 1.69/5.57; V34 OTA 9.76/6.90; no-clip controls stopped | Head, assignment and clipping changes were insufficient |
| V35 OTA/spatial/timed/early/q 200/latent-only | Box-only; latest spatial subset K1/K4 22.46/18.16; query-count/timing controls fail to rescue it | Insight 5; geometry and exposure failures survive beyond scoring |
| V36 box/box+c/normalized variants | Original contract omits frame size; normalized variants still poor | Initial failure is not a clean rejection of image-free geometry codes |
| V37/reweight/V38/V39/live-source | V37 full-test box K1/K4 3.49/2.05; other controls weak; live-source V39 K4 about 5.16 | Matching, KL or one gradient-route change did not suffice |
| V40/EMA | Full-test box K1/K4 48.00/25.83; EMA 47.75/30.74 | Strong state/score audit; EMA helps K4 but stays below K1 |
| V40 SIGReg/pilot/live-target full run | Initial scale explosion; pilot briefly stable; longer live-target run hits scale guard | No validated regularized detector |
| V41 | Coordinate-box/JEPA/SIGReg proposal architecture; latest stopped epoch- 9 subset box K1/K4 32.54/29.00 | No completed pose result; combined intervention |
| V42 absolute/box-center/t0/uniform | Box-center repair helps early; t0-only run later fails Gaussian start; uniform full-test pose K1/K4/K8 51.77/50.87/50.78 | Important strong reference; source/timestep contract matters |
| V42 matched/noisy/paired/remapping | Several cancelled failures; paired geometry/score interventions verified | Matching alone does not solve inference; Insight 1 |
| V42 image-confidence/JEPA | Full-test pose K1/K4/K8 49.98/39.38/31.58 and 49.71/34.32/23.24 | Counterevidence to simple scoring/auxiliary-loss repairs |
| Geometry AE/VAE/fidelity matrix | High clean reconstruction; noise/masking/prior tradeoffs remain | Insight 6; representation fidelity is feasible but insufficient |
| V43 frozen VAE/small-noise/alignment | Full-test pose K1/K4/K8 23.30/19.39/16.72; conditional controls; alignment frontend damaged | Codec fidelity does not explain the whole failure; gradient/conditioning confounds measured |
| V44/x0-v/rollout | Selected full-VAL baseline K1/K4 pose 28.59/25.50; velocity comparison fails; rollout has a tradeoff | Insights 6/7; fixed codec/proposals support cleaner tests |
| V45 progressive | Stopped after epoch 13; fixed32 pose AP 11.74 | Separate from task-pose V45; no positive final endpoint |
| V45 task-pose control/JEPA | Completed matched full-test pair; JEPA K4 +3.11 AP | Insight 9; plausible narrow training result |
| V45 ordered JEPA | Cancelled after epoch 8; fixed32 pose AP approximately zero | Rank-slot supervision did not immediately produce a useful detector |
| V46/V47 | Full 80-epoch training completed; selected full-VAL pose AP 0.730/0.036 | Negative endpoint for those ordered recipes; Insight 10 |
| V48/AdaLN | Active; refreshed fixed32 snapshots around 1 AP | Interim only; birth diagnostics informative; no final gain claim |
| Image-information/ownership/depth suite | State/hidden-arm results; ownership admission fails; depth has ceiling/shortcut limits | Insights 2/8; behaviour, accessibility and 3D claims differ |
| Depth/point-cloud/visible-depth ordering | Earlier 3D geometry inadequate; visible-only rank coverage incomplete | No ready 3D supervision route with a two-week evidence advantage |
| RTMPose joint completion | Full-VAL proposal cache; missing-label accuracy unknown; teacher evaluation in-sample | Useful audit; adjudication required before annotation-quality claims |
The inventory rows are identifiers and result pointers. They should not be quoted as matched comparisons across families. Detailed protocols and raw artifacts, rather than a version number alone, determine whether two numbers can be compared.

## Appendix B. Practical evidence sources and open items

**Historical overview:** [experiment diary](/home/gasparyanartur/dev/pose-graph-diff/docs/experiments.md), [original chronology](/home/gasparyanartur/dev/pose-graph-diff/reports/diffusionregpose-to-poseflow-chronology.md), [project conventions](/home/gasparyanartur/dev/pose-graph-diff/docs/project.md), [architecture](/home/gasparyanartur/dev/pose-graph-diff/docs/architecture.md).

**Evaluation-only investigations with mature evidence:** [native/image-information report](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/2026-10-24-image-information-report.md), [V40 stateaudit](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v40_state_audit.md), [V42 scoreintervention](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/poseflow_v42_image_confidence.md), [V19 gradientaudit](/home/gasparyanartur/dev/pose-graph-diff/tmp/v19_sampling_analysis/findings.md), [fullstepcomparison](/home/gasparyanartur/dev/pose-graph-diff/tmp/v19_v5_full_steps/findings.md).

**Main unresolved evidence:** generality beyond the diagnostic cohorts; second training seeds for the central training ablations; full-size/released-checkpoint state-route replication; a practical repair that improves a strong baseline; held-out coordinate-reader performance for hidden joints; finalizedV 48 outcomes. These are not all required for the primary closeout. The chosen claim determines which must be completed.

**Operational scope:** this report creates a research artifact and records a plan. It does not authorize or perform model edits, evaluation extensions, training, cancellation, annotation export, or publication. Those actions require selecting and reviewing the concrete next iteration under the repository protocol.
