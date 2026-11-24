# Person proposal interference on CrowdPose TEST

Frozen ED-Pose R50 evaluation, 17 November 2026.

This report examines where person proposals compete, whether joint predictions mix nearby people, and which image and decoder information sustains the predictions. It uses **512 official TEST images containing 2270 annotated people, of whom 1869 are eligible under the native evaluator**, with controlled interventions on the same **32 TEST images containing 162 annotated people, of whom 133 are scored**. The reported weights are the released ED-Pose checkpoint. The 512 images include the 32 intervention images. Their IDs were fixed before the analyses.

The same weights score **51.33 pose AP at 256 / 427** and **70.53 at 800 / 1333** on the 512 images. The main text uses the larger subset for scene patterns and the smaller paired subset for causal interventions. Every table states its population. All evaluation annotations come from official TEST.


## What the experiments show

**The existing person-to-person attention primarily controls competition between hypotheses in these weights.** On the paired 32 images, blocking every inter-person edge lowers pose AP from 65.00 to 40.86. The 127 baseline matched proposals keep virtually identical mean OKS, 0.8774, and move only 0.33 original-image pixels on average. Restoring the baseline scores to the isolated decoder's geometry recovers 65.18 AP. Applying only the isolated scores to the baseline geometry yields 40.59 AP. The intervention therefore damages pose ranking and suppression much more than the matched joint coordinates. Box geometry also changes, as the separate box AP counterfactual shows.

**Duplicates can take useful slots, and global messages help prevent that.** In the raw top-N budget on those 32 images, isolation increases duplicate slots from 5 to 39 and reduces unique people covered from 106 to 68. Native postprocessing reduces the collision count, but postprocessed top-N coverage still falls from 106 to 84. Across the 512-image high-resolution baseline, raw proposals average 12.56 qualifying hypotheses per scored person, while the raw top-N budget spends 71 of 1,869 slots on duplicates. The large low-confidence pool and the smaller competitive duplicate set have different consequences.

**Resolution and person scale explain much of the crowding pattern.** Pose AP on the 512 images rises from 51.33 to 70.53 with the same weights. In scenes with six or more annotated people, it rises from 36.36 to 63.34. At low resolution, raw recall for people below 64 pixels tall is about 40–48% across the person-count groups, whereas people at least 128 pixels tall reach about 98.4–98.7%. The small-person cells have uneven sample sizes. These observations make local image detail a substantial part of the crowded-scene deficit.

**Structured displacement can change who a proposal represents even when AP hides the difference.** A full early coherent drift toward a neighbor changes the best person identity for 71 of 87 affected proposals, versus 23 of 87 under matched Gaussian displacement. Their aggregate pose AP is nevertheless similar, 51.69 and 51.20. Other proposals can still cover a person whose tracked proposal migrated. Proposal lineage and owner-specific recovery reveal a failure that a single AP score conceals.

**Midpoint-like and neighbor-endpoint errors occur in a small fraction of the inspected joints.** At high resolution, 127 of 8,528 eligible joint comparisons are midpoint-like and 37 land near a neighbor endpoint. Among the 1,235 eligible joint errors, those counts are 100 and 37. Only 2 of 3,853 eligible two-joint limbs put both joints near the same neighbor endpoint. This supports targeted diagnosis of limb ownership errors. The identity filters exclude ambiguous-owner skeletons, so these frequencies do not cover every possible cross-person mixture.

**The decoder still depends strongly on local image evidence and its carried features.** Masking labeled limbs lowers AP to 51.93, while masking the same number of background pixels leaves 64.44. Blur and JPEG degradation yield 40.81 and 43.32. Zeroing joint image-read messages leaves approximately zero pose AP, and zeroing carried token features before pose stage two leaves 8.73. The box and person streams continue to carry other information, which is visible in their separate metrics.


## Proposal duplication and missing people

| Resize   | Output   | People   | Top N covered   | Duplicate slots   | Recovered below N   | All one-to-one coverage   |
|:---------|:---------|:---------|:----------------|:------------------|:--------------------|:--------------------------|
| 256      | raw      | 1869     | 1359            | 209               | 241                 | 1612                      |
| 256      | post     | 1869     | 1399            | 119               | 171                 | 1580                      |
| 800      | raw      | 1869     | 1631            | 71                | 179                 | 1820                      |
| 800      | post     | 1869     | 1650            | 38                | 138                 | 1794                      |

Each raw proposal is assigned independently to its best ground-truth person when OKS is at least 0.50. A duplicate slot is an additional qualifying proposal assigned to a person already covered in that budget. “Top N” uses the number of evaluator-eligible annotated people in each image as an oracle budget. “Recovered below N” counts additional best-assignment identities available farther down the score ranking. “All one-to-one coverage” uses maximum-cardinality bipartite matching, allowing an ambiguous proposal to cover another eligible person. These last two quantities can differ because independent best assignment can send several ambiguous proposals to the same person.

A missed person is recoverable only if an eligible candidate exists. A duplicate appearing alongside a missed person does not by itself establish that removing the duplicate would recover that person. The score-budget curves show how many people could be covered as more candidates are retained. Raw proposals are the 100 expanded person groups. Postprocessed outputs use the native score threshold and pose suppression. Native inference keeps a variable number of detections; the top-N budget is an oracle diagnostic.


![Same-weight resolution comparison and score-budget coverage on 512 TEST images. N and 2N are image-dependent oracle budgets.](../../results/person_interference/20261117/figures/resolution_and_coverage.png)

Same-weight resolution comparison and score-budget coverage on 512 TEST images. N and 2N are image-dependent oracle budgets.


Threshold sensitivity for raw top-N proposals on 512 TEST images provides a check on whether the conclusion depends on the OKS cutoff.

| size   | threshold   | people   | covered   | duplicates   | all_cardinality   |
|:-------|:------------|:---------|:----------|:-------------|:------------------|
| 256    | 0.25        | 1869     | 1437      | 263          | 1732              |
| 256    | 0.50        | 1869     | 1359      | 209          | 1612              |
| 256    | 0.75        | 1869     | 1064      | 108          | 1256              |
| 800    | 0.25        | 1869     | 1667      | 79           | 1849              |
| 800    | 0.50        | 1869     | 1631      | 71           | 1820              |
| 800    | 0.75        | 1869     | 1441      | 42           | 1640              |

![Multiplicity at OKS ≥ 0.50, including people with zero qualifying raw proposals. The final bin includes ten or more proposals. Thresholding confidence exposes how many geometric duplicates remain competitive.](../../results/person_interference/20261117/figures/proposal_multiplicity.png)

Multiplicity at OKS ≥ 0.50, including people with zero qualifying raw proposals. The final bin includes ten or more proposals. Thresholding confidence exposes how many geometric duplicates remain competitive.


| Resize   | Candidates   | Mean per person   | Median   | People with zero (%)   | People with ≥2 (%)   | Maximum   |
|:---------|:-------------|:------------------|:---------|:-----------------------|:---------------------|:----------|
| 256      | All raw      | 10.21             | 6.00     | 14.39                  | 78.81                | 83        |
| 256      | Score ≥0.05  | 6.40              | 3.00     | 15.09                  | 64.42                | 69        |
| 256      | Score ≥0.50  | 0.50              | 0.00     | 50.24                  | 0.16                 | 2         |
| 800      | All raw      | 12.56             | 7.00     | 3.16                   | 85.66                | 95        |
| 800      | Score ≥0.05  | 5.81              | 2.00     | 3.75                   | 60.67                | 78        |
| 800      | Score ≥0.50  | 0.72              | 1.00     | 28.46                  | 0.32                 | 2         |



Before pose expansion, box candidates provide the following coverage at IoU ≥ 0.50 on the 32 intervention images. This is box localization. Pose coverage is measured separately after expansion.

| budget      | people   | all_covered   | all_cardinality   |
|:------------|:---------|:--------------|:------------------|
| 900         | 133      | 133           | 133               |
| selected100 | 133      | 131           | 131               |

![Raw person confidence versus qualifying pose geometry on 512 TEST images. A geometrically correct duplicate still counts as qualifying here, so this is a quality diagnostic rather than detection precision.](../../results/person_interference/20261117/figures/confidence_and_quality.png)

Raw person confidence versus qualifying pose geometry on 512 TEST images. A geometrically correct duplicate still counts as qualifying here, so this is a quality diagnostic rather than detection precision.


![Coverage and duplicate slots across the four pose decoder stages on 512 TEST images. Identity is preserved through the pose stages.](../../results/person_interference/20261117/figures/decoder_coverage.png)

Coverage and duplicate slots across the four pose decoder stages on 512 TEST images. Identity is preserved through the pose stages.


A query can change its best qualifying person between stages. The following counts require OKS ≥ 0.50 at both adjacent stages, which excludes unlocalized queries.

| size   | stage   | eligible   | switches   | Switch rate (%)   |
|:-------|:--------|:-----------|:-----------|:------------------|
| 256    | 2       | 17149      | 39         | 0.23              |
| 256    | 3       | 18445      | 30         | 0.16              |
| 256    | 4       | 18932      | 7          | 0.04              |
| 800    | 2       | 20728      | 80         | 0.39              |
| 800    | 3       | 22551      | 45         | 0.20              |
| 800    | 4       | 23224      | 19         | 0.08              |

## Sharp poses, mixed limbs, and midpoint errors

A limb’s owner is assigned using the remaining joints, excluding that entire limb. We require complement-joint OKS ≥ 0.50 and a margin of at least 0.10 over the next person, then use one-to-one assignment among the native top-N detections. For each elbow, wrist, knee or ankle, the nearest labeled corresponding joint on any other annotated person defines the neighbor, including people ignored by the native AP evaluator. We exclude pairs separated by less than 0.1 times the square root of the owner’s box area. This reduces cases where the two annotations are geometrically indistinguishable.

An owner endpoint is within 15% of the owner-to-neighbor separation from the owner joint. A neighbor endpoint is within the same relative radius around the neighbor joint. A midpoint-like prediction lies between 25% and 75% along their connecting segment and within 15% of the separation perpendicular to it. These relative regions describe geometry. A point near the owner endpoint can still exceed the separate error threshold of 0.1 times the square root of box area when the neighbor is far away.


| Resize   | Population   | Joint comparisons   | owner_endpoint (%)   | midpoint_like (%)   | neighbor_endpoint (%)   | other (%)   |
|:---------|:-------------|:--------------------|:---------------------|:--------------------|:------------------------|:------------|
| 256      | All eligible | 7395                | 67.88                | 2.26                | 0.62                    | 29.24       |
| 256      | Errors only  | 1743                | 13.02                | 6.94                | 2.64                    | 77.40       |
| 800      | All eligible | 8528                | 80.07                | 1.49                | 0.43                    | 18.01       |
| 800      | Errors only  | 1235                | 15.95                | 8.10                | 3.00                    | 72.96       |

![Joint-to-neighbor geometry on 512 TEST images. The scatter uses 800-resolution errors. Points outside the displayed limits remain included in the tables.](../../results/person_interference/20261117/figures/joint_ownership_geometry.png)

Joint-to-neighbor geometry on 512 TEST images. The scatter uses 800-resolution errors. Points outside the displayed limits remain included in the tables.


When both distal joints qualify, we can ask whether the limb moves coherently. Both-neighbor requires the same neighboring person for both joints. Mixed owner-neighbor means one joint lies near its owner and the other lies near a neighbor.

| size   | eligible_limbs   | both_owner   | both_neighbor   | both_midpoint   | mixed_owner_neighbor   |
|:-------|:-----------------|:-------------|:----------------|:----------------|:-----------------------|
| 256    | 3347             | 1946         | 4               | 15              | 2                      |
| 800    | 3853             | 2800         | 2               | 11              | 7                      |

These frequencies are conditional on the identity, score-budget and joint-separation filters. They exclude ambiguous-owner skeletons and therefore cannot estimate the prevalence of all cross-person failures. These are deterministic prediction geometries. A midpoint-like output supports an interpolation error description, but does not establish that the network internally averages a multimodal posterior. The error categories and coherent-limb counts test the observable consequences of that idea.


![An illustrative TEST error selected by the largest normalized error within its category at 800 resolution. Blue is the independently assigned owner, green the neighboring person, and orange the prediction. This is a selected failure example, not a prevalence estimate.](../../results/person_interference/20261117/figures/midpoint_like_example.png)

An illustrative TEST error selected by the largest normalized error within its category at 800 resolution. Blue is the independently assigned owner, green the neighboring person, and orange the prediction. This is a selected failure example, not a prevalence estimate.


![An illustrative TEST error selected by the largest normalized error within its category at 800 resolution. Blue is the independently assigned owner, green the neighboring person, and orange the prediction. This is a selected failure example, not a prevalence estimate.](../../results/person_interference/20261117/figures/neighbor_endpoint_example.png)

An illustrative TEST error selected by the largest normalized error within its category at 800 resolution. Blue is the independently assigned owner, green the neighboring person, and orange the prediction. This is a selected failure example, not a prevalence estimate.


## Scene crowding, image quality, and resolution

Person count includes all annotated people, including those ignored by native pose AP. Metric denominators retain the native evaluator’s eligible people. Local box overlap measures spatial competition. CrowdPose crowd index provides an additional annotation-level crowding measure. These variables are related to person size and occlusion, so natural-image strata describe associations. The controlled image edits below measure sensitivity while keeping the same scene and annotations.


| size   | field            | group          | images   | people   | Pose AP   | Recall50 (%)   | Recall 95% interval   |
|:-------|:-----------------|:---------------|:---------|:---------|:----------|:---------------|:----------------------|
| 256    | crowd_index      | easy <0.2      | 100      | 294      | 48.43     | 75.51          | 67.3–83.9             |
| 256    | crowd_index      | medium 0.2–0.8 | 329      | 1271     | 52.99     | 86.00          | 83.4–88.5             |
| 256    | crowd_index      | hard >=0.8     | 83       | 304      | 46.96     | 87.17          | 82.7–91.8             |
| 800    | crowd_index      | easy <0.2      | 100      | 294      | 71.49     | 96.26          | 92.8–99.0             |
| 800    | crowd_index      | medium 0.2–0.8 | 329      | 1271     | 71.68     | 95.75          | 94.6–96.9             |
| 800    | crowd_index      | hard >=0.8     | 83       | 304      | 64.52     | 96.71          | 94.3–98.6             |
| 256    | annotated_people | 1–2            | 213      | 403      | 69.69     | 96.77          | 95.0–98.5             |
| 256    | annotated_people | 3–5            | 179      | 625      | 59.46     | 91.36          | 88.9–93.9             |
| 256    | annotated_people | 6+             | 120      | 841      | 36.36     | 73.60          | 69.0–77.8             |
| 256    | all_max_overlap  | low <0.2       | 191      | 571      | 54.75     | 83.36          | 78.9–87.7             |
| 256    | all_max_overlap  | medium 0.2–0.5 | 246      | 1028     | 49.81     | 84.14          | 80.8–87.4             |
| 256    | all_max_overlap  | high >=0.5     | 75       | 270      | 50.20     | 88.52          | 84.4–92.7             |
| 800    | annotated_people | 1–2            | 213      | 403      | 79.08     | 97.77          | 96.2–99.3             |
| 800    | annotated_people | 3–5            | 179      | 625      | 73.97     | 97.28          | 96.1–98.4             |
| 800    | annotated_people | 6+             | 120      | 841      | 63.32     | 94.17          | 92.3–96.1             |
| 800    | all_max_overlap  | low <0.2       | 191      | 571      | 72.46     | 94.92          | 92.8–96.8             |
| 800    | all_max_overlap  | medium 0.2–0.5 | 246      | 1028     | 70.44     | 96.98          | 95.7–98.1             |
| 800    | all_max_overlap  | high >=0.5     | 75       | 270      | 68.30     | 94.44          | 91.7–97.0             |

![Native pose AP by image strata on 512 TEST images. n is the number of images in each bin. Luminance, blur proxy and visibility-composition bins use sample tertile boundaries. Ties can produce unequal bin sizes.](../../results/person_interference/20261117/figures/scene_strata.png)

Native pose AP by image strata on 512 TEST images. n is the number of images in each bin. Luminance, blur proxy and visibility-composition bins use sample tertile boundaries. Ties can produce unequal bin sizes.


![Per-person raw one-to-one recall by resized height and overlap on 512 TEST images. Each resolution changes the resized-height bin membership, so the left-hand curves do not compare the same people at every x-position.](../../results/person_interference/20261117/figures/person_scale_and_overlap.png)

Per-person raw one-to-one recall by resized height and overlap on 512 TEST images. Each resolution changes the resized-height bin membership, so the left-hand curves do not compare the same people at every x-position.


![Crowding by person-size strata on 512 TEST images. Cells show one-to-one raw recall at OKS ≥ 0.50, with person counts. These descriptive controls reduce scale mixing but do not equalize pose difficulty, occlusion or image content.](../../results/person_interference/20261117/figures/size_density.png)

Crowding by person-size strata on 512 TEST images. Cells show one-to-one raw recall at OKS ≥ 0.50, with person counts. These descriptive controls reduce scale mixing but do not equalize pose difficulty, occlusion or image content.


The image brightness proxy is mean grayscale intensity. The blur proxy is Laplacian variance after thumbnailing to at most 256 × 256. Clipping is the fraction of near-black or near-white pixels. These measurements also depend on scene content and texture. The visibility-composition stratum records the fraction of labeled joints carrying flag 1. Its nonmonotonic association is treated descriptively rather than as a calibrated occlusion-severity measure. All natural-image strata, including clipping and person height, are available in the accompanying CSV.


## Controlled changes to the image

All rows use the same 32 TEST images, the 800 / 1333 baseline, native preprocessing and postprocessing. Brightness multiplies original RGB intensities and clips to the valid range. Blur has a standard deviation of three pixels in the original image. JPEG uses quality 10 before resizing. Limb masks follow labeled arm and leg segments with width proportional to person size. Masked pixels receive the image’s mean RGB color. Background means pixels outside every annotated person box and every drawn limb segment. This conservative definition can include scene context and unannotated people.


| Condition             | Images   | Pose AP   | Box AP   | Δ pose AP   | Recall50 (%)   | Δ recall 95% interval   |
|:----------------------|:---------|:----------|:---------|:------------|:---------------|:------------------------|
| baseline800           | 32       | 65.00     | 60.20    | 0.00        | 94.74          | +0.0 to +0.0            |
| baseline512           | 32       | 62.03     | 57.61    | -2.97       | 91.73          | -5.9 to -0.7            |
| baseline256           | 32       | 45.33     | 40.78    | -19.67      | 71.43          | -34.4 to -10.0          |
| brightness_0.35       | 32       | 64.05     | 57.44    | -0.95       | 90.98          | -8.6 to +0.0            |
| brightness_1.8        | 32       | 60.84     | 52.77    | -4.15       | 90.98          | -7.2 to -0.8            |
| blur_3                | 32       | 40.81     | 36.71    | -24.19      | 68.42          | -38.1 to -12.9          |
| jpeg_10               | 32       | 43.32     | 34.84    | -21.68      | 75.19          | -27.2 to -11.4          |
| mask_limbs            | 32       | 51.93     | 48.26    | -13.07      | 88.72          | -10.8 to -2.1           |
| mask_background_equal | 32       | 64.44     | 59.63    | -0.56       | 94.74          | +0.0 to +0.0            |
| mask_people           | 32       | 0.00      | 0.12     | -65.00      | 0.00           | -100.0 to -88.5         |
| mask_all_background   | 32       | 63.02     | 59.49    | -1.98       | 92.48          | -4.5 to +0.0            |



Equal-area background masking is possible in **32 of 32 images**. The common-population comparison below excludes images with insufficient background. The masks match pixel area, while their shape and semantic content differ. Whole-person-box and whole-background masks alter different amounts of the image and serve as broad information-removal interventions.



| condition             | images   | Pose AP   |
|:----------------------|:---------|:----------|
| baseline800           | 32       | 65.00     |
| mask_limbs            | 32       | 51.93     |
| mask_background_equal | 32       | 64.44     |

![The most-overlapping scene in the intervention subset, chosen by maximum annotated box IoU. Orange predictions have native score above 0.30. The original image is shown for the resolution comparison, which changes the model input resize.](../../results/person_interference/20261117/figures/masking_and_resolution_example.png)

The most-overlapping scene in the intervention subset, chosen by maximum annotated box IoU. Orange predictions have native score above 0.30. The original image is shown for the resolution comparison, which changes the model input resize.


## Breaking attention and testing earlier state

ED-Pose predicts people simultaneously. Its earlier poses are the hypotheses carried across decoder stages. We test both that carried state and messages from other person tokens. Each expanded group has one person/box token and 14 joint tokens. A joint normally attends within its own group. A person token attends to its own joints and all person tokens.

The nearest, random and farthest conditions remove 20 directed inter-person edges per receiving token at every pose stage. Geometry is taken from the current box centers, with self edges preserved. Isolation removes all 99 other-person edges. Joint-self-only removes a joint’s access to its own box token and other joints. Zero-self zeros the attention output while retaining the residual and normalization computation. Shuffling cyclically reassigns the person-token attention outputs among the 100 groups. The mid-feature reset zeros all carried token features before pose stage two while retaining reference geometry and image memory. Image-read conditions zero the deformable-attention output for all joints, or only the left elbow and wrist, in all four pose stages.


| Condition            | Images   | Pose AP   | Box AP   | Δ pose AP   | Recall50 (%)   | Δ recall 95% interval   |
|:---------------------|:---------|:----------|:---------|:------------|:---------------|:------------------------|
| null                 | 32       | 65.00     | 60.20    | 0.00        | 94.74          | +0.0 to +0.0            |
| isolate              | 32       | 40.86     | 31.65    | -24.14      | 95.49          | +0.0 to +2.5            |
| near20               | 32       | 62.90     | 56.48    | -2.10       | 95.49          | +0.0 to +2.5            |
| random20             | 32       | 64.51     | 59.25    | -0.49       | 94.74          | +0.0 to +0.0            |
| far20                | 32       | 64.85     | 60.39    | -0.15       | 94.74          | +0.0 to +0.0            |
| shuffle_box_messages | 32       | 48.65     | 49.64    | -16.35      | 94.74          | +0.0 to +0.0            |
| joint_self_only      | 32       | 1.94      | 54.74    | -63.05      | 36.84          | -69.6 to -43.2          |
| zero_self            | 32       | 58.21     | 54.66    | -6.79       | 93.23          | -3.4 to +0.0            |
| reset_mid_features   | 32       | 8.73      | 13.16    | -56.27      | 78.95          | -24.9 to -5.5           |
| zero_joint_image     | 32       | 0.01      | 53.83    | -64.99      | 1.50           | -98.4 to -87.2          |
| zero_limb_image      | 32       | 46.18     | 56.86    | -18.82      | 92.48          | -4.6 to +0.0            |

![Paired intervention effects on 32 TEST images. Bars show AP point changes. The tables give image-bootstrap uncertainty for recall changes.](../../results/person_interference/20261117/figures/intervention_ap.png)

Paired intervention effects on 32 TEST images. Bars show AP point changes. The tables give image-bootstrap uncertainty for recall changes.


### Direct joint image reads in early and final pose stages

This follow-up uses the same 32 CrowdPose TEST images, frozen full ED-Pose R50 weights, and 800 / 1333 native evaluation. We replace the joint rows of deformable cross-attention with its output-projection bias in pose stages 1–3 together or in stage 4 alone. Pose stages follow the two box-only decoder layers. The projection bias and box-token image reads are preserved at each intervention site. Later box outputs can still change through decoder interactions. The earlier all-stage `zero_joint_image` condition removed the entire joint output, including the bias.


| Condition                               | Images   | Pose AP   | Δ pose AP   | Box AP   | Δ box AP   | Recall50 (%)   | Δ recall 95% interval   |
|:----------------------------------------|:---------|:----------|:------------|:---------|:-----------|:---------------|:------------------------|
| Native baseline                         | 32       | 65.00     | 0.00        | 60.20    | 0.00       | 94.74          | +0.0 to +0.0            |
| Remove joint image values in stages 1–3 | 32       | 11.82     | -53.18      | 53.91    | -6.29      | 59.40          | -46.5 to -22.4          |
| Remove joint image values in stage 4    | 32       | 64.39     | -0.61       | 60.16    | -0.04      | 94.74          | +0.0 to +0.0            |

**The full checkpoint depends much more on the combined early joint reads than on the final joint read alone.** Removing stages 1–3 drops pose AP by 53.18 points and reduces Recall50 from 94.74% to 59.40%. With those reads intact, removing stage 4 costs 0.61 AP and leaves Recall50 unchanged. The final joint read alone cannot recover the pose quality supported by the earlier reads in this inference setting. The grouped intervention does not resolve which of stages 1–3 supplies the evidence or how they interact. The earlier reduced DiffusionRegPose checkpoint showed the opposite pattern across its two pose stages.

The no-op replay exactly reproduced the saved raw and postprocessed baseline on all 32 images. Initial query IDs and references matched in both interventions. The stage-4-only run reproduced the first three pose outputs exactly.

[Metrics](../../results/person_interference/20261117/stage_image_values/metrics.csv) · [Verification and provenance](../../results/person_interference/20261117/stage_image_values/verification.json) · [Replay code](../../results/person_interference/20261117/analysis_code/stage_image_values.py)

From the repository root, run `venv/bin/python results/person_interference/20261117/analysis_code/stage_image_values.py` before rebuilding the report.


### Effects on proposal geometry and confidence


Holding the baseline proposal-to-person assignment fixed distinguishes geometry changes from changes to confidence and output selection. This table includes baseline raw proposals matched one-to-one at OKS ≥ 0.50. An identity switch means the query’s best-OKS person differs from its own baseline best-OKS person. The no-op control has zero such switches.

| condition            | matched_proposals   | baseline_oks   | changed_oks   | baseline_score   | changed_score   | mean_shift_pixels   | identity_switches   |
|:---------------------|:--------------------|:---------------|:--------------|:-----------------|:----------------|:--------------------|:--------------------|
| far20                | 127                 | 0.88           | 0.88          | 0.31             | 0.31            | 0.02                | 0                   |
| isolate              | 127                 | 0.88           | 0.88          | 0.31             | 0.41            | 0.33                | 0                   |
| joint_self_only      | 127                 | 0.88           | 0.36          | 0.31             | 0.18            | 45.47               | 4                   |
| near20               | 127                 | 0.88           | 0.88          | 0.31             | 0.41            | 0.05                | 0                   |
| null                 | 127                 | 0.88           | 0.88          | 0.31             | 0.31            | 0.00                | 0                   |
| random20             | 127                 | 0.88           | 0.88          | 0.31             | 0.33            | 0.02                | 0                   |
| reset_mid_features   | 127                 | 0.88           | 0.72          | 0.31             | 0.12            | 13.43               | 1                   |
| shuffle_box_messages | 127                 | 0.88           | 0.88          | 0.31             | 0.22            | 0.11                | 0                   |
| zero_joint_image     | 127                 | 0.88           | 0.20          | 0.31             | 0.19            | 110.06              | 16                  |
| zero_limb_image      | 127                 | 0.88           | 0.78          | 0.31             | 0.27            | 12.92               | 0                   |
| zero_self            | 127                 | 0.88           | 0.82          | 0.31             | 0.40            | 9.41                | 0                   |

The score and geometry swaps below reapply native postprocessing to the same 100 groups. Original-scores uses isolated boxes and joints with baseline confidence. Only-scores uses baseline geometry with isolated confidence. Only-boxes changes the boxes while keeping baseline joints and scores. Reconstructing the unmodified baseline from these arrays preserves every retained detection within 1e-5 numerical tolerance.

| condition               | images   | Pose AP   | Box AP   |
|:------------------------|:---------|:----------|:---------|
| isolate_original_scores | 32       | 65.18     | 54.79    |
| isolate_only_scores     | 32       | 40.59     | 38.80    |
| isolate_only_boxes      | 32       | 65.00     | 54.77    |

![Score and geometry counterfactuals on 32 TEST images. Isolated geometry includes the boxes and joints from the decoder with all inter-person attention blocked.](../../results/person_interference/20261117/figures/score_geometry_counterfactual.png)

Score and geometry counterfactuals on 32 TEST images. Isolated geometry includes the boxes and joints from the decoder with all inter-person attention blocked.


The same intervention increases competition from high-scoring duplicate geometry. The following counts are summed across the 32 images. Qualifying score50 counts raw proposals with OKS ≥ 0.50 and confidence ≥ 0.50, including duplicates.

| condition            | people   | raw_top_covered   | raw_top_duplicates   | post_top_covered   | post_top_duplicates   | post_count   | qualifying_score50   |
|:---------------------|:---------|:------------------|:---------------------|:-------------------|:----------------------|:-------------|:---------------------|
| baseline800          | 133      | 106               | 5                    | 106                | 3                     | 523          | 80                   |
| far20                | 133      | 107               | 5                    | 107                | 4                     | 521          | 79                   |
| isolate              | 133      | 68                | 39                   | 84                 | 19                    | 1140         | 439                  |
| near20               | 133      | 95                | 19                   | 104                | 6                     | 600          | 133                  |
| random20             | 133      | 101               | 9                    | 106                | 5                     | 551          | 92                   |
| shuffle_box_messages | 133      | 86                | 18                   | 88                 | 10                    | 1103         | 72                   |

![Attention between the 20 highest-scoring final person tokens in the most-populated intervention scene. Rows receive messages from columns. Values are probabilities averaged across heads, with colors saturating at 0.25. Each row can also attend to 14 own joints and 80 omitted person tokens, so the displayed rows need not sum to one.](../../results/person_interference/20261117/figures/attention_matrix.png)

Attention between the 20 highest-scoring final person tokens in the most-populated intervention scene. Rows receive messages from columns. Values are probabilities averaged across heads, with colors saturating at 0.25. Each row can also attend to 14 own joints and 80 omitted person tokens, so the displayed rows need not sum to one.


Average person-token attention mass across all 100 proposals and all 512 images is shown below. Neighbor entropy is normalized by log(99). Large attention mass establishes use of an available route; the intervention results establish sensitivity to changing it.

| size   | stage   | other_person_mass   | self_person_mass   | own_joints_mass   | neighbor_entropy   |
|:-------|:--------|:--------------------|:-------------------|:------------------|:-------------------|
| 256    | 1       | 0.88                | 0.01               | 0.11              | 0.91               |
| 256    | 2       | 0.58                | 0.01               | 0.41              | 0.91               |
| 256    | 3       | 0.37                | 0.01               | 0.62              | 0.87               |
| 256    | 4       | 0.51                | 0.02               | 0.47              | 0.82               |
| 800    | 1       | 0.89                | 0.01               | 0.10              | 0.90               |
| 800    | 2       | 0.60                | 0.01               | 0.39              | 0.89               |
| 800    | 3       | 0.41                | 0.01               | 0.59              | 0.85               |
| 800    | 4       | 0.51                | 0.02               | 0.47              | 0.78               |

There is no ordered person-birth history in this model. These measurements address previous decoder hypotheses and other-person context. Sequentially emitted poses or previous diffusion calls require a model with those mechanisms.


## Gaussian errors versus structured errors

The corruption targets 87 baseline raw proposals in 26 images assigned one-to-one to people at OKS ≥ 0.50. AP remains measured over all 32 paired images, including images where no proposal qualifies for corruption. A target must have a neighboring scored annotated person and labeled left elbow and wrist for both people. Arm swapping moves the elbow and wrist references toward the neighbor’s annotated elbow and wrist. Coherent drift translates all 14 joint references by the displacement between the two person-box centers. Half and full doses apply 50% and 100% of those displacements. These interventions use ground truth to construct diagnostic errors. The ordinary baseline inference receives the image alone.

Errors are injected either immediately after joint expansion or after the first pose stage, leaving carried features available. Gaussian controls use independent Gaussian directions over the same affected coordinates and match the structured displacement norm in original pixel units for each proposal after applying coordinate bounds. This makes them bounded, norm-matched Gaussian-direction controls. Matching displacement magnitude does not equalize the resulting distance from ground truth, so injected joint error is reported alongside final recovery. One direction in the main 696-vector dose check required resampling because the image bounds prevented it from attaining the target displacement. At the half dose, two additional Gaussian seeds test direction sensitivity.


| Condition                | Images   | Pose AP   | Box AP   | Δ pose AP   | Recall50 (%)   | Δ recall 95% interval   |
|:-------------------------|:---------|:----------|:---------|:------------|:---------------|:------------------------|
| early_swap_0.5           | 32       | 60.07     | 59.30    | -4.93       | 94.74          | +0.0 to +0.0            |
| early_swap_1.0           | 32       | 56.36     | 59.67    | -8.64       | 93.98          | -2.3 to +0.0            |
| early_gaussian_swap_0.5  | 32       | 60.92     | 59.26    | -4.07       | 94.74          | +0.0 to +0.0            |
| early_gaussian_swap_1.0  | 32       | 58.41     | 59.31    | -6.59       | 94.74          | +0.0 to +0.0            |
| early_drift_0.5          | 32       | 55.09     | 56.86    | -9.91       | 87.97          | -9.6 to -3.6            |
| early_drift_1.0          | 32       | 51.69     | 54.58    | -13.31      | 87.97          | -11.3 to -2.9           |
| early_gaussian_drift_0.5 | 32       | 54.42     | 57.75    | -10.58      | 90.23          | -7.7 to -1.5            |
| early_gaussian_drift_1.0 | 32       | 51.20     | 57.03    | -13.79      | 85.71          | -12.9 to -4.8           |
| mid_swap_0.5             | 32       | 58.14     | 59.53    | -6.85       | 94.74          | +0.0 to +0.0            |
| mid_swap_1.0             | 32       | 55.84     | 60.04    | -9.16       | 93.98          | -2.3 to +0.0            |
| mid_gaussian_swap_0.5    | 32       | 58.91     | 59.57    | -6.09       | 93.98          | -2.3 to +0.0            |
| mid_gaussian_swap_1.0    | 32       | 57.54     | 59.33    | -7.46       | 94.74          | +0.0 to +0.0            |
| mid_drift_0.5            | 32       | 49.40     | 57.44    | -15.60      | 86.47          | -12.4 to -4.0           |
| mid_drift_1.0            | 32       | 49.04     | 56.34    | -15.96      | 86.47          | -13.5 to -3.1           |
| mid_gaussian_drift_0.5   | 32       | 48.86     | 57.84    | -16.14      | 87.22          | -10.3 to -4.1           |
| mid_gaussian_drift_1.0   | 32       | 47.82     | 57.81    | -17.18      | 84.96          | -13.8 to -5.2           |

![AP response to structured errors and matched Gaussian controls on the same 32 TEST images. Different points retain the same affected proposal identities, while the clipping-aware Gaussian direction can change when a larger dose is infeasible.](../../results/person_interference/20261117/figures/structured_vs_gaussian.png)

AP response to structured errors and matched Gaussian controls on the same 32 TEST images. Different points retain the same affected proposal identities, while the clipping-aware Gaussian direction can change when a larger dose is infeasible.


Recovery is measured on the injected proposals and affected labeled joints. Pixel distances use original-image coordinates. RMS is the mean per-proposal input displacement magnitude. Baseline final error makes ordinary decoder improvement visible when assessing recovery.

| condition                | proposals   | rms_pixels   | injected_error   | final_error   | baseline_final_error   | final_oks   | identity_switches   | neighbor_wins   |
|:-------------------------|:------------|:-------------|:-----------------|:--------------|:-----------------------|:------------|:--------------------|:----------------|
| early_drift_0.5          | 87          | 61.15        | 76.68            | 64.33         | 10.57                  | 0.41        | 40                  | 36              |
| early_drift_1.0          | 87          | 122.21       | 129.93           | 115.56        | 10.57                  | 0.18        | 71                  | 70              |
| early_gaussian_drift_0.5 | 87          | 61.15        | 73.98            | 38.90         | 10.57                  | 0.59        | 4                   | 2               |
| early_gaussian_drift_1.0 | 87          | 122.21       | 121.61           | 88.81         | 10.57                  | 0.34        | 23                  | 16              |
| early_gaussian_swap_0.5  | 87          | 68.02        | 87.42            | 50.84         | 11.72                  | 0.82        | 0                   | 0               |
| early_gaussian_swap_1.0  | 87          | 136.04       | 144.77           | 110.83        | 11.72                  | 0.78        | 0                   | 0               |
| early_swap_0.5           | 87          | 68.02        | 77.13            | 54.89         | 11.72                  | 0.81        | 1                   | 1               |
| early_swap_1.0           | 87          | 136.04       | 132.14           | 120.51        | 11.72                  | 0.76        | 1                   | 1               |
| mid_drift_0.5            | 87          | 61.11        | 65.81            | 58.73         | 10.57                  | 0.33        | 43                  | 38              |
| mid_drift_1.0            | 87          | 122.08       | 125.45           | 120.65        | 10.57                  | 0.11        | 77                  | 72              |
| mid_gaussian_drift_0.5   | 87          | 61.11        | 59.38            | 46.54         | 10.57                  | 0.46        | 5                   | 4               |
| mid_gaussian_drift_1.0   | 87          | 122.08       | 113.17           | 102.41        | 10.57                  | 0.23        | 22                  | 15              |
| mid_gaussian_swap_0.5    | 87          | 66.35        | 66.85            | 55.66         | 11.72                  | 0.80        | 0                   | 0               |
| mid_gaussian_swap_1.0    | 87          | 132.70       | 127.91           | 119.59        | 11.72                  | 0.76        | 0                   | 0               |
| mid_swap_0.5             | 87          | 66.35        | 68.97            | 63.58         | 11.72                  | 0.79        | 1                   | 1               |
| mid_swap_1.0             | 87          | 132.70       | 132.14           | 127.37        | 11.72                  | 0.76        | 1                   | 1               |

![The arm-swap example with the largest excess final joint error. Top row shows the incoming pose references and bottom row the final poses. Blue is the owner, green the neighbor, and orange the proposal. The Gaussian control has matched input displacement magnitude.](../../results/person_interference/20261117/figures/corruption_example.png)

The arm-swap example with the largest excess final joint error. Top row shows the incoming pose references and bottom row the final poses. Blue is the owner, green the neighbor, and orange the proposal. The Gaussian control has matched input displacement magnitude.


![Excess joint error relative to the uncorrupted run through the four pose stages. Before a mid-stage injection, the trajectories coincide.](../../results/person_interference/20261117/figures/recovery_trajectories.png)

Excess joint error relative to the uncorrupted run through the four pose stages. Before a mid-stage injection, the trajectories coincide.


The half-dose Gaussian repeats use the same target proposals and displacement norms.

| Condition                       | Images   | Pose AP   | Box AP   | Δ pose AP   | Recall50 (%)   | Δ recall 95% interval   |
|:--------------------------------|:---------|:----------|:---------|:------------|:---------------|:------------------------|
| early_gaussian_swap_0.5         | 32       | 60.92     | 59.26    | -4.07       | 94.74          | +0.0 to +0.0            |
| early_gaussian_swap_0.5_seed43  | 32       | 60.61     | 59.23    | -4.39       | 94.74          | +0.0 to +0.0            |
| early_gaussian_swap_0.5_seed44  | 32       | 60.43     | 59.32    | -4.57       | 94.74          | +0.0 to +0.0            |
| early_gaussian_drift_0.5        | 32       | 54.42     | 57.75    | -10.58      | 90.23          | -7.7 to -1.5            |
| early_gaussian_drift_0.5_seed43 | 32       | 56.05     | 58.05    | -8.95       | 90.98          | -7.1 to -0.8            |
| early_gaussian_drift_0.5_seed44 | 32       | 55.50     | 57.73    | -9.50       | 91.73          | -6.1 to +0.0            |

## Implications for a geometry-aware pose graph

The current model already uses person tokens to regulate which hypotheses receive high confidence. Nearby-token removal changes AP more than removal of equally many random or faraway tokens, but those tokens can represent duplicate hypotheses as well as distinct people. A graph should therefore represent relations between hypotheses, with person identity treated as uncertain.

The most direct target for geometry-aware edges is duplicate competition and ownership under coherent displacement. Relative joint geometry could tell the model that two confident hypotheses agree on an entire person, or that one arm is being borrowed from a neighbor. The current inter-person route barely changes the tracked joint coordinates in the isolation test. Improving joint ownership would require the relation information to influence joint updates effectively during training.

I would judge a future learned relation module by three simultaneous outcomes on a fresh declared TEST subset, comprising fewer competitive duplicates at the same detection budget, fewer lineage transfers under coherent drift, and preserved high-resolution native AP. A ranking gain alone would already be measurable, but it would leave the joint-recovery question open. The full-drift stress test is deliberately severe and uses annotated neighbors to construct the corruption, so its large identity-transfer rate is a mechanism diagnostic rather than the natural frequency of that error.



## Measurement contract and limits



- The released checkpoint is `/home/gasparyanartur/dev/pose-graph-diff/external/DiffusionRegPose/pretrain_models/edpose_r50_crowdpose.pth` with SHA256 `0f50dae126a73ad2a15c41906c9ba6d8efa3421ef6cc2fbdc878238bd26b8f93`. It uses the native full six-encoder, two-box-stage and four-pose-stage path, with 900 candidate box queries and 100 expanded person groups. The model has 14 joints per person.
- Inference is FP32 with TF32 disabled. The native CUDA deformable-attention implementation is required. The fixed intervention seed is 42 plus image ID. Additional Gaussian repeats use 43 and 44 plus image ID.
- The annotation audit found 401 zero-count records in this subset, including 262 with labeled coordinates. These records contribute to physical crowding and foreground masking, while the native pose evaluator ignores them. The dataset annotation SHA256 is `67019ba94560f7a791265260d31f2bbb2e0d2484dd4711ed39cb2b6cba588190`. Sampling selects the established fixed32 TEST cohort plus 480 uniformly sampled remaining TEST IDs with seed 42. The 512 sample therefore contains a fixed component. It covers 1 to 13 eligible people per image. An eligible person has a positive `num_keypoints` field and `iscrowd=0`, matching the native evaluator’s ignore rule. This is distinct from recounting coordinate visibility flags.
- Native AP uses `crowdposetools.COCOeval`, including CrowdPose’s `0.53 × box area` normalization. Custom OKS agrees numerically with that implementation to less than 1e-12. Custom coverage uses eligible people and maximum-cardinality matching. Its recall is a separate diagnostic from the native AP/AR summary.
- Native postprocessing includes its score threshold 0.05 and pose suppression. Exact saved-prediction replay was checked for both 256 and 800 baselines on all 32 intervention images. Disabled hooks preserve raw joint coordinates exactly. Decoder-only conditions preserve the initial selected query IDs. Image changes can change proposal selection, and are evaluated as full-pipeline interventions.
- Tables show native AP point estimates. Recall intervals resample whole images with replacement, using 1,000 draws for descriptive strata and 2,000 paired draws for intervention differences. Small strata and the 32-image intervention cohort limit precision. These exploratory comparisons are not adjusted for multiple testing.
- Natural brightness, blur, crowding and overlap correlate with scene content, scale and occlusion. The image edits estimate sensitivity to the declared edits. Inference-time attention and state ablations also alter the distribution seen by a trained model. Their effects identify dependencies in these weights, and do not establish the performance of a retrained replacement architecture.
- Midpoint-like coordinates and limb mixtures are observable error patterns. They do not identify an internal probabilistic averaging mechanism. Annotated box background is an approximation to irrelevant pixels, and can contain useful context.


## Reproduction and artifacts

The report is accompanied by per-image native detections, unsuppressed outputs, raw proposal coordinates, scores, original query IDs, decoder references, person-token attention matrices, perturbation doses and image-mask counts. The image and checkpoint manifests record the exact inputs. Machine-readable tables contain the full threshold, budget, scene-stratum and condition results.



```bash
venv/bin/python tmp/person_interference/checks.py
venv/bin/python tmp/person_interference/verify.py
venv/bin/python tmp/person_interference/run.py --mode baseline
venv/bin/python tmp/person_interference/run.py --mode interventions
venv/bin/python tmp/person_interference/supplement.py
venv/bin/python tmp/person_interference/repair_background.py
venv/bin/python tmp/person_interference/analyze.py --mode baseline
venv/bin/python tmp/person_interference/analyze.py --mode interventions
venv/bin/python tmp/person_interference/annotation_audit.py
venv/bin/python tmp/person_interference/ownership_audit.py
venv/bin/python tmp/person_interference/extra_analysis.py
venv/bin/python tmp/person_interference/context_counterfactual.py
venv/bin/python tmp/person_interference/figures.py
venv/bin/python tmp/person_interference/make_report.py
```


[Subset and checkpoint manifest](../../results/person_interference/20261117/manifest.json) · [Machine-readable tables](../../results/person_interference/20261117/tables) · [Verification](../../results/person_interference/20261117/checks.json) · [Perturbation magnitude checks](../../results/person_interference/20261117/dose_checks.json) · [Analysis source fingerprints](../../results/person_interference/20261117/analysis_sources.json)
