# Off-the-shelf CrowdPose missing-joint completion

9 November 2026. Approved first evaluation iteration; fill visibility=0 only.

## Finding

RTMPose produces finite coordinates for every missing joint in the evaluated posed-person subset, but this does not establish correct full annotations. It recovers existing labels well, while missing joints show much weaker flip agreement, many crop-edge predictions, and visible examples of collapsed legs or wrong-person capture. Recommend a reviewed annotation proposal workflow rather than accepting all predictions as ground truth.

No existing annotation was changed. No TRAIN inference, TEST access, teacher fine-tuning, model integration or full-dataset export was performed.

## Frozen teacher and protocol

- Teacher: local `checkpoints/rtmpose-m_crowdpose.pth`, CrowdPose 14 RTMPose-M, native 256×192 crop, official config and strict state-dict loading. Dataset top_head matches this repository's head index 12; no COCO 17 conversion.
- Weight SHA256: `fd7041be5f9ebac8866e700fdfdc368aed543c5bf7483154eadd94d41fff016c`. Checkpoint metadata: epoch 200, iteration 16601.
- Native original-image person boxes, positive-size checked, `iscrowd==0` and `num_keypoints>0`. No keypoint coordinates enter teacher inference.
- Primary prediction uses the released `flip_test=True`. Consistency uses two separate single passes with flip TTA disabled, mirrored boxes, and native left/right joint permutation. It measures original/flip sensitivity rather than comparing an averaged prediction with itself.
- Seed 42 uniform 256-image VAL pilot; then all 2,000 VAL images and 7,074 valid posed people. Full pass reuses 877 pilot people only after annotation/config/checkpoint/evaluation-source hash checks.
- Teacher training config includes TRAIN+VAL, including the evaluated VAL images. All measured label-recovery values are in-sample and GT-box-assisted; none is end-to-end detector AP or missing-joint accuracy.
- Raw flag 1 and flag 2 are reported separately. Head/neck never have flag 2 when labeled in these files. Treat physical occlusion semantics as unresolved rather than assuming every flag 1 is truly invisible.
- Runtime versions: torch 2.4.1+cu121, numpy 1.26.4, mmcv 2.2.0, mmengine 0.10.7, mmpose 1.3.2, mmdet 3.3.0.

## Reference-label recovery

PCK@.05/.10: fraction of eligible joints whose Euclidean pixel error is at most .05/.10 times max(original person-box width,height). Joint-mean OKS uses native CrowdPose sigmas and area=.53*box width*height. These are conditional coordinate diagnostics, not AP. Intervals use500 image-cluster bootstrap draws (seed 42); joints/people from the same image stay together.

| Reference stratum | Joints | PCK@.05 | 95% image-bootstrap CI | PCK@.10 | Mean joint OKS |
|---|---:|---:|---:|---:|---:|
| All labeled | 76,924 | 90.28% | 89.81–90.78% | 96.19% | 0.9314 |
| Raw flag 1 | 49,883 | 89.54% | 88.95–90.10% | 95.30% | 0.9270 |
| Raw flag 2 | 27,041 | 91.65% | 91.04–92.22% | 97.83% | 0.9395 |
| Flag1 limbs only | 37,020 | 86.61% | 85.81–87.40% | 94.18% | 0.9099 |
| Flag2 limbs only | 27,041 | 91.65% | 91.04–92.22% | 97.83% | 0.9395 |
| Head and neck | 12,863 | 97.96% | 97.62–98.27% | 98.52% | 0.9762 |

Labeled ankles are harder than head/neck: left/right ankle PCK@.05 = 85.51%/86.20%, versus 98.12%/97.81% for head/neck. All 14 per-joint/flag results and CIs are in `summary.json`. Even these supervised-label results do not measure the missing-joint population.

## Missing-joint proposals

There are 22,112 flag 0 joints on VAL. All receive finite coordinate proposals, preserving every original zero flag and all labeled coordinates in the source copy. Missing-joint accuracy remains undefined: no reference coordinates or human adjudications have been supplied.

| Diagnostic | Missing joints | Fraction of missing joints |
|---|---:|---:|
| Independent flip disagreement ≤2% box size | 6,234 | 28.19% |
| Independent flip disagreement ≤5% box size | 11,798 | 53.36% |
| Independent flip disagreement ≤10% box size | 15,184 | 68.67% |
| Native score ≥0.5 | 4,968 | 22.47% |
| Native score ≥0.6 | 2,541 | 11.49% |
| Native score ≥0.7 | 1,080 | 4.88% |
| Score ≥.6 AND flip disagreement ≤2% | 1,825 | 8.25% |

At score ≥ .6 and flip disagreement ≤ 2%, existing labeled-joint PCK@.05 is 97.95%, but missing-joint precision is unknown. Those thresholds are illustrative diagnostics, not an approved acceptance rule or a calibrated probability. Agreement can be consistently wrong.

Missing-joint median flip disagreement is 4.40% of box size, 90th percentile 33.90%, and 95th percentile 54.29%. The all-missing raw mean score is .3516. All confidence and consistency curves are retained, including the unfiltered baseline.

Of the missing proposals, 1,840 (8.32%) lie outside the original image and 10,674 (48.27%) outside the original box. These are location counts, not errors; out-of-frame joints may legitimately be outside an image. No coordinates were clamped.

| Joint | Missing VAL joints | Flip agreement within 2% |
|---|---:|---:|
| left_shoulder | 588 | 36.73% |
| right_shoulder | 589 | 38.20% |
| left_elbow | 1,467 | 33.95% |
| right_elbow | 1,466 | 34.65% |
| left_wrist | 1,744 | 24.60% |
| right_wrist | 1,734 | 24.22% |
| left_hip | 1,142 | 28.72% |
| right_hip | 1,158 | 29.36% |
| left_knee | 2,318 | 28.39% |
| right_knee | 2,347 | 29.10% |
| left_ankle | 3,113 | 23.23% |
| right_ankle | 3,161 | 25.25% |
| head | 736 | 32.20% |
| neck | 549 | 31.15% |

## Crop support and visual failures

The configured SimCC decoder uses a finite coordinate-bin grid within a native person crop, with `use_dark=False`. GetBBoxCenterScale expands the supplied box by 1.25; TopdownAffine adjusts its aspect ratio to 192/256. Valid bin peaks therefore cannot represent arbitrary joint positions far outside that crop. If a truncated person's true missing ankle lies beyond it, ordinary decoding cannot reach that coordinate. This is a code-backed output-support limitation; no expanded-box ablation was run.

34.32% of missing predictions lie within 5% of a native crop edge, compared with 1.98% of labeled predictions. That concentration supports inspecting crop-boundary effects but is not an accuracy metric or proof that any particular missing coordinate is wrong.

The four random sheets contain 48 reproducibly sampled people with missing joints. All four sheets and the 12-person worst-labeled-error sheet were inspected qualitatively. Examples include:

- Image 110718 / annotation 186360 (random sheet 3): the player's missing lower limbs collapse around the lower torso/crop boundary.
- Image 107762 / annotation 122940 (random sheet 4): missing leg joints cluster near the lower torso in a truncated-person image.
- Image 104889 / annotation 154996 (failure sheet): a partial-person crop yields a strongly misbound/full-body guess.
- Several less-occluded examples retain good visible-limb alignment and plausible hidden-joint proposals. Plausibility does not supply a reference coordinate.

This is visual diagnosis, not a human-validated missing-joint accuracy study. The review CSV leaves adjudication/reference-coordinate fields blank. Tiny original crops and points extending outside a tile require checking the original image before adjudication.

[Random missing proposals, sheet 1](../../results/joint_completion/20261109_rtmpose/fullval/random_missing_01.jpg) · [sheet 2](../../results/joint_completion/20261109_rtmpose/fullval/random_missing_02.jpg) · [sheet 3](../../results/joint_completion/20261109_rtmpose/fullval/random_missing_03.jpg) · [sheet 4](../../results/joint_completion/20261109_rtmpose/fullval/random_missing_04.jpg) · [labeled failures](../../results/joint_completion/20261109_rtmpose/fullval/labeled_failures_01.jpg) · [human review CSV](../../results/joint_completion/20261109_rtmpose/fullval/human_review.csv)

## Full annotation coverage and decision

On existing posed people, TRAIN has 112,562 missing joints (22.55%) and VAL 22,112 (22.33%). A complete all-person supplement would also have to cover 6,757 noncrowd TRAIN boxes and 1,347 noncrowd VAL boxes with no pose labels at all. Another 558/106 TRAIN/VAL records are crowd records, excluded from this person-pose task. No quality/identity claim is established for the box-only population.

Recommended next iteration: adjudicate a small missing-joint sample with a correct/incorrect/ambiguous label and reference coordinates where defensible; then compare a stronger CrowdPose 14 teacher on identical people before choosing an acceptance/export policy. Keep unknown and out-of-frame cases explicit. A future supplement should preserve original GT, expose prediction source/confidence/review state separately, and leave official evaluation annotations unchanged.

ViTPose-L is the scoped stronger candidate: its official repository releases a CrowdPose 14 config/weight link and reports 76.6 aggregate test AP. That does not establish missing-joint accuracy. Its legacy TopDown implementation requires MMCV 1.3.8–1.5.0, whereas this environment has 2.2.0; an isolated legacy runtime or verified conversion needs a separately reviewed follow-up. Repository code is Apache 2.0; no checkpoint was downloaded or redistributed in this iteration. [Official models/config](https://github.com/ViTAE-Transformer/ViTPose#results), [runtime constraint](https://github.com/ViTAE-Transformer/ViTPose/blob/main/mmpose/__init__.py), [code license](https://github.com/ViTAE-Transformer/ViTPose/blob/main/LICENSE).

## Verification and artifacts

- 6 focused tests pass, plus 4 related evaluator checks (10 total). Missing-module red captured before implementation; a native xyxy/xywh flip-box regression failed before its correction. Final inference uses the corrected path.
- Independent official `crowdposetools.COCOeval.computeOks` parity on 42 pilot people: maximum absolute difference 2.22e-16. The official API constructor uses `keypoints`; this is distinct from the MMPose metric wrapper's `keypoints_crowd` spelling.
- All 7,074 output identities and entire original annotation copies match the raw VAL annotations exactly. TRAIN/VAL annotation hashes still match the pre-task audit. Native output shape, input-image dimensions, box order, finite coordinates and strict checkpoint loading checked.
- Two-image corrected native smoke precedes the pilot. An earlier smoke with incorrect consistency-box formatting is retained as invalid for consistency and was not reused by pilot/fullVAL.
- Pilot inference 23.24 s; incremental fullVAL inference 157.18 s with 256 images reused. Full-pass own peak CUDA allocation 140,213,248 bytes (about 134 MiB), small person batches of 8. Shared-GPU timing is descriptive, not an isolated speed benchmark. No existing jobs were stopped or reconfigured.
- Evaluation-source/input-annotation/config/checkpoint hashes are recorded in each manifest. Source snapshots are archived. Original-image and extra native-source hashes were captured after inference; no pre-run image-byte hashes were recorded, so do not claim before/after image-byte parity.

- [Full summary](../../results/joint_completion/20261109_rtmpose/fullval/summary.json), [manifest](../../results/joint_completion/20261109_rtmpose/fullval/manifest.json), [prediction cache](../../results/joint_completion/20261109_rtmpose/fullval/predictions.jsonl), [additional audit](../../results/joint_completion/20261109_rtmpose/fullval/additional_diagnostics.json).
- [Pilot summary](../../results/joint_completion/20261109_rtmpose/pilot/summary.json), [final test log](../../results/joint_completion/20261109_rtmpose/regressions_final.log), [official metric parity](../../results/joint_completion/20261109_rtmpose/official_metric_parity.json).

Run from the configured repository environment; output directories must be new:

```bash
python -m scripts.evaluation.evaluate_joint_completion \
    --limit-images 256 --output results/joint_completion/new_pilot

python -m scripts.evaluation.evaluate_joint_completion \
    --limit-images 0 \
    --reuse-predictions results/joint_completion/new_pilot/predictions.jsonl \
    --output results/joint_completion/new_fullval
```

CLI defaults to raw VAL annotations. It writes proposal records, metrics and review sheets only; there is no mutation/export-to-training mode.
