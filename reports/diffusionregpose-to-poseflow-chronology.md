# From DiffusionRegPose to PoseFlow: experiment chronology

Status: 2026-09-27. AP values below are fractions, so a difference of 0.01 is
one AP point. The main benchmark is the 8,000-image CrowdPose test annotation.
This is a history of measured work, including unsuccessful and unfinished
experiments, rather than a claim that every run was a matched ablation.

## Summary of results

- The released ResNet-50 DiffusionRegPose checkpoint reached **0.7268 keypoint
  AP** in our native evaluator, consistent with the [CVPR 2024 paper's 72.7
  AP](https://openaccess.thecvf.com/content/CVPR2024/papers/Tan_DiffusionRegPose_Enhancing_Multi-Person_Pose_Estimation_using_a_Diffusion-Based_End-to-End_Regression_CVPR_2024_paper.pdf).
  Our own external-code 80-epoch training reached **0.6880**; native training
  peaked at **0.7005** in training-loop validation and ended near **0.674**.
  The released weights work in our evaluator, but we have **not reproduced the
  paper's trained-model AP**.
- A small, fast model was useful for profiling but failed as a learning proxy.
  Preserving EDPose-compatible weights and then selecting transferable layers
  produced a usable 256/427 development model. In a controlled selected-layer
  comparison, pose-aware OTA improved final test keypoint AP from **0.5157**
  to **0.5490** over Hungarian matching.
- The first selected-layer PoseFlow reached **0.5501** test AP at one step.
  V3 lost substantial AP with repeated steps. The later paper-scale V5 reached
  **0.6813** at one step, but its 1–16-step sweep had **no meaningful positive
  keypoint AP response**: the largest gain over S=1 was only **0.00031**.
  DiffusionRegPose's own sweep gained **0.01396** at S=2. These are single
  checkpoint, single-seed observations; no statistical significance test was
  performed.

## Chronology

### 8–14 August: install and reproduce the external implementation

The [external DiffusionRegPose checkout](../external/DiffusionRegPose/) was
integrated into the shared environment, its CUDA extension and CrowdPose layout
were made runnable, and the released checkpoint was evaluated on all 8,000
test images. It returned **0.727 keypoint / 0.631 bbox AP**, establishing that
the data, inference path, and external evaluator could reproduce the supplied
model's result. Early local training, by contrast, produced effectively zero
AP. Matching the reference's batch size 1 with gradient accumulation 8 was a
turning point: the external reproduction reached **0.57645** keypoint AP by
epoch 30. The later 80-epoch external-code reproduction reached **0.6880
keypoint / 0.6206 bbox AP**, still below the released checkpoint. See the
[training-gap analysis](diffusionregpose-training-gap.md) and its
[external run configuration](../external/DiffusionRegPose/logs/crowdpose_r50_effective_batch8_resume80/config_args_all.json).

### 16 August–9 September: native implementation and parity

We built a native DiffusionRegPose pipeline with the ResNet-50/ED-Pose model,
deformable encoder and decoder, diffusion sampling, Hungarian criterion,
CrowdPose post-processing, and standalone evaluation. The implementation was
checked against external checkpoint loading and inference: a full prediction
comparison produced identical output records, and the native evaluator gave
**0.726776** keypoint AP for the released checkpoint. We then aligned training
transforms, crowd filtering, target construction, parameter sharing, random
initialization, and optimizer scheduling with the external code. The corrected
native paper-scale run used 80 epochs, batch 1 with accumulation 8, and the
reference learning-rate milestones. It peaked at **0.7005 keypoint / 0.6202
bbox AP** around epoch 50 in training-loop validation, then ended at **0.6745
/ 0.5925**. Late validation loss decreased while AP regressed. Some early AP
history was reconstructed after a logger overwrite; the [training-gap
report](diffusionregpose-training-gap.md) labels those points. The later
[standalone evaluation](../results/diffusionregpose_paper_full_standalone_eval/full_s1_seed42/metrics.json)
of the final checkpoint measured **0.673763 / 0.595505** at S=1. The
[resolved native configuration](../results/diffusionregpose_paper_full/diffusionregpose_paper_full/20260903_000016/.hydra/config.yaml)
records the training and test annotation paths.

### 9–12 September: faster models and the cost of reducing capacity

Profiling found that image resolution alone did not remove the decoder and
matching costs. The first 31.50M-parameter compact preset, with reduced
transformer depth, feature levels, FFN width, and query counts, measured
**38.58 images/s at batch 16** versus **2.63 images/s at batch 1** for the
paper-sized profile. This is a whole training-setting comparison, including
different image sizes and batch sizes. Its 256-pixel learning run reached only
about **0.0033 peak AP** through epoch 60 because much of the EDPose
transformer could not be transferred and only one pose-refinement layer
remained. The [runtime audit](diffusionregpose-runtime-audit.md) documents
both the measured speed and this failure.

The next compatible presets retained checkpoint-compatible weights while
reducing resolution and query-side work. At 256/427 pixels, a 200-query run
reached **0.5634 AP at epoch 20** before stopping; the 128/192 version reached
**0.5458 at epoch 80**. A 3-encoder/3-decoder selected-layer model then
transferred EDPose layers 2, 4, and 6 explicitly, used 200 initial queries
and 50 person groups, and reached **0.6302 validation keypoint / 0.5964 bbox
AP** after 80 epochs. Those altered-resolution runs are development results,
not paper-parity results. Architecture, timing, and AP provenance are collected
in the [efficiency journey](diffusionregpose-efficiency-journey.ipynb).

### 11–13 September: change matching and loss normalization

The native criterion gained an optional pose-aware Dynamic-K OTA matcher.
It assigns complete person groups using box and pose costs, allows multiple
clean-query positives per person, and normalizes each clean output by its own
positive count. Denoising queries keep their explicit target correspondence.
The original Hungarian path remained the control.

The matched 3-encoder/3-decoder, 256/427, single-seed comparison changed only
clean-query assignment in its resolved training configuration. At epoch 80,
OTA versus Hungarian validation keypoint AP was **0.703043 vs 0.630153**.
Independent full-test evaluation of the final checkpoints, however, gave
**0.548950 vs 0.515739** keypoint AP and **0.518699 vs 0.508753** bbox AP.
The test advantage is real for this pair of runs but smaller than the
validation difference; it has not been shown to transfer to the full paper
model or a second seed. The [OTA comparison](diffusionregpose-ota-comparison.md)
contains the resolved-config audit, trajectories, and test metric artifacts.

### 13–18 September: PoseFlow and successive state designs

The first PoseFlow replaced the diffusion state path with endpoint flow of
keypoint coordinates while retaining the ED-Pose architecture, OTA task
criterion, denoising, and pose NMS. Its selected-layer 256/427 checkpoint
reached [**0.5501265 keypoint / 0.5196890 bbox AP**](../results/poseflow_training/poseflow_training/20260914_020329/evaluation_epoch079/metrics.json)
at one Euler step. A bare S=1–8 sweep stayed within **0.000073 keypoint AP**;
renewal and corrected cross-step candidate aggregation gave no persuasive
improvement. A shared learned pose-codebook branch was implemented and
diagnosed, but the C=128 run was stopped at epoch 61 to pursue V3; there is no
completed final-AP comparison for that branch.

V3 transported box corners and joints together, carrying 50 selected states
and refilling 150 at each step. Its [fixed-seed full-test
sweep](../results/poseflow_v3_evaluation/step_sweep_summary.md) fell from
**0.5690 AP at S=1** to **0.5336 at S=2** and about **0.25–0.27 at S=4–16**.
V4 rebuilt coordinate-conditioned image hypotheses at each call rather than
carrying a latent hypothesis; its design and diagnostic training were
implemented, but no completed full-test V4 comparison is used here. V5 added
a two-cycle cosine schedule, same-image resampled/interpolated training
sources, and auxiliary box-center supervision. Those changes were introduced
together, so later V5 results cannot isolate any one of them. A separate
256/427 V5 [S=1 test](../results/poseflow_v5_evaluation/s01_seed1729/metrics.json)
was **0.5576 AP**. The model-family contracts are documented in
[project assumptions](../docs/project.md) and [architecture](../docs/architecture.md).

### 19–27 September: paper-scale V5 and direct step-response comparison

We trained V5 for 100 epochs with six encoder/six decoder layers, 900 initial
queries, 100 groups, 800/1333 resizing, batch 1 with accumulation 8, OTA,
and the requested learning-rate reset after epoch 50. The run used a fixed
1,000-image subset of the test annotation for periodic validation AP; the
table below uses separate full 8,000-image test evaluations of the immutable
[epoch-99 checkpoint](../results/poseflow_v5_paper_100e_training/poseflow_v5_paper_100e_training/20260919_115201/checkpoints/epoch099.ckpt).
Its [resolved config](../results/poseflow_v5_paper_100e_training/poseflow_v5_paper_100e_training/20260919_115201/.hydra/config.yaml)
records the training contract. A later first-and-final decoder-loss ablation
was launched, but it has no completed final test result as of this report and
is not included in the table.

The DiffusionRegPose column uses the final 80-epoch native checkpoint,
deterministic DDIM (`eta=0`), and global seed 42. The V5 column uses its final
100-epoch checkpoint, per-image seed 1729, fixed-source Euler flow, and no
renewal or ensemble. Both use 800/1333 resizing and the same 8,000-image
annotation. The models, training recipes, source distributions, samplers, and
seed contracts differ; the informative comparison is **each model's change
from its own S=1 baseline**.

| Steps | DiffusionRegPose keypoint AP | Change vs S=1 | PoseFlow V5 keypoint AP | Change vs S=1 | DiffusionRegPose bbox AP | PoseFlow V5 bbox AP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.673763 | 0 | 0.681345 | 0 | 0.595505 | 0.581246 |
| 2 | **0.687720** | **+0.013957** | 0.681522 | +0.000177 | 0.598977 | 0.580477 |
| 4 | 0.686809 | +0.013046 | **0.681657** | **+0.000312** | **0.600660** | 0.580388 |
| 8 | 0.683941 | +0.010178 | 0.681614 | +0.000269 | 0.596142 | 0.579280 |
| 16 | 0.679760 | +0.005997 | 0.680575 | -0.000770 | 0.590207 | 0.579127 |

All ten evaluations completed on the full test set. The
[DiffusionRegPose S=1 metrics](../results/diffusionregpose_paper_full_standalone_eval/full_s1_seed42/metrics.json)
and [S=2](../results/diffusionregpose_paper_full_standalone_eval/s02_seed42/metrics.json),
[S=4](../results/diffusionregpose_paper_full_standalone_eval/s04_seed42/metrics.json),
[S=8](../results/diffusionregpose_paper_full_standalone_eval/s08_seed42/metrics.json),
and [S=16](../results/diffusionregpose_paper_full_standalone_eval/s16_seed42/metrics.json)
are saved separately. The V5 metrics are likewise saved for
[S=1](../results/poseflow_v5_paper_100e_evaluation/s01_seed1729/metrics.json),
[S=2](../results/poseflow_v5_paper_100e_evaluation/s02_seed1729/metrics.json),
[S=4](../results/poseflow_v5_paper_100e_evaluation/s04_seed1729/metrics.json),
[S=8](../results/poseflow_v5_paper_100e_evaluation/s08_seed1729/metrics.json),
and [S=16](../results/poseflow_v5_paper_100e_evaluation/s16_seed1729/metrics.json).

The V5 curve is essentially flat: its best tested step count gains **0.0312
AP points** over S=1, while bbox AP declines slightly with more steps.
DiffusionRegPose improves by **1.3957 AP points** at S=2 before the gain
shrinks at longer sampling. In this sweep, extra flow steps did not deliver a
meaningful advantage, and neither our final DiffusionRegPose model nor V5
reaches the paper's 72.7 AP. This does not establish that one model family is
intrinsically better: training recipes and random-source contracts were not
matched, and the sweep has no repeated seeds or uncertainty estimate.

## Outstanding questions

- Why both independent external and native training trajectories fall short
  of the released checkpoint remains unresolved. The existing
  [training-gap analysis](diffusionregpose-training-gap.md) proposes a paired
  sample-order, augmentation, noise, and optimizer-step comparison.
- The selected-layer OTA gain needs a full-size matched Hungarian/OTA test and
  another seed before it can be generalized.
- The first-and-final V5 loss ablation and any seed-sensitivity runs should be
  reported only after their final, complete evaluation artifacts exist.
