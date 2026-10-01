# From DiffusionRegPose to PoseFlow: experiment chronology

Status: 2026-09-30. AP values below are fractions, so a difference of 0.01 is
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
  DiffusionRegPose's own sweep gained **0.01396** at S=2 for seed 42. Its
  separate five-inference-seed S=1/S=2 test was positive for keypoint AP in
  every paired seed (mean gain **0.01276**); it does not measure training-seed
  or test-set uncertainty.
- V6 tested whether frozen multiscale DINO image features could improve the
  image-conditioned decoder: its reduced 3-encoder/3-decoder epoch-79 model
  at 256/427 evaluation resize reached **0.5630** full-test keypoint AP at one
  step and **0.5643** at two (per-image seed 42).
  V7 reduces the two-backbone design; V8 lets DINO alone drive coarse person
  proposals and fine pose refinement. V7/V8 have no completed full-test AP
  comparison in this report.

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
reducing resolution and query-side work. A **6-encoder/6-decoder Hungarian**
model at 256/427 pixels and 200 initial queries reached **0.5634 validation
AP on `crowdpose_val.json` at epoch 20** before stopping; its 128/192 version
reached **0.5458 validation AP at epoch 80**. A different **3-encoder/3-decoder
Hungarian** model transferred EDPose layers 2, 4, and 6 explicitly, used 200
initial queries and 50 person groups, and reached **0.6302 validation
keypoint / 0.5964 bbox AP at epoch 80**. These are development runs with
different architectures and checkpoints, not paper-parity results. Their
architecture and AP provenance are in the
[efficiency journey](diffusionregpose-efficiency-journey.ipynb).

### 11–13 September: change matching and loss normalization

The native criterion gained an optional pose-aware Dynamic-K OTA matcher.
It assigns complete person groups using box and pose costs, allows multiple
clean-query positives per person, and normalizes each clean output by its own
positive count. Denoising queries keep their explicit target correspondence.
The original Hungarian path remained the control.

The matched 3-encoder/3-decoder, 256/427, single-seed comparison changed only
clean-query assignment in its resolved training configuration. At epoch 80,
OTA versus Hungarian validation keypoint AP was **0.703043 vs 0.630153**.
Standalone evaluation of those final checkpoints on the full test split gave
**0.548950 vs 0.515739** keypoint AP and **0.518699 vs 0.508753** bbox AP.
The test advantage holds for this pair of runs but is smaller than the
validation difference. The [OTA comparison](diffusionregpose-ota-comparison.md)
contains the resolved-config audit and full metrics.

These AP values describe **different runs or different evaluation splits**:

| Run and evaluation point | Architecture | Assignment | Evaluation | Keypoint AP |
| --- | --- | --- | --- | ---: |
| [Compatible 256, epoch 20](../results/diffusionregpose_development_compatible_256/dev256-compatible-full/20260910_075409/csv_logs/metrics.csv) | 6 encoder / 6 decoder | Hungarian | 2,000-image `crowdpose_val.json`, training-loop validation | 0.563411 |
| [Selected-layer, epoch 80](../results/diffusionregpose_development_compatible_256_keep_layers/diffusionregpose_development_compatible_256_keep_layers/20260912_000637/csv_logs/metrics.csv) | 3 encoder / 3 decoder | Hungarian | 2,000-image `crowdpose_val.json`, training-loop validation | 0.630153 |
| [Selected-layer, epoch 80](../results/diffusionregpose_development_compatible_256_keep_layers/test_eval_256_v3/metrics.json) | 3 encoder / 3 decoder | Hungarian | 8,000-image `crowdpose_test.json`, standalone S=1 | 0.515739 |
| [Selected-layer, epoch 80](../results/diffusionregpose_development_compatible_256_keep_layers_ota/test_eval_256_v3/metrics.json) | 3 encoder / 3 decoder | OTA | 8,000-image `crowdpose_test.json`, standalone S=1 | 0.548950 |

Thus **0.5634 is not the Hungarian control for the 0.5490 OTA result**. It is
an earlier, deeper model measured on validation after 20 epochs. The
controlled test comparison is **0.548950 OTA versus 0.515739 Hungarian** for
the two selected-layer models. The large validation-to-test drops warrant
caution; these artifacts do not by themselves establish their cause. Nor does
this one-seed, reduced-model result establish a full paper-scale OTA gain.

### 13–18 September: PoseFlow and successive state designs

The first PoseFlow replaced the diffusion state path with endpoint flow of
keypoint coordinates: a direct source-to-pose trajectory was meant to let
repeated decoder calls refine a simple, fixed-source state. It retained the
ED-Pose architecture, OTA task criterion, denoising, and pose NMS. Its
selected-layer 256/427 checkpoint
reached [**0.5501265 keypoint / 0.5196890 bbox AP**](../results/poseflow_training/poseflow_training/20260914_020329/evaluation_epoch079/metrics.json)
at one Euler step. A bare S=1–8 sweep stayed within **0.000073 keypoint AP**;
renewal and corrected cross-step candidate aggregation gave no persuasive
improvement. A shared learned pose-codebook branch tested whether reusable
pose-hypothesis tokens could enrich image-conditioned queries; it was
implemented and diagnosed, but the C=128 run was stopped at epoch 61 to pursue
V3. There is no
completed final-AP comparison for that branch.

V3 transported box corners and joints together, carrying 50 selected states
and refilling 150 at each step. Its [fixed-seed full-test
sweep](../results/poseflow_v3_evaluation/step_sweep_summary.md) fell from
**0.5690 AP at S=1** to **0.5336 at S=2** and about **0.25–0.27 at S=4–16**.
Carrying a small selected state set made the current coordinates available at
the next call, but the observed collapse showed that this transport rule did
not make repeated refinement reliable. V4 therefore rebuilt 200
coordinate-conditioned, image-attending hypotheses at every call rather than
carrying a latent hypothesis; its 50 selected states still carried their
coordinates and fixed sources, and 150 fresh states refilled the pool. This
made the image available when interpreting the current state, but V4 has no
completed full-test comparison here. V5 kept that structure and aimed to
reduce training/source mismatch: it added same-image resampled and interpolated
sources, a two-cycle cosine schedule, and auxiliary box-center supervision.
Those changes were introduced together, so later V5 results cannot isolate
any one of them. A separate
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
matched. The V5 sweep has one inference seed; the DiffusionRegPose S=1/S=2
seed study below measures a different, narrower source of variation.

### 27–30 September: inference seeds and the V5 loss ablation

We repeated DiffusionRegPose S=1 and S=2 with inference seeds 42–46, using the
same final native epoch-79 checkpoint, 800/1333 resize, deterministic DDIM
(`eta=0`), and standalone evaluator on all 8,000 CrowdPose test images. Each
seed pairs its S=1 and S=2 evaluations; the evaluator seeds proposal sampling
before model construction, so the paired comparison holds the initial random
draw fixed within that seed. AP values remain fractions.

| Inference seed | S=1 keypoint AP | S=2 keypoint AP | S=2 minus S=1 | S=2 minus S=1 bbox AP |
| ---: | ---: | ---: | ---: | ---: |
| 42 | [0.673763](../results/diffusionregpose_paper_full_standalone_eval/full_s1_seed42/metrics.json) | [0.687720](../results/diffusionregpose_paper_full_standalone_eval/s02_seed42/metrics.json) | +0.013957 | +0.003472 |
| 43 | [0.674178](../results/diffusionregpose_paper_full_standalone_eval/s01_seed43/metrics.json) | [0.687044](../results/diffusionregpose_paper_full_standalone_eval/s02_seed43/metrics.json) | +0.012866 | +0.002143 |
| 44 | [0.675250](../results/diffusionregpose_paper_full_standalone_eval/s01_seed44/metrics.json) | [0.686468](../results/diffusionregpose_paper_full_standalone_eval/s02_seed44/metrics.json) | +0.011217 | -0.000034 |
| 45 | [0.675040](../results/diffusionregpose_paper_full_standalone_eval/s01_seed45/metrics.json) | [0.687061](../results/diffusionregpose_paper_full_standalone_eval/s02_seed45/metrics.json) | +0.012021 | +0.000180 |
| 46 | [0.673520](../results/diffusionregpose_paper_full_standalone_eval/s01_seed46/metrics.json) | [0.687236](../results/diffusionregpose_paper_full_standalone_eval/s02_seed46/metrics.json) | +0.013716 | +0.002032 |

The paired keypoint gain is positive in all five seeds, with mean **0.012756**,
sample standard deviation **0.001149**, and observed range **0.011217–0.013957**.
The S=1 and S=2 AP means are **0.674350** and **0.687106**, respectively.
Bbox AP increases in four seeds and is nearly unchanged but slightly negative
in one; its mean paired gain is **0.001559**. These are inference-seed
descriptives for one checkpoint and one fixed test set, not a formal confidence
interval or evidence about retraining variation.

The paper-scale V5 `decoder_loss_mode=first_last` ablation was stopped after
epoch 55; its [resume checkpoint](../results/poseflow_v5_paper_first_last_100e_training/poseflow_v5_paper_first_last_100e_training/resume_20260925_2133/checkpoints/last.ckpt)
is preserved, but no final full-test evaluation supports a step-response
conclusion for it.

### 29–30 September: DINO-conditioned V6, smaller V7, and DINO-only V8

The V3 and V5 step sweeps motivated a different question: could stronger
image features help the decoder interpret each current pose hypothesis, where
additional coordinate updates alone had not helped? V6 keeps V5's flow state,
source construction, OTA losses, and sampler, and adds a frozen DINOv3
ConvNeXt-Small branch. At every configured decoder layer, separate deformable
attention reads DINO stages 2/3/4 at the current query references (strides
8/16/32). An MLP combines the three readouts with the existing query; a
zero-initialized gate adds the result after ordinary image attention. This is
query-side conditioning, not an added encoder pyramid: ResNet still creates
the person proposals. The [resolved V6 config](../results/poseflow_v6_training/poseflow_v6_training/20260929_024515/.hydra/config.yaml)
uses 3 encoder/3 decoder layers, 200 initial queries, 50 proposals, 256/427
evaluation resizing, all-stage losses, and separate main/ResNet cosine
warmups. The epoch-79 [checkpoint](../results/poseflow_v6_training/poseflow_v6_training/20260929_024515/checkpoints/epoch079.ckpt)
completed training.

The V6 standalone evaluator, using that checkpoint, the full 8,000-image test
split, per-image flow seed 42, no renewal, and no ensemble, has completed these
steps:

| Flow steps | Keypoint AP | Change from S=1 | Bbox AP |
| ---: | ---: | ---: | ---: |
| 1 | [0.563047](../results/poseflow_v6_training/poseflow_v6_training/20260929_024515/evaluation_retry_20260930/crowdpose_test_seed42_steps1_metrics.json) | 0 | 0.511975 |
| 2 | [0.564296](../results/poseflow_v6_training/poseflow_v6_training/20260929_024515/evaluation_retry_20260930/crowdpose_test_seed42_steps2_metrics.json) | +0.001250 | 0.512319 |
| 3 | [0.563995](../results/poseflow_v6_training/poseflow_v6_training/20260929_024515/evaluation_retry_20260930/crowdpose_test_seed42_metrics.json) | +0.000949 | 0.511525 |
| 4 | [0.564200](../results/poseflow_v6_training/poseflow_v6_training/20260929_024515/evaluation_retry_20260930/crowdpose_test_seed42_steps4_metrics.json) | +0.001153 | 0.511362 |

This is a small positive step response for one V6 seed, far smaller than the
DiffusionRegPose five-seed S=2 gain above. V6's training-loop validation on
the 2,000-image `crowdpose_val.json` split recorded **0.772751 keypoint AP**
at epoch 79 ([CSV](../results/poseflow_v6_training/poseflow_v6_training/20260929_024515/csv_logs/metrics.csv)); the lower full-test AP is a
different split and evaluator path. The cause of that gap has not been
established. S=6/8 evaluation is still in progress and is excluded here.

V7 asks whether a smaller two-backbone version can keep the proposed DINO reads
at lower model size. It uses frozen ConvNeXt-Tiny stages 2 and 4 (strides 8
and 32), with two separate per-layer DINO readouts, while the base transformer
uses only two ResNet maps (strides 8 and 16). Thus its four attended maps are
split across two paths, not a four-level shared encoder. Its reduced
[resolved config](../results/poseflow_v7_training/poseflow_v7_resume_20260930/20260930_180743/.hydra/config.yaml)
retains V6's flow, all-stage losses, 3/3 transformer depth, 200/50 query
setup, 256/427 resizing, and optimizer schedule. The resumed training run
had an [epoch-29 checkpoint](../results/poseflow_v7_training/poseflow_v7_resume_20260930/20260930_180743/checkpoints/epoch029.ckpt)
at this report's snapshot; no V7 full-test AP result is used here.

V8 tests the broader hypothesis that frozen DINO features should also drive
the *proposal* path, instead of appearing only as a decoder residual after
ResNet has selected people. It removes ResNet and the V7 fusion branch. The
same frozen ConvNeXt-Tiny supplies four native stages (strides 4/8/16/32),
without synthesizing another pyramid. Independent `1x1 Conv + GroupNorm`
adapter banks keep coarse encoder/query features separate from fine decoder
features. The current [resolved V8 config](../results/poseflow_v8_coarse234_training/poseflow_v8_coarse234_training/20260930_191659/.hydra/config.yaml)
uses stages 2/3/4 for coarse encoder attention and person-proposal selection,
and all four stages for fine decoder deformable attention. Omitting the
stride-4 map from the encoder avoids its dense token cost while retaining its
spatial detail for pose refinement. The earlier all-four-coarse V8 startup
stopped during validation without a checkpoint; its
[launch log](../results/poseflow_v8_training/launch_20260930.log) was preserved, and
the revised 3-level-coarse/4-level-fine run started fresh from compatible
EDPose transformer/head tensors. The [V8 architecture and parameter
accounting](../docs/architecture.md) document this checkpoint boundary.
At this snapshot, revised V8 had an [epoch-2 checkpoint](../results/poseflow_v8_coarse234_training/poseflow_v8_coarse234_training/20260930_191659/checkpoints/epoch002.ckpt)
and no full-test AP result. V7 and V8 are architectural tests, not established
improvements over V5 or V6.

## Outstanding questions

- Why both independent external and native training trajectories fall short
  of the released checkpoint remains unresolved. The existing
  [training-gap analysis](diffusionregpose-training-gap.md) proposes a paired
  sample-order, augmentation, noise, and optimizer-step comparison.
- The selected-layer OTA gain needs a full-size matched Hungarian/OTA test and
  another seed before it can be generalized.
- V5 first-and-final loss training stopped before final evaluation. A matched
  completed ablation would be needed to attribute any change to the loss mode.
- V6's validation-to-test AP gap needs an evaluator/split audit. Its full step
  sweep and V7/V8 final test evaluations are pending; their DINO designs have
  not yet established an AP or runtime advantage in a matched comparison.
