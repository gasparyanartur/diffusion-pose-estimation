# Fidelity-first pose-and-box codec

Implemented 2026-11-04 from the approved fidelity-first plan. The acceptance gate
is **>=97.5 clean pose AP and >=97.5 clean bbox AP** on all CrowdPose validation
images. This threshold is not a zero-coordinate-error guarantee. No PoseFlow
integration or Gaussian-prior regularization is included.

## Diagnosis and protocol

The preceding VAE's frozen epoch049 checkpoint measured 63.5254 clean-mean pose
AP and 86.0017 bbox AP on validation; posterior samples measured 51.1568/78.6434.
Ground-truth copies measured 100/100. These are known-person **codec** AP, with
one reconstructed output per usable annotation, constant scores, annotation
order, and no NMS. They are not detector AP from images.

The evaluator uses native `crowdposetools.COCOeval`, `keypoints` and `bbox` modes,
thresholds .50:.05:.95, pose maxDets20 and bbox maxDets100. Targets are non-crowd
people with num_keypoints>0; boxes are actual annotations and pose OKS uses the
native denominator `8 * joint_sigma^2 * .53 * bbox_width * bbox_height`. Visibility
1 and 2 supply labeled coordinates; visibility0 does not. All 2,000 validation
images and 7,074 usable people are included in full runs. Constant confidence
scores remain independent of reconstruction error.

AP/AP75/AP95 are supplemented by paired-person OKS/IoU means and lower fifth
percentiles, and median/p95 joint and box errors. Joint errors are Euclidean,
in pixels and divided by the target maximum box extent. Center errors use
per-axis target width/height; size errors are absolute log ratios. Paired metrics
retain source-person identity even if native set matching can swap neighbors.
Ground-truth-copy and normalization-only ceilings must reach 100/100 before
training starts. Subset wiring smokes explicitly restrict evaluated annotations
and cannot satisfy the full-run gate.

## Deterministic control

`PoseBoxAutoencoder` retains width256/latent32, two residual MLP blocks on each
side, the isotropic box-centered joint representation, and independent box and
joint heads. It has **566,336 parameters**. All geometry passes through the
latent; decode accepts no coordinates or clean reference. No posterior sampling,
KL, input-coordinate jitter or synthetic masking is used. Natural missing labels
remain masked; training flips clean geometry with the CrowdPose permutation.

The public interfaces are:

```python
z = model.encode(keypoints_px, boxes_cxcywh_px, joint_mask, image_wh)
prediction = model.decode(z, image_wh)
```

They use pixel joints/boxes and image-size metadata, as in the VAE. The encoder
input has 46 values and the latent has 32. The decoder recomposes joints using its
predicted box. Existing VAE classes, checkpoints and runs are preserved.

The objective averages people equally and sums four terms with unit weights:

```
pose = mean_labeled_joints(squared_pixel_error / (8*sigma_joint^2*.53*box_area))
center = mean_xy(((predicted_center-target_center)/target_width_height)^2)
size = mean_wh(log(predicted_size/target_size)^2)
loss = mean_people(pose + center + size + 1 - aligned_GIoU)
```

Train 100 epochs, seed 42, batch 512, AdamW lr .001/weight decay .0001, cosine to .00001,
FP32 and gradient clipping 1. Best checkpoint maximizes the smaller of clean pose
and box AP. Save that checkpoint and last. If either AP is below 97.5, robustness
training does not start; report train/validation AP and tails instead.

## Gated decoder robustness

After a passing clean checkpoint, compute per-dimension population mean/std from
unflipped training records only and save them with the encoder fingerprint. Freeze
both encoder trunk and latent projection. Only the decoder and output heads train.
Mix equal clean and perturbed-latent losses, with independent per-person Gaussian
noise amplitude sampled uniformly from 0–5% of fixed training latent std. Because
the encoder and noise scale stay fixed, the model cannot evade this perturbation
by expanding its latent scale.

Fine-tune 50 epochs with AdamW lr .0001, cosine to .000001 and otherwise the control
recipe. Each epoch evaluates clean AP and 5% noise with fixed seed 1042. Checkpoint
selection maximizes the smaller noisy AP among candidates passing both clean AP
gates; failing candidates receive a score below every passing candidate.

Final curves measure 0/1/2.5/5/10% noise with paired draws across seeds 1042/2042/3042.
Accept the robust checkpoint only if both clean AP gates hold and the three-seed
mean 5% pose AP **and** bbox AP improve over the clean control. Otherwise retain
the passing clean checkpoint. Simulated Gaussian latent perturbations do not
establish robustness to actual downstream student prediction errors.

## Execution, verification and artifacts

Run `venv/bin/python scripts/training/train_pose_autoencoder.py`. The robust
preset is applied internally after the full clean gate passes. All fitting and
assessment use train/val only; test annotations are not loaded, hashed or scored.
The development runner rejects test-after-fit/evaluation-only runtime requests.

Verification: 11 new checks plus 33 VAE/shared-runner/data regressions pass. New checks
include native AP ceilings, exact normalization, natural missing labels and
outside-box joints, batch size one, fixed scaled noise, native CPU/CUDA model/optimizer/
scheduler/RNG restoration with exact next update, and frozen-encoder invariance.
Two-epoch GPU smokes exercise both phase lifecycles; they are wiring evidence.

Each experiment saves `ceilings.json`, clean/robust phase configurations and
source/annotation manifests, best/last checkpoints, validation AP histories and
qualitative/raw reconstruction grids. Final assessment includes training/validation
AP and error tails, training-only latent statistics, three-seed noise curves and
`completion.json` with explicit gate/acceptance decisions. Test stays untouched.

The first full experiment launched at 21:53 CET on 2026-11-04, PID 773969, under
`results/pose_autoencoder/20261104_215332`. Full-validation GT-copy and normalization
ceilings are both 100/100. All 395 source/config and two annotation hashes verified
with zero mismatches; train 35,659/val 7,074 people, zero image overlap. Test is untouched.
The 100-epoch clean phase completed, followed by full train/validation assessment
and three-seed latent-noise curves. No test data were accessed.

## Completed initial experiment: gate failed

All 100 epochs and **7,000 optimizer updates** completed with finite losses and
gradients. The selected checkpoint is epoch099, maximizing the minimum of clean
pose and box AP. Results are:

| Split | Pose AP | Box AP | Pose AP95 | Box AP95 |
| --- | ---: | ---: | ---: | ---: |
| Train, 35,659 people | 93.8084 | 96.9536 | 67.7343 | 79.7630 |
| Validation, 7,074 people | 92.4676 | 96.6083 | 62.9982 | 77.7481 |
| Required validation gate | 97.5 | 97.5 | — | — |

**Neither validation AP gate passed.** The 50-epoch robustness stage was not
started, no checkpoint was accepted for integration, and test remains untouched.
The candidate remains available for inspection at
`clean/checkpoints/epoch099-score92.4676.ckpt` within the run directory.

Validation paired-person mean OKS is .96245, with lower fifth percentile .86791;
paired mean box IoU is .97487. Median/p95 labeled-joint error is .01247/.04963
of target max box extent; p95 pixel error is 18.8994. Training median/p95 joint
error is .01201/.04680. Thus this recipe has not learned a sufficiently accurate
clean round trip even on training data. The similar train/validation shortfall
does not support attributing the failure solely to generalization. It does not
isolate architecture, optimization, loss weighting or convergence as the cause.

The clean-only checkpoint's latent-noise curve, averaged over three fixed seeds:

| Noise std / per-dimension training latent std | Pose AP | Box AP |
| --- | ---: | ---: |
| 0% | 92.4676 | 96.6083 |
| 1% | 91.9034 | 96.0473 |
| 2.5% | 89.3539 | 93.5378 |
| 5% | 78.6407 | 86.7229 |
| 10% | 49.5514 | 73.3029 |

These are perturbations of encoded validation latents, not coordinate corruption,
Gaussian-start generation, or actual downstream student outputs. Decoder
robustness training is implemented and wiring-tested but has not been evaluated
as a learned full-run intervention because its prerequisite failed.

Artifacts include `completion.json` with `clean_gate=false`, `robust_started=false`
and `accepted_checkpoint=null`; `clean_assessment.json`; `latent_statistics.pt`
and JSON; `clean_noise_curve.json`; `results_summary.json`; `verification.json`;
`clean_learning_curves.png`; `latent_noise_curves.png`; native best/last checkpoints;
and final grids/raw geometry under `clean/validation/epoch099.*`. The runtime
snapshot is under `source/`, and the plotting/summary script is under `analysis/`.

The combined deterministic/objective/schedule intervention improved validation
AP relative to the preceding VAE, but this run does not attribute gains to an
individual change and does not satisfy the requested quality target. No
unapproved extra training or architecture change followed the failed gate.

### Follow-up requiring a new iteration

First test whether this exact path can overfit a small fixed batch to near-perfect
paired geometry and AP, logging the separate pose/box terms and error tails.
That distinguishes a basic optimization/representation problem from a broader
dataset-fitting problem before choosing more capacity, different objectives or
a reversible geometry control. Keep the current failed candidate out of
downstream training while that issue is unresolved.

## Follow-up diagnostics and unchanged-code refinement

After the user instructed “Keep going”, temporary diagnostic scripts tested a
fixed, evenly spaced set of128training people. With the unchanged codec and loss,
a copy of the full-data checkpoint reached100/100pose/box AP after2500updates;
a fresh initialization reached100/100 after2750updates. This demonstrates that
the path can fit that batch to the AP ceiling. It is neither full-validation
evidence nor a guarantee of zero coordinate error. No diagnostic weights were
used to initialize the next full-data run.

Frozen full-validation readout substitutions measured:

| Readout | Pose AP | Box AP |
| --- | ---: | ---: |
| Original predictions |92.4676|96.6083|
| Ground-truth box, predicted relative joints |93.2854|100.0000|
| Predicted box, ground-truth relative joints |98.1912|96.6083|

These oracle substitutions identify a larger remaining contribution from joint
offsets under this intervention; they do not establish the effect of retraining
either branch. Error slices also show worse paired OKS for sparsely labeled and
small people. All diagnostic artifacts and scripts are retained under the next
run's `diagnosis/` directory.

A new experimental refinement starts from the original full-data epoch099
checkpoint with the exact same production codec and losses, seed42/batch512/FP32/
clip1/flip recipe. It uses fresh AdamW lr1e-4, weight decay1e-4 and100epochs of
cosine decay to1e-6. Model weights are initialized strictly; optimizer and RNG
are reset, so this is not an exact optimizer-state resume. The97.5/97.5 clean
gate, conditional frozen-encoder50epoch stage and three-seed acceptance policy
remain unchanged. Test data and previous runs remain untouched.

Refinement run: `results/pose_autoencoder_refinement/20261104_225527`, PID785828.
All395source/config and two annotation hashes match the frozen control; no
production model/config change was made. The experiment script and its hash,
initialization checkpoint hash, diagnostics, configuration and source snapshot
are retained in the run.

### Completed refinement: pose gate still fails

All 100 additional epochs completed. The selected checkpoint is epoch098
(6,930 updates in this phase), `clean/checkpoints/epoch098-score96.0492.ckpt`.
Its SHA256 is
`e7358ca464155a3a086ee2b6d92bc8501483dfb17bea3dee1dac646cc1cb8dcf`.

| Split | Pose AP | Box AP | Pose AP95 | Box AP95 |
| --- | ---: | ---: | ---: | ---: |
| Train, 35,659 people | 96.9882 | 98.3604 | 84.8808 | 90.3728 |
| Validation, 7,074 people | 96.0492 | 97.9153 | 80.0902 | 86.7823 |

Validation pose AP improved by 3.5816 points and box AP by 1.3071 points over
the original control. **Box AP passes; pose AP misses the gate by 1.4508 points.**
Training pose AP also remains below 97.5, so generalization alone does not explain
the remaining shortfall. The robustness phase was skipped, no checkpoint was
accepted for integration, and test remained untouched.

Validation paired mean/p05 OKS is .97772/.91795 and paired mean/p05 IoU is
.98239/.94178. Joint-error median/p95 is .00925/.03731 of target maximum box
extent, or 2.6976/13.9414 pixels. Training median/p95 is .00883/.03486 of box
extent. These tails remain material despite high average AP.

Three-seed latent-noise results for the selected clean checkpoint:

| Noise std / per-dimension training latent std | Pose AP | Box AP |
| --- | ---: | ---: |
| 0% | 96.0492 | 97.9153 |
| 1% | 95.7295 | 97.5255 |
| 2.5% | 93.0263 | 94.6784 |
| 5% | 80.3275 | 87.4597 |
| 10% | 47.1122 | 73.6263 |

This control remains sensitive to latent noise. No decoder robustness intervention
was trained in this run, so these curves do not establish its effectiveness.
Full-validation copy and normalization ceilings again reached 100/100. All 395
source/config and two annotation hashes verified without mismatches. The 44
previously passing tests apply to the unchanged production source; they were
reused rather than rerun for this experiment.

The run contains `completion.json`, `results_summary.json`, training-only latent
statistics, raw three-seed curves, learning/noise plots and native checkpoints.
`analysis/` adds selected-checkpoint raw reconstructions, varied/worst-case grids,
per-joint and person-size/label-count error slices, and the retained analysis
scripts. Previous experiments are preserved.

The user requested finishing this experiment before reviewing changes. The
conditional proposal in `.plan` adds a masked relative-joint loss at weight 1,
with default weight 0 preserving existing behavior, then measures one controlled
variant. **That proposal is deferred and has not been implemented or launched.**
