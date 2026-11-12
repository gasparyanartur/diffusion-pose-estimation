# Image-free pose-and-box denoising VAE

Implemented 2026-11-04 following the approved standalone-codec plan. This learns
individual CrowdPose people from annotation JSONs without loading images. It has
no existence output, background examples, metric-learning objective or PoseFlow
integration. Reconstruction metrics do not measure detector AP.

## Geometry and model

For supplied pixel-space center/size box `(c,w,h)`, define `s=max(w,h)` and
`relative_joint=(joint-c)/s`. The same scalar divides both axes, preserving
proportions. Box placement is `(cx/W,cy/H,log(w/W),log(h/H))`; original image WH
comes only from annotation metadata. Encoder normalization uses the supplied,
possibly corrupted box. No target box is consulted by the encoder.

Input has 46 values: 28 relative coordinates, four placement values, and 14
observation-mask bits. Missing coordinates become zero after normalization.
The width-256 encoder and mirrored decoder each have an input projection,
two residual two-linear-layer blocks, SiLU and LayerNorm. Gaussian heads output
32 means and logvariances (bounded to `[-10,2]`); means are not normalized.
Total trainable parameters: **574,560**.

Separate decoder heads return box placement and 14 relative joints. Box log sizes
are bounded to `[-10,2]` before exponentiation for positive, finite sizes; center
and joint readouts are unconstrained. Absolute joints are recomposed using the
predicted box center and its maximum pixel extent. Boxes are never derived from
joint extrema. Labeled joints outside boxes or image bounds remain valid.

## Data, corruption and objective

The default loader uses 35,659 train people and 7,074 validation people, with zero
train/val image overlap. It keeps non-crowd people with labeled joints; visibility
1 (occluded) and 2 (visible) both supervise geometry. Visibility 0 never supplies a
coordinate target. Test is loaded for final evaluation after fitting, using the
best validation checkpoint. No test metric selects the training checkpoint.

Training first flips both clean targets and input geometry with probability .5,
including CrowdPose left/right permutation. It leaves 25% of inputs uncorrupted.
For other inputs, per-person joint and box-center Gaussian standard deviations
are independently drawn from `[0,.05]*s`, box log-size jitter has standard
deviation .1, and joint-drop probability is uniform in `[0,.3]`. Dropout retains
at least one labeled observation. Randomly hidden labels remain clean targets.

The reconstruction objective gives each person equal weight:

```
pose = mean_labeled_coordinates(((predicted_joint-clean_joint)/clean_s)^2)
box = mean_xy(((predicted_center-clean_center)/clean_s)^2)
      + mean_wh(log(predicted_size/clean_size)^2)
reconstruction = mean_people(pose+box)
KL = mean_people(.5*sum_latent(mean^2+exp(logvar)-1-logvar))
loss = reconstruction + beta*KL
```

Reconstruction uses posterior samples. Beta is `1e-4*min(epoch/10,1)`, starting
at zero in epoch 0. Seed 42, AdamW lr .001/weight decay .0001, cosine decay to
.00001 over 50 epochs, batch 512, FP32, gradient clipping 1.0. Save best noisy
sample validation reconstruction and last checkpoints. Annotation records are
memory-resident; the default uses zero data-loader workers.

## Inference

```python
import torch
from src.models.pipelines.pose_vae.training import PoseVAEPipeline

pipeline = PoseVAEPipeline.load_from_checkpoint(checkpoint, weights_only=False).eval()
with torch.no_grad():
    mean, logvar = pipeline.model.encode(
        keypoints_px, boxes_cxcywh_px, joint_mask, image_wh)
    reconstructed = pipeline.model.decode(mean, image_wh)
    generated = pipeline.model.decode(torch.randn_like(mean), image_wh)
```

Inputs have shapes `[B,14,2]`, `[B,4]`, `[B,14]` and `[B,2]`; output keys are
`keypoints_px` and `boxes_cxcywh_px`. Use a bool observation mask. These are
pixel-space APIs, an explicit boundary distinct from the detector's canonical
joints and normalized boxes. Generation uses only latent draws and image-size
metadata. It supplies one person's geometry, without a person-presence score.

## Validation and artifacts

Fixed private validation corruption/posterior RNG resets each epoch and never
advances training RNG. Validation compares clean mean/sample and noisy mean/sample
decoding. It reports reconstruction terms, labeled-joint Euclidean errors in
pixels and clean-scale units, artificially masked versus observed errors, box
IoU, per-joint errors, and size strata (`max(box_wh)/max(image_wh)` below .1,
.1–.3, and >=.3). Empty strata are omitted rather than reported as zero error.
Per-joint and masked metrics aggregate valid-joint numerators/counts exactly.
Latent diagnostics use clean posterior means: KL, posterior standard deviation,
clamp fraction, per-dimension variance and active dimensions (variance >.01).

Epoch 0, every tenth completed epoch and the final epoch export reconstruction
and fixed-prior grids, raw `geometry.pt`, and metrics JSON. Test exports the same
artifacts once. Prior samples and latent activity require inspection; neither
guarantees broad pose coverage or downstream diffusion quality.

The training generator state is checkpointed alongside native model, optimizer
and scheduler state. Fixed-batch next-update parity was verified on CPU and CUDA;
this does not assert exact shuffled-data-order continuation across restarts.

## Verification and initial run

- Eight geometry/model checks and five training checks, including native CPU/CUDA
  checkpoint, optimizer, scheduler and exact next-update parity.
- Twenty shared-runner/dataset regression checks; **33 total pass**.
- Two-epoch, 3% train/val GPU smoke: six finite updates, best/last checkpoints,
  rendered reconstruction/prior grids. This is wiring evidence only.
- Full frozen-source run: `results/pose_vae/20261104_212036`, launched with the
  approved 50-epoch recipe. Initial launch PID 767063. All previous jobs preserved.

Each run records `resolved.yaml`, `manifest.json` (source and annotation SHA256,
parameter count, split sizes/overlap), `launch.json`, TensorBoard/CSV logs, and
best/last checkpoints. The initial full run additionally contains its source
snapshot. `completion.json` and `test_results.json` are written after completion.

## Completed initial recipe

The full 50 epochs completed on 2026-11-04 with **3,500 optimizer updates**.
Best validation reconstruction selected epoch 49 (the final epoch):
`checkpoints/epoch049-reconstruction0.01041.ckpt`. The best and last checkpoints
include native optimizer/scheduler state and private training RNG.
All 388 source/config hashes and three annotation hashes verified against the
frozen run, with no mismatches. The model was evaluated once on all 29,122 usable
people in the held-out 8,000-image test split after training.

| Posterior-mean metric | Validation | Test |
| --- | ---: | ---: |
| Clean joint error / box scale | 0.04180 | 0.04103 |
| Corruption-mixture joint error / box scale | 0.05449 | 0.05333 |
| Artificially hidden joint error / box scale | 0.09697 | 0.09542 |
| Clean box IoU | 0.92862 | 0.92878 |
| Corruption-mixture box IoU | 0.86093 | 0.86186 |

The corruption mixture retains 25% uncorrupted inputs, matching training;
hidden-joint metrics score only artificially removed labeled observations.
Naturally unlabeled joints are never included. Test sampled reconstruction
objective is 0.010414; clean posterior-mean reconstruction is 0.002955. Test
latent diagnostics show 32 dimensions above the .01 activity threshold, mean
per-dimension mean variance .71690, posterior standard deviation .25412, summed
KL 57.74099, and zero logvariance clamp saturation. These are codec metrics.

### Fidelity and generation limitations

Visual inspection of the final fixed-prior grids shows malformed poses. This
recipe therefore provides an encoder/reconstructor, but does **not** establish
a useful unconditional Gaussian pose generator. All dimensions being active
does not establish that the aggregate posterior matches the prior.

A frozen, full-validation comparison using the exact same corruption draws
measured observed-joint error of **0.02266 for the supplied coordinates versus
0.04893 after reconstruction** (68,026 observed labeled joints). Thus the codec
does not improve the precision of these lightly corrupted observed joints.
For 8,898 synthetically hidden joints, reconstruction error is **0.09697 versus
0.18380** for a per-joint mean relative skeleton estimated only from training
annotations and placed using the noisy input box. This supports completion
relative to that baseline; it does not imply reliable completion of every pose.
No additional test evaluation or training change was made for this comparison.

Artifacts under the initial run directory:

- `results_summary.json`, `validation_baselines.json`, `test_results.json` and
  `verification.json` contain measured results and provenance.
- `learning_curves.png` shows all 50 validation epochs.
- `artifacts/val/epoch049/` and `artifacts/test/epoch050/` contain final grids,
  raw geometry and metrics. The test directory's epoch050 denotes evaluation
  after 50 completed epochs; its weights are the selected epoch049 checkpoint.
- `source/` retains the exact runtime source/configuration snapshot.

The next separate iteration should assess the reconstruction/KL tradeoff against
observed-joint fidelity and Gaussian sample quality before any PoseFlow
integration. No hyperparameter sweep, integration or extra training is included
in this completed iteration.
