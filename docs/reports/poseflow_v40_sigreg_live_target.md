# V40-SigReg live-target correction pilot

The approved correction is implemented and verified. The fresh seed-42 pilot
completed two epochs and 3,000 optimizer updates without a guard failure. It
avoided the original scale explosion and zero-AP failure over this interval.
Four-step AP remains substantially below one-step AP.

Only the latent target gradient changes. The new SigReg recipe explicitly sets
`detach_latent_target: false`, so matched latent MSE differentiates through both
the recovered prediction and the online clean target. The corruption source
remains live. SIGReg weight 0.05, identity clean output normalization, proposal
selection, reference conditioning, task losses, masks and inference are unchanged.

Historical SigReg configurations without the new flag default to `true`, retaining
their detached-target behavior. V35, V39, V40 and V40-EMA retain their existing
training rules. The default V40 loss hook calls the original matched-only loss.
No checkpoint tensor shapes change.

Four intended-red checks preceded 50 passing focused checks. The final SigReg run
passed 14 checks, including one additional whole-batch integration test: 51 distinct
checks across the runs. These cover direct target gradients, matched/background
masks, empty/single/full-50 cases, a contracting-predictor gradient example, exactly
unchanged forward losses, historical configuration behavior, and strict model,
pipeline, optimizer, RNG and next-update restoration for both flag values.

Native FP32 CUDA verification passed two optimizer updates with batch size 4 and
accumulation 2. All 342 eligible ED-Pose tensors loaded; shared initial model tensors
and global RNG match V40. Independent latent/task/SIGReg encoder gradients and live
reference gradients are present. Strict restore reproduces the next CUDA update
within FP32 tolerance (maximum parameter difference 2.38e-7; CPU restoration is
exact). Gaussian K1/K4 sampling produced 100 detections for two images each.
These are wiring checks, not learned-performance evidence.

The pilot uses fresh seed 42 and the original data, augmentation, optimizer and
learning-rate recipe: resize 256/427, FP32, batch size 4, accumulation 2 and gradient
clipping 0.1. The Trainer stops after two epochs; the module retains the 80-epoch
scheduler horizon so early learning rates match the original experiment. Loss
weights remain proposal 1, clean/recovered task 0.5/0.5, matched latent MSE 10 and
SIGReg 0.05, with detection coefficients 2/5/2.

A guard saves a forensic checkpoint and stops if matched clean-code RMS exceeds 10
or a numerical check fails. It does not modify valid updates. Fixed32 held-out bbox
evaluation runs after each epoch with independent Gaussian initialization, K1/K4,
all 50 outputs and no NMS. A fixed 17-person clean-code probe uses evaluation-only
preprocessing and records scale, drift and final variance.

Pilot: `tmp/v40_sigreg/live_target/pilot_20261103_144558`, resolved under
`/library/tmp-folders/pose-graph-diff/`. The namespace contains frozen source,
configuration, manifest, persistent logs, probes, evaluations and checkpoints.
All 609 frozen source files and five input/configuration hashes were verified.
TensorBoard stream `V40-SigReg-live-target-pilot` preserves all 18 prior streams
and existing AP pins. The failed original experiment remains at
`/storage/outputs/pose-graph-diff/results/poseflow_v40_sigreg/20261103_123752`.

No 80-epoch launch follows automatically. The pilot must establish bounded,
noncollapsed codes and useful held-out detections before a longer run is reviewed.
The frozen gradient counterfactual motivated this correction; it did not prove that
target detachment alone caused the original AdamW trajectory. See
[the failure diagnosis](poseflow_v40_sigreg_failure.md) for that distinction.

Measured fixed32 results (AP in percentage points):

| Epoch | Target gradient | K1 AP | K4 AP | Held-out clean RMS |
| --- | --- | ---: | ---: | ---: |
| 0 | Detached, original | 1.19 | 1.01 | 14.357 |
| 0 | Live, correction | 8.35 | 5.98 | 0.405 |
| 1 | Detached, original | 0.00 | 0.00 | 246.464 |
| 1 | Live, correction | 13.15 | 5.34 | 0.492 |

After epoch 1, raw IoU >= 0.5 coverage is 101/133 people for K1 and 104/133
for K4; the original run has zero coverage at that epoch. Each condition has
1,600 finite detections covering the same 32 images. Seed, resize, sampler,
precision, TF32 settings and postprocessing match the original evaluator.

The fixed 17-person probe has mean coordinate variance 0.1601, MSE 0.02498
from epoch 0 and cosine similarity 0.9432 to epoch 0. These measurements do
not show code collapse or the earlier scale runaway. They do not establish
a unit-Gaussian distribution or stability beyond two epochs.

The final checkpoint has finite floating-point state, and every initialized
AdamW parameter state records 3,000 updates. Both final checkpoint paths
(`last.ckpt`, `epoch001.ckpt`) are preserved. All frozen hashes remain intact;
the trainer process has exited. `result.json` and `comparison.json` in the
pilot namespace contain the results. Dashboard history remains available.

The pilot supports the correction as a remedy for the observed early
divergence. It does not resolve the multistep performance gap: K4 AP decreases
from 5.98 to 5.34 while K1 improves from 8.35 to 13.15. Scores also change
substantially with sampling depth (at epoch 1, 3 K1 versus 458 K4 predictions
score above 0.5), despite similar raw coverage. No causal explanation for that
remaining gap is established here. This is a fixed32 box-detection diagnostic,
not full-test or pose-estimation evidence. No further training was launched.


After reviewing this pilot, the user explicitly authorized a fresh 80-epoch run.
See the [full-run launch record](poseflow_v40_sigreg_live_target_run.md).
The pilot measurements and limitations above remain unchanged.
