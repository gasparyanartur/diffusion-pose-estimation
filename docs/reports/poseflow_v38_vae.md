# V38: variational clean latents with mean diffusion

Approved experiment based on V37-reweight. Implementation, verification, fresh launch and live dashboard checks completed.
No learned-performance claim is made.

## Intervention

Keep V35's image-conditioned spatial model, existing clean LayerNorm, 50 slots
on a 5x10 grid, construction/sorting, deterministic denoiser, decoder and DDIM.
Append an identity/zero-bias mean head and zero-weight logvariance head with
bias -4, clamp [-10,2], without normalizing the mean. Remove explicit alignment
and all six calibration parameters. Shared initialization and global RNG must
remain exactly unchanged.

Clean reconstruction decodes a reparameterized posterior sample. Existing clean
outputs decode the mean for monitoring. Diffusion corrupts detached means;
matching uses mean squared distance to detached clean means with unchanged V37
Dynamic-K exp(-distance), conflict handling, nearest fill and unique-real-person
coverage. One assignment supplies recovered task and latent targets.

```
total = 0.5 * sampled_clean_task + 0.5 * recovered_task
      + 100 * matched_latent_MSE + 0.1 * clean_prior_KL
clean_prior_KL = mean_BPD[0.5 * (mean^2 + exp(logvar) - 1 - logvar)]
```

Existing focal/L1/GIoU coefficients 2/5/2 and positive-count normalization stay
unchanged. Real-only geometry; all-slot existence and KL. No denoised prior KL.
Train jointly from scratch, with the original EDPose image-only initialization.
Recovered supervision detaches clean targets and corruption sources; shared
image features still receive gradients. This diffuses means and is not an exact
replication of LDM's sampled-code diffusion. Fake geometry can collapse under
the confirmed task mask without constituting a reconstruction failure.

## Reproducibility and diagnostics

Explicit posterior noise uses a separate CPU generator seeded1042 with saved
pipeline state. Validation uses1042+batch index without advancing training state.
Proposal/timestep/diffusion generators and global RNG remain unchanged. Inference
never executes clean posterior heads and preserves [B,50,256] states and signatures.
Resolved config saves posterior settings; strict restoration saves both heads,
optimizer state and the independent posterior RNG buffer.

Raw/weighted KL, real/fake mean-code variance, posterior standard deviation and
clamp saturation distinguish regularization behavior. Mean versus sampled clean
box losses distinguish reconstruction behavior. Recovered L1 by existing
low/middle/high timestep bins aggregates by positive-assignment count. Retain
assignment utilization/coverage diagnostics and fixed32 K1/K4 Gaussian AP.

## Verification

- Nine intended failing tests preceded implementation.
- 19 focused checks pass: nine V38 checks and all ten V37 regressions. Analytical
  KL/gradients, finite clamped extremes, sampled clean gradients to both heads,
  detached recovered routes, posterior-noise isolation, fake existence/KL and
  real-only geometry, duplicates/empty GT, exact shared initialization/global
  and original RNG sequences, initial mean outputs, K1/K4 inference parity,
  strict model/pipeline/optimizer restoration and continuation of all draws.
- Native CUDA batch4/accum2: two finite updates, strict model/pipeline/optimizer
  restoration and proposal/diffusion/posterior RNG continuation pass. K1/K4
  two-image smoke emits100 detections per condition. Positive real-person
  assignment coverage1. Peak allocated1.106GiB/reserved1.348GiB.
- Shared native model tensors and global RNG exactly match V37-reweight. New
  model has41,722,565 parameters, no calibration parameters. Image-only import
  copies282/282 tensors with SHA256
  `9ff5a0930cab53c42a903d1f10fec1ebbb49457a127987afab54e1c72d8d71d9`.
- Resolved recipe diff confirms only namespace, V38 class and approved posterior/
  alignment settings change; data, optimizer, augmentation and schedule match.

## Launch and outcome criteria

Launched one fresh seed42/80epoch V38 run with the V37-reweight
data/augmentation/optimizer/schedule, clipping0.1, fixed32 K1/K4 per epoch and
final epoch79 full8000 K1/K4 evaluation. Preserve both V37 runs and stopped
cancelled jobs. Freeze source/config/input hashes and use short local IPC tmp.
Add V38 to the existing33330 dashboard with history and AP pins retained.

Compare matched-age checkpoints with the same split/resize/evaluator/seed and
Gaussian sampling calls. Report AP/coverage and recovered geometry versus
V37-reweight. Lower KL or better clean reconstruction alone is not evidence
that detection is improving. The CUDA smoke is a wiring/finite-update check,
not a learned result; fixed32 monitoring and full8000 evaluation are distinct.

## Frozen run

Run: `/storage/outputs/pose-graph-diff/results/poseflow_v38_vae/20261101_001343`.

- Config: `seed42/.hydra/config.yaml`; source/config/input hashes and objective:
  `manifest.json`. Hashed inputs include the EDPose checkpoint, all three
  annotation files and resolved config. Dataset roots retain the original links.
- Native verification: `preflight.json`, `preflight.log`, `tests.log` and
  `config_diff.json`. Initial complete model SHA256:
  `a2d467a905cada9b5fae893a513f83717aa98b241eda4bf99f98eacfc11f7c62`.
- Frozen training source: `workspace/`; short local IPC directory:
  `/library/tmp-folders/pose-graph-diff/v38/ipc`.
- Progress: `status.json`, `launch.log`; viewer:
  `venv/bin/python tmp/v38/watch.py --root /storage/outputs/pose-graph-diff/results/poseflow_v38_vae/20261101_001343`.
- Baseline: `/storage/outputs/pose-graph-diff/results/poseflow_v37_reweight/20261031_232257`.
- Existing [33330 dashboard](http://lynx.tail481642.ts.net:33330/) adds `V38`
  as its13th run, retaining the12 prior entries and fixed32 K1/K4 AP pins.
- Final evaluator writes same-age fixed32 AP/coverage and validation recovered
  geometry from both runs into `comparison.json`; full8000 comparison is added
  when the baseline's final report is available. Fixed32 and full8000 remain
  separate, with the original256x427 resize, FP32, seed42+image_id, K1/K4 DDIM,
  all50 queries, noNMS and COCO maxDets[1,10,100].

## Initial live verification

At160 finite optimizer updates, V38 is training with
unique-real-person assignment coverage1 and clipping0.1 verified. HTTP200 and
62 V38 scalar tags are visible through the actual TensorBoard API, including
real/fake posterior diagnostics, mean/sample reconstruction and positive-count
noise-bin geometry. Run history and both K1/K4 AP pins are preserved.

Original V37 PID3477966 continues at43290 updates; V37-reweight PID3582414
continues at5000. Previously cancelled early/timed/q200 jobs remain stopped with
`cancelled_by_user` status. These are launch-time snapshots, not current totals.
`verification.json` records the live evidence. First-epoch fixed32 AP is pending;
finite updates and coverage do not establish later detection improvement.

The repository `.plan` entry is retained for outcome review.
