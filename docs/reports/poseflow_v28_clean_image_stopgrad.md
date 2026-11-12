# V28: clean-branch image stop-gradient

Launch date:2026-10-25, seed42, fresh80epoch pilot from V26 latent-only.

- Root: `/storage/outputs/pose-graph-diff/results/poseflow_v28_clean_image_stopgrad/20261025_192620`.
- Frozen source: `/storage/outputs/pose-graph-diff/results/poseflow_v28_clean_image_stopgrad/20261025_192620/workspace`.
- Config: `/storage/outputs/pose-graph-diff/results/poseflow_v28_clean_image_stopgrad/20261025_192620/seed42/.hydra/config.yaml`.
- Shared dashboard: http://lynx.tail481642.ts.net:33329/ ; label `V28-clean-image-stopgrad`.
- Worker status/log: `/storage/outputs/pose-graph-diff/results/poseflow_v28_clean_image_stopgrad/20261025_192620/status.json`, `/storage/outputs/pose-graph-diff/results/poseflow_v28_clean_image_stopgrad/20261025_192620/launch.log`.
- Progress: tmux `poseflow-v28`; visible terminal titled V28 clean-image stopgrad training progress.

## Single intended change

The clean latent encoder reads detached image content AND image positions/level
embeddings. Clean task loss trains its latent encoder and shared readout but does
not update image extraction. The denoiser reads live image features; latent MSE
trains it and the image encoder. Clean h0 is still detached before noise and as
the recovery target. Forward computation still uses image features in both
branches, so clean targets can change across updates as image features change.

Retain V26 latent-only's four concatenated feature levels,3+3 latent blocks,
50x256 latents, time modulation, shared final heads, proposal construction and
objective0.5clean-task + latent-MSE1. Denoised task metrics remain monitoring-only.
No V27 scale scheduling or attention masking. Historical families unchanged.

## Verification

- Four TDD tests first failed because V28 did not exist; final48 V24–V28 focused
  and regression tests passed. `git diff --check` clean.
- Full seed42 state SHA256 `268bdfe42eadf545bcddc5d1223af05fb8a0fd152389d7e34eff0a122e73abe2`
  identical to V26 latent-only;41,994,721 parameters. No new parameters/RNG draws.
- Same-weight clean and Gaussian K1/K4 predictions bitwise equal before training.
  Clean remains sensitive to image features despite the gradient stop.
- Isolated losses verify no clean-task gradients to backbone, input projections,
  level embeddings; latent recovery still trains those through denoising.
- Strict pipeline/model restoration, missing-init-asset evaluator loading and
  next proposal/timestep/noise draws verified.
- CUDA real batch4/accum2 forward/backward/optimizer smoke passed; peak allocated
  0.916GiB, reserved1.045GiB.
- Actual Lightning training/validation/checkpoint smoke passed. Smoke AP is wiring
  evidence only, not learned performance.
- Image import282 tensors100%, SHA256 `9ff5a0930cab53c42a903d1f10fec1ebbb49457a127987afab54e1c72d8d71d9`.
- Frozen recipe exactly matches V26 latent-only excluding run name/output/config
  target; runtime progress-bar override matches the prior frozen run.

## Training and evaluation

CrowdPose trainval,256/427 evaluation resize, batch4 accumulation2, FP32/TF32off,
AdamW LR2e-4/backbone1e-4, decay1e-4, milestones31/61, clipping0.1.80epochs/
120000updates. Image initialization from original EDPose source; fresh latent modules.
Fixed32 Gaussian K1/K4 monitoring after every epoch; no monitor checkpoint selection.
Automatic final epoch079 full8000 held-out test K4(primary)/K1, eta0 DDIM,
seed42+imageID and existing corrected pose NMS. Validation loss overlaps trainval.

Compare V28 with stopped V26 latent-only at the same training age. Original V26
full-test is available but includes an extra denoised task loss; V24 differs in
architecture/compute. Stopped V26 latent-only has no full-test endpoint.
`manifest.json`, `recipe_parity.json`, source/config hashes, tests/smokes and original
working-tree diff/status preserve launch provenance. Existing debug tag prefix
`debug/v26` is inherited within the distinct V28 dashboard label.

## Prior runs stopped as requested

V25-h-DINO, V26-latent-only and V27 exited after graceful SIGINT; process trees
verified absent. Retained checkpoints epoch050/epoch049/epoch018 respectively.
Live state at interruption can be newer than saved checkpoint state. Hashes,
original status and explicit user-stop records are preserved in
`tmp/v28/stopped_runs.json` and each experiment root. Logs/checkpoints were not
removed; the shared dashboard retains their labels.

## Status and interpretation

Pilot launched; learned performance pending. Stop-gradient prevents one clean-task
gradient route into image extraction; it does not guarantee elimination of all
shortcuts, target drift or improved pose generation. Final conclusions must separate
clean reconstruction, conditional latent recovery and Gaussian-start detection.

Launch verified at global_step310, epoch0, PID2684920; finite updates, live peak allocated1.470GiB. First-update gradients nonzero for backbone, input projections, clean/denoiser stacks and shared heads. Actual scalar samples verified over shared dashboard; HTTP200 local and Tailscale. See `launch_verification.json` and `dashboard_verified.json` in the run root. No learned AP yet; first fixed32 AP occurs at epoch end.
