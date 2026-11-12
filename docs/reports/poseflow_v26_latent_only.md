# V26 latent-only denoising pilot

Launched 2026-10-25, seed42,80 epochs. Fresh initialization from the same EDPose
image-extraction source as original V26; architecture and resolved data/training
recipe match the original, with only the denoised task coefficient changing.

Objective: `0.5*clean_task + latent_MSE1`. The clean encoder stays jointly trained.
Denoised box/joint/existence losses remain monitoring-only and have no autograd
graph; the denoiser trains only through recovery of detached clean latents.
Original V26 defaults and checkpoints retain their original objective.

- Config: `conf/benchmark_poseflow_v26_latent_only.yaml`.
- Frozen run: `/storage/outputs/pose-graph-diff/results/poseflow_v26_latent_only/20261025_094303`.
- Checkpoints/logs: `seed42/checkpoints`, `launch.log` and `status.json` under that root.
- Shared TensorBoard: http://lynx.tail481642.ts.net:33329/, label `V26-latent-only`.
- Live terminal viewer: `V26 latent-only training progress`; worker tmux session
  `poseflow-v26-latent-only`. Both monitor the same run.
- Training batch4/accum2, FP32/TF32off, resize256/427; K4 DDIM primary/K1 diagnostic.
- Fixed32 Gaussian monitoring; epoch079 automatically evaluates all8000 test
  images at K4/K1. Frozen worker compares original V26 and V24 reference metrics.
- Validation/monitoring do not select a checkpoint. Latent MSE alone is not success.

Verification:17 V26 tests and20 V24/V25 regression tests passed, including exact
legacy objective and ablated gradient contracts, RNG and strict restoration.
CUDA batch4/accum2 update and actual Lightning fit/validation/checkpoint smoke
passed. Preflight peak reserved0.902GiB; live peak allocated at update
300 was 1.470GiB. All first-update
branch/head gradients were finite and nonzero. Training is running; no learned
AP conclusion is available yet. Existing V25 training remains running.

Future user-selected sequence, excluded from this run: review layer structure and
FiLM/AdaLN locations, then REPA-style DINO-feature prediction. No frozen-autoencoder
experiment, conditioning change or teacher alignment is included here.
