# V27 coarse-to-fine scale attention pilot

Launched2026-10-25: seed42,80epochs, fresh frozen source/config. Running at
update250 when verified. This is scale scheduling only:
no RT/RF-DETR encoder, image fusion neck, DINO target or new conditioning.

Each block in both clean and denoising stacks reads one level in stride32/16/8
order. The same50 proposal slots traverse all3 blocks. Image memories, positions
and padding masks remain separate; no multiscale token concatenation is performed.
The schedule repeats inside every DDIM call. Synthetic stride64 is omitted.

Objective remains0.5 clean-task + latent-MSE1. The clean encoder is jointly
trained; denoised task losses are monitoring-only. Block design, time modulation,
FFN, latent normalization/projection, proposal construction and native readout
are inherited from V26. Resolved data/training config and all other pipeline
parameters match V26 latent-only, apart from family target and image level count.

Verification:
-44 focused/regression tests passed across V24/V25/V26/V27, including routing,
  fine-level isolation, B2/B4 independence, padding, permutation, gradients,
  strict checkpoint/RNG restoration and evaluation without initialization assets.
- CUDA batch4/accum2 update and actual Lightning fit/validation/checkpoint smokes
  passed. Preflight peak allocated0.661GiB/reserved0.680GiB; live peak allocated
  1.107GiB at verification.
- Image source levels0/1/2 explicitly selected;278 tensors imported at100%
  coverage from original EDPose source. Source embedding rows0/1/2 retained.
-37,275,105 parameters; removing stride64 projection changes parameter count and
  RNG consumption versus V26. Latent/head initialization is fresh and is not
  claimed bitwise identical across the two architectures. No decoder transfer.

Operational artifacts:
- Frozen root: `/storage/outputs/pose-graph-diff/results/poseflow_v27_coarse_to_fine/20261025_145027`.
- Config: `conf/benchmark_poseflow_v27.yaml`.
- Checkpoints/logs/status under root `seed42/checkpoints`, `launch.log`, `status.json`.
- Shared dashboard: http://lynx.tail481642.ts.net:33329/, run label
  `V27-coarse-to-fine`; real loss/latent/monitoring scalar samples and HTTP200
  local/Tailscale verified. Inherited debug tags retain the V26 training-protocol
  prefix within the distinct V27 run label.
- Persistent tmux worker: `poseflow-v27`; visible terminal viewer:
  `V27 coarse-to-fine training progress`.
- Existing V25 and V26 latent-only trainers remain running and untouched.

Fixed32 Gaussian K1/K4 AP monitoring begins at epoch end. Epoch079 automatically
evaluates all8000 held-out images at K4/K1, FP32/TF32off and corrected pose NMS.
The worker reports original V26/V24 references and V26 latent-only if its endpoint
metrics are available; pending references are labeled explicitly. No checkpoint
selection from the monitoring subset. No learned AP conclusion yet.
