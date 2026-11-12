# V26 implementation and pilot launch — 2026-10-24

Status: **RUNNING**, epoch0, verified at optimizer update320.
This is launch/wiring evidence; final learned AP is pending.

- Experiment root: `/storage/outputs/pose-graph-diff/results/poseflow_v26_latent_diffusion/20261024_194251`.
- Frozen workspace/config/input hashes: `manifest.json` in that directory.
- Live PID: 1857586; tmux session `poseflow-v26`.
- Dashboard: http://lynx.tail481642.ts.net:33329/ (V26-latent and V24-control).
- Separate read-only progress viewer reads `status.json`; persistent `launch.log`.
-42 tests pass across V26/V24/V25/V12; actual B4/accum2 FP32 CUDA optimizer,
 Lightning validation/checkpoint and native CLI inference checks pass.
-41,994,721 model parameters.282 original EDPose image tensors imported at100%
 coverage;124 fresh state tensors; no decoder transfer or deformable modules.
- Both stacks and all readout heads have finite nonzero first-update gradients.
- Gaussian K1/K4 smokes and all required live loss/diagnostic tags verified.
- One initial zero-update launch attempt failed because its runtime TMPDIR made
 Unix socket paths too long. Shortened to local `tmp/v26-runtime`, retained attempt
 logs and restarted only V26 with unchanged frozen source/config. No trained state
 was discarded, and no other jobs were signaled.

The run is seed42,80 epochs, original V24 data/optimizer recipe. Fixed32 K1/K4 is
monitoring only. The worker automatically evaluates epoch079 over all8000 held-out
images at K4/K1, with corrected native NMS, FP32/TF32off and seed42+imageID. It
verifies image coverage and annotation provenance against the completed V24-control
reference. Architecture/call-budget differences prevent a single-change causal claim.

Accepted research limitations: moving detached targets; borrowed negatives are not
overlap-filtered; scale normalization does not establish semantic latent quality.
Clean reconstruction, conditional denoising and Gaussian-start detection metrics
must be assessed separately. No full-test V26 performance result is available yet.
