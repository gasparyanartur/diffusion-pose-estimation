# V33 no-clipping variants

User approved parallel fresh seed42 experiments on2026-10-27. Both retain the
V33 token readout, live clean-image gradients, dual DINO, constructed fixed
correspondence and both h0 detaches. Existing V33/V34 jobs are preserved.

| Run | Gradient clipping | Latent-MSE weight |
|---|---|---:|
| V33 control | global norm0.1 |1|
| v33-no-clip | disabled |1|
| v33-no-clip-rescale | disabled |10|

Configs inherit V33; no new model/pipeline/state schema. All task/DINO weights,
learning rates/schedule, batch4accum2, augmentations, FP32/TF32off and80epochs
remain unchanged. Fixed32 held-out Gaussian-start K1/K4 monitoring and final
epoch79 full8000 K1/K4 evaluation retain native corrected NMS and image seed42+ID.
Training/ordinary validation use the original overlapping trainval/val recipe.

Eight focused config/runner/V33 checks pass. Both native CUDA batch4accum2
smokes completed two finite optimizer updates, quick evaluation and strict
teacher/init-free evaluator restoration. A spy on clip_grad_norm_ confirms no
norm clipping call occurred; Lightning gradient_clip_val is None. Both initial
model hashes match exactly. Measured smoke peak reserved2.323GiB each, allocated
2.138GiB or less. Full evidence: tmp/v33_variants/tests.log, preflight.json,
smoke.log. These checks establish wiring, not learned performance.

Run-local callbacks observe accumulated gradient norms before AdamW every
update and log global/module norms, readout squared-norm share and cumulative
would-clip-at0.1 frequency every50 updates, without changing gradients. Native
loss/scalar streams remain. Nonfinite losses/gradient norms abort explicitly;
periodic parameter-finiteness checks catch invalid updates. Disabling clipping
does not guarantee proportionally larger AdamW updates.

Frozen runs:

- v33-no-clip: `/storage/outputs/pose-graph-diff/results/poseflow_v33_no_clip/20261027_222707`
- v33-no-clip-rescale: `/storage/outputs/pose-graph-diff/results/poseflow_v33_no_clip_rescale/20261027_222710`

Each has source/config/input hashes, native initialization report, persistent
launch.log/status.json, seed42 checkpoints and automatic final evaluation worker.
Both launch processes verify Lightning clipping is disabled. Existing V33/V34
jobs continue. Progress windows are in tmux73; training sessions are
poseflow-v33-no-clip and poseflow-v33-no-clip-rescale. Shared Tailscale33329
TensorBoard labels are V33-no-clip and V33-no-clip-rescale. Startup is finite;
launch scalar verification is recorded in each run's launch_verification.json.

Hypothesis: removing global clipping may reduce the influence of readout norm
on upstream learning; latent weight10 separately increases direct recovery
supervision. Compare the two new runs to isolate that weight change, and the
weight1 run to the original V33 for clipping. Single-seed results cannot establish
general reliability. Diagnose clean reconstruction, conditional denoising and
Gaussian-start detection separately; do not select checkpoints using fixed32.

Launch verified: both runs reached70 optimizer updates with finite losses and
gradients; actual training peak allocated2.280GiB each. Both have explicit
clipping_disabled_verified=true and actual_gradient_clip_val=null. Actual
TensorBoard gradient/module/would-clip-frequency and native task/latent loss
streams are populated; Tailscale HTTP200 verified on33329. Per-run
launch_verification.json records the status and sampled scalar events. This
confirms launch and monitoring only; first fixed32 AP follows epoch0 validation.

2026-10-28: both variants stopped at the user's request during epoch43
(zero-based). Epoch042 and last checkpoints remain, with logs/events and stop
request/status records. Processes exited after SIGINT; existing V34 was not
stopped. These are incomplete pilots, not final80epoch/full8000 endpoints.
