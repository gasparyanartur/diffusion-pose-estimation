# V31: denoised task supervision with detached h0

Approved2026-10-26. Fresh V28-based fixed-correspondence model; V30 stopped on
replacement, retaining epoch025.ckpt and last.ckpt.

Loss: `0.5*clean_task + 0.5*recovered_task + latent_MSE`. Recovered task gradients
train readout/denoiser/image extraction. Clean h0 is detached before corruption
and as latent target. Clean encoder learns only through clean task, with clean
image features detached. Shared readout learns from both task branches.

No model parameter, sampler or correspondence changes. Same seed42/80epochs,
256/427 resize, batch4/accumulation2,FP32/TF32off,AdamW2e-4/backbone1e-4,
weight decay1e-4,milestones31/61,clip0.1. Gaussian-start K4 DDIM primary, K1
diagnostic; fixed32 AP monitoring every epoch and all8000 epoch79 evaluation.
No tuning or checkpoint selection from fixed32.

Three tests first failed before implementation, then passed: isolated recovered
task/latent/clean gradients and direct h0 retain_grad check; exact initialization
plus same-forward losses versus V28 configured weight0.5; strict restore/next
proposal+noise/evaluation without initialization assets; recipe parity with V30.
All133 regression tests passed. CUDA batch4/accumulation2 optimizer smoke passed
(peak allocated0.889GiB/reserved1.055GiB), and Lightning optimizer/validation/
quickAP smoke passed. Initial model state matches V28 exactly. Launch verified: 100 optimizer updates with finite losses/nonzero
gradients. This is launch evidence, not learned performance.

Frozen run: `/storage/outputs/pose-graph-diff/results/poseflow_v31_denoised_task/20261026_142350`. Source/config and input hashes are in `manifest.json`;
resolved config is `seed42/.hydra/config.yaml`; persistent output is `launch.log`,
live progress is `status.json`. Shared TensorBoard label **V31-denoised-task**
has actual training scalars and Tailscale HTTP200 verified. First fixed32 AP is
scheduled after epoch0 validation. The visible V31-progress window is selected
in attached tmux session73. No separate TensorBoard server was added.

Automatic epoch79 full8000 deterministic K1/K4 evaluation is configured in the
frozen runner with checkpoint/config/annotation provenance. Results pending.

Hypothesis: task gradients into the denoiser/image route provide a useful signal
that heads-only adaptation did not. Compare against V30/V28 at matched ages and
report held-out Gaussian AP separately from conditional/reconstruction losses.
The original V26 also differs in clean-image gradient routing.

## Final outcome — 2026-10-27

Training and epoch79 full8000 K1/K4 evaluation completed. See
[final comparison](2026-10-27-v26-v31-v32-final-comparison.md) for results, matched
protocol verification and limits. Earlier launch-only evidence above is retained
as provenance.
