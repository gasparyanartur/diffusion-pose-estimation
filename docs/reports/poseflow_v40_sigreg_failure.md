# V40-SigReg zero-AP investigation — 2026-11-03

Run: `/storage/outputs/pose-graph-diff/results/poseflow_v40_sigreg/20261103_123752`. Status: failed during epoch3; process278823 is no longer alive.
Last logged optimizer step4940. The run stopped on its nonfinite gradient-norm
check. That check squares float32 gradients, so its error does not establish
that every individual gradient tensor contained NaNs. The scale divergence is
already conclusive from earlier finite logs and saved checkpoints.

## Confirmed failure

| Measurement | Initial | Epoch0 | Epoch1 | Epoch2 |
|---|---:|---:|---:|---:|
| Fixed held-out clean-code RMS |0.849|14.357|246.464|2136.947|
| Fixed32 K1 bbox AP (%) |—|1.195|0|0|
| Fixed32 K4 bbox AP (%) |—|1.007|0|0|

At epochs1/2 both samplers have zero IoU0.5 GT coverage out of133 people. Each
prediction file still contains1600detections for all32correct held-out image IDs.
Saved boxes degenerate towards sigmoid boundaries; this is not merely score
ranking. In epoch1,1234/1600 K1 boxes have width or height at most one pixel.
The five relevant production files match the frozen run source and manifest.

Training initially looked finite: step380 clean RMS1.225, latent MSE0.316. By
step1250 RMS7.064; step1500 RMS14.009; step4500 RMS1796.892. Last logged batch
has clean RMS7628.787, latent MSE3184114.5, and weighted SIGReg1.169.
The initial finite-update smoke did not establish sustained training stability.

## Loss-design investigation

The approved variant removed the final clean LayerNorm, kept h0 live through
corruption, and retained stopgrad(h0) as latent regression target. V39's live
source experiment retained its final LayerNorm, which fixes output scale.
SIGReg's weighted characteristic-function statistic is bounded and did not
supply a sufficient scale anchor in this configuration. A fixed-direction
rescaling diagnostic on epoch2 codes changes the weighted penalty from1.267
at RMS10 to1.236 at RMS10000. This is not an arithmetic mismatch with the pinned
SIGReg formula; reference quadrature tests passed.

A scalar example exposes a problematic update: with prediction k*h and target
stopgrad(h), the source gradient is2*k*(k-1)*h, outward for0<k<1. A live target
instead gives2*(k-1)^2*h. This demonstrates a possible expansion bias, not the
complete dynamics of the trained neural network.

Frozen actual-model diagnostics use17matched people in the same4held-out images,
same weights/noise/predictions. At epoch2,t900 the detached-target loss has a
latent radial gradient dot of-2.449e6(outward), versus+1.105e7(inward) with the
live target, at exactly the same forward loss3495133. At the two-update smoke
checkpoint,t900 signs are also outward/inward. At epoch2,t600/t999 the target
change also reverses the latent radial sign. Weighted latent-loss gradient norm
at t900 is89.41, versus0.00744 for SIGReg.

Important limitation: projecting through the actual clean-encoder parameter
Jacobian at epoch2,t900 gives inward small-SGD scale effects for BOTH target
choices (stronger with a live target). Therefore this experiment does not prove
that target detachment alone caused the full AdamW training trajectory, nor
that removing it will by itself fix training. Normalization removal and weak
scale control remain coupled factors. No learned correction result is claimed.

The original LeJEPA objective uses no target stop-gradients:
https://arxiv.org/html/2511.08544v3#S5.SS1 . Our retained target detach was an
intentional hybrid choice, not an implementation of LeJEPA's complete objective.

Artifacts: `tmp/v40_sigreg/diagnosis/gradient_routes_late.json`,
`gradient_routes_early.json` and associated script/logs. No production gradient
rule was changed and no new training run was launched during this investigation.

## Proposed next iteration — awaiting review

Change only SigReg latent regression to propagate through both prediction and
clean target, retaining live corruption, identity output normalization, SIGReg
weight0.05, masks and all task losses. Preserve historical family gradients and
allow old SigReg configs without the new flag to retain their detached-target
semantics. Record the new flag explicitly in the corrected preset and manifest.

Add tests for identical forward MSE values and corrected target-side gradients,
a contracting scalar predictor's gradient direction, historical restore and
matched/background masks. Run a bounded two-epoch seed42 CUDA pilot with the
existing recipe, fixed32 K1/K4, clean-code drift and scale diagnostics. Stop on
nonfinite values or matched clean RMS>10, with logs/checkpoint preserved. Require
bounded noncollapsed codes and nonzero held-out geometry/AP before proposing
another80epoch run. If the pilot fails, use its measurements to separate source
routing from normalization/regularization rather than launching another long run.


Follow-up: the approved [live-target correction pilot](poseflow_v40_sigreg_live_target.md)
completed two epochs with finite training and held-out clean RMS 0.492.
Fixed32 K1/K4 AP is 13.15/5.34 at epoch 1. The original failed artifacts and
results above are preserved; the multistep gap remains.
