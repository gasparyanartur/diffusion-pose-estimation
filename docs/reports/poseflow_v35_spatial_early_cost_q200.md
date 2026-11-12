# V35-spatial-early-cost-q200

Approved query-budget experiment, launched 2026-10-31 Europe/Stockholm.

- Run: `/storage/outputs/pose-graph-diff/results/poseflow_v35_spatial_early_cost_q200/20261031_004325`
- Config: `conf/benchmark_poseflow_v35_spatial_early_cost_q200.yaml`, inherited from V35-spatial-early-cost.
- Fresh seed42,80epochs;200 proposals,10x20 normalized anchors. No200-to50 selection. Historical50querydefaults unchanged.
- All learned parameters start exactly equal to fresh seed42 early-cost50. The anchor buffer changes; parameter count remains41,590,981. Image extraction imports282/282compatible tensors from the same EDPose source (SHA `9ff5a0930cab53c42a903d1f10fec1ebbb49457a127987afab54e1c72d8d71d9`). No V35 checkpoint warmstart.
- Early spatial ramp max(0,1-(999-t)/500), cutoff500/max1; objective.5clean+.5decoded+10latentMSE; both clean-source/target detaches retained; clip.1. Pixelbox/100 input, original image-conditioned clean stack, denoiser-only Fourier positions, direct box/existence heads, native1000cosinediffusion/deterministicDDIM. No flow conversion.
- Original data augmentations/256x427 resize/batch4/accum2/FP32, AdamW2e-4/backbone1e-4, weightdecay1e-4, milestones31/61 preserved by inheritance.
- Proposal spares retain floor-half perturbed positives/remainderforeign negatives and original fallback behavior. Additional queries increase both duplicates/negatives and change normalized task supervision/OTAcompetition; this is not a pure capacity ablation. No latent-loss rescaling.

## Verification

Three intended missing-config failures, then21 focused tests pass (q200/early/late/legacyV35). Tests cover config-only changes,200counts/oddspares/coverage/foreigngeometry/globalcanonicalindices/emptyrecipient/zero andpositivefallback, strictpipeline/RNGrestore and200-queryGaussianinference. Existing matcher/gradient/isolation tests retained.

Native CUDA batch4accum2 smoke completed2optimizerupdates, finitegradients, decodedtaskactive, isolatedlatentMSEcannottraincleansource/readout, K1/K4twoimagebboxeval and strictmodelrestore. Sharedlearnedtensors EXACT50control parity. Peakallocated1.220GiB/reserved1.420GiB; amplemeasuredheadroom. Smoke AP is wiring validation, not a performance result.

## Runtime and evaluation

Frozen source/config/hashes, preflight evidence and console logs under run root. Session `poseflow-v35-spatial-early-cost-q200` includes a progress window. The two active50-queryearly/late jobs remain running.

Use existing object-only TensorBoard33330, label `V35-spatial-early-cost-q200`, retaining pinned quick-testbboxAP K1/K4. No new server/port or overfitstreams. First AP arrives after first epoch; initial livechecks only establish finite training.

Fixed32K1/K4 Gaussian-start monitoring and automaticepoch79full8000testK1/K4 retained: seed42+imageID, FP32/TF32off, all200queryboxes passed to nativebboxCOCOeval, personclass1/noNMS. Existing maxDets[1,10,100] unchanged, so scoring internally considers atmost100detections/image. Coverage diagnostics can inspect all200. Do not compare raw confidentpredictioncounts directly with50queryruns without accountingforbudget.

No AP gain is claimed before results. Compare checkpoints at matched epochs and distinguish fixed32 monitoring from final8000 evaluation.

Live verification: 120 optimizer updates completed with finite losses/parameters; peak allocated 1.529GiB. The first 120 updates took 114.9s including startup under concurrent GPU load; this is a short runtime observation, not an isolated throughput comparison. Event file contains46scalar tags. Native preflight K1/K4 each emitted400detections for2images, confirming200predictions/image through evaluation.
