# V40: reference-conditioned latent box diffusion

Approved architecture: ED-Pose backbone/projections, source encoder layers2/4/6,
native grid-anchor encoder proposal heads, top200 valid candidates, then a single
Hungarian assignment. No native decoder or pose modules. Training keeps all
matches plus top unmatched references to50; inference takes top50.

V35's box clean encoder, shared final box/existence heads and latent DDIM remain.
The denoiser retains full-image attention and receives its own live reference
feature plus normalized-box embedding through each block's timestep modulation.
The proposal stage runs once per image, with fixed references throughout sampling.

Matched GT input: resized unpadded pixelcxcywh/100. Unmatched inputs: zero boxes.
Both h0 corruption sources and latent targets are detached. Proposal/clean/recovered
classification includes background; geometry and latent reconstruction are matched
only. Proposal classification supervises both native sigmoid channels (person1,
other channel negative). L = proposal_task +0.5clean_task +0.5recovered_task
+10positive_latent_MSE. Detection coefficients: focal2, L1 5, GIoU2. Final bbox
ranking uses denoiser existence only. No quantized codebook, DN, native decoder,
foreign negative construction or post-denoising assignment.

Verification and launch results will be appended after native preflight.


## Verification and launch

- Seven tests failed before implementation; all31 focused V40/V35/V39 checks now pass.
- Native FP32 CUDA real-data preflight: batch4/accumulation2, two finite optimizer
  updates, clipping0.1, Gaussian K1/K4 with100 detections over two images per condition.
- Exact V35 shared initialization/global RNG parity;342 required ED-Pose tensors
  copied with100percent coverage, encoder sources2/4/6. Total45,699,659 parameters.
- Isolated live-reference gradients verified for both features and boxes; strict
  model/pipeline/optimizer restoration and next RNG draws verified without init assets.
- Fresh seed42/80epoch training launched: `/storage/outputs/pose-graph-diff/results/poseflow_v40/20261101_235355`.
  tmux: `poseflow-v40-seed42`; trainer PID3813829. Frozen source520files and
  five checkpoint/annotation/config inputs hash-verified. Live100
  optimizer updates verified with finite loss and gradients; no training restart.
- Detection dashboard now contains16runs, preserving prior15runs/history/APpins.
  V40 exposes51scalar tags. Fixed32K1/K4 every epoch and full8000K1/K4 after epoch79
  are configured. First-epoch AP remains pending; smoke/launch evidence is not a
  learned-performance result.
- Run-local `runtime_overrides.json` records the short local worker TMPDIR, disabled
  progress bar, and replacement of generic post-fit task-loss testing with explicit
  held-out Gaussian bbox evaluation. The optimizer/data/training recipe is unchanged.

Primary artifacts: `manifest.json`, `preflight.json`, `live_verification.json`,
`status.json`, `launch.log`, `seed42/.hydra/config.yaml`, and frozen `workspace/`.


[2026-11-04T19:31+01:00] V40 actual state-dependence audit COMPLETE
BaseV40 epoch079/120000/fixed32 seed42/resize256427/nativebboxallqueries/noNMS; savedK1/K4 metrics exactlyreproduced and frozen core methodsverified. K1boxAP43.2876; differentnoise42.9505; zerostate43.5072; zerocompleteproposalconditioning.2316. K4boxAP20.6213; zeroinginputeveryK4call42.8150. Pairedgeometry/confidenceswaps: K4boxes+K1scores43.4169; K1boxes+K4scores20.3638. Samecachedreferences/queryorder/no rematching. Thus measuredK4AP deterioration tracks scores, not geometry; generatedstatefeedbackcausallyimplicated by zero-inputcounterfactual, exact underlyingtrainingrouteunisolated. Acceptedqueries@.5 rise2.66->49.75/50, centerchange1.14resizedpx, probabilityMAE.621.
GTconditioned matchedreferencebank tests (not inferenceAP) show informative stateuse: t0boxAP69.89->41.16whenzeroed; matchedboxMAE.0050->.0179, positivescore.904->.529 andnegativescore.000112->.05866. t500AP50.32->44.35. ThereforeV40 maintained trainingcorrespondence and learned low-noise stateuse, while generatedtrajectoryconfidence remainsfailed. Full8000nativeK1/K4boxAP47.9955/25.8320verified; ablationsarefixed32only. No model/run/configchanges. Report docs/reports/poseflow_v40_state_audit.md; artifacts tmp/v40_state_audit/report.json. No new fix or run approved by this question.
