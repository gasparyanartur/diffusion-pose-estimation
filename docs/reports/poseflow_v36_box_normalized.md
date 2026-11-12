# V36-box-normalized

User-approved2026-10-29 control branching from V36-box. Only clean input format
changes: use constructed input_boxes=(cx/W,cy/H,w/W,h/H) directly, without/100.
W/H are resized unpadded image dimensions. No c input. This removes the missing
frame-size information in the previous pixel-to-normalized reconstruction task.
Input noise on duplicate centers remains5resizedpixels before conversion;
centers remain unclamped. Foreign negative input geometry is retained.

Architecture/state/initialization are unchanged: MLP4->256->256,GELU,nonaffineLN,
separate3block spatial denoiser, shared direct box/existence heads. Config
clean_box_input_format defaults pixels for historical checkpoints; new mode is
normalized. Training and strict restoration explicitly carry the format flag.
No added parameters. Both clean corruption source and OTA latent targets detached.

Objective remains.5clean task+10OTA latentMSE, recovered task monitoring-only.
NoDINO/joints/tokenreadout, no clean image/attention route. Spatial5x10anchors,
Fourier denoiser conditioning, global Hungarian construction, OTA cost1, clip.1,
batch4 accumulation2, FP32/TF32off, AdamW2e-4/backbone1e-4,wd1e-4,milestones31/61,
resize256/427/originalaugmentation,80epochs/seed42 unchanged. Clean existence
ambiguity and OTA identity/background remapping are not addressed by this control.

23focusedtests pass after3intendedredfailures: exactMLPinput(no double scaling),
resize invariance, samepixel/differentframe ambiguity removal, actualunmasked
foreigninputs, detachedgradientroutes, configisolation, initializationparity,
strictpipeline restoration/nextproposal-noise draws and legacyV35/V36 behavior.
NativeCUDAverification and launch evidence appended below. Source/config frozen
perrun, persistentconsole/status/progress/shared33329TailscaleTensorBoard.
Fixed32GaussianK1/K4 monitoring and automatic epoch79all8000bboxK1/K4,seed42+imageID,
all50queries/noNMS, unchanged. Ordinaryvalidationisfitdiagnostic, notheldoutAP.
Hypothesis: clean reconstruction improves; recovered/Gaussian detection must be
measured separately. No learned performance claim from startup checks.

Native CUDA preflight passed twooptimizerupdates, quick2imagebboxevaluation,
isolatedMSEgradientchecks andstrictrestorationwithnormalizedformat. Peakallocated
1.005GiB/reserved1.170GiB.35,675,077parameters. Initial fullmodelSHA256
61e3c460539d1229b2f02c1b8f4bffb8b61048e74bc1843abcb6e7d987886bf6
matches V36-box exactly. Imported282compatibleimage tensors/100% image coverage;
all latent modules/heads fresh. Exact fullinitialization hash asserted at launch.

Frozen root:
`/storage/outputs/pose-graph-diff/results/poseflow_v36_box_normalized/20261029_094245`.
Persistentlaunch.log/status.json/manifest.json,source+inputhashes/resolvedconfig,
preflightevidence andautomaticfinalevaluationworker retained. Training session
poseflow-v36-box-normalized; visibleprogress73:V36-box-normalized. Source/config
snapshots protect active historical jobs. Dashboard labelV36-box-normalized on
the existing33329TailscaleTensorBoard; launch verification recorded separately.

Launch verified beyond50finiteoptimizerupdates. Existing TensorBoard33329 exposes
40populatedscalartags forV36-box-normalized, including loss,latentMSE andgradient
samples fetched throughTailscaleHTTP200. Actualstatus/scalars andpeakmemory saved
inlaunch_verification.json. AllhistoricalTBlabels andactivejobs retained.
