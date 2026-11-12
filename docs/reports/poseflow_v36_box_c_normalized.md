# V36-box-c-normalized

User-approved2026-10-29 follow-up based on V36-box-c. Clean inputs are
[cx/W,cy/H,w/W,h/H,c], with geometry relative to resized unpadded image dimensions
and c an unscaled binary constructed existence label. No /100. Same model code,
MLP5->256->256/GELU/nonaffineLayerNorm, shared direct box/existence heads and
three-layer spatial denoiser. c enters only clean encoding, never denoiser or
Gaussian inference. Initial weights must match original V36-box-c exactly.

Only clean_box_input_format=normalized and output namespace change relative to
V36-box-c. Objective.5clean task+10OTA latentMSE, no recovered task gradients;
source/targetdetach, spatial5x10queries/cost1, clip.1, proposals/foreignnegatives/
perturbedpositives/fallbacks/OTA remapping, original resize256/427/augmentations,
FP32/TF32off, batch4 accumulation2, AdamW2e-4/backbone1e-4/wd1e-4,milestones31/61,
80epochs freshseed42 all retained. Existingruns continue.

Hypothesis: normalized geometry and explicit clean existence input allow both
clean task components to be represented consistently. That does not establish
successful image-conditioned denoising or repair OTA source-target remapping.
Clean/recovered fit metrics and Gaussian-start detection must remain separate.

Verification includes the intended missing-config red failure, exact five-input
hook (normalizedbox,c without scaling), initial tensor parity tobox+c, detached
latent-MSE gradientroutes, strictrestore plus focused V35/V36/normalizedregressions.
NativeCUDA batch4accum2 two-update preflight/quickbboxeval/memory measurement and
strictloadingwithbothflags required before frozen launch.

Run uses a fresh frozen source/config/input hash manifest, persistentlaunch.log,
atomicstatus, visibleprogress and the existing33329TailscaleTensorBoard. Fixed32
GaussianK1/K4 and automatic epoch79all8000heldout bboxK1/K4 use seed42+imageID,
all50queries/noNMS; no checkpointselection from monitoring. Ordinaryvalidation
is an overlappingtrainval fit diagnostic. Launch verification appended below.

Verification passed:25focusedtests, nativeCUDA2optimizerupdates/quick2imagebboxeval,
strict modelrestorewithnormalized+cflags andisolatedlatentgradientroutes.
InitialfullmodelSHA2560bc219d6e613802c4f3885f5b5f2e11f56fa8c49d559e7f0fc94b00bb4fe840d
matches V36-box-c exactly.35,675,333parameters, peakallocated1.005GiB andreserved
1.170GiB duringpreflight. Imageinitimports282compatibletensors,100%coverage.

Fresh frozen root:
`/storage/outputs/pose-graph-diff/results/poseflow_v36_box_c_normalized/20261029_140927`.
Training sessionposeflow-v36-box-c-normalized, progress73:V36-box-c-normalized.
Source/config/input hashes andfullinitialhash assertedbeforetraining. Label
V36-box-c-normalized onexisting33329TensorBoard; otherjobsandhistoricalrunlabels
retained. Final launch/scalar verification saved inlaunch_verification.json.

Launch verified beyond50finiteupdates,40populatedscalartags(loss/MSE/gradient
samples fetched viaTailscaleHTTP200). launch_verification.json retains actual
status/peakmemory/scalars. AllfourV36variants confirmed fromfrozenconfigs:
spatialqueriestrue,grid5x10,spatialcost1,latentweight10,recoveredtaskweight0.
No queryconditioning incleanencoder. No performanceclaim fromstartupverification.
