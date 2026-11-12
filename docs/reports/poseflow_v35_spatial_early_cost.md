# V35 spatial early-cost control

User requested the opposite of the late noise-timestep ramp on2026-10-31.
Base remains V35-spatial WITH decoded task supervision. NoV36/c/inputchanges.

ForT1000 andrampwidth500:
- late:g(t)=max(0,1-t/500).
- early:g(t)=late(999-t)=max(0,1-(999-t)/500).

Early is zero t<=499, .002att500, .5att749 and1att999. Both schedules average
.2505 over the1000native uniformtimesteps. Thus maximum andmean coefficient
are equal; direction alone changes. Actual matching assignment frequencies may
still differ. This is a trainingmatching cost, not an inferenceguidance term.

Everything else retained: spatial5x10queries/Fourierdenoiserpositions,
globalHungarianproposals, originalclass/boxcosts and100/10000candidatepenalties,
shareddecodedheads, image-conditioned3blockcleanencoder and3blockdenoiser,
pixelboxinputs/100, bothh0detaches, oneOTAassignmentfortask+latenttargets.
Objective.5clean+.5recovered+10latentMSE, clip.1, batch4accum2, FP32/TF32off,
AdamW2e-4/backbone1e-4,wd1e-4,milestones31/61, resize256/427/originalaugmentation,
freshseed42/80epochs andoriginalimage-onlyEDPoseinit unchanged.

TDD:3expectedredfailures then21focusedearly/late/legacytests pass. Testsverify
exactfullschedule reflection, endpoints/equalmean, actualperimageOTAweights,
oneassignment, intendedgradientroute, strictpipeline/restorednexttrainingand
validationdraws, config-onlydirection/namespacechanges andlegacyparity.
NativeCUDA2updates, quick2imagebboxeval, exactfullinitparity andstrictmodelrestore
requiredbeforelaunch. No newmodeltensors/RNGdraws, oldconstant/latedefaults preserved.

Freshsource/config/inputhashmanifests, persistentconsole/status/progress,
existingobject-only33330dashboard(labelV35-spatial-early-cost), sameK1/K4APpins.
Fixed32GaussianK1/K4 monitoring andautomatic epoch79all8000bboxK1/K4,
seed42+imageID/all50queries/noNMS unchanged. No checkpointselectionfrommonitoring.
Ordinaryvalidationlossisoverlappingtrainvalfitdiagnostic. NoAPgainassumed.
Existing late run is preserved. Launch evidence appendedbelow.

NativeCUDApreflightpassed2updates/quick2imagebboxeval, strictmodelrestore and
isolatedlatentdetach.41,590,981params, peakallocated1.244/reserved1.405GiB.
FullinitialSHA083e35a988ba78d79cfbd62dbc5590cf63690293fde1ae890540287cc70c1a8f
EXACTLYmatcheslate-ramp/control. Imported282/282compatibleimage tensors.

Freshfrozenroot:
`/storage/outputs/pose-graph-diff/results/poseflow_v35_spatial_early_cost/20261031_001634`.
Trainingtmuxsessionposeflow-v35-spatial-early-cost withprogresswindow. Persistent
launch.log/status.json/manifests andautomaticfinalevaluationworker retained.
Sameexistingobject-only33330dashboardlabelV35-spatial-early-cost andK1/K4APpins;
launch/status/scalarverification appendedwhencomplete. Existinglatejobuntouched.

Launchverifiedbeyond200finiteupdates.46scalarstreams populateearlyrun;
loggedlow-noiseweights areexactly0andhigh-noiseweightspositive. Actualstatus,
peakmemory/scalars andlatePIDliveness savedinlaunch_verification.json.
Existingobject-only33330TensorBoard nowincludes9runs, withsameK1/K4APpins,
nooverfits/failedattempts,TailscaleHTTP200verified. Historicalreloadcompleted;
latejobpreserved. No detectionperformanceclaimbeforefirstepochmonitoring.
