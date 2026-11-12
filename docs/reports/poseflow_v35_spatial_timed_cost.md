# V35 spatial matching cost scheduled by timestep

Approved2026-10-30. Base is V35-spatial WITH decoded task supervision, not V36.
Clean boxes use pixelcxcywh/100 with the original image-conditioned3block clean
encoder; denoiser has3blocks, spatial Fourier queries and shared direct linear
box/existence heads. No c input or normalized MLP-only clean path.

Only additional fixed-anchor OTA distance cost is multiplied per image by
max(0,1-t/500):t>=500 gives0,t250 gives.5,t0 gives1, maximumspatialweight1.
Uses actual noise timestep, not epoch. Class/box matching costs, native100/10000
candidate penalties, globalHungarian proposal construction and denoiser positions
are retained. No matching runs at Gaussian inference. One OTA assignment supplies
recovered task labels and detached canonical-person latent targets. Source-target
identity/existence remapping remains a limitation, not addressed by this change.

Objective.5clean+.5recovered+10latentMSE, bothh0detaches, clip.1 and the complete
V35seed42/80epochrecipe unchanged:batch4 accumulation2, FP32/TF32off,
AdamW2e-4/backbone1e-4,wd1e-4,milestones31/61,originalaugmentation/resize256/427.
Only image extraction initialized from originalEDPose; latent modules/heads fresh.
Schedule/defaultcutoff saved in config; historical configs defaultconstant cost.
No newmodeltensor/schema. Existing frozenjobs stay untouched.

Verification:4intendedredfailures then29focusedtests; endpoints/monotonicity,
mixedtimestepbatch/emptytargets/invalidmultipliers, exacthighnoisecost0 and
lownoisecost1assignmentparity, constantlegacyparity, oneOTAcall, intendedgradients,
strictpipeline/restorednextdraws/configisolation. Final4schedulechecks alsoverify
trainingandvalidationRNGconstruction. NativeCUDA batch4accum2 twooptimizerupdates,
quick2imagebboxevaluation, strictmodelrestorewithoutinit andisolatedMSEdetachchecks
passed. Peakallocated1.244GiB/reserved1.405GiB,41,590,981parameters. FullinitialSHA
083e35a988ba78d79cfbd62dbc5590cf63690293fde1ae890540287cc70c1a8f
matches V35-spatial exactly. Imagecoverage282/282tensors.

Diagnostics report actualeffectiveweight/weightedcost and meanweight bynoisebin;
bin means are aggregated by actual bin image counts. Fixed32GaussianK1/K4 boxAP
monitoring and automatic epoch79all8000heldout bboxK1/K4, seed42+imageID,
all50queries/noNMS retained. Ordinaryval losses are overlappingfitdiagnostics.
No checkpointselection from monitoring and no AP gain inferred from tests.

User explicitly requested a NEWobject-onlyTensorBoard forV35andlater detection
runs, excludingoverfit tests. V35controls and V36variants retained; failedattempt
excluded. Pin quick-test/noise_var_100/steps_1/bbox/AP and steps_4 equivalent.
Separate port from historical dashboard; persistentserverlog/launchmanifest and
TailscaleHTTPverification required. Launch evidence appended below.

Frozen run:
`/storage/outputs/pose-graph-diff/results/poseflow_v35_spatial_timed_cost/20261030_221528`.
Training tmuxsessionposeflow-v35-spatial-timed-cost, with own progresswindow;
historical73session is absent in the current environment. First attempt tocreate
thatoldviewerfailedaftertrainstart; keptsamePID andcreatednewprogresswindow,
without restarting/duplicatingtraining. Console/status/manifests retained.

Launch verified beyond100finiteoptimizerupdates. Actualpeakallocated~1.467GiB.
46scalarstreams populate the newrun, including weightedcost, effectiveweights,
low/mid/highbin weights andgradients. Allloggedhigh-noise weights areexactlyzero.
launch_verification.json records actualstatus andAPI samples.

Newdashboard:http://lynx.tail481642.ts.net:33330/ (TailscaleHTTP200 verified).
Eightlabels:V35spatialtimedcost,V35OTA,V35spatial,V35spatiallatentonly andfourV36
boxvariants. Nooverfit/failedattempts. AllsevenhistoricalquickAPstreams loaded.
One/fourcallquick-testbboxAPcards pinned andverifiedvisually inIABdeliverabletab.
Launchmanifest/pinnedURL/serverconsole/screenshot:tmp/detection_dashboard/.
Newrun's own AP begins afterfirstepoch; no AP improvementclaimedatlaunch.
