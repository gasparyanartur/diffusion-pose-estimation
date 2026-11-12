# V37 reweighting control

Approved 2026-10-31. Config-only implementation and verification complete; fresh training launched.

## Experiment

Parent: benchmark_poseflow_v37, the aligned latent-OTA V35-spatial model.
The resolved configuration changes only alignment_existence_weight4→1,
latent_loss_weight10→100 and the run namespace. Objective:

0.5 clean_task +0.5 recovered_task +100 latent_MSE +alignment(1pos+1size+1exist).

Matching, detach boundaries, image-conditioned clean encoder, box-only inputs,
spatial sorting/queries, 50queries/5x10, six sigmoid calibration parameters,
Gaussian DDIM and direct box/existence heads remain unchanged. This is a fresh
seed42 run rather than a continuation of the existing V37 checkpoint.

Hypothesis: reducing relative existence alignment and emphasizing matched latent
recovery improves denoised geometry and Gaussian-start detection. This combined
intervention cannot isolate the two weight changes. Scalar loss magnitude alone
does not establish shared-parameter gradient dominance. With clipping0.1,
relative gradient contributions matter more than absolute preclip norm.

## Verification and evaluation

Hydra parent/control comparison allows exactly two weight and two namespace
differences. Existing10 V37 tests pass. Native CUDA verification additionally
checks identical initial model/calibration tensors and RNG, two finite
batch4/accum2 updates, isolated latent gradients10× before clipping on the same
forward, strict model/pipeline restore and next draws, and K1/K4 inference.
No production model/loss/matcher/sampler code is changed.

Training retains the original80epoch seed42 data/augmentation/resize256-max427,
batch4/accum2 FP32, AdamW2e-4/backbone1e-4, weightdecay1e-4, milestones31/61,
clipping0.1, per-epoch fixed32K1/K4 and final epoch79 full8000K1/K4.
Evaluation uses seed42+imageID, FP32/TF32off, bboxCOCOeval, all50predictions,
noNMS and maxDets[1,10,100]. Compare matched checkpoint ages and raw loss
components, not weighted total losses. No AP improvement is established yet.

Existing V37 continues. Previously cancelled early/timed/q200 jobs remain stopped;
their artifacts/dashboard history are retained. Existing object-only33330
dashboard and pinned K1/K4 AP cards are preserved.

Native verification PASS: two finite optimizer updates, exact original V37
model/calibration state and global RNG initialization, strict model/pipeline
restoration and next proposal/noise draws. Isolated latent backward gradients
on the same frozen forward scaled10× in denoiser, query projection and image
parameters. Native K1/K4 two-image evaluation emitted100detections per condition.
Peakallocated1.106GiB/reserved1.344GiB. Preclip accumulated norm on the second
update was300.108, coefficient0.000333 at unchanged clip0.1; this verifies
relative gradient reweighting, not10× optimizer updates. No performance claim.

## Run

Run: `/storage/outputs/pose-graph-diff/results/poseflow_v37_reweight/20261031_232257`

Session: `poseflow-v37-reweight`, training and progress windows.
Frozen source/config/input hashes and preflight are under the run root.
Initial model SHA083e35a988ba78d79cfbd62dbc5590cf63690293fde1ae890540287cc70c1a8f,
identical to original V37; image initialization282/282 with SHA
9ff5a0930cab53c42a903d1f10fec1ebbb49457a127987afab54e1c72d8d71d9.
Dashboard33330 label `V37-reweight`, original run labels and K1/K4 AP pins retained.
Short resolved local tmp/v37_reweight/ipc is used for multiprocessing sockets.
The first source-freeze attempt stopped on a stale helper copy path before any
launch; the successful frozen namespace above contains the corrected helper.

Initial live check: 90 finite updates, real-person coverage1,
peak allocated1.422GiB. No epoch-level AP yet.

Final launch verification: 190 finite updates; dashboard HTTP200 and59
scalar tags under V37-reweight. Original V37 continues at update
38480; all three previously cancelled jobs remain stopped.
No epoch-level AP is available yet. verification.json records this snapshot;
status.json and launch.log continue updating.
