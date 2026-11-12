# V37: aligned latent OTA

Approved initial experiment, 2026-10-31. Implementation and verification complete; fresh training launched.
No learned detection result is available yet.

## Training contract

Branch: V35-spatial (50 queries, 5x10 grid), original image-conditioned clean
encoder and box-only pixel/100 inputs. No existence input or architecture change.
One distance, mean squared embedding difference, supplies OTA matching and
latent supervision. All padded real/fake codes participate as candidates.
Dynamic-K sums the top5 exp(-distance) qualities, floors and clamps to min1.
Resolve conflicts among proposed assignments by cost, fill unassigned predictions
with nearest targets, then repair unique-real-person coverage using movable
fake or duplicate-person assignments. Ties use index order. Targets can be
reused; fake and duplicate slots need not all be selected. Matching excludes
decoded boxes, scores, anchor costs and candidate penalties.

Recovered task and latent losses share this mapping. Latent targets and
corruption sources are detached. The objective is .5clean_task + .5recovered_task
+10latent_MSE+alignment. Clean reconstruction uses constructed correspondence.

Alignment uses unordered clean-clean nonself pairs inside each image. Position
and width/height similarities are exp(-squared_normalized_distance/.25^2);
existence targets are label equality. Positives use GT geometry, fakes retain
constructed geometry (including zero fallback). BCE logits are bias minus
softplus(raw_scale) times the same latent distance. Three scales initialize1,
three biases0, property weights1/1/4. Each property averages real-real, fake-fake
and mixed groups separately, then available groups and images equally. Empty
groups are omitted; a single-slot image contributes a differentiable zero.
Alignment trains the clean/image path. Six calibration parameters are
checkpointed/optimized in the pipeline and absent from inference.

## Hypothesis and limits

The combined intervention may improve assignment and Gaussian-start detection
by giving latent distances useful geometric/existence relationships. It cannot
isolate matching from alignment. Strong existence weighting may cluster classes;
learned sigmoid calibration can absorb alignment changes. Pairwise losses and
correlations are diagnostics, not evidence of detection improvement.

Compare V35-spatial at matched checkpoint ages on identical fixed32 K1/K4, then
final epoch79 full8000 K1/K4. Evaluator: CrowdPose bbox COCOeval, all50 predictions,
noNMS, maxDets[1,10,100], seed42+imageID, FP32/TF32off, resize256/max427.

## Verification

10 V37 tests and35 V35/V36 regression tests passed. The native CUDA smoke
completed2 batch4/accum2 optimizer updates with finite losses/gradients.
Model/global RNG initialization matched fresh seed42 V35-spatial exactly.
Strict model/pipeline restoration, next proposal/noise draws, calibration
optimizer membership/state, detached latent targets and alignment gradients
through clean/image modules passed. Two-image K1/K4 evaluation emitted100
detections per condition (50 per image); its AP is not performance evidence.
Peakallocated1.106GiB/reserved1.344GiB. Model parameters41,590,981 plus6
training-only calibration parameters. Image import282/282; SHA
9ff5a0930cab53c42a903d1f10fec1ebbb49457a127987afab54e1c72d8d71d9.

## Run and monitoring

Run: `/storage/outputs/pose-graph-diff/results/poseflow_v37_latent_ota/20261031_170101`

Frozen source, resolved configuration, source/input hashes, CUDA preflight and
persistent console/status are stored in the run root. Fresh seed42,80epochs,
original data augmentation/256-max427 resize, batch4/accum2 FP32, AdamW
2e-4/backbone1e-4, weightdecay1e-4, milestones31/61, clipping.1. Shared
initial-model SHA083e35a988ba78d79cfbd62dbc5590cf63690293fde1ae890540287cc70c1a8f.

Session: `poseflow-v37-latent-ota`, with a progress window. Existing object-only
TensorBoard33330 includes `V37-latent-OTA`, retaining earlier runs and K1/K4 AP
pins. The launcher automatically evaluates epoch079 on full8000K1/K4 after
training.

The first process reached zero optimizer updates: its long run-root TMPDIR
exceeded AF_UNIX socket path limits in DataLoader workers. Only that V37 process
was interrupted and restarted with the same frozen source/config/model, using
short local `tmp/v37/ipc`. Initial-attempt logs/status and runtime_override.json
record the correction. No earlier training process was restarted.

Live initial check: 50 finite updates, real-person coverage1,
peak allocated1.381GiB. These are launch checks,
not evidence of learned detection quality.

Final launch verification: 90 finite updates, real-person coverage1, dashboard
HTTP200 and59 scalar tags including alignment components. All three earlier
early/timed/q200 training PIDs remained alive and progressing.
`verification.json` stores this snapshot; `status.json` continues updating.
No epoch-level AP is available yet.
