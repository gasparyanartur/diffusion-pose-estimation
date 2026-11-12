# V29: OTA person-latent targets

Launch:2026-10-25, fresh seed42/80epoch pilot alongside V28.
Root: `/storage/outputs/pose-graph-diff/results/poseflow_v29_ota/20261025_201803`; frozen source `/storage/outputs/pose-graph-diff/results/poseflow_v29_ota/20261025_201803/workspace`.
Config: `/storage/outputs/pose-graph-diff/results/poseflow_v29_ota/20261025_201803/seed42/.hydra/config.yaml`.
Shared dashboard: http://lynx.tail481642.ts.net:33329/ ; label `V29-OTA`.
Status/log: `/storage/outputs/pose-graph-diff/results/poseflow_v29_ota/20261025_201803/status.json`, `/storage/outputs/pose-graph-diff/results/poseflow_v29_ota/20261025_201803/launch.log`.
Visible progress terminal and tmux session `poseflow-v29`.

## Target assignment

The model is unchanged `PoseFlowV28Model`: original V26 four-level3+3 stacks,
50x256 person latents, clean image memory AND positions detached at clean input,
h0 detached before corruption and as recovery target. Clean construction and task
supervision retain known source indices and recipient annotation-row provenance.

`DiffusionRegPoseOTAMatcher` assigns recovered predictions against unique original
receiving-image GT people, not duplicate clean slots or foreign-image negatives.
Dynamic-K uses topk5 box IoUs; box/center candidate prior radius2.5; conflict repair
preserves at least one query perGT forM<=50. One GT may receive several queries,
but each query has at most oneGT. Existing pose-present matcher cost is class +
jointL1 + OKS with box candidate penalties; boxL1/GIoU cost is disabled in that
existing pose-present cost path. No historical matcher behavior is changed.

Matched queries target the assigned person's detached original clean latent from
canonical firstM slots. Unmatched queries target their nearest detached clean
negative latent from this image's constructed negative slots. Distance selection
and matching run under no_grad. No negative geometry is supervised. Multiple
background queries can select the same clean negative latent. All50slots/channels
contribute uniformly to latent MSE. Empty recipient uses all negative targets;
full50person budget has no unmatched query and needs no negative bank.

Objective stays0.5clean-task + latent-MSE1. Denoised task losses are monitoring-only
under OTA labels. They are not directly comparable to fixed-correspondence task
losses, since assignment/count normalization changed. No direct recovered pose
loss trains denoiser/heads. Image extraction receives latent-recovery gradients
through denoiser conditioning; clean encoder/readout learn from clean task.

## Evidence before launch

- Six TDD tests initially failed for missing helper/config, then passed.
-127 V24–V29 and DiffusionRegPose-training regression tests passed; diff check clean.
- Tests cover swapped GT latent targets, source-negative->positive, nearest
  background selection, empty/full budgets, Dynamic-K/coverage/conflict repair,
  query permutation, detached sources, clean-image gradient stop, model/pipeline
  strict restore, next proposal/noise draws and source-free evaluator loading.
- Full seed42 model-state SHA256 `268bdfe42eadf545bcddc5d1223af05fb8a0fd152389d7e34eff0a122e73abe2`
  equals V28 exactly;41,994,721 parameters. Matcher adds pipeline buffer only.
- Strict restored Gaussian K1/K4 predictions match unchanged V28 sampling bitwise.
- Real CUDA batch4/accum2 optimizer smoke passed: allocated0.843GiB,
  reserved1.023GiB. Smoke reports100% uniqueGT assignment.
  This is a wiring/optimizer check, not learned performance.
- Actual Lightning fit/validation/checkpoint smoke passed.
- Image import282 tensors100%, image SHA256 `9ff5a0930cab53c42a903d1f10fec1ebbb49457a127987afab54e1c72d8d71d9`.
- Frozen resolved config equals V28 except namespace/family/matching strategy and
  new center-radius field; topk5 and all data/optimizer settings retained.

## Training/evaluation

Same CrowdPose trainval recipe,256/427 resize, batch4 accumulation2, FP32/TF32off,
AdamW LR2e-4/backbone1e-4, decay1e-4, milestones31/61, clip0.1.80epochs/120000updates,
fresh original EDPose image initialization and fresh person-latent modules.
Fixed32 deterministic Gaussian K1/K4 monitoring every epoch; no monitor checkpoint
selection. Final epoch079 automatically evaluates all8000 held-out test images
with native K4(primary)/K1 eta0 DDIM, seed42+imageID and corrected pose NMS.

Main comparator is V28 at the same training age; V28 is left running with its
frozen source. Original V26 includes denoised pose loss; V24 differs in architecture.
The final evaluation records V28 endpoint if available, otherwise explicitly pending.
Existing shared TensorBoard labels are retained; no additional server is created.
Inherited tags use `debug/v26` under distinct V29 label. Additional `ota/` diagnostics
record assigned/background counts, uniqueGT coverage, multiplicity, source-person
reassignments, converted negative slots, negative bank usage and positive/negative
latent MSE. Denoised confidence uses OTA labels; clean/construction tags retain
source labels. V28 tags retain their old meanings.

## Interpretation limits

This tests OTA person assignment AND explicit nearest-negative background targets.
OTA also changes the number of positive latent targets under uniform all-slot MSE,
so a gain would not isolate reduced identity ambiguity as the sole cause. Low-noise
states can also be reassigned; no timestep-dependent matching schedule is added.
Decode costs at initialization can be noisy. Monitor clean reconstruction,
conditional recovery and Gaussian generation separately; decreasing reassigned MSE
alone is not success. Validation losses overlap trainval; fixed32 is a small
monitoring subset, not the final endpoint. Current status: pilot launched,
learned results pending.

Launch verified: PID2720265, step120, epoch0, finite losses; live peak1.425GiB allocated. V28 remains active at step11640, PID2684920. Shared TensorBoard actual training and OTA scalar samples verified; local/Tailscale HTTP200. No learned AP yet.
