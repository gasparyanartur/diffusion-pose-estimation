# V42 whole-pose feature baseline

V42 replaces DiffusionRegPose's expanded box/joint query groups with one learned
feature vector per person. Gaussian diffusion remains over pose XY coordinates:
noisy coordinates are encoded into a person vector; a learned reader predicts
clean coordinates, which drive the native cosine DDIM update. JEPA is deferred
until baseline pose performance has been reviewed.

The control is `benchmark_diffusionregpose_development_compatible_256_keep_layers`:
three encoder and three decoder layers initialized from EDPose layers 2/4/6,
width 256, four image levels, eight heads, 200 initial queries, 50 selected
people and 50 DN queries. One box-only layer precedes two pose layers. The
existing resize/crop recipe uses shortest side 256 and maximum side 427;
batch four with accumulation two, FP32, seed 42 and an 80-epoch schedule.

The native image encoder, proposal heads, scored selection, class/box heads,
reference detach boundary, Hungarian matching and task/auxiliary/proposal/DN
losses remain. The whole-pose encoder is a two-layer MLP, 28→256→256. Its
output is added to selected image-conditioned person features at the former
expansion boundary. Each pose layer has a three-layer 256→256→256→28 MLP with
an absolute sigmoid XY readout. The constant joint confidence retains the native
sigmoid(1) convention. There are no joint query embeddings or joint reference
updates. All image attention uses the current person box.

The box stage retains four offsets per head/level. Each ordinary pose query
uses 60: the original 15 tokens × four offsets. DN queries retain their
four-offset attention. Thus normal and DN image sample counts match at every
stage. New offsets use native radial initialization and the native division by
point count; retaining sample count does not retain the original joint-centered
sampling footprint, attention capacity, FLOPs or wall time.

The earlier runs incorrectly retained timestep-zero-only training. The corrected
`benchmark_poseflow_v42_uniform_t` samples independent per-image timesteps0..999
for coordinate corruption. See the correction and launch evidence below.
Inference starts from Gaussian XY coordinates, re-encodes each reverse step and
carries coordinates only. It recomputes the original image/proposal stage per
call. Full-pose task supervision establishes the baseline; there is no latent
reconstruction, JEPA, EMA or SIGReg loss.

Configuration: `conf/pipeline/poseflow_v42.yaml` and
`conf/benchmark_poseflow_v42.yaml`. Model, training and quick-evaluation code live
in `src/models/pipelines/pose_estimators/poseflow/{model,training,quick_eval}_v42.py`.
The native training entry point supports this configuration:

```bash
python scripts/training/train_diffusionregpose.py --config-name benchmark_poseflow_v42
```

The native `scripts/evaluation/evaluate_diffusionregpose.py` supports strict V42
restore using its `--config`, `--sampling-timesteps`, `--shortest-side 256` and
`--max-size 427` arguments. The historical Euler-oriented PoseFlow evaluator's
model-builder path is not used for V42.

Quick evaluation uses a fixed seed-42 sample of 32 test images after each epoch.
K1/K4/K8 use the same Gaussian source per image (`42 + image_id`), native
pose NMS, and pose/box AP from the same detections. Model modes, RNG state and
sampling-step settings are restored. Tags retain the historical layout:
`quick-test/noise_var_100/steps_{1,4,8}/{pose_AP,bbox_AP}`. The launch wrapper
also requests full 8,000-image K1/K4/K8 evaluation after epoch 79. Subset AP and
full-test AP are distinct artifacts.

Verification: seven V42 tests cover shapes, no expansion, noisy-input gradients,
sample counts including DN, reference detach, batch isolation, empty targets,
native losses, strict restore, coordinate sampling and paired evaluation/RNG.
The two focused suites contain 102 distinct tests. Native CUDA verification on
a real augmented batch four with accumulation two completed two finite updates.
All 481 surviving original model tensors exactly match a separately seeded
native control; this includes buffers and is not a count of pretrained tensors.
The original pretrained coverage gate runs before conversion. Model/optimizer
restore reproduces predictions exactly, and a controlled next update agrees
within 3.78e-8 in FP32. A separate two-image K1/K4/K8 smoke checks evaluator
wiring; its AP is not a learned-performance result. Peak reserved CUDA memory
in that verification was approximately 2.15 GiB.

V40-SIGReg was already stopped by its numerical guard. V41 was explicitly
cancelled by the user; its checkpoints, logs and fixed-subset results remain.
The V42 comparison TensorBoard includes previous pose-estimation runs and V41;
V41 contributes box metrics only. Missing historical K8 measurements are not
filled with invented values. Dedicated dashboard and frozen-run manifests live
under local `tmp/v42/` and the timestamped V42 output directory.

Remaining scientific decisions: define acceptable matched-control pose AP before
adding JEPA, then review feature target/correspondence, noise coverage and
collapse prevention. A shared person vector alone does not guarantee invariance
across noisy inputs. Strict weight/optimizer restoration does not establish
exact replay of dataloader workers and prefetched augmentation batches.

## Verified launch, 2026-11-03 20:35 Europe/Stockholm

Fresh run: `/storage/outputs/pose-graph-diff/results/poseflow_v42/20261103_203019`. Trainer PID 447947 in tmux `poseflow-v42`.
At verification, epoch 0 had reached 300 optimizer steps with finite
loss 49.13892. All 623 frozen source files and five input hashes match
the run manifest. Training is ongoing; learned pose AP has not yet been evaluated.

The dedicated comparison dashboard runs on port 33331 and includes 33 run roots
(36 indexed streams, including resumed and diagnostic streams). All six requested
K1/K4/K8 pose/box pins were verified through the UI and HTTP scalar index.
Its URL, manifest, tag index and screenshot are under `tmp/v42/dashboard/`.
Initialization evaluation is stored separately at `seed42/initial_quick_test/`,
with the verified initial-model hash and genuine step-zero measurements for
all six tags. Epoch-end evaluations use `seed42/quick_test/`.

V41 stopped gracefully after the cancellation request; its last checkpoint is
preserved and its status is explicitly cancelled. V40-SIGReg was already
guard-stopped and was not restarted. Historical logs and dashboards remain.


## Approved V42 box-center reader repair

The original V42 absolute reader produced initial pose centers almost identical
across people, which damaged pose-based Hungarian correspondence. The repaired
reader computes `sigmoid(reader(h) + inverse_sigmoid(reference_center))` for
each joint, using the same incoming box reference as the layer's image read.
There is still one vector per person, no joint expansion, and no new gradient
route through detached reference construction. Noise schedule, losses and
initial weights are unchanged. The parameter/state keys are unchanged.

`pose_readout_mode` defaults to `absolute` for old configs; the V42 preset now
explicitly selects `box_center_residual`. The mode is recorded in checkpoint
hyperparameters. Existing frozen runs retain their source and original config.
The approved learning check is two full-data epochs with the original 80-epoch
scheduler horizon, followed by same-age fixed32 K1/K4/K8 pose/box AP and stage
quality comparison. Longer training requires outcome review.

Three intended-red tests then 18 focused checks passed. Native CUDA batch4 with
accumulation2 completed two finite updates, strict model/optimizer/prediction
restore, and a controlled next update (max parameter difference4.10e-8). All
initial model tensors have the identical hash to the failed run. On the same
32 images/133 people, the actual repaired initialization increases mean matched
box IoU from0.1096 to0.3714 and pose/box assignment agreement from1.50% to20.30%.
These are correspondence diagnostics, not learned AP improvement.

Repair pilot launched 2026-11-03 21:34 Europe/Stockholm at `/storage/outputs/pose-graph-diff/results/poseflow_v42/box_center_pilot_20261103_213148`.
Trainer PID459492, tmux `poseflow-v42-repair`; two full-data epochs, scheduler
horizon80. The original run has a preserved interruption checkpoint and explicit
cancelled status. The comparison dashboard retains its six pins and labels the
streams `V42-failed-absolute` and `V42-repair-box-center`. Learned results are
pending; this pilot will not extend itself beyond two epochs.

### First repaired epoch, 2026-11-03 21:52 Europe/Stockholm

At1500updates, identical32testimages/seed/resize/sampler versus the failed run
(AP points out of100):

| Calls | Failed pose | Repaired pose | Failed box | Repaired box |
|---|---:|---:|---:|---:|
| 1 | 0.098 | 7.434 | 2.323 | 19.766 |
| 4 | 0.204 | 7.256 | 2.281 | 19.645 |
| 8 | 0.122 | 7.024 | 1.611 | 19.624 |

The improvement is a measured learned result on this subset, not a full-test
result or proof of the eventual baseline quality. The second epoch is ongoing.
The first checkpoint is preserved separately with a SHA256 manifest. Its saved
mode is box_center_residual and scheduler horizon remains80. A separate legacy
restore check reproduces the original checkpoint's CPU predictions bitwise
under frozen and updated source (B2).

### Completed two-epoch repair pilot, 2026-11-03 22:09 Europe/Stockholm

The approved pilot completed3000optimizer updates and stopped. The trainer
process exited. Both fixed32 epoch evaluations and the final clean-input stage
diagnostic completed successfully. This tests only the box-center readout
correction, with identical initial parameters and unchanged training recipe.

Matched epoch1/globalstep3000 results,32testimages, seed42, resize256/427,
paired Gaussian sources and native pose NMS (AP points out of100):

| Calls | Failed pose | Repaired pose | Failed box | Repaired box |
|---|---:|---:|---:|---:|
| 1 | 0.013 | 20.094 | 0.490 | 26.120 |
| 4 | 0.010 | 16.453 | 0.797 | 24.094 |
| 8 | 0.057 | 15.229 | 0.866 | 23.603 |

The near-zero failure is substantially improved in this bounded learning test.
One call performs best here; additional calls reduce AP on this subset. These
are subset measurements after two epochs, not final full-test performance or
evidence that iterative denoising is beneficial. JEPA remains deferred.

In the separate GT-conditioned near-clean t0 diagnostic, mean box IoU under
pose-based matching rises from0.2059
to0.6773. Raw top50 box AP at the encoder,
first box stage and final stage is respectively
25.92,
29.96 and
19.81. Final refinement still
reduces box AP relative to the first box stage in that diagnostic; this is a
remaining limitation, not a reason to silently change the approved architecture.
The diagnostic is distinct from Gaussian-start inference AP above.

Verification:18focused checks passed, nativeFP32CUDA training/restore checks
passed, and the real historical checkpoint's B2 CPU predictions are bitwise
identical under frozen and updated source. All624frozen source files and five
input hashes match. The final checkpoint and AdamW states are finite; all
optimizer steps are3000. The saved readout mode is box_center_residual and the
scheduler horizon is80.

Final checkpoint: `/storage/outputs/pose-graph-diff/results/poseflow_v42/box_center_pilot_20261103_213148/seed42/checkpoints/epoch001.ckpt`.
SHA256: `db567c5151df56faa0156cb47b65ac5566f692c3ba5ccd7d2063628416922b02`.
The epoch0checkpoint is preserved under `seed42/preserved_checkpoints/`.
Exact metrics/comparison: `/storage/outputs/pose-graph-diff/results/poseflow_v42/box_center_pilot_20261103_213148/comparison.json`; final audit:
`tmp/v42/repair/final_verification.json`. The TensorBoard scalar endpoint was
checked for all six AP values at step3000. The browser comparison view filters
V42 runs and disables outlier clipping to show the full AP range. All previous
run roots and six pins remain available. Screenshot: `tmp/v42/repair/dashboard.png`.

The pilot is stopped pending outcome review. A longer run of the repaired
baseline is the next suggested iteration; no80epoch continuation was started.

2026-11-03 23:58 CET: User explicitly authorized full training continuation and rejected further two-epoch pilot gates. Resume repaired epoch001/step3000 from preserved checkpoint, same frozen source/recipe, total80epochs, native strict model/AdamW/scheduler restore. K1/K4/K8 fixed32 checks continue after each epoch; full8000 evaluation remains scheduled at completion. Persistent tmux session `poseflow-v42`; continuation provenance/logs under the existing repaired run (`continuation.json`, `continuation.log`, `run_v42_continue.py`). No JEPA or architecture change.

2026-11-03 23:58 CET: V42 full training continuation RUNNING, PID497679, tmux poseflow-v42. Resumed preserved repaired epoch001/step3000 with strict native model/AdamW/scheduler restore verified; live epoch2/step3050, finite loss/gradients. Total80epochs, same frozen source/initialization/data/loss/noise/optimizer. No further pilot cutoff or review gate. Fixed32 K1/K4/K8 after every epoch and full8000 final evaluation retained. Existing TensorBoard33331 indexes V42-repair-box-center/resume; historical streams and sixpins preserved. Pilot endpoint/status/manifest/config remain preserved; continuation.json and continuation.log record the launch.


## Corrected box-center training, 2026-11-04

All earlier V42 runs (absolute reader, repaired box-center and matched-target)
used the native target builder's hardcoded timestep zero. This did not implement
the requested diffusion training distribution. Their historical results remain
measurements of those t0-only models. The matched-target run was cancelled at
epoch33/50,950 recorded updates; its process exit and saved interruption
checkpoint are verified. Its artifacts are preserved.

The corrected `benchmark_poseflow_v42_uniform_t` uses ordinary output-Hungarian
supervision with the box-center reader and independent per-image uniform integer
timesteps0..999. Coordinates are corrupted at the sampled timestep, then encoded;
the same timestep is passed into the model. No JEPA or noisy-input target matching
is added. Architecture, initialization, optimizer, augmentation and DDIM remain
unchanged. Legacy configs/default target-builder calls retain t0 for exact restore.

Six intended-red checks preceded implementation. The final focused/native suite
passes107 tests, covering mixed t0/500/999 corruption, invalid timesteps, exact
legacy corruption/RNG, actual forwarded random timesteps, empty images and
output-Hungarian use. Native FP32 CUDA batch4accum2 verification passed finite
updates, model/AdamW/scheduler strict restoration and a controlled continued
update (maximum parameter delta1.19e-7). Observed training forward timesteps were
643,409,798,250,92,207,443,973,989,991,646,432,33,388,871,284. The initial model
SHA256 is exactly the previous box-center initialization:
`ef71a0f0543d7109117605ee9e162052d81aaff935c3ea685e7e21db27373628`;
481 surviving native tensors match. Two-image K1/K4/K8 checks are wiring checks.

A fresh full80epoch run, fixed32 K1/K4/K8 each epoch and final8000 K1/K4/K8 are
authorized without a pilot cutoff. Local verification/launch helpers live under
`tmp/v42_uniform_t/`; live source/config/run provenance will be recorded below.


2026-11-04: Corrected V42 box-center uniform-t full80epoch training RUNNING, PID666506, tmux poseflow-v42-uniform-t, root /storage/outputs/pose-graph-diff/results/poseflow_v42_uniform_t/20261104_133538. Fresh EDPose/seed42 initialization exact prior box-center hash; no t0 checkpoint resumed. Native coordinate corruption samples per-image integer t0..999 and same actual t conditions model; ordinary output-Hungarian, no JEPA. Live epoch0/step200, finite loss/gradients; observed 1600 images/790 distinct timesteps, range1..998, all10 time bins sampled.107tests/nativeCUDA/strictmodel-AdamW-scheduler restore PASS.631frozen source+5input hashes worker/audit verified. Failed matched-target PID552828 cancelled with interrupted checkpoint preserved; old V42 control PID497679 remains running. TensorBoard0.0.0.0:33332 preserves prior42roots/sixpins and adds V42-box-center-uniform-t (43roots/47streams); scalar history reload ongoing. Fixed32 K1/K4/K8 each epoch/full8000 afterepoch79 retained, no pilot cutoff. Evidence launch_verification.json and local tmp/v42_uniform_t; first learned AP pending. Next follow-up: epoch-end sampled AP and actual-t histogram, without another launch gate.

Corrected V42 launch final verification: PID666506 active at epoch0/step250; actual forward timestep range1..999. TensorBoard33332 fully reindexed47scalarstreams, new V42-box-center-uniform-t training-loss data populated and all6pose/boxK1/K4/K8 pins have existing measurements; prior43configured roots including new stream preserved. Corrected run AP populates after each epoch; no invented values. No pending launch gate or pilot cutoff; training proceeds through80epochs.


[2026-11-04T17:53+01:00] V42 uniform-t similarity audit complete
Same-age epoch11/18000 fixed32 poseAP K1/K4/K8: uniform32.830/31.091/30.917; t0control29.640/29.662/28.950. Averaged epochs8..11: uniform31.051/29.309/29.342 vs t0control31.450/30.830/30.531; no consistent pose gain demonstrated yet. Frozen config/source match manifest and uniform training histogram covers all1000 timesteps roughly evenly.
Frozen epoch011 K1 baseline reproduces all saved evaluator metrics exactly. Same32/seed42/256427/nativeposeNMS: baselineposeAP32.830; alternateGaussianseed32.744 (joint shift0.166 resizedpixels mean); zero coordinate state32.765 (0.129px); zero pose encoder31.729 (2.595px); sameGaussian input/timestep0 gives32.617. At t0 GT-derived native-corrupted input31.414 vs zero-state32.756 (0.851px); t500 GT-derived32.085 vs zero-state32.511 (0.619px). Conditional GT-input AP is a recovery diagnostic, not inference AP. Coordinate deltas average all50 ordinary slots with fixed image/timestep and unchanged preselection, so slot order is matched.
Finding: weak coordinate-state dependence at this checkpoint; image/proposal-driven solution is supported. Coordinate inputs only add an MLP feature at selection, while box-referenced image proposals/readout are rebuilt each call; output-Hungarian task loss allows this solution. This is an empirical inference-ablation interpretation, not proof of exact training causality or all-checkpoint behavior. Uniform-t correction is active; useful iterative state-guided refinement remains unestablished. No production/model/job/dashboard changes; both live jobs retained. Evidence tmp/v42_uniform_t/sensitivity/{report,same_age,audit_summary}.json and sensitivity_conditional/report.json. No further run/architecture change proposed as approved scope.


[2026-11-04T19:00+01:00] Proposal–pose correspondence audit COMPLETE; no production changes
Frozen corrected V42epoch014/22500, fixed32/133people plus32 seeded augmentedtrain images/112validpeople, native refs/inputsource/finalHungarian assignments captured. SavedGaussianK1 poseAP32.573 and all metrics reproduced exactly. On augmentedtrain sample: knownsource slots17sameGT/30differentGT/65unmatched (15.2/26.8/58.0percent); frozen eval-mode diagnostics, not live training-gradient labels. Fixed32t0:22same/32different/79unmatched. Source coverage100percent using exact raw masked joint match. Some proposal IoU>=.5 covers94.0percentGT, but source's own slot covers30.8percent; mean ownslotIoU.305 vs geometryaligned.807. GTassisted bijective input-bank permutation preservesboxes/readcount, raises sourceboxcoverage93.2percent, but conditionalAP31.045->30.940: correspondence mismatch confirmed, frozen alignment doesnotrestoreAP; causal learned-suppression effect needs controlled retraining.
Historical answer: V40 explicitly proposal-Hungarian->gather assignedGT->encode/noise->same recovered task/latent targets, verified in original frozen worker/helper;2targeted tests PASS. V41 replaced this with independent source/reference banks and separate final/output JEPA alignment (source/proposal identity not fixed), verified current=frozen. V33/V34 latent state itself is query, no rankedimageproposal additivebank; frozen methods verified. V19 state defines sampling references/residual, native DiffusionRegPose has learned coordinate-fusion attention but no hard source/proposal identity alignment. V40forcedmatchedcandidate inclusion differs from score-top50 inference and is a selection-policy tradeoff, not drop-in proof of AP benefit. No new fix/run approved or implemented; jobs retained. Full report docs/reports/poseflow_v42_correspondence_audit.md; artifacts tmp/v42_uniform_t/correspondence/{report,details,v40_checks.log}. Proposed next design decision is consistent proposal/source/supervision mapping, with explicit selection policy, before another experiment.


[2026-11-04] Repaired V42 late Gaussian-AP collapse diagnosed (read-only)
Historical t0-only control K1 fixed32 poseAP33.815(epoch9)->4.385(epoch63), boxAP37.738 and val loss15.697->7.542. Frozen epoch1/63 baseline metrics exactly reproduced. Early Gaussian/GTt0/zeroAP20.094/17.148/19.368; late4.385/54.718/2.147. Holding Gaussian coordinates fixed while changing t999->0 gives4.385->4.336, ruling out time embedding alone. Holding native background padding fixed while randomizing only GT-source slots gives54.718->17.951 (erasing gives7.814); informative pose dependence is learned late. Clamped Gaussian6 partial recovery17.223 at999 also identifies input-scale mismatch. Interpretation: t0-only training/validation learn conditional pose recovery, while Gaussian initialization is untrained; source-content/scale mismatch drives observed sampling degradation. No sole-cause optimization claim or full-test generalization. Earlier weak state-use applied to younger uniform-t model; late repair clearly uses pose content despite rematching. Correcteduniform t0..999 removes this distribution omission; its late outcome remains unmeasured. No model/job/dashboard changes. Report docs/reports/poseflow_v42_box_center_collapse.md; artifacts tmp/v42/repair/collapse_audit.
