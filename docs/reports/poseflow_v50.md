# V50: proposal attention with noise-gated residual geometry

V50 addresses two failures observed in V49: randomized pure-noise birth lacked a person identity, and repeated absolute reconstruction could move an already good pose. It retains V49-balanced's robust geometry, explicit joint/box means, fixed TRAIN gradient scales, equal person/age buckets and box/pose priorities. This combined experiment does not isolate their contributions.

## Model

The original ED-Pose retained backbone, three image encoder layers and one native box decoder layer produce image memory and 50 local candidate features once per image. The frontend uses V44's unchanged cached-context helper and native box/encoder/DN losses. A single selected-slot Hungarian assignment maps each GT person to a local proposal, held fixed across all teacher-forced rows; refined geometry is never rematched in training.

A prefix-conditioned query attends over remaining proposal keys plus a learned EOS key. Candidate indices are sample-local bookkeeping, with no integer-ID embedding. Conditional candidate cross-entropy trains selection among proposals; the marginal EOS probability has the balanced person/first-EOS/tail objective. Geometry receives one hard-gathered candidate feature and reference, detached on this identity path. Candidate keys and native detector remain trainable. Random teacher order is sampled once per image/forward. No future teacher identity or geometry enters the selector query.

Two fresh dense decoder layers (width256, eight heads, FFN2048, pre-LN/GELU/no dropout) process independent training rows, with shared image K/V and RNG-preserving activation checkpointing. All active people attend bidirectionally. Coordinate encoding/decoding is jointly trained. Birth predicts absolute geometry conditioned on the selected proposal. Existing people receive a deattenuated coordinate bypass and noise-gated residual: XY delta coefficient sigma/(2*scale), positive size reference times exp(clamp(sigma*delta,-10,10)). Noisy nonpositive sizes clamp to1e-6. Sigma0 returns the input coordinates exactly. Small nonzero residual-head initialization permits shared-decoder gradient calibration.

Inference executes50 rows with capacity49 and a final refinement/EOS row. The first EOS latches generation; remaining rows refine. Existing states follow coordinate DDIM eta0, with sigma .1 at age1, then halving, and zero at the final row. Transformer hidden states do not persist. Selected local/native identities persist. Confidence is native objectness times non-EOS probability at birth and is retained during refinement. All image-only evaluations use per-image seeded noise.

## Objective and recipe

Retained objective: robust OKS-relative pose/box geometry plus masked coordinate means; equal person totals; refinement groups ages1/2/older; four fixed median shared-decoder gradient scales from32 augmented TRAIN batches without updates. Box:pose priorities are3:1 at birth,1:1 at age1,1:3 for older tokens. Conditional selection and native frontend losses are added once. Empty images retain EOS/frontend losses; missing joints keep their masks.

Fresh original ED-Pose initialization, seed42, full80 epochs, TRAIN10k/VAL2k, original resize/augmentation, FP32, physical batch4/accumulation2 (effective8), AdamW2e-4/backbone1e-4, weight decay1e-4, clip.1 and LR drops31/61. No V49 warmstart. TEST stays unopened. Best=min(pose_AP,bbox_AP) on the same fixed32VAL images, last checkpoint retained, selected best evaluated on all2000VAL after training.

## Verification and status

Implementation uses failing-first helper/model/pipeline tests. Checks cover sample-local permutation, used masks, detached identity, residual endpoints, native frontend parity, row independence, padding exclusion, bidirectional influence, teacher identity exclusion from selection, one context/assignment, both-layer gradients, calibrated restore, seeded batching, EOS latch and capacity. Native FP32 full-capacity verification and launch measurements are recorded below. Unit tests/preflight are implementation evidence, not learned detection quality.

TensorBoard tags: `quick-test/proposal_attention_residual/rows_50/{pose_AP,bbox_AP}`; separate teacher-forced and generated-assignment diagnostics. Strict V50 architecture/calibration contract rejects old model contracts and restores without initialization/calibration assets.

## Verified launch (2026-11-11 22:03, Europe/Stockholm)

Fresh run: `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v50/20261111_215450`; tmux `poseflow-v50`, PID1102983, seed42/full80. Live first epoch, optimizer step380, finite loss6.308419, about8.55images/s. Physical batch4/accumulation2; retained AdamW/resize/augmentation recipe. V42timestep trainer continues; cancelled V49 artifacts remain intact.

79 distinct focused/historical checks passed across the full suite and targeted reruns. Final immutable native verification passed all50rows/49people, FP32/batch4/accum2/non-square256x416/missing joints: peak2055795712B (1.91GiB), 9.42training images/s and6.20inference images/s. Shared-GPU timings vary; these are synthetic full-capacity measurements, not controlled speed comparisons. Resumed next-update loss46.38261604309082 exactly matches, max state difference1.64e-07. All retained backbone/projection/transformer/box/class components have100% exact measured transfer (386 state tensors including aliases). Unused native diffusion-time MLPs were removed; V50 sigma conditioning remains active.

TRAIN32 calibration leaves model and RNG unchanged; fixed scales `[2.5276451110839844, 2.4136385917663574, 0.0008402460953220725, 0.001334645552560687]`. Fresh initial model SHA256 `257b5f15fe25ddfed0fc7b3b076a4dbeadb8e4f64056879c8cbb226893c98110` agrees between verification and training;594 source/config hashes and881 archived source/test/doc files. Native2VAL evaluator passed. Initial fixed32VAL pose/bboxAP=0/0 is a pre-training baseline. Same32 IDs as V49; all2000VAL selected-best evaluation remains scheduled after80epochs. Raw TRAIN/VAL maximum annotations15/14 fit capacity49; TEST unopened.

TensorBoard33332 includes V50 plus all72 historical roots; all prior pins retained, both new AP tags added (23 total including the browser's prior training-loss pin). Live selection/frontend/total loss and native AP scalars verified; browser visibility/screenshot proof under `tmp/v50/dashboard.png`. AP cards update after each completed epoch. Launch is not evidence of learned detection improvement or successful generated refinement. Report detection quality, generated versus teacher-forced refinement, and computational cost separately as the run progresses.

## First-EOS and refinement diagnostic (2026-11-12)
- Frozen selected best epoch047 checkpoint (48 completed epochs), SHA d9ccf2b2471e7a8841e809cbe9f0194aa38046eb3959657c1f68b90ad7ac3d29; same fixed32VAL/image seed42+ID, native evaluator/postprocessing, same generation IDs/counts/confidences. FirstEOS includes that row's refinement. No production source, optimizer updates, sampler policy or active job changes; TEST unopened; model state unchanged and existing final AP exactly reproduced.
- Same-trajectory birth snapshots pose/bboxAP15.5264/44.6146; firstEOS15.0337/42.0360; row50 exactly15.0337/42.0360 (0-100). EOS mean row3.625, all32 stop byEOS; max normalized EOS-final coordinate delta.00058818. Tail offers no measured AP benefit on this checkpoint/subset; end-to-end early-stop speed not benchmarked. Birth snapshots are not a separate birth-only generation policy.
- Teacher-forced learned pose-refine task.01113920 versus zero-residual input bypass.01114463, only.0487% improvement. Joint coordinate MSE2.922841e-5 versus2.922817e-5 (essentially identical); box-refine task.02098725 versus.02087676 (learned.529% worse). Both-layer averaging and original task masks/scales retained. Pose heads' optimizer states show60000 updates and nonzero first moments; no evidence of frozen heads, but no useful denoising improvement measured here.
- TensorBoard60epoch trajectory: train total21.8948->16.2967, birth-pose2.77077->.77489, native frontend5.45878->2.07514 and selection3.10029->1.22765 improve; pose refinement.0222036->.0231691 stays flat/noisy. VALpose-refine.01973396->.01968242 improves only.26%; weighted TRAINpose-refine contribution7.314->7.628 remains a large nearly constant floor. Prior train/inference mismatch explanation alone cannot explain flat TRAIN refinement.
- Archive diagnostics/first_eos_epoch047 under immutable run root; scratch tmp/v50/first_eos/report.json and compare.py. Keep current training policy; any production early-stop/refinement redesign needs its own reviewed plan. Retain .plan pending outcome review.


## Refinement failure audit (2026-11-12)

Read-only production investigation on frozen epoch047 (48 completed epochs), SHA256
`d9ccf2b2471e7a8841e809cbe9f0194aa38046eb3959657c1f68b90ad7ac3d29`.
Independent diagnostic optimizers used four augmented TRAIN images
119068/106143/119073/101307, with3/1/3/6people. Native image memory and projected
K/V were cached/frozen; row decoder, codec and heads could train. FP32,
AdamW2e-4/weight-decay1e-4, clipping.1. These are bounded wiring/learning controls,
not a new run, native AP evaluation or convergence study. Existing V50/V42
trainers, production sources and frozen checkpoint are unchanged. TEST unopened.

### Confirmed loss/gradient routes

An analytic residual oracle reduces pose-refinement task loss from.02650994
to9.43e-14 and joint coordinate MSE to0; box loss reaches1.66e-7.
Direct optimization of emitted residual tensors reduces pose-refinement loss
to6.74e-6 after80updates (>99.97% reduction). Gradients reach residual heads,
coordinate encoder and dense decoder. Native checkpointed/direct joint-path
gradients differ by at most1.75e-10. These tests reject a disconnected loss or
wrong correction sign on this example; they do not rule out every possible bug.

### Measured conditioning degeneration

For age1 people, added time-embedding RMS is104.9047, versus.4332 for coordinate
features, .6635 for proposal content and.9565 for rank features. A single time
channel contains99.834% of time-vector energy (maximum absolute activation1677).
Newborn time-embedding RMS is only.2842. Age1 row state RMS is105.302 before
the first decoder layer; normalized readout RMS is.01362/.01380 across layers.
The large finite time offset dominates normalization and suppresses content
variation. The mean joint residual-head diagonal sensitivity to normalized
input coordinates is-4.87e-5 in one age1 token; the exact fixed-GT-identity
noise correction would have diagonal-120. This oracle derivative is a wiring
reference, not an expected Bayes-optimal derivative for uncertain image poses.

Frozen-weight interventions increase joint residual sensitivity (Jacobian
Frobenius norm) from.0047399 to.086488 (18.25x) with time RMS capped at1,
.090521 with time multiplied by.01, and.080383 by removing its largest channel.
These interventions establish a contribution to suppressed sensitivity;
they do not establish improved AP or a complete fix.

### Task gradients and failed fresh-noise learning

At the coordinate encoder, weighted pose-refinement gradient norm is.006183,
box-refinement.006063, birth-pose.01381, birth-box.01636, EOS.11126 and selection
.18932. Selection is30.6x stronger than pose refinement and their encoder
cosine is-.296 on this batch. At the residual heads pose-refinement norm is1.192;
at the dense decoder it is.06951. The existing calibration observes only shared
decoder layers, so it does not balance the shared coordinate encoder. This is
local evidence of imbalance/interference, not proof it is the sole cause.

On fixed rows and fixed noise,160updates reduce pose-refinement loss by63.5%
with refinement alone (.02651->.009674), and57.8% with the joint row objective
(.02651->.011195). Thus the branch can optimize/memorize this task.
With independent noise each update on the same13 identities,400updates fail
to beat the bypass: pose-refinement loss is0.03445984 vs.0344184 for refinement-only,
and.0344711 vs.0344184 with joint row losses. A capped-time refinement-only
control also fails (.0347831 vs.0344184); fixing the time amplitude alone is
insufficient in this400-update test. Reported comparisons use the same fresh
noise at each measured point, not different-noise loss trajectories. No native
frontend losses were recomputed in these cached-context controls.

### Information survives but is poorly conditioned

For the same13 TRAIN identities, freeze the model and generate256 Gaussian
training draws plus128 held-out draws per identity, sigma.1. Fit an independent
standardized linear regression for each identity. Held-out noise MSE is.09965
from final readouts versus.99969 for zero prediction (~90.0% explained),
.07999 from coordinate-encoder features (~92.0%), and4.07e-12 from raw coordinates.
The unstandardized readout coefficient RMS is236031, versus331 for encoder
features. This probe has identity-specific supervised maps and known training
identities; it demonstrates accessible local information and poor scaling,
not a deployable denoiser or unseen-image generalization. The actual learned
age1 joint residual RMS is.00631 versus oracle.9664, correlation-.0288.

Conclusion: useful refinement learning has collapsed toward the input bypass.
Confirmed contributors include a runaway additive time channel and a poorly
conditioned shared input/readout path; encoder task imbalance is also measured.
No single production repair is validated. A proposed fix must address bounded
conditioning and usable coordinate sensitivity, then pass independent-noise
fixed-identity learning before another full run. Candidate implementation
requires a reviewed plan and regression test; current trainer remains active.

Artifacts: `tmp/v50/refinement_audit/` and immutable run
`diagnostics/refinement_audit_epoch047/`, including scripts, logs, all numeric
reports and hashes. Augmentation/data RNG comes from the restored checkpoint;
explicit row/noise generator42112, fresh-noise CUDA generator98712 and
linear-probe generator112112 are recorded separately.


Gradient audit follow-up (2026-11-12): full native TRAIN/AdamW/accumulation audit on32images confirms encoder classification dominance and98.25–99.65% gradient energy in native image parameter blocks. Separate clipping budgets plus protected geometry routing improve local first-order pose-refinement descent estimates3.39–9.58x, with no learned-performance claim. See [full audit and recommended implementation sequence](poseflow_v50_gradient_audit.md). Production training unchanged.
