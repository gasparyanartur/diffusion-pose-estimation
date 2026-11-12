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
