# V47: ordered coordinate diffusion

V47 replaces V46's frozen latent diffusion with coordinate diffusion and a jointly trained person encoder/reader. This tests the combined state/representation/objective change, not its individual contributions. Implementation, native verification and the learnability gate passed. The fresh80-epoch TRAIN/VAL run is active; learned VAL outcomes remain pending.

## Model and targets

After augmentation, valid people retain V46's canonical order:512 normalized horizontal columns, vertical center, horizontal center, box dimensions, masked joint coordinates and stable original order. Collisions preserve all annotations. Each of64 parallel training rows contains its noisy prefix and one distinct PREDICT token; capacity is50. Rows beyond EOS continue refinement and EOS supervision. Bidirectional attention is restricted to the valid row; no hidden state carries between inference rows.

Each person's32 coordinates are jointXY28 and boxCXCYWH4, normalized by the unpadded image width/height. Missing input joints are filled with their own box center before corruption; original observation masks control losses, and are not encoder inputs. Diffusion state is x=6(2c-1). V46's cosine row schedule has exact pure-noise/clean endpoints. Noisy box sizes may be negative and are accepted by the encoder without clipping.

One encoder shared over people/rows is32->256GELU->32 with non-affine LayerNorm. A256D adapter, sequence/PREDICT embeddings and additive sinusoidal time conditioning feed a fresh2-layer256D/8head/2048FFN pre-LN zero-dropout dense decoder. Each layer has its own normalized256->32 latent and EOS head. A shared32->256GELU->256GELU->32 reader decodes both heads; predicted sizes are softplus+1e-6 and other coordinates unconstrained. These latents are internal features, not targets or sampling state.

The original ED-Pose backbone/projections/encoder source layers2/4/6 initialize the retained image frontend under the existing freezing policy. The codec and decoder initialize freshly. Image encoding and each layer's image K/V projections are shared across rows, including sequential inference. RNG-preserving non-reentrant checkpointing wraps the batched decoder during training. No frozen VAE, latent statistics, teacher, KL, latent loss, matching, proposals, DN or native detection objectives are used.

## Objectives and sampler

For each person, coordinate MSE is the squared observed joint-coordinate plus box-coordinate sum, divided by 2*labeled_joints+4. Both decoder layers contribute. Average each person's birth and subsequent refinement predictions separately, then average people. Total is10*(coordinate_birth+coordinate_refine)/2 +.5*(geometry_birth+geometry_refine)/2 +balancedEOS. Geometry preserves the existing masked OKS-exponent and relative box-center/log-size/GIoU terms. EOS balances present classes within each image, then averages images. Empty images receive EOS only.

Inference runs64 sequential rows. Existing noisy coordinate states are encoded afresh, decoded clean predictions update them with DDIM eta0, and newly born people receive forward noise once at the next row's level. EOS latches permanently; capacity50 also stops births while refinement continues. Final outputs are row64/layer2 geometry, birth confidence, valid masks, counts and sequence ranks. Per-image RNG keeps sampling independent of evaluation batching. Optional traces contain decoded coordinates and row/layer/rank/birth/stop information.

## Verification and training gate

The approved four-image TRAIN probe selects one image each with1/2/3/4 valid people, disables augmentation and allows2000 optimizer updates, evaluating every100. Both seeds42/43 must produce exact counts on every image and pose/bbox AP>=90 on the same four training images. This is memorization evidence only. Failure blocks the full launch for diagnosis; it does not authorize silent tuning. Full training always starts freshly.

Native preflight uses FP32/all64rows/50people/batch4/accum2/nonsquare images, checks restored next updates and reports training/inference throughput and peak allocation alongside existing jobs. Resource failure must be reported before changing precision, rows, batch or gradients. Existing trainers/artifacts and TEST are preserved.

After passing, launch fresh seed42/full80 on TRAIN10k/VAL2k with V46's augmentation/256short-side427max resize, AdamW2e-4/backbone1e-4,wd1e-4,clip.1,FP32,batch4accum2,drops31/61. Fixed32 VAL image-only AP before/per epoch selects best min(pose_AP,bbox_AP); retain best/last and evaluate best on full2000 VAL after80. ED-Pose pretraining exposure remains disclosed; VAL is model-selection data.

## Interfaces, restore and usage

Model APIs: prepare_context(images,padding_mask=None); forward_rows(context,noisy_coordinates,row_times,token_types,padding_mask); predict(context,image_wh,generator=None,collect_trace=False); image-only sample_keypoints. Outputs preserve native geometry/confidence conventions and add coordinate predictions, validity/count/rank/stopping fields.

Strict V47 checkpoints include conventions/scale/missing-joint policy, row/order/noise, learned codec/decoder, objectives, DDIM/newborn/EOS behavior, initialization, private and global RNG. Normal trainer checkpoints also restore optimizer/scheduler. Restore creates architecture without reading initializer assets. Exact prefetched DataLoader replay is not established by fixed-batch next-update agreement.

Run native verification and the learnability gate from a frozen workspace:

```bash
TMPDIR=$PWD/tmp venv/bin/python -m scripts.training.verify_poseflow_v47 --output <fresh-verification-directory>
```

Only after a passed verification with matching sources/config/input/init hashes:

```bash
venv/bin/python -m scripts.training.train_poseflow_v47 verification_file=<verification.json> hydra.run.dir=<fresh-seed42-directory>
```

TensorBoard run:V47-ordered-coordinate; AP tags:quick-test/ordered_coordinate/rows_64/{pose_AP,bbox_AP}. Teacher-forced and generated rank errors remain distinct; teacher-forced final row receives clean GT. Native AP uses fractions0..1 unless explicitly multiplied by100. Trace-collected shared-GPU evaluator timing is not deployment latency.

## Evidence

53 focused/legacy checks passed; a subsequent preset-construction regression caught and verified correction of the initial Hydra target typo. Native FP32/all64rows/50people/batch4accum2/256x416 verification passed at `20261109_145640`: peak training allocation2,342,332,416bytes (2.18GiB), identical resumed next-update loss627.7232666015625 and maximum state delta1.49e-8. The measured first update34.59seconds includes CUDA startup/shared-GPU contention; it is not steady-state throughput. Trace-free inference measured2.321images/s with two resident verification models.

The initial four-image diagnostic was invalid because image100000 has four posed people plus nine noncrowd box-only people. Native bbox evaluation scores those extra targets, whereas the pose model training contract excludes them. Perfect posed-person outputs achieve pose100/bbox52.475AP on that initial subset, so the90bbox gate is impossible there. Artifacts and the native checkpoint are preserved; only that diagnostic verifier was stopped. The corrected probe selects one image each with1/2/3/4 people and matching native noncrowd box/valid-pose coverage, then requires an oracle pass through the full native output/postprocessing path before optimization. The model, gate thresholds, full TRAIN/VAL split and native evaluator remain unchanged. The corrected learnability gate and fresh full80 launch subsequently passed; details follow.

This annotation-coverage difference also affects interpretation of native bbox AP on the full/fixed32 VAL protocol: it includes box-only targets that are outside the current pose-sequence supervision. Keep the metric unchanged and disclose the limitation; do not interpret oracle posed-person outputs as a100bboxAP ceiling guarantee.


## Learnability gate and full80 launch (2026-11-09)

Frozen run root: `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v47/20261109_150341`. Native verifier3475182 completed, and the full trainer3490732 runs in tmux `poseflow-v47`. The earlier startup snapshot145428 and coverage-invalid diagnostic145640 remain archived.

All55 focused/new/V46/V44 checks pass. The corrected probe oracle gives100/100 pose/bbox AP. At500 optimizer updates, both evaluation seeds42/43 pass with pose98.80198/bbox91.72560 AP0–100 and exact person counts4/4. This is four-image TRAIN memorization, not VAL quality. No overfit weights enter full training.

Refinement behavior on the passing probe is mixed. Averaged over the same10 generated people, coordinate MSE changes5.89453e-5 at birth ->7.58170e-5 final, while geometry loss changes.0705176 ->.0690794. High memorization AP does not establish uniformly improving refinement or rollout stability. All rank-wise teacher/generated errors and both-seed native predictions are retained under verification/.

Final native preflight repeats FP32/64rows/50people/batch4/accum2/256x416 with peak2,342,332,416B, equal next-update losses627.7232360839844 and max parameter delta1.49e-8. All74decoder/49encoder/10codec/42trainablebackbone tensors have finite nonzero native gradient state. Cold/shared-GPU update32.23seconds (.248images/s) includes initialization; trace-free inference2.134images/s has two resident verification models, so neither is an isolated deployment benchmark. Early actual full-run throughput is5.61images/s at step70 with peak3,167,055,360B; these are shared-GPU early measurements and may change.

The full trainer rebuilds freshly and verifies initial model SHA256 `0f3930206699cbb62b9471ec89571adf7b22a09cd26ade6b2e1af78417f2c2d0` against preflight. All549 source/config/script hashes plus input hashes are recorded. TRAIN10000/VAL2000 are disjoint. Initial fixed32VAL uses exactly the V46 image IDs and gives AP0/0 before training. Every-epoch fixed32VAL selection, best/last retention and selected full2000VAL after80 are wired into the entrypoint; those outcomes are asynchronous and pending.

TensorBoard33332 contains `V47-ordered-coordinate/.` plus the distinct coverage-valid and archived invalid diagnostic streams.67 roots preserve all64 pre-V47 roots plus both diagnostic streams; all12 pre-V47 pins remain, with four new AP pins added. Native AP0 baseline and live coordinate/geometry/total loss scalar data are indexed. Existing task-pose/ordered/V46 trainers remain alive; TEST remains unopened.

Artifacts: `verification/verification.json`, `preflight.json`, `native_gradient_coverage.json`, `subset_oracle.json`, `overfit.json`, `refinement_summary.json`; `seed42/manifest.json`, `status.json`, `quick_test/`; launch and dashboard manifests in the run root. Raw timing includes existing workloads. See `.plan` for retained outcome-review TODOs.
