# V53: birth-only generation with native joint queries

V53 removes coordinate diffusion and inter-generation refinement. Each new person
uses the original ED-Pose keypoint expansion and two retained pose layers. Earlier
people remain fixed clean geometry, and inference stops individually at EOS.
The objective is competitive birth pose AP on the existing fixed32 VAL quick-test;
verification alone is not evidence of detection quality.

## Architecture

The native image frontend retains encoder layers 2/4/6 and box decoder layer 2.
The selected image-local proposal expands into one person query and 14 joint
queries. Native keypoint embeddings, learned joint reference extents, positional
MLP, post-normalization, ReLU FFNs, box heads, pose heads and extent heads are
retained with pose decoder layers 4/6. Features have width 256 and eight heads.
Projected image values are cached once per image for each pose layer.

Clean completed people supply conditioning tokens through a fresh 32→256→256
coordinate MLP plus detached proposal feature/reference inputs. Native group
masking restricts joint queries to their own new-person group; person queries
exchange context with completed-person tokens. Hidden features are recomputed
per call. Decoder layers update only the new person's predictions; previously
emitted geometry is never replaced. The model has no codec, noise input, sigma
embedding, DDIM transition, consistency loss, refinement loss or post-EOS tail.

The selector retains candidate exclusion and sample-local identities. Birth
confidence is native objectness times the birth non-EOS probability. Capacity
remains 49 people, with up to 50 teacher rows including the first EOS. Capacity
stopping requires no additional EOS call. No interference scheduler is added.

## Training and evaluation

GT people are randomly ordered once per forward. One selected-slot Hungarian
assignment fixes proposal identities across clean teacher-forced rows. Only the
new person is supervised, equally over both pose layers and then people.
Robust pose/box definitions are inherited from V52, with equal normalized birth
task priorities and frozen TRAIN-only median gradient scales over 32 batches.
Native frontend/DN losses occur once per batch; candidate CE occurs on birth
rows; EOS BCE balances person and first-EOS rows per image. Empty images retain
EOS and frontend supervision. Separate native/selector/geometry clipping bounds
are each 0.1/√3. There is no online task reweighting.

Recipe: fresh original ED-Pose transfer, seed 42, full 80 epochs, FP32, batch 4
with accumulation 2, resize 256/max427, existing flip/crop augmentation, AdamW
2e-4/backbone1e-4, weight decay1e-4, drops31/61. TRAIN10000/VAL2000 and the fixed32
quick-test image IDs are retained; TEST is unopened. Best selection uses
min(pose_AP,bbox_AP), with best/last retained and selected fullVAL evaluation at
completion. Birth and final geometry are identical and use a single run stream
with shared `quick-test/pose_AP` and `quick-test/bbox_AP` tags.

`prepare_context`, `forward_rows`, `predict` and image-only `sample_keypoints`
are implemented separately from legacy families. Checkpoints record native
configuration, expansion, clean-prefix policy, objectives, initialization,
calibration and RNG, and restore without loading external initialization assets.

## Verification and evidence limits

Failing tests preceded the native adapter, model/training, and evaluator. The
adapter matches a native expanded group exactly for batch two with no prefix.
CPU checks cover independent row outputs/gradients, padding exclusion, both-layer
birth supervision, frozen calibration/RNG, empty/missing-joint data, immutable
births, individual stopping, deterministic inference and strict checkpoint restore.
V50/V52 regression checks preserve existing behavior.

Native full-capacity FP32 CUDA verification, initialization coverage, source/input
hashes, checkpointed gradient parity, resumed update and shared-GPU memory/cost
measurements are required before launch. Their results and run location will be
recorded below. Initial or subset AP is not a competitive full-run result. This
version changes pose decoding and removes refinement together; it cannot isolate
their individual contributions.


V53 native verification passed: FP32/all50 teacher rows/49people/batch4accum2/non-square256x416. Peak training allocation 2201446400 bytes; update 1.130s (7.08 images/s); full-capacity inference 2.36 images/s alongside existing jobs.255 parameter tensors pass checkpointed gradient parity (maxabs2.56e-06); next-update loss 17.3021297454834 matches resume exactly, maxstate7.33e-08.399 frontend and94 pose-decoder tensors match the original native transfer exactly, with100% retained coverage. Frozen birth pose/box scales [2.3932335,1.0367477] measured on32 augmented TRAIN batches; initial model hash d03d00d283159d9717e3efc2cb61c01976aa849830ede4055f335d8e5d9dd1aa. No resource failure and no recipe reduction. Initial two-image evaluator smoke is wiring evidence only.

The initial launch guard caught tuple/list serialization in the native checkpoint contract before training. JSON-normalized configuration and an explicit roundtrip regression fix it. Repeated verified source: `tmp/v53/verification_json/verification.json`; same initial model hash. Repeated measurements:0.961s/update,8.33images/s,2.48full-capacity inference images/s,2.05GiB allocation; resumed maxstate4.19e-8. Failed namespace20261113_231254 retained.

The next launch archive omitted native kernel sources; only V53 was stopped while status was starting, preserving that namespace. The corrected archive includes C++/CUDA sources/headers, all629 source files are hash-pinned, and the trainer rejects fallback before training.22 tests and repeated full native verification pass. Authoritative report: `tmp/v53/verification_native/verification.json`;2.05GiB allocation,0.926s/update,8.64images/s,2.39full-capacity inference images/s, resumed maxstate5.59e-8 and identical loss. Both failed delivery attempts are preserved and are not training experiments.

V53 training launched in `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v53/20261113_231953` from629 verified executable/native source hashes (902 archived files); initial model hash matches preflight. PID2842214, tmux`poseflow-v53`,80epochs/FP32/batch4accum2. Parameters39,404,025 (39,181,625 trainable). Baseline32VAL image IDs match V52 exactly. Confirmed finite optimizer progress through step150; existingV51/V52/observer/bridge preserved. Native fallback is forbidden by the entrypoint. TensorBoard adds V53-native-joint-birth-only under shared measurement tags while preserving77roots/5pins. Live AP/train scalar API verified; full visual/history indexing proof pending. Training outcome and50AP milestone remain unproven.

V53 DELIVERY COMPLETE: native-backed trainer2842214 active, epoch index0, optimizer step320, finite loss5.022033.22checks and native FP32 full-capacity/strict next-update verification passed. Restored native joint expansion with fixed clean completed-person geometry;629 verified source/native hashes match immutable workspace. ExistingV51/V52/observer/bridge preserved. Shared quick-test pose/bbox cards visibly overlayV50/51/52/53 with initialV53 AP and live training scalar data; all12 prior comparison series/614points preserved,77old roots/5pins retained. Screenshot tmp/v53/dashboard.jpg and native API/AX/history proofs archived. Full80 run, epoch1 trained AP and50poseAP outcome pending; retain plan for outcome review. No further architecture changes.

V53 final live-refresh proof: TensorBoard caught up to76loss samples through step759, trainer step760 with finite loss4.475401. Startup history indexing delay resolved; shared AP/table visual proof and614historical comparison points preserved. Implementation/launch complete; full80/learned AP outcome pending.
