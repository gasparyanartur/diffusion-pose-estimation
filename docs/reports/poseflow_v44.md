# V44: native human proposals with frozen-codec latent diffusion

V44 preserves native DiffusionRegPose up to human selection and replaces joint
expansion with one diffused 32D pose-and-box latent per person proposal. This is
one training baseline, without a separate alignment stage.

## Contract

The reduced frontend is ResNet, three retained ED-Pose encoder layers, 200
queries, one native box-decoder layer and top-50 human selection. Its encoder
uses `timesteps=None`, making its image memory/proposals independent of the
noise level. Two retained native decoder layers receive
`human_feature + latent_MLP(h_t) + time_MLP(t)` and predict clean latents.
Their deformable image reads use fixed detached proposal boxes. Native joint
expansion/readout modules are removed. The image frontend remains trainable.

The SHA-pinned High-KL codec is
`sample_kl_high_clean/checkpoints/epoch099-score97.2048.ckpt`, SHA256
`7168bffc7452f83e5e7c70b062d55ff867d2cb7d1eb0932cd1339cdc9e0ee951`.
Both encoder and decoder are frozen/eval. Clean targets use posterior means,
standardized by fixed TRAIN-only posterior-mean moments, with no per-code norm
normalization. Statistics cover all 10,000 TRAIN images, one deterministic
augmented view per image, seed42 plus image ID: 34,096 retained person views.
The decoder is differentiable with respect to predicted codes.

Training and inference use the same top-50 selection. Hungarian class/L1/GIoU
costs 2/5/2 assign selected slots once per training forward pass. One explicit
assignment controls all latent decoder targets, geometry masks and confidence
labels. There is no denoised-output matching. Native box/encoder supervision
retains separate matching, and the native box-layer DN branch retains known
DN labels. Unmatched selected slots have zero clean codes, negative confidence
labels and no geometry loss. Confidence is the image-only selected class logit.

Each latent decoder layer receives weight10 balanced positive-plus-background
latent MSE and weight0.5 existing V43 geometry loss. Native frontend losses and
weight2 selected-slot focal loss are added once. Geometry uses naturally labeled
joints; all pose and box predictions pass through the frozen decoder.

Forward corruption is 1,000-step cosine VP latent Gaussian noise, uniformly
sampled per image. DDIM eta0 carries only the latent state. Image memory,
references, slot identities and scores are computed once and reused. There is
no coordinate corruption, re-encoding between steps, coordinate bypass or EMA.

## Recipe and evidence

Seed42, 80 epochs, FP32, batch4/accumulation2, AdamW2e-4 (backbone1e-4),
weight decay1e-4, clipping0.1, LR drops at31/61, native 256/427 resize and TRAIN
flip/crop augmentation. TRAIN/VAL are disjoint (10,000/2,000). The unused eager
test-loader slot points to VAL, so development does not open TEST. Codec and
ED-Pose pretraining may already have trainval exposure; VAL is not an independent
pretrained-detector test.

Focused tests verify native pre-expansion feature/box/index parity, fixed
assignment/masks, latent diffusion arithmetic, codec invariance and input
geometry gradients, empty/missing-joint/B=1/non-square cases, capacity rejection,
query caching, strict state restoration and forced validation transforms.

Native FP32 CUDA preflight completed two optimizer updates with accumulation2.
Codec/statistics were unchanged. Restored optimizer/scheduler/RNG produced equal
next-update losses (68.19273376464844), maximum state difference5.96e-8.
Model parameters:41,167,480 total,40,370,520 trainable. Preflight peak CUDA
allocated memory:2,431,850,496 bytes (includes the restoration comparison).
These are finite/restore checks, not learned-performance results. Native
initialization coverage: input projections/level embeddings/encoder/heads 100%,
decoder 99.95%; native strict-load precedes replacement of expansion modules.

## Evaluation and artifacts

The live TensorBoard stream is `V44-native-HighKL-LDM`. K1/K4/K8 Gaussian-start
VAL pose/bbox AP uses the existing six `quick-test/noise_var_100/steps_K/*_AP`
cards, native CrowdPose evaluator and native pose NMS. An initial step0 baseline
is logged before updates. The same fixed32 VAL images are evaluated each epoch.
Best checkpoint maximizes the best-over-K minimum of pose/bbox AP; only best and
last are retained. The selected checkpoint receives full2,000VAL K1/K4 evaluation
when training ends.

Small-noise curves use fixed matched GT-derived codes/noise, nearest discrete
sigma to1/2.5/5/10%, with actual sigma/time recorded. Input/output latent MSE,
paired OKS/IoU and error tails are logged under `quick-test/noise_*`. These are
conditional reconstruction metrics, separate from Gaussian-start detector AP.
Exported figures use the same fixed slot-to-person correspondence.

Run root and launch health are recorded below after launch. Source/config/input
hashes, TRAIN statistics, exact initial model identity, preflight evidence and
checkpoints are preserved in a fresh run namespace.

Run: `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v44/20261105_215551`.
The implementation passed 43 focused/regression checks. A source snapshot and
verified initialization are retained in that run. Training progress and dashboard
verification are recorded in `launch_verification.json` when available.

Launch verified at325 optimizer updates (epoch0/80), finite loss9.4159.
TensorBoard port33332 exposes `V44-native-HighKL-LDM/.` on all six existing
quick-test AP cards and live training scalars. All51 previous roots/pins remain;
archived streams reload with four threads. Existing trainers remain alive.
`launch_verification.json` verifies625 frozen source files,5 input hashes, exact
initial model identity, disjoint splits and the accessible dashboard. Step0 AP
is an untrained initialization baseline; first learned VAL AP awaits epoch end.
