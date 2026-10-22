# Experiment diary

Research recipes and investigation notes moved from the README. These describe
separate experiments, including historical controls and diagnostic runs; they are
not a recommended sequence to execute. Commands assume the environment, data and
weights described in the [README](../README.md) are already available. Run them
from the repository root.

Use each experiment's own configuration and checkpoint family. Recorded smoke,
subset, oracle and low-resolution results do not establish full-test performance.
Some notes describe instrumentation added for subsequent processes; they do not
assert the current state of a running job. Local session instructions and private
dashboard addresses have been removed.

## Model and training recipes

```bash
# Select the graph-pose pipeline
python scripts/training/train_graph_pipeline.py pipeline=graph_pose

# Select the tensor-native box-diffusion pipeline
python scripts/training/train_box_diffusion.py pipeline=box_diffusion

# Train the native DiffusionRegPose implementation
python scripts/training/train_diffusionregpose.py

# Train PoseFlow. It retains the ED-Pose/DiffusionRegPose architecture and
# task losses, uses pose-aware OTA by default, and transports only keypoint XY
# with continuous endpoint flow. It is a new training path: do not evaluate a
# DiffusionRegPose checkpoint as though it were a trained PoseFlow model.
python scripts/training/train_poseflow.py

# Train PoseFlow V3: each state carries two signed box corners plus pose XY.
# It runs a 200-slot pool, carries 50 selected slots, and refills 150 fresh
# slots at each later Euler call. Its fixed gate is image/state = 1/0 at t=0
# and 0.5/0.5 at t=1. Checkpoints are incompatible with V1.
python scripts/training/train_poseflow.py --config-name benchmark_poseflow_v3

# Matched V3 control: identical pool/refill/loss setup with state conditioning off.
python scripts/training/train_poseflow.py --config-name benchmark_poseflow_v3_control

# Train PoseFlow V4: all encoder locations are position-aware image tokens;
# every call independently constructs 200 timestep-conditioned hypotheses from
# image evidence and the Fourier-encoded coordinate flow state, selects 50,
# and refines box and pose at every decoder layer.
python scripts/training/train_poseflow.py --config-name benchmark_poseflow_v4

# Matched V4 control: same tokens, pool, losses, and timestep path with only
# the Fourier coordinate-prior path disabled.
python scripts/training/train_poseflow.py --config-name benchmark_poseflow_v4_control

# V4 additionally logs detached activity diagnostics under
# debug/{train,val}/ (the TensorBoard run label identifies the model):
# coordinate/image/hypothesis RMS, initial score, and initial-to-final
# movement. Canonical train/ and val/ tags contain every stage loss.

# Train PoseFlow V5: V4 hypotheses with a two-cycle cosine schedule,
# augmented/interpolated GT-state training sources, and auxiliary box-center
# supervision. It is a distinct checkpoint family; do not resume V4 weights.
python scripts/training/train_poseflow.py --config-name benchmark_poseflow_v5

# Train PoseFlow V13: independently matched latent/geometry supervision.
# Start fresh; every epoch also evaluates 32 fixed test images at noise
# variance 25/50/100 percent with 1/2/3 DDIM calls.
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v13

# V13 reconstruction/gradient experiment: weight 1 -> 0.01 over epochs 0-39,
# then hold; gradient norm cap 2.0. Fresh run with bounded gradient diagnostics.
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v13_reconstruction_decay

# Matched OTA experiment: same reconstruction schedule and cap; change matcher.
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v13_ota_reconstruction_decay

# V14: box +14 joint diffusion tokens, per-call hidden global tokens,
# and 50 persistent hypothesis slots. Shared image-evidence/memory updates.
# Fresh checkpoint family; OTA top-k 5, reconstruction decay and cap 2.0.
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v14
# Uses evaluate_poseflow with --flow-sampling-steps for latent DDIM calls.
# Main quick-test tags match V13; reset control uses quick-test-reset/.
# Extra diagnostics use debug/quick-test/. JSON retains memory condition names.
# Hypotheses carry between calls; global tokens reset. Training uses one call.
# Artifacts include raw box coverage/duplicates/confidence and per-step quality.

# V15: direct local image updates + carried 16-token person representations.
# Hypothesis memory uses learned scalar retention (initially 0.9).
# Separate checkpoint family; inherits V14's OTA/data/optimizer recipe.
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v15

# Same V15 architecture; 50/50 single-call / differentiable 2–3-call training.
# Fresh run, Gaussian rollout starts and random descending timestep schedules.
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v15_rollout

# Fresh rollout run with non-affine normalization of hypothesis writer queries only.
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v15_rollout_query_norm training.resume=null

# Fresh rollout run with tied candidate memory in 17-token local self-attention.
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v17 training.resume=null

# Fresh tied-memory rollout run with pre-image FFN and normalization restored.
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v18 training.resume=null

# TensorBoard: quick-test/noise_var_100/steps_3/pose_AP, etc.
# The 25/50 percent conditions use GT-derived inputs; 100 percent starts
# from Gaussian noise. These are preliminary subset metrics, not full-test AP.
# Disable the bounded diagnostic with pipeline.quick_eval_enabled=false.
# Summary/prediction JSONs: <run>/quick_test/epochNNN/; fixed IDs: subset.json.
# V13 uses the same standalone evaluate_poseflow CLI and --flow-sampling-steps.
# V13 decoder classification uses focal sum / matched-person count; clean
# reconstruction keeps slot-mean classification.
# Individual loss layers use _0/_1/_2; unsuffixed tags show the final layer;
# *_mean tags show layer averages. loss_latent_mean enters the total loss.
# loss_reconstruction stays raw; loss_reconstruction_weighted enters the total.
# debug/optimization/ shows objective gradients, accumulated preclip norms,
# clip_factor, expected_postclip_norm, learning rates and reconstruction weight.

# Train PoseFlow V12: fifty fixed groups, 17 Gaussian latent states plus a
# global helper token per group. Coordinates are readouts, never sampler inputs.
# Separate checkpoint family; this command starts a fresh full training run.
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v12

# V12 evaluation uses the existing evaluate_poseflow CLI. Its
# --flow-sampling-steps option controls latent DDIM calls (eta=0), not Euler
# coordinate updates. Renewal and ensemble flags are rejected for V12.
# Bounded debug/latent logs separate conditional losses from sampled quality.

# Train PoseFlow V11: box/existence coarse latent, then 17 structured tokens.
# Refine with image -> within-person -> existence-only cross-person attention.
# Center/size flow, shared point/size/existence MLPs, and V5 LR schedule.
# Requires a new V11 checkpoint; no corner or auxiliary center losses.
# V11 sampling skips the extra noisy-joint feature addition by default;
# training and conditional validation retain it. Existing V11 checkpoints load
# unchanged. For historical sampling, set pipeline.sampling_joint_conditioning
# to true in the config passed to the evaluator. Box conditioning stays active.
python scripts/training/train_poseflow.py --config-name benchmark_poseflow_v11

# Two-epoch wiring smoke: full architecture, 12 train / 2 validation images.
# Restart at epoch 1 only to exercise both LR cycles during the smoke.
python scripts/training/train_poseflow.py --config-name benchmark_poseflow_v11 \
    run_name=poseflow_v11_smoke out_dir=results/poseflow_v11_smoke \
    data.subset_frac=0.001 data.num_workers=0 data.ap_batch_size=2 \
    training.max_epochs=2 training.test_after_fit=false \
    pipeline.lr_cycle_restart_epoch=1 pipeline.val_ap_every_n_epochs=1

# Train PoseFlow V10: one person latent and shared absolute pose/bbox readout.
# Retains V5 initialization/losses; uses single cosine decay with main/backbone
# peak LR 1e-4/1e-5, warmups 1/10 epochs and group-relative 1% minima.
# Use a distinct V10 checkpoint family.
python scripts/training/train_poseflow.py --config-name benchmark_poseflow_v10

# Train V10-tiny with model width 128 and transformer FFN width 256.
# It retains the same EDPose ResNet weights; width-dependent layers retrain.
python scripts/training/train_poseflow.py --config-name benchmark_poseflow_v10_tiny

# Retry V10-tiny with V5's two-cycle LR schedule in a separate run.
python scripts/training/train_poseflow.py --config-name benchmark_poseflow_v10_tiny_v5lr

# Train PoseFlow V9: V5 with learned skeleton-biased joint attention
python scripts/training/train_poseflow.py --config-name benchmark_poseflow_v9

# The standalone PoseFlow evaluator accepts optional --step-probe-dir,
# --stage-probe-dir, --feature-probe-dir, and (V9 only) --attention-probe-dir.
# Use --flow-seed to pair source draws, --limit for a fixed diagnostic subset,
# and separate output directories.

# Future V5/V9/V10/V11 runs log `debug/solution/val_first_batch/...` on periodic AP
# epochs. The fixed first validation batch records proposal/refiner GT quality,
# flow-slot turnover, score/NMS counts, and (V9) per-layer attention behavior.
# `pipeline.solution_feature_diagnostics=true` also runs four paired feature
# interventions on that batch; set it to false to avoid the extra forwards.
# Existing processes retain their loaded instrumentation; these additions
# apply to subsequent processes. V9 skeleton-bias attention remains V9-only.
# V10 also records latent update/spread, readout saturation, decoded movement
# and tiny boxes under the first-batch latent/step_<s>/layer_<l>/ tags.
# V11 reports coarse box coverage (no coarse pose metrics) and role/update RMS,
# decoded width/height, reference saturation, and cross-person existence updates
# under first-batch tokens/step_<s>/layer_<l>/ tags.
# debug/lr/{main,backbone} records each epoch's effective learning rates.

# The preset uses the external-compatible variable-size multi-scale training
# transforms, batch size 1 with gradient accumulation 8 (effective batch size
# 8), and reference trainable parameter set (49,304,924 parameters).
# The paper-compatible schedule runs for 80 epochs, initializes from the
# EDPose checkpoint, clips gradients at 0.1, uses keypoint loss
# weight 10, and applies reference LR milestones [31, 61]. It includes 100
# denoising queries and uses FP32. Validation loss runs every epoch and
# validation AP is calculated every 10 epochs. Sanity validation is disabled,
# and validation preserves the training RNG state.
# CrowdPose person targets use category label 1, matching the annotations and
# the external implementation; retrain native checkpoints created before this
# label-contract correction.
# Checkpoints include the latest completed training epoch as `last.ckpt`.

# Train PoseFlow V6: V5's flow/source/loss contract with frozen DINOv3
# ConvNeXt-Small stages 2/3/4 read by separate deformable attention modules
# after image attention at every configured decoder layer.
python scripts/training/train_poseflow.py --config-name benchmark_poseflow_v6

# V6 is a distinct checkpoint family. DINO stage features are sampled at
# each layer's current references, concatenated with the decoder query, and
# fused through an MLP and zero-initialized residual gate. The DINO backbone
# weights must be available through Hugging Face before building the model.
# V6 uses LR 1e-4 for model parameters and 1e-5 for ResNet parameters. Their
# warmups (1 and 10 epochs) and cosine decay starts are independent; the
# ResNet reaches 1e-5 at the start of epoch 10.

# Train PoseFlow V7: frozen DINOv3 ConvNeXt-Tiny stages 2/4 plus the first
# two ResNet maps (internal layers 2/3). DINO stage 2 shares stride 8 with
# ResNet layer 2; DINO stage 4 adds stride-32 context.
python scripts/training/train_poseflow.py --config-name benchmark_poseflow_v7

# V7 uses two ResNet feature levels and two separately attended DINO maps at
# every decoder layer. It is a separate checkpoint family from V5/V6.
# Its measured model count is 67,266,675 total parameters: 39,224,147
# trainable and 28,042,528 frozen. The frozen Tiny backbone means total
# parameters are not matched to the DINO-free V5 baseline.

# Train PoseFlow V8: frozen DINOv3 ConvNeXt-Tiny, stages 2-4 for coarse
# encoder/query generation and all four stages for fine decoder attention.
python scripts/training/train_poseflow.py --config-name benchmark_poseflow_v8

# V8 removes the ResNet path and V7's per-decoder DINO fusion. At the reduced
# setup it has 42,384,848 total / 14,564,720 trainable / 27,820,128 frozen
# parameters. Encoder deformable attention uses 3 levels; decoder attention
# uses all 4. V8 is a distinct checkpoint family from V7.

# Fast checkpoint-compatible validation loop (full architecture, tiny data and
# resolution; this is an AP smoke test, not a paper benchmark). The lower LR
# avoids destroying the downloaded checkpoint on the tiny subset.
python scripts/training/train_diffusionregpose.py \
    --config-name benchmark_diffusionregpose_fast \
    run_name=diffusionregpose_fast_full_lr \
    out_dir=results/diffusionregpose_fast_full_lr \
    training.max_epochs=20 pipeline.lr=1.0e-5 pipeline.lr_backbone=5.0e-6
# Evaluate its checkpoint on the matching subset using the run config:
#   python scripts/evaluation/evaluate_diffusionregpose.py \
#       --checkpoint results/diffusionregpose_fast_full_lr/.../checkpoints/last.ckpt \
#       --config results/diffusionregpose_fast_full_lr/.../.hydra/config.yaml \
#       --annotations data/crowdpose/annotations/crowdpose_test.json \
#       --image-root data/crowdpose/images \
#       --shortest-side 256 --max-size 384 \
#       --output tmp/diffusionregpose-fast/predictions.json \
#       --limit 26 --metrics tmp/diffusionregpose-fast/metrics.json

# The reduced benchmark_diffusionregpose_debug preset is reserved for wiring
# and tensor-contract smoke tests. Its reduced decoder is not a valid submodel
# of the downloaded checkpoint and should not be used as an AP gate.

# Historical runtime-only compact model. Its 256-pixel full-data run remained
# near zero AP, so this preset must not be used as a learning-quality gate.
python scripts/training/train_diffusionregpose.py \
    --config-name benchmark_diffusionregpose_development

# EDPose-compatible 256-pixel development learning gate: 49.5M parameters,
# all pretrained encoder and pose-refinement layers, 200 initial queries,
# 50 proposal groups, batch 8, and deterministic AP every 10 epochs.
python scripts/training/train_diffusionregpose.py \
    --config-name benchmark_diffusionregpose_development_compatible_256

# Assignment-only OTA comparison at the same compatible 256-pixel resolution.
# This keeps model tensors, EDPose initialization, data, optimizer, schedule,
# denoising queries, sampling, and evaluation identical to the Hungarian gate.
python scripts/training/train_diffusionregpose.py \
    --config-name benchmark_diffusionregpose_development_compatible_256_ota

# Same checkpoint-compatible development architecture at 128/192-pixel
# resolution. This is a full-data learning gate with batch size 8, no
# gradient accumulation, validation loss every epoch, and AP every 10 epochs.
python scripts/training/train_diffusionregpose.py \
    --config-name benchmark_diffusionregpose_development_compatible_128 \
    run_name=dev128-compatible-full \
    out_dir=results/diffusionregpose_development_compatible_128

# Controlled 256-pixel comparison with every ResNet parameter frozen. This
# preserves the 49.5M-parameter architecture but leaves approximately 26.1M
# trainable parameters.
python scripts/training/train_diffusionregpose.py \
    --config-name benchmark_diffusionregpose_development_compatible_256_freeze_backbone

# Diagnostic head-only comparison: initialize from EDPose and freeze the
# backbone, input projections, label embedding, and transformer computation.
# Only prediction/refinement heads train (approximately 1.6M parameters), so
# this is a runtime/transfer experiment rather than a paper-scale recipe.
python scripts/training/train_diffusionregpose.py \
    --config-name benchmark_diffusionregpose_development_compatible_256_head_only

# Reduced three-layer transfer model. It explicitly initializes local encoder
# and decoder layers from EDPose checkpoint layers 2, 4, and 6. This is useful
# for architecture iteration, but is not an exact full-checkpoint evaluator.
python scripts/training/train_diffusionregpose.py \
    --config-name benchmark_diffusionregpose_development_compatible_256_keep_layers

# OTA counterpart to the selected-layer fast run. It keeps the 3E/3D,
# EDPose-layer-2/4/6, 256/427, batch-4/accumulation-2 setup and changes only
# the clean-query assignment strategy.
python scripts/training/train_diffusionregpose.py \
    --config-name benchmark_diffusionregpose_development_compatible_256_keep_layers_ota

# Reproduce the CPU/static and CUDA profiling artifacts described in
# reports/diffusionregpose-runtime-audit.md. Runtime profiling exits before CUDA
# allocation when less than 18 GiB is free.
python scripts/investigation/profile_diffusionregpose_runtime.py \
    --runtime-config benchmark_diffusionregpose_development \
    --output reports/diffusionregpose_runtime/runtime_development_preset.json

# Full-data low-resolution learning gate. This keeps the full model structure
# but uses 128/192-pixel train and validation resizing, batch size 8 with no
# accumulation, and AP every 10 epochs. Its 80 epochs and `[31, 61]` LR
# milestones match the paper recipe's effective batch and optimizer-update
# schedule. It remains a diagnostic run, not the paper benchmark, because it
# uses lower image resolution. The training AP uses the same crowdposetools
# evaluator as the standalone evaluation script.
python scripts/training/train_diffusionregpose.py \
    --config-name benchmark_diffusionregpose_lowres \
    run_name=diffusionregpose_lowres out_dir=results/diffusionregpose_lowres

# Continue the 128/192 diagnostic to 80 total epochs from its latest checkpoint.
python scripts/training/train_diffusionregpose.py \
    --config-name benchmark_diffusionregpose_lowres \
    training.max_epochs=80 \
    training.resume=path/to/run/checkpoints/last.ckpt \
    run_name=diffusionregpose_lowres80 out_dir=results/diffusionregpose_lowres80

# Reduced CPU wiring smoke run
python scripts/training/train_diffusionregpose.py \
    data.subset_frac=0.00005 data.batch_size=1 data.num_workers=0 data.max_size=128 \
    pipeline.backbone_pretrained=false pipeline.num_queries=30 \
    pipeline.num_proposals=2 pipeline.num_group=2 \
    pipeline.num_encoder_layers=1 pipeline.num_decoder_layers=3 \
    pipeline.num_box_decoder_layers=1 \
    training.max_epochs=1 training.test_after_fit=false \
    hardware.accelerator=cpu hardware.devices=1 hardware.precision=32

# Override hardware and training settings
python scripts/training/train_pose_refinement.py \
    training.max_epochs=100 hardware.accelerator=gpu hardware.devices=1
```


## Research investigations and later experiments

The [depth and segmentation compatibility report](../reports/3d-prior/depth-segmentation-compatibility.md)
compares pretrained models and identifies which feature backbones their heads
expect. The [point-cloud notebook](../reports/3d-prior/pointcloud-exploration.ipynb)
uses Depth Anything 3 Base on three CrowdPose images and saves aligned RGBD
arrays, RGB-colored point-cloud figures, colored PLY files, and DA3's GLB
exports under `reports/3d-prior/assets/`. The notebook checks its projection
against DA3's exporter and records DA3's estimated intrinsics. Install the
[official DA3 package](https://github.com/ByteDance-Seed/Depth-Anything-3)
in a separate environment, then open the notebook from the repository root.
It explicitly runs on CPU with two threads to avoid the training GPU. Its
single-image clouds have relative scale and show only visible surfaces.
On the three inspected CrowdPose examples, their person geometry was too
incomplete for a 3D skeletal prior; the notebook records this negative result.
The [SHARP feasibility pilot](../reports/3d-prior/sharp-feasibility.md) compares
SHARP's two depth layers and a sampled Gaussian scene on one of those images.
The [executed SHARP RGBD notebook](../reports/3d-prior/sharp-rgbd-pointcloud.ipynb)
projects final Gaussian centers into a sparse depth raster, combines valid
pixels with source RGB, and shows interactive scene and person-box point
clouds plus fixed top and side views. The first depth layer is an intermediate
visible-surface estimate; the center-derived raster is an approximation to
final rendered depth. The inspected depth-based geometry is too unreliable for
the intended skeletal prior, so this route is closed and no 3D prior was
integrated into the pose model. Future work in this thread will use DINO image
features; the notebooks remain as evidence for the decision.

The [DiffusionRegPose DINO multi-stage investigation](../reports/diffusionregpose-dino-multistage-investigation.md)
traces the existing deformable-attention memory, compares DINO fusion options,
and includes the prior simple-head backbone benchmark: frozen DINOv3
ConvNeXt-Small stage 4 scored 0.008 AP, versus 0.049/0.050 for trainable
ResNet-18/34. The [executed backbone sweep notebook](../notebooks/visualization/visualize_backbone_sweep.ipynb)
contains the full results table and visualizations. The DINO analysis explains
the feature-grid and freeze-policy caveats, compares ConvNeXt stage taps with
ViT block taps, and treats four ResNet levels plus DINO stages 2–4 (seven
shared levels) as the direct full-fusion experiment. A separate
query-conditioned decoder branch remains an optional lower-cost diagnostic.
Runtime effects for DiffusionRegPose remain unmeasured pending a matched
profile.

For an already-running trainer with older tags, maintain a separate comparison
view without restarting training or modifying its original event files:

```bash
python -m scripts.utils.quick_test_logging --source RUN/tb_logs --target results/tensorboard_comparison/RUN
```

Point TensorBoard at the target directory. The bridge follows new events every
10 seconds, preserves summary values, steps and timestamps, and resumes without
duplicating copied events. Keep it running alongside the dashboard.

PoseFlow V16 uses separate confidence, box-center and box-size diffusion tokens,
plus 14 joints. Its V15 decoder and persistent hypothesis memory are retained;
this baseline trains with single calls to isolate the token change. Run fresh:

```bash
venv/bin/python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v16 training.resume=null
```

Outputs/checkpoints use `results/poseflow_v16_training`. Decoder/codec components
initialize freshly; compatible image components transfer as in V15. Bone loss is
deferred. Validation overlaps trainval; use held-out test AP for generalization,
with fixed32 quick tests labeled as subset diagnostics.

V15 rollout with joint image/self-attention message dropout:

```bash
venv/bin/python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v15_rollout_path_dropout training.resume=null
```

The two configurable probabilities are
`pipeline.joint_image_path_dropout_probability=0.25` and
`pipeline.joint_self_attention_path_dropout_probability=0.25`. Masks are
independent per joint/layer/call; residuals, hypothesis reads, box/global receiving
paths and FFNs remain active. Evaluation enables all paths. Training uses the
V15 rollout mixture and its unchanged parameter count, in
`results/poseflow_v15_rollout_path_dropout_training`, separate from V16 and bone
loss experiments. Mask frequencies are logged alongside rollout diagnostics.

### PoseFlow V19: coordinate refinement and tied memory

V19 diffuses box center/size and joint coordinates, uses pretrained decoder
components for reference-relative refinement, and carries one hypothesis token per
candidate. It has no global token or learned geometry codec. Training combines
single-call examples and differentiable rollouts; all iterations/calls receive OTA
geometry/confidence supervision. Use the separate V19 configuration when restoring.

```bash
venv/bin/python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v19
```

See [architecture details](architecture.md#poseflow-v19-coordinate-refinement-with-tied-memory).

### V19 detached geometry

Run the controlled inter-layer geometry-gradient comparison with:

```bash
venv/bin/python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v19_detached
```

This fresh experiment keeps V19's architecture, parameter count, seed, image
initialization and rollout recipe. Only coordinate logits and auxiliary joint
extents detach between decoder layers. Feature and hypothesis recurrence, final
predictions and cross-call DDIM gradients stay attached. Intermediate predictions
retain their losses; unsupervised extent heads lose inter-layer gradients. Use the
saved detached configuration for strict restoration and the separate
`V19-detached-geometry` dashboard label for comparison. Forward values for identical
weights are unchanged; AP improvement is an experimental outcome.

### V19 sampling-reference-only detach

```bash
venv/bin/python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v19_sampling_detach
```

This fresh controlled comparison retains gradients through coordinate residual
additions while detaching box/joint sampling references before later decoder
layers' positional embeddings and image attention. Initial references, hidden
features, hypothesis memory and cross-call DDIM stay differentiable. Sampling
references include auxiliary extents, so their unsupervised heads lose later-layer
gradients as in full-detach. Both detach flags cannot be enabled together. The
parameter count, forward computation and training recipe are unchanged. Use the
saved configuration for strict restoration and the `V19-sampling-detach` label for
matched-age comparisons with `V19-detached-geometry`; inference parity alone does
not establish learned AP improvement.

### V19 evidence-ranking supervision

The separate `benchmark_poseflow_v19_evidence_rank` configuration derives from
full geometry detachment, with the same coordinate/rollout recipe. It adds
training-only noisy evidence branches at RMS-relative levels `[0, .25, .5]` and
one shared bias-free linear hypothesis-confidence head. The OTA-positive
three-pair hinge uses variance-scaled margins, margin1 and weight0.1. Validation
and inference use clean evidence, with no confidence reranking change. A zero
`pipeline.evidence_rank_weight` disables the auxiliary branches and loss.

Inspect the resolved recipe without starting training:

```bash
venv/bin/python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v19_evidence_rank --cfg job
```

Use the saved configuration for strict evaluator restoration. Identify this
family as `V19-evidence-ranking` in comparisons. The extra head adds256 parameters;
image reads are shared, while local updates run three times during training.
Implementation smokes and fixed-image evaluator checks do not establish learned
AP or confidence quality. Full-run launch is recorded below. Future confidence
controls should include independent noise
directions and unseen strengths, since ranking synthetic noise can be a shortcut.

The fresh V19 evidence-ranking run was launched on 2026-10-19 with the inherited
80-epoch recipe, batch4/accumulation2, OTA5 and the 50/50 single-call/rollout mixture.
It uses the compatible pretrained-source initialization with fresh new components,
not a resumed V19 optimizer/checkpoint. Ranking weight remains0.1. The full-detach
reference continues independently; assess the new objective at matched training
ages and distinguish synthetic ranking success from pose-confidence quality.

### V19 dense hypothesis image read

`benchmark_poseflow_v19_dense_read` extends the V19 full geometry-detach baseline.
Once at the start of each denoising call, the 50 hypotheses read all valid encoded
image tokens through dense cross-attention. Keys include spatial and level
positions; values contain encoded image features. The residual update is
`H + gamma * (.1 + .9 * (1-t)) * MHA(LayerNorm(H), F+P, F)`.
The shared scalar `gamma` starts at zero; no normalization or FFN follows the
addition. The time multiplier remains .1 at pure noise and reaches 1 at the clean
endpoint. Attention projections initialize normally, allowing the gate to learn
first and attention weights to receive gradients after it opens.

Coordinate diffusion, full between-layer geometry detachment, existing local
refinement, hypothesis-bank attention, and the 50/50 single-call versus 2/3-call
rollout objective stay unchanged. There is no confidence-ranking head or loss.
The inherited recipe remains 80 epochs, batch4/accumulation2, OTA top5 and gradient
cap2. The dense hypothesis enters geometry features through the existing local
attention after the first deformable read; the added read does not directly change
that first read's sampling coordinates.

```bash
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v19_dense_read
```

The configuration has its own `poseflow_v19_dense_read_training` output namespace.
Use its saved configuration for strict evaluator restoration. Deliberate import of
compatible V19 model weights uses `model.load_baseline_state_dict(baseline_state)`;
it reports copied state coverage and new `dense_read.*` keys, and rejects ranking
or already-extended states. This import does not restore optimizer state or select
a fine-tuning schedule.

The user authorized a fresh full run on 2026-10-19. It uses the compatible pretrained
source initialization with new dense-read components and `resume=null`, keeping the
80-epoch recipe. The confidence-ranking training run was cancelled gracefully;
its completed epoch44 and last checkpoints and logs were retained. TensorBoard
identifies the replacement as `V19-dense-read` alongside the historical comparisons.

The dense read adds263,681 parameters at width256, bringing V19 from42,008,567 to
42,272,248 parameters. Initial zero-gate predictions and baseline-path gradients
are tested for parity. Diagnostics record gate/time strength, message/update and
memory RMS, plus separately labeled per-head attention entropy. Small optimizer
smokes and fixed-image grids verify implementation and evaluator wiring; they do
not establish an AP improvement.

## PoseFlow V20: 3D guidance for dense-read hypothesis memory

V20 retains V19 dense-read's image-conditioned hypothesis update and coordinate
DDIM. It adds a shared linear 17-joint pose readout from persistent memory and
trains the student interaction alongside two final-call auxiliary terms:
projected masked 2D L1 (weight 1) and frozen GFPose score distillation (weight 1).
Only OTA-positive candidates receive these terms. The head/neck correspondence
is excluded; the twelve shared limb joints receive the available 2D labels,
including labeled occluded joints. No metric 3D labels are invented.

GFPose's U3D state is normalized image-plane XY plus scaled depth. V20 transfers
normalized image XY through the release's virtual 1000-pixel frame and scale 30;
this may introduce source-dataset framing bias and is not a calibrated camera.
The auxiliary head does not alter sampling. It could be ignored by the detector,
so evaluate native AP and intact/reset-memory controls rather than treating the
auxiliary poses or score-surrogate values as performance evidence.

The teacher uses strictly restored EMA weights and an explicit null mask. It is
frozen, evaluated in FP32 and loaded lazily only during positive training batches.
Teacher time is uniform in [.1,.6], independent of student time. The injected
noise-residual gradient uses no teacher backward. Guidance RNG state is saved in
student training checkpoints separately from attention/sampling RNG.
The normal evaluator and sampler do not need the teacher weights.

```bash
python scripts/training/train_poseflow.py --config-name benchmark_poseflow_v20
```

This is a separate checkpoint family; use its resolved configuration when
restoring. Defaults preserve the dense-read initialization, OTA, rollout,
optimizer and 80-epoch schedule. This command documents usage, not a training
launch. Comparing prior weight zero while retaining projected supervision can
separate the learned prior from the extra 2D objective; disabling both provides
exact dense-read task/sampling parity after importing the same baseline state.

Implementation verification (2026-10-19): 36 focused V20/V19 regressions passed,
including all eight V20 tests; final-weight V20 checks were repeated. The adapted
teacher exactly matches upstream EMA score outputs at three noise levels.
Reduced CPU and full-width CUDA real-data checks each passed two optimizer
updates, including strict evaluator restoration. BF16 training and sampling were
finite with FP32 teacher execution. The student has 42,285,355 parameters; the
teacher is excluded from its optimizer/checkpoint and inference path.

The provisional prior weight .01 produced a very small memory gradient in a
bounded preflight. The selected starting weight 1 produced a prior gradient
about 14% of the task gradient on first-call outgoing memory in one batch after
two smoke updates (about 1.6% on initial hypothesis slots). These are gradient
scale diagnostics, not an optimal weight or evidence of improved localization.
The full 80-epoch V20 run was launched on 2026-10-19 with fresh EDPose
initialization, seed 42, batch size 4 and accumulation 2, retaining the verified
dense-read recipe and projection/prior weights 1/1. The V19 dense-read reference
continues in parallel. Initial optimizer updates and 3D guidance diagnostics are
finite; learned performance and AP improvement remain unestablished.

## V21: V1 detector with fixed-slot hypothesis-derived proposals

V21 restores the reduced V1 coordinate-flow detector and training recipe while
introducing persistent H as an additional decoder token. Its 50 hypotheses read
image features densely with V20-style time scaling, predict initial boxes/classes,
and remain aligned with the same 50 candidates through one box and two pose
stages. There is no top-k reordering, 200-to-50 memory routing, box diffusion or
3D guidance. The original coordinate fusion, DN50, two classes, stage-loss sum,
multistep learning rates and `.1` gradient cap are retained. Candidate identity
is not ground-truth identity. The smaller initial pool and replacement of the
spatially anchored initializer are explicit differences from the V1 checkpoint.

```bash
venv/bin/python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v21
```

This first experiment uses single-call V1 training. Persistent-versus-reset
inference measures dependence on memory, not a matched trained ablation. A later
rollout-training comparison must apply the same schedules and loss aggregation
across its control arms. GFPose/projection objectives are deferred.

Implementation verification (2026-10-21): 158 focused regressions passed.
Reduced CPU and full CUDA real-data checks each completed two optimizer updates;
H and dense-image attention gradients were finite/nonzero after gate opening.
Strict evaluator restoration matched one/three-call predictions exactly; BF16
sampling was finite. The student has 40,788,233 parameters. A fresh 80-epoch run
was launched as 20261021_020829 with the verified V1 recipe and no resume.
Initial optimizer progress is finite; no held-out AP improvement is established.

V21 monitoring repair (2026-10-21): the initial trainer omitted V20-style quick
test logging. The repair adds native V21 Euler diagnostics for the same fixed32
held-out images at K1/K2/K3, with persistent and reset H. Sampling begins from
Gaussian noise; V20's GT-conditioned 25/50-percent DDIM diagnostics are not
transferred to this different sampler. The current run receives checkpoint-backed
logging without a trainer restart or changes to its optimizer, loss or weights.
For an existing V21 run, start the monitor with:

```bash
venv/bin/python -m scripts.evaluation.watch_poseflow_v21_quick_test --run /absolute/path/to/run
```

The monitor evaluates the latest completed epoch and subsequent checkpoints,
records checkpoint/config hashes and writes six conditions to the existing
`tb_logs` at the checkpoint global step. The default cadence assumes evaluation
finishes before the next checkpoint; missed earlier checkpoints are not backfilled.
Fixed32 AP is a development diagnostic, not a replacement for full held-out test
evaluation. Newly constructed V21 trainers have the epoch-end hook enabled by
default; `quick_eval_enabled: false` disables it for bounded wiring checks.

The first repaired diagnostic uses `epoch021.ckpt` (22 completed epochs,
step 33000), strictly restored with its saved configuration. On the fixed32 subset,
pose AP at K1/K2/K3 was 32.79/12.74/10.72 with persistent H, versus
32.79/31.56/33.43 with reset H. K1 prediction files were byte-identical, and
image IDs exactly matched V19-detached's existing manifest. This suggests a
repeated-call H failure to investigate; it is not full-test or causal evidence.
Checkpoint hashes, original predictions, numerical flags and metrics are saved
with the run. The dashboard retains only V5, V19-detached-geometry and V21-fixed-H.

V21 H-first correction (2026-10-21): the original parallel decoder wrote its final
H alongside geometry, leaving that returned H outside the current prediction's
loss path. The new `h_then_g` preset completes each layer's H attention/FFN before
geometry attends to it. The returned H is exactly the state consumed by geometry.
This adds no parameters or extra image-encoder call and keeps single-call V1
training, DN, OTA, losses, coordinate solver and optimizer recipe. Two attention
query groups reuse the original weights. Legacy saved configurations without the
new field retain parallel behavior. The replacement run starts fresh in its own
namespace; old checkpoints/logs remain available. Gradient wiring and optimizer
smokes are not evidence of improved AP.

Verification: 203 focused/base regression checks passed. A full-width real-data
CUDA smoke completed two optimizer updates, with finite nonzero gradients to the
returned final H on all four backward passes (norms .0083–.1642). The native
epoch-end quick-test hook ran at both smoke epochs (two held-out images, six
conditions); strict evaluator K1/K3 predictions matched exactly, and BF16 sampling
was finite. Parameter count remains 40,788,233. These are wiring checks only.

The old parallel run was cancelled gracefully with epoch030 and last checkpoints
preserved. Replacement H-first run: `20261021_113014`, under
`poseflow_v21_h_then_g_training`, fresh80epochs with `training.resume=null`.
The saved recipe differs only in H/G order and the previously approved quick-test
fields; source initialization, data, optimizer and losses match the original run.
Dashboard current-run entry is V21-H-first alongside V5 and V19-detached-geometry.

### V22: explicit timestep conditioning on V21 H-first

The model retains H-derived proposals, 50 fixed H/state pairs and the original
single-call detector recipe. FiLM conditions coordinate self-attention/fusion and
H/G decoder attention; AdaLN modulates the existing H/proposal and decoder
normalizations/FFN paths. Time embeddings use physical flow time multiplied by999;
flow-state geometry and solver times remain unchanged. New modulation heads start
at identity. A selective new-adapter coordinate gradient path is needed because
V21 detaches expanded references; inherited geometry projections remain detached.
No ranking, matching, loss, teacher or rollout change is included.

```bash
venv/bin/python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v22
```

The V21 checkpoint family remains separate for exact historical restoration.
V22 reports the same fixed32 held-out K1/K2/K3 persistent/reset quick-test metrics.
This is a combined conditioning-and-embedding-scale experiment; a gain cannot be
attributed separately to FiLM, AdaLN or time scaling without further ablations.

Verification (2026-10-21): 209 focused/base checks passed. Full-width CUDA
real-data training completed two optimizer updates, with nonzero finite gradients
to the coordinate conditioners and final returned H on all four backward passes.
The two-image/six-condition native epoch-end hook ran at both smoke epochs;
strict K1/K3 restoration matched exactly and BF16 sampling was finite. The model
has 44,603,401 parameters. These checks establish wiring, not learned AP.

The prior V21 H-first trainer was stopped gracefully with epoch009/last checkpoints
and logs preserved. Fresh80epoch V22 run `20261021_143304` uses original source
initialization and `resume=null`. Its resolved recipe changes only the pipeline
family, embedding scale and output namespace. Dashboard retains V5/V19-detached
and replaces the current entry with V22-time-conditioned.

V22 batch-layout correction (2026-10-21): frozen probes found that the inherited
coordinate helper flattened[group,batch,joint,XY] directly into a batch-leading
shape. Batched training consequently mixed other images' states while retaining
per-image reference coordinates and times. Earlier parity checks shared this
reference defect, and B1 held-out evaluation hid it. The corrected preset now
selects `coordinate_batch_layout: per_image`; missing fields in historical saved
V22 configurations still mean `legacy`. No further conditioning, proposal,
matching, loss or sampler changes are included. Corrected training starts fresh
in its own namespace. The correction's learned AP effect is unmeasured at launch.

Correction verification: 35 focused tests passed, including B2/B4 independent-image
execution, other-image state/reference/timestep isolation under nonzero FiLM,
legacy-config restoration, and exact B1/K3 sampling parity. Full-width real-data
CUDA training completed two optimizer updates, with finite coordinate-conditioner
and final-H gradients, strict checkpoint restoration, finite BF16 sampling and
the native quick-test hook at both smoke epochs. Parameter count stays44,603,401.
The prior run stopped gracefully with epoch002/last preserved. Fresh corrected
run `20261021_153657` uses `coordinate_batch_layout: per_image`, original source
initialization,80epochs and `resume=null`; remaining recipe unchanged.

### V23: one supervised call with optional detached warm-up

V23 tests whether initial layers can learn to read actual carried H. Two iid
uniform times are ordered and the batch randomly supervises either the first or
second with equal probability. First-step batches use one ordinary training call.
Second-step batches use a no-grad/eval warm-up, carry detached H and the native
advanced state, and supervise only the second call. The selected time remains
uniform marginally. One gradient-bearing call/criterion per batch preserves the
supervision budget; extra warm-up compute means runtime is not matched.
Architecture, batch-layout fix, FiLM/AdaLN, H-derived proposals, OTA, DN, summed
within-call losses, optimizer recipe and native held-out evaluator stay unchanged.

```bash
venv/bin/python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v23
```

Warm-ups start from the original interpolated flow training states and receive
no targets/DN queries. This is not a pure-Gaussian inference rollout and does not
supply temporal gradients to the warm-up. Full temporal credit assignment would
be a separate experiment. Training RNG state is checkpointed; validation stays
single-call and held-out quick tests use the native persistent/reset sampler.

V23 verification: 184 focused/base checks passed, including both supervision
branches, one criterion call per batch, no warm-up gradients/targets, exact source
transport, uniform-time statistics, strict schedule-RNG continuation and unchanged
validation. Full-width real-data CUDA training exercised both branches across two
optimizer updates; final-H/coordinate gradients, strict K1/K3 restoration, BF16
and two epochs of native quick-test hook passed. Parameters remain44,603,401.
Fresh80epoch run `20261021_155200` starts from original source initialization with
`resume=null`; the only recipe changes are training pipeline and schedule seed.
The previous batch-fixed V22 run was stopped before its first completed epoch;
its logs/config/source are retained but it had no epoch checkpoint. Dashboard
current entry is V23-random-step alongside V5 and V19-detached-geometry.
