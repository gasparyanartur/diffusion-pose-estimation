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

### V24: matched single-call h addition (2026-10-22)

The one-step audit found V23 epoch079 full-test AP51.926 versus historical V1
55.105 and V5 55.948 under matched evaluation. Those models change proposal
initialization, candidate counts, conditioning and training together; removing h
from V23 only measures reliance and does not isolate adding h.

V24 therefore trains a fresh pair with the V1 spatial initializer and geometry
path preserved. `benchmark_poseflow_v24_control` omits h;
`benchmark_poseflow_v24_h` adds only a zero-gated image-conditioned residual h
branch to the two pose stages. Both use corrected coordinate batching, identical
source/shared initialization, data and single-call optimizer/loss/sampling recipe.
Seeds42/43/44,80epochs per arm; final epoch079 full-test8000 K1 AP is primary.
The h arm has extra compute. Exact initial parity is a wiring requirement, not
learned-performance evidence. Frozen source, paired initialization fingerprints,
queue status and full-test evaluations live under the experiment result root
`results/poseflow_v24_matched/`; existing experiment artifacts remain retained.

V24 verification/launch:96 focused/base checks passed, plus full-width real-data
CUDA two-update checks for both arms, strict evaluator restoration, BF16 finite
sampling and two-epoch native monitoring. Shared initialization and RNG match for
all three seed pairs. The first two microbatch losses agree exactly; after the
first update h gates open and its image/writer/reader parameters receive gradients.
Control/h parameter counts are40,511,752/42,353,674. These are wiring checks.
The frozen six-run experiment is `results/poseflow_v24_matched/20261022_192822`;
both seed42 trainers advanced to optimizer99 with finite losses. Seeds43/44 and
final full-test evaluations are queued. Learned AP results remain pending.

V24 budget correction (2026-10-22): the user approved retaining only the current
seed42 control/h pair. Four unstarted seed43/44 repeats were cancelled. Both
original trainer PIDs and optimizer progress were preserved; at verification both
were at optimizer7349, epoch index4. Replacement single-run supervisors retain
the two final full-test evaluations and cannot launch additional training. The
initial three-seed manifest is archived; active protocol/manifest/comparison and
dashboard now describe one pair. Replication is deferred pending that result.

## Frozen image-information diagnostic (2026-10-23)

[Report: six image-information questions](reports/2026-10-24-image-information-report.md).
V24 control epoch063,64 annotation-selected held-out test images,12 paired image
conditions,K1 FP32. This is a localization diagnostic, not full-test AP. The same
63 people matched across all conditions. Masking wrist+elbow+shoulder increased
wrist error by7.20 percentage points of sqrt person-box area versus equal-area
wrist+distant-body masks (image-bootstrap95% interval4.09–10.89); an alternate
fill agreed. This demonstrates context dependence, while anatomical specificity,
person ownership and 3D remain unresolved. Final pose-stage mean joint error fell
6.80->5.99. Grayscale, outside-box masking and flip changed individual predictions
with uncertain net mean error changes. No new model training or production changes.

## Discriminating h/control suite specification (2026-10-23)

[Concrete paired-model protocol](reports/2026-10-24-image-information-report.md).
Same-age seed42 control/h epoch065 checkpoints are pinned and restore strictly.
The suite specifies native person/limb ownership and hidden-wrist ambiguity tasks,
matched-complete-2D depth readouts, initialization/stage controls and incremental
h readouts. A task-specific readout is required to investigate native failure;
depth-sign accessibility alone cannot explain a depth-magnitude task. Shared2D
controls exclude depth-derived visibility labels. Registered construction pilots
must pass before any larger evaluation. This entry records specification and
checkpoint validation only: no new challenge images, learned probe results or
main-suite results exist yet. No additional full model training is scheduled.

## DiffusionRegPose decoder and diffusion inspection (2026-10-23)

[Executed notebook](../notebooks/investigations/diffusionregpose_diffusion_inspection.ipynb)
uses the reduced 3E/3D OTA epoch079 checkpoint from
`diffusionregpose_development_compatible_256_keep_layers_ota/20260912_224837`,
32 randomly selected test images (subset seed42), resize256/427, native pose NMS,
and deterministic DDIM with common Gaussian proposals seeded1729+image_id.
Checkpoint/config/annotation/source hashes and ordered image IDs are recorded
under local `tmp/diffusionregpose_diffusion_inspection/`.

| Diffusion steps | 32-image keypoint AP | 32-image bbox AP |
| --- | --- | --- |
| 1 | 0.505718 | 0.477583 |
| 2 | 0.507381 | 0.477054 |
| 3 | 0.504237 | 0.478515 |

These are subset diagnostics from one checkpoint and proposal seed. The top
three image IDs are111057/115025/100537 and the bottom three100656/102109/104488
at every S, ranked by per-image CrowdPose keypoint AP (undefined scores excluded,
image-ID tie break). Native standalone aggregate evaluation agrees with the
notebook's S1 keypoint/bbox AP. All18 selected image/settings replays match the
untraced predictions exactly, covering36 model calls and108 decoder layers.

Reference points and predictions are plotted separately, with all three decoder
layers, actual query masks and sparse deformable image-attention samples/weights.
In the six selected images' traces, all25,200 incoming joint-reference entries
at the first pose layer are outside the image; the next layer's entries are
inside. These are repeated image/settings/call entries, not independent samples.
Expanded axes preserve the actual coordinates; this observation does not identify
the cause of the subset AP or justify a model change. Group IDs are local to a
call and do not establish person tracking across diffusion steps.

Validation:25 focused tests passed; notebook schema and executed cells verified,
with15 embedded figures and no error outputs. Static galleries/decoder sheets
are saved; live exploration requires a Jupyter kernel and Update view. No model
training or production model changes were made.

## 2026-10-23 — Information-suite A admission pilot

[Protocol](reports/2026-10-24-image-information-report.md) ·
[Pilot report](reports/2026-10-24-image-information-report.md)

Executed the first staged iteration against pinned V24 epoch065 control/h,
step99000, K1, seed42, resize256/427. Newly rendered synthetic controls only; no
CrowdPose validation/test evaluation. Initial construction failed visual validity;
the single allowed repair used fresh32-family physical-occluder controls.
It passes implemented scoped geometric/pixel audits and RGB discrimination63/64,
but fails85% native competence in both models. Matching-only reference-box audit
with original pixel tolerance gives occluded8/64 control,5/64 h; visible47/64,49/64.
The audit fixes scoring rejections, not localization failure, and does not
retroactively admit the benchmark. A512 main was not launched.

712 native calls including parity; seven focused implementation tests pass.
Original scores/predictions and failed construction history preserved. No model
training or production architecture changes. B/C pilots and T2–T6 results remain
pending; this is an admission result, not a finding about3D or relational reasoning.

## 2026-10-24 — Remaining information probes executing

[Live report](reports/2026-10-24-image-information-report.md). A stays rejected.
B/C32-family admission pilots pass native visible-wrist competence64/64 in both
pinned epoch065 models; family-held-out RGB controls B63/64 and C64/64. C raw2D
controls tie. B native hidden-wrist pilot0/64 each; task-specific frozen readouts
are needed to distinguish accessibility from native utilization. An outside-arm
pair-order shortcut warning remains explicit; no coherent3D reasoning conclusion.
Main B512 test(+256/64 reader train/val) and C1024/128/512 splits are now executing.
Ten focused tests, real h feature-hook parity and cached synthetic AP smoke pass.
No estimator training or production-model changes; results pending.

## 2026-10-24 — V24 training and remaining probes complete

[Completion review](reports/2026-10-24-image-information-report.md).
Final epoch079/step120000 seed42, full held-out8000 K1 pose AP: control54.9141,
h55.2949 (+0.3808points). Fixed32 monitor and same32 cached final predictions
still favor control; their ordering does not transfer to the full test. Validation
losses close but distinct; the2000 validation images overlap trainval training.
B/C probes finish on pinned epoch065. B hidden wrists0/1024 each model, visible
1023/1024 control and1024/1024 h. B primary token-reader coverage only156/512;
post-hoc all-family map audit retains512/512 and yields encoded-reader86.3%/96.9%
both-member classification. C is ceiling/specificity-limited: pretrained maps100%,
outside-arm pair ordering97.5%, shuffled-h augmentation ties100%. No true3D or
reasoning claim. Original scores and failed A construction retained. No new
estimator training or inference by this completion audit.


## 2026-10-24 — Consolidated image-information report

[Single current report](reports/2026-10-24-image-information-report.md) combines
attempt history, A admission failure, B native/readout results, C specificity and
ceiling controls, and final matched training. Earlier diary entries above remain
chronological snapshots; their pending statements are superseded by this report.
Six original reports are archived unchanged, with their former paths linking to
the consolidated report. No model changes or new experiments in this consolidation.

## 2026-10-24 — Full probe-report consolidation and notebook integration

[Consolidated investigation](reports/2026-10-24-image-information-report.md) is now
the single report for the23–24October native DiffusionRegPose inspection/state
probes and V24 information/h-control studies. It includes an outline, integrated
methods, result/contrast tables, interpretation, construction history and11
captioned figures, including the recent notebook's geometry and attention views.
Eight unique source reports, six archived duplicates and redirect/index files
were removed after integration. Earlier diary notes describing separate reports
or archives are historical and superseded by this consolidation. The notebook,
its saved code/outputs, raw numerical evidence and checkpoints remain preserved.
The ongoing full-test multistep comparison still updates the same report.

## 2026-10-24 — V26 joint clean-latent diffusion pilot

Approved experiment: V24-control image/data recipe, new dense-attention clean and
noisy stacks,50 indexed person proposals with repeated positives and foreign-pose
negatives. No EMA, matching, query expansion, deformable attention or layerwise
losses. Final task loss is averaged across clean/recovered branches plus latent
MSE1. Negatives receive existence/latent supervision only. Synthetic fallback
averages two poses plus sigma5-pixel noise; donor overlap is not filtered.

One seed42,80epoch pilot; K4 DDIM primary and K1 diagnostic. Final epoch079 full
held-out test8000 is the endpoint; fixed32 and overlapping validation remain
monitoring/fit diagnostics. V24-control is an architectural reference, not a matched
single-change ablation. CUDA optimizer smoke and regression checks are recorded
in the V26 launch artifacts; no learned-performance conclusion is available.

V26 completed endpoint and diagnostic update (2026-10-25): full8000 epoch079
K1/K4 pose AP0.63/7.33, box AP3.39/14.71. Additional held-out128 K1–64 sweep
peaks atK4 (8.46 poseAP); clean GT-conditioned fixed32 readout92.45 poseAP.
Image interventions show strong dependence, while high-noise denoising loses
slot diversity and generated coverage remains poor. These scopes are distinct.
See [V26 latent diagnostics](reports/2026-10-25-v26-latent-diagnostics.md) for
all measurements, limitations, raw-artifact paths and the conditional-averaging
hypothesis. This supersedes the pre-launch statement above that no learned
performance conclusion is available.

## V26 latent-only denoising ablation (2026-10-25)

Approved loss-only follow-up: retain0.5 clean task + latent MSE1 and remove the
recovered-latent task gradient. Clean encoder stays jointly trained. Fresh seed42,
80epochs with original architecture/data/optimizer/init and K4/K1 evaluation.
37 focused/regression tests and CUDA/Lightning smokes passed; training launched.
See [launch record](reports/poseflow_v26_latent_only.md). Shared TensorBoard label
`V26-latent-only` on33329. Layer/FiLM–AdaLN review and REPA-style DINO targets are
separate later steps, excluded from this experiment.

V26 latent-only failure audit at epoch011 (12 completed epochs): same-age
fixed32 K4 AP0.470 versus original4.493 despite lower reported latent MSE.
Clean GT readout43.30AP becomes19.98 after near-clean denoising; equal-MSE
random perturbations retain35.62. Long sampling restores latent variation
without accurate person coverage. See the [frozen-checkpoint failure audit](reports/2026-10-25-v26-latent-only-failure-audit.md)
for geometry/ranking controls, limitations and provenance. Training remains
unchanged; no architecture fix or new experiment was launched by this audit.

V26 latent-only near-clean layer trace: [pinned epoch011 report](reports/2026-10-25-v26-near-clean-layer-trace.md).
Exact native parity across fixed32/B4; the entire learned transformation degrades
near-clean readout43.43→19.98AP. Blocks and final projection use different internal
representations, and no single sublayer bypass restores input accuracy. Time
conditioning and same-person self-attention are active. No architecture change
was made; low-noise identity preservation is a candidate for a later ablation.

## V27 coarse-to-fine scale attention (2026-10-25)

Scale-only successor to V26 latent-only: each clean/denoiser block reads one
separate image feature level, stride32→16→8. No multiscale token concatenation,
image fusion neck or RT/RF-DETR modules. Retains0.5 clean-task + latent-MSE1.
Fresh seed42/80epoch pilot launched after44 checks and CUDA/Lightning smokes.
See [launch record](reports/poseflow_v27_coarse_to_fine.md) for initialization,
parameter/RNG differences, frozen provenance and shared-dashboard access.

## 2026-10-25: V28 clean-branch image stop-gradient

Fresh seed42/80epoch pilot from V26 latent-only, with only detached image memory/positions entering clean encoding. Exact full initialization/recipe parity;48 tests and CUDA/Lightning smokes passed. Root `/storage/outputs/pose-graph-diff/results/poseflow_v28_clean_image_stopgrad/20261025_192620`. Shared TensorBoard label `V28-clean-image-stopgrad`; fixed32 K1/K4 and final epoch079 full8000 K1/K4 retained. Prior V25/V26-latent-only/V27 stopped at user request with checkpoints/logs preserved. Pilot running; no learned-performance claim. [Launch report](reports/poseflow_v28_clean_image_stopgrad.md).

## 2026-10-25: V29 OTA person-latent targets

Parallel fresh seed42/80epoch pilot fromV28: clean correspondence unchanged; recovered queries use existing Dynamic-K OTA to uniqueGT clean latents, unmatched queries use nearest detached clean negative latents. Uniform latentMSE with0.5clean-task, no recovered-pose gradients.127regressions and CUDA/Lightning checks passed; full initialized model identical toV28. Root `/storage/outputs/pose-graph-diff/results/poseflow_v29_ota/20261025_201803`, shared dashboard label `V29-OTA`. V28 remains running; fixed32K1/K4 and final epoch079 full8000K1/K4 retained. Assignment hypothesis unproven, learned results pending. [Launch report](reports/poseflow_v29_ota.md).

## 2026-10-25: Four-image V28/V29 overfit capability checks

Two fresh matched diagnostics on trainval IDs113269/106326/111895/103487 (1/2/4/8people). No augmentation, fixed proposals, fresh latent noise/times,5000constant-LR updates. Evaluate clean, t0/499/999 and3seed GaussianK1/K4/K16 on these SAME training images with raw OKS50/75 coverage and native AP. Root `/storage/outputs/pose-graph-diff/diagnostics/latent_overfit4/20261025_224813`. Shared labels Overfit4-V28/Overfit4-V29; production jobs unchanged. Optimizer/strict-restore/RNG/parity smokes passed; memorization only. [Setup/provenance](reports/2026-10-25-latent-overfit4.md).

## 2026-10-26 — V30 decoder tolerance

On user approval, stop V29 and launch V30 from V28 fixed-correspondence training.
Recovered task supervision is restored only for shared heads by detaching hhat0
before readout; clean task0.5, recovered task0.5, latent MSE1. This tests whether
readout tolerance explains part of the gap between task-supervised and latent-only
models. See [V30 report](reports/poseflow_v30_decoder_tolerance.md). No learned
performance claim from initialization/optimizer checks.

## 2026-10-26 — V31 restores denoised task gradients

User approved returning to task supervision through the denoiser while keeping
clean h0 detached before corruption and as MSE target. V31 retains V28 clean-image
stopgrad and fixed correspondence, with clean/recovered task weights0.5/0.5 and
latent MSE1. V30 stopped with epoch025/last retained. This tests task gradients
into denoiser/image extraction relative to V30's heads-only supervision. See
[V31 report](reports/poseflow_v31_denoised_task.md).

## 2026-10-26 — V32 dual DINO and clean-image gradients

User requested DINO supervision on both h0/hhat0 and live clean-image gradients.
V32 adds a shared7x7x384 prediction head and frozen pinned V25 person-region
teacher, retains V31 task weights and both h0 detaches, and uses original V26
clean image conditioning. Added auxiliary loss is0.5*(clean_DINO+recovered_DINO),
weight1; only positive constructed slots are supervised. V31 remains the running
reference. See [V32 report](reports/poseflow_v32_dual_dino.md).

## 2026-10-27 — V33 structured task decoder

Approved readout experiment from V32:14joint+box+existence+global tokens per
person, three within-person self-attention/FFN blocks, final linear outputs.
Direct image-space joints avoid arithmetic box-error propagation. Diffusion,
DINO/gradient routing and losses remain V32. See
[V33 report](reports/poseflow_v33_token_readout.md). A null result would weaken
this decoder hypothesis, not establish the latent representation as sole cause.

## 2026-10-27 — V34 actual OTA task supervision

User approved the missing OTA+task-supervision experiment. V34 retains V33 and
optimizes recovered pose/box/existence against OTA, reusing that mapping for latent
and DINO targets. Clean correspondence unchanged; unmatched existence0 and
nearest-negative latent policy retained from V29. V33 remains active as control.
See [V34 report](reports/poseflow_v34_ota_task.md).

## 2026-10-27 — V33 clipping and latent-weight controls

User approved parallel v33-no-clip (latent weight1) and v33-no-clip-rescale
(weight10). Both retain V33 architecture and all other recipe/loss settings.
Eight focused checks and two real-teacher CUDA optimizer/strict-restore smokes
passed, clipping calls absent. Fresh seed42/80epoch launch uses shared33329TB.
See [experiment report](reports/poseflow_v33_no_clip_variants.md) for launch state.

## 2026-10-28 — V35 box-only OTA and spatial query pair

Approved two fresh runs with box inputs, direct box/existence heads, latent
weight10, clipping0.1 and no DINO/joints. Spatial adds one global Hungarian
proposal sort, denoiser-only Fourier queries and soft OTA cost. Fixed8 cost
calibration selected weight1 from native/spatial same-gate cost gaps (.124/.131),
without AP tuning. Native CUDA optimizer/strict restoration/box evaluator checks
passed. See [V35 report](reports/poseflow_v35_box_detectors.md) for run evidence.

## 2026-10-29 — V35 spatial latent-only control

User requested removing denoised decoder task supervision and clarified spatial
base. New control uses0.5clean task+10OTA-matched latentMSE; no recovered task
gradients, same anchors/sorting/spatialcost1/clip0.1. Fourteen focused checks pass.
See [report](reports/poseflow_v35_spatial_latent_only.md) for verification/launch.

### V36 clean box MLP controls (2026-10-29)

Approved pair: independent clean MLP4->256->256 versus MLP5->256->256 with binary
existence input. Both retain LayerNorm, spatial V35 latent-only denoiser/OTA,
latent10/clean.5/recovered0 and clip.1. No image/attention in clean encoding.
20 focused tests pass; native CUDA verification/launch evidence is recorded in
the V36 report. Compare clean reconstruction, conditional latent recovery and
Gaussian K1/K4 box AP separately; no performance conclusion from wiring checks.

### V36-box-normalized (2026-10-29)

User approved the clean coordinate-contract correction: use constructed normalized
box inputs directly, rather than pixels/100 without image dimensions. Everything
else matches V36-box, including no c input and no recovered task loss.23focused
tests pass; native CUDA preflight and fresh frozen launch evidence are recorded
in docs/reports/poseflow_v36_box_normalized.md. No AP gain assumed.

### V36-box-c-normalized (2026-10-29)

Approved follow-up to normalized box-only: apply normalization to V36-box-c,
retaining its clean existence input. Only format and namespace differ from
V36-box-c; same latent-only objective and recipe. Evidence is recorded in
`docs/reports/poseflow_v36_box_c_normalized.md`; no AP gain assumed.

### V35 spatial cost by noise timestep (2026-10-30)

Approved V35-spatial control with decoded task loss: extra anchor cost weight
max(0,1-t/500); original model/recipe/init retained.29focusedtests pass after
4intendedredfailures. NativeCUDA/launch evidence is recorded in
`docs/reports/poseflow_v35_spatial_timed_cost.md`. New object-only TensorBoard
includes V35-and-later box experiments, excludes overfit/failed attempts, and
pins fixed32quick-test bboxAP at1/4calls.

### V35 early versus late spatial cost (2026-10-31)

Approved exact reflection of the late ramp, with equal native discrete mean.2505
and maximum1. New early_linear schedule/config only; no V36/c changes.21focused
tests pass, including reflection, per-image matching dispatch and strictRNGrestore.
NativeCUDA and freshseed42 launch evidence:docs/reports/poseflow_v35_spatial_early_cost.md.

2026-10-31: launched fresh seed42 V35-spatial-early-cost-q200 (200queries,10x20grid), sameearlyramp/decodedloss/latent10/clip.1.21focusedchecks+nativeCUDApass; exactsharedparameterinitialization vs50control. Existingjobsretained. See [report](reports/poseflow_v35_spatial_early_cost_q200.md). NoAPresultyet.


2026-10-31: launched V37-latent-OTA, fresh seed42/80epochs, branching from
V35-spatial with unchanged model/50queries/5x10. All-padded-target latent OTA
(exp(-MSE) Dynamic-K, real-person coverage), clean sigmoid metric alignment
(position/size/existence1/1/4), objective.5clean+.5recovered+10latent+alignment1.
10 V37+35 legacy checks and nativeCUDA two updates/K1K4/strictrestore pass.
Run `/storage/outputs/pose-graph-diff/results/poseflow_v37_latent_ota/20261031_170101`. Existing33330 dashboard label V37-latent-OTA; final8000K1K4 scheduled.
See [report](reports/poseflow_v37_latent_ota.md). No AP result yet.


2026-10-31: launched V37-reweight, a fresh seed42/80epoch config-only control
of V37. Alignment1:1:1, latentMSE100; decoded task0.5/0.5, matching, model,
clipping0.1 and recipe unchanged.10 V37 checks plus nativeCUDA2updates,
strictrestore/RNG/initialization and isolated10xlatentgradient verification PASS.
Run `/storage/outputs/pose-graph-diff/results/poseflow_v37_reweight/20261031_232257`; existing33330 label V37-reweight, fixed32/final8000K1K4.
Original V37 continues; older cancelled runs remain stopped.
See [report](reports/poseflow_v37_reweight.md). No performance claim.

2026-11-01: launched V38 VAE clean-latent experiment, fresh seed42/80epochs,
based on V37-reweight. Replace alignment with posterior heads after clean
LayerNorm; sampled reconstruction, detached mean diffusion/matching, latent100
and all-slot prior KL0.1. Geometry stays real-only, existence all-slot. Separate
checkpointed posterior RNG1042; Gaussian inference unchanged.19 focused V38/V37
checks and nativeCUDA batch4accum2 two finite updates/K1K4/strictrestore/RNG pass.
Run `/storage/outputs/pose-graph-diff/results/poseflow_v38_vae/20261101_001343`.
Live160updates, coverage1,62scalar tags on existing33330,13runs/history/APpins
retained. Original V37/reweight active; cancelled early/timed/q200 remain stopped.
Source/config/input hashes frozen, fixed32 per epoch and final epoch79 full8000
K1/K4 configured. Compare matched-age AP/coverage/recovered geometry; no AP result
or learned-performance claim yet. See [report](reports/poseflow_v38_vae.md).


2026-11-01: fixed-source one-to-one V35-OTA branch approved for implementation
and fresh seed42/80epoch training. Exactly one real slot/person; fake slots
receive classification only; permanent source targets replace OTA/Hungarian.
V35 model, image initialization, optimizer/data/FP32/clip0.1 and K1/K4 sampler
retained. Objective0.5clean+0.5recovered+10positive-onlylatentMSE.17 focused
checks and nativeCUDA two-update/strictrestore/RNG/init parity verification
passed. V37-reweight(epoch55) and V38(epoch53) cancelled at user's request;
existing checkpoints/logs and dashboard history preserved. Original V37 was
already complete. See [report](reports/poseflow_v39.md). Launch
provenance and live verification follow there; no performance claim.

V39 fresh run launched at `/storage/outputs/pose-graph-diff/results/poseflow_v39/20261101_193746`; exact frozen initialization/source/config/input manifests, native preflight and progress retained. Shared33330 label V39,14configured entries with historical AP pins retained. FirstepochAP pending.


2026-11-01: V39-denoiser-grad approved and implemented as a V39 subversion.
Only `detach_corruption_source=false` changes training behavior; latent targets
remain detached and clean/recovered decoded tasks keep their original weights.
Four intended-red checks then40 focused V26/V35/V39/gradient-boundary checks
passed. Native FP32 CUDA batch4/accum2 two finite updates, K1/K4, strict
model/pipeline/optimizer/RNG restore, exact original V39 initialization and
identical training draws verified. Isolated fixed-forward clean-encoder
gradients from both recovered objectives are zero with source detach and
nonzero with live corruption. Fresh seed42/80epoch launch alongside V39 follows
verification; see [report](reports/poseflow_v39_denoiser_grad.md).

V39-denoiser-grad freshseed42/80epoch run launched: `/storage/outputs/pose-graph-diff/results/poseflow_v39_denoiser_grad/20261101_212530`; V39 continues. Frozen source/config/input/initialization and isolated gradient evidence retained, with epoch12 V39 reference pinned for later matched-age diagnostics. Shared33330 label V39-denoiser-grad, existing history/AP pins retained; firstepochAP pending.


### V40 — encoder reference proposals before latent corruption

New V35-OTA-derived box experiment: ED-Pose-initialized deformable encoder and
proposal heads, top200 Hungarian assignment, matched-preserving50 references,
live per-slot reference conditioning, fixed correspondence and positive-only
latent/box supervision. Denoiser keeps its box/existence output and full-image
attention. Fresh seed42/80epoch launch follows verification; no AP claim yet.
See `docs/reports/poseflow_v40.md`.

V40 verification/launch:31focused checks and native CUDA preflight pass; fresh
seed42/80epoch run `/storage/outputs/pose-graph-diff/results/poseflow_v40/20261101_235355` is active. See the V40 report for provenance;
first-epoch Gaussian AP is pending.


### V40-EMA — slow full clean-code targets

Approved fresh seed42/80epoch V40 comparison with EMA0.999 applied after optimizer
updates to the complete image-conditioned clean encoder. Teacher h0 supplies both
corruption and latent targets; existing task weights/proposals/sampling stay intact.
Full teacher/optimizer/RNG restoration and online-only inference are verified before
launch. SigReg deferred; lower latent loss alone is not a successful result.
See `docs/reports/poseflow_v40_ema.md`.

V40-EMA verification/launch:37focused tests and native CUDA/teacher cadence/strict
restore checks pass. Freshseed42/80epoch run `/storage/outputs/pose-graph-diff/results/poseflow_v40_ema/20261102_112758` is active; teacher updates
match optimizer steps. See V40-EMA report. FirstepochAP pending; SigReg deferred.


### V40-SigReg — live clean-code gradients and Gaussian regularization

Approved one freshseed42/80epoch run, no separate long-running control. Replace
EMA with one online clean encoder, remove final clean LayerNorm, leave h0 live
through corruption and preserve detached matched MSE targets. Add0.05SIGReg
on actual matched clean codes, using pinned author weighted characteristic
function statistic. Eight intended-red then45focused tests pass; native preflight
and launch verified below. Compare K1/K4AP and fixed held-out clean-code drift, not
regularization loss alone. Existing EMA comparisons combine three changes.
See `docs/reports/poseflow_v40_sigreg.md`.

V40-SigReg launch verified: `/storage/outputs/pose-graph-diff/results/poseflow_v40_sigreg/20261103_123752`, fresh seed42/80epochs, 380 finite optimizer updates at verification;45focused checks and native CUDA/restore/probe checks pass.607source/doc+5input hashes verified; existing dashboard history/pins preserved. AP pending; see the V40-SigReg report.

2026-11-03 V40-SigReg learned outcome FAILED: fixed32 K1/K4 AP0 and IoU0.5 coverage0 at epochs1/2; fixed held-out clean RMS .849->14.357->246.464->2136.947; run stopped duringepoch3 on nonfinite gradient-norm check(lastlogged4940). Predictions cover correct32images/1600outputs; frozen source matches current. SIGReg weighted penalty remains order1 while MSE explodes. Frozen target-gradient counterfactual removes outward latent component at high noise, but parameter-Jacobian check does not prove target detach alone caused AdamW scale growth. Production training rules unchanged; report docs/reports/poseflow_v40_sigreg_failure.md; propose SigReg-only live MSE target and bounded2epoch pilot, awaiting review.


## V40-SigReg live-target correction (two-epoch pilot)

The corrected preset explicitly sets detach_latent_target=false, restoring
latent-regression gradients through the online clean target as well as the
corruption source. Missing flags default true for historical checkpoint/config
restoration. All other families retain detached targets. Objective values,
normalization, SIGReg weight and sampling are unchanged at fixed weights.
See docs/reports/poseflow_v40_sigreg_live_target.md for tests and the approved
bounded2epoch pilot; no automatic80epoch launch.


2026-11-03 approved live-target pilot completed: fresh seed 42, two epochs,
3,000 optimizer updates, with the original 80-epoch scheduler horizon. No scale
or numerical guard triggered. Fixed32 bbox K1/K4 AP: epoch 0 = 8.35/5.98,
epoch 1 = 13.15/5.34. Held-out 17-person clean RMS = 0.405/0.492 (original
14.357/246.464); final mean coordinate variance 0.1601. Epoch-1 raw IoU >= 0.5
coverage = 101/133 (K1), 104/133 (K4). Final checkpoint/optimizer finite and at
step 3000, hashes intact, trainer exited. This avoids the observed early
scale/zero-AP failure over two epochs; K4 still trails K1. Pilot artifacts:
`tmp/v40_sigreg/live_target/pilot_20261103_144558`; see
`docs/reports/poseflow_v40_sigreg_live_target.md`. No further training launched.


2026-11-03 15:22 Europe/Stockholm: user authorized the fresh seed42/80epoch live-target
SigReg run after pilot review. Launched `/storage/outputs/pose-graph-diff/results/poseflow_v40_sigreg/20261103_151941` with frozen source/configuration,
unchanged pilot objective/data/optimizer, clean-scale/numerical guard and fixed32
K1/K4 each epoch plus full8000 afterepoch79. Startup finite at step290;
609 source + 5 input hashes verified, 20 dashboard streams preserve history/pins.
This is launch health; learned results are pending. See
`docs/reports/poseflow_v40_sigreg_live_target_run.md`.


## V41 coordinate box diffusion with retained proposals

V41 keeps V40's complete ED-Pose-initialized proposal front end (top200 candidates,
top50 scored reference boxes/features), with live references fixed across reverse
steps. Both train and inference select by score. Proposal Hungarian supervision
is separate from final predicted-box Hungarian matching; the latter chooses the
box and live clean-embedding target together. Shared clean/noisy coordinate
encoding uses three full-image blocks; the reference/timestep-conditioned
predictor has three blocks. State is normalized cxcywh mapped by z=6*(2*b-1),
with uniform t0..999 coordinate Gaussian corruption and coordinate DDIM eta0.
Embeddings are encoded anew each step, not noised or carried. Objective is
proposal +.5clean +.5denoised +10matched-JEPA +.05unique-clean SIGReg. All-slot
classification, matched geometry/JEPA and existing SIGReg RNG/masks remain.
Use `benchmark_poseflow_v41`; see `docs/reports/poseflow_v41.md` for interfaces,
verification and launch provenance. Previous paths/experiments remain intact.


2026-11-03 18:18 Europe/Stockholm: V41 implementation and authorized fresh seed42/80epoch
launch verified at `/storage/outputs/pose-graph-diff/results/poseflow_v41/20261103_181429`. Proposals retained (200candidates/50score-selected
references), coordinate Gaussian state, shared-view encoder, output Hungarian
and live JEPA/SIGReg. 65distinct focused checks and native FP32CUDA strict
restore/global RNG/next-update/K1K4 verification pass;617source+5input hashes
verified. Live step420 finite, cleanRMS0.2945;21dashboard streams
preserve history/pins. Fixed32K1K4 each epoch and full8000 afterepoch79 configured.
First evaluated AP pending; see `docs/reports/poseflow_v41.md`.


## V42 whole-pose coordinate-diffusion baseline

V42 preserves the original DiffusionRegPose coordinate corruption/DDIM, image
proposals, scored person selection, box refinement/detach and original
Hungarian task/auxiliary/proposal/DN losses. At the former expansion boundary,
a whole-pose MLP encodes noisy XY into one person vector. Two pose-refinement
layers use current box references and a learned absolute full-pose reader,
without joint query expansion. Each person gets60 sampling points per head/level
in place of15tokens x4; DN retains4. The native timestep-zero training rule is
unchanged. JEPA remains deferred. Use `benchmark_poseflow_v42` (3E/3D EDPose2/4/6,
256/427,200/50queries,DN50,FP32,batch4accum2,seed42/80epochs).
Fixed32 test-subset pose/box AP runs after each epoch at K1/K4/K8 with paired
Gaussian sources; full8000 afterepoch79 is configured by the launch wrapper.
See `docs/reports/poseflow_v42.md` for contracts, transfer and verification.

2026-11-03 20:35 Europe/Stockholm: V42 fresh80epoch launch verified at `/storage/outputs/pose-graph-diff/results/poseflow_v42/20261103_203019`; live step300, finite. 102focused checks/nativeCUDA restore+nextupdate passed; six K1/K4/K8 pose/box pins verified in dedicated comparison dashboard. Initialization measurements are separate from trained AP, which is pending. See V42 report.


V42 approved repair: the current preset explicitly uses
`pose_readout_mode: box_center_residual`, decoding full-pose residuals against
the incoming current box center. Old configs default to `absolute` for correct
historical restore. No joint expansion or loss/noise change. Initial assignment
quality improves with identical weights; learned benefit is being tested in a
bounded two-epoch run with an80-epoch scheduler. Longer training requires
outcome review. See `docs/reports/poseflow_v42.md`.

2026-11-03 22:09 Europe/Stockholm: V42 box-center reader pilot completedtwoepochs/3000updates andstopped. Onsame-agefixed32/seed42/256427, K1poseAP0.013->20.094 andboxAP0.490->26.120; K4/K8poseAP16.453/15.229. 18focused/nativeCUDA/restore checks pass,624source+5inputhashesverified. This is subsetevidence; morecalls currentlydegradeAP, finalfulltestunmeasured, JEPAdeferred. Pilotstoppedpendingreviewbeforelongertraining; seeV42report forcomparison/checkpoints andremainingstagegap.


2026-11-04T20:40+01:00: V42 paired-proposal full80 RUNNING, PID758206, tmux poseflow-v42-paired-proposal, root /storage/outputs/pose-graph-diff/results/poseflow_v42_paired_proposal/20261104_203731. Live epoch0/step100, finite loss/gradients;2751 paired positive sources checked with zero source/target mismatch,400 actual timesteps spanning5..995/all10bins. Fresh initial weights match control;129tests/nativeCUDA/strictrestore/nextupdate checks PASS.638source+5input hashes verified. TensorBoard0.0.0.0:33332 adds distinct stream, preserves43previousroots/sixpins; indexing ongoing. Full val/fixed32K1/K4/K8 each epoch and final8000 afterepoch79; no pilot cutoff. Prior controls preserved. Learned state-use/AP effect pending. Report docs/reports/poseflow_v42_paired_proposal.md.


### V46 ordered person generation (2026-11-08)

Approved:512-column canonical GT order,64 independent parallel denoising rows,
50-person capacity and EOS; sequential generated-prefix re-noising at inference.
Fresh dense decoder over original ED-Pose image encoder, frozen High-KL codec,
equal birth/refinement task weights, no Hungarian/proposals. Full80 TRAIN/VAL
run follows native full-capacity verification. Implementation/resource/launch
evidence and outcomes: [V46 report](reports/poseflow_v46.md). No learned result yet.

V46 launch verified: `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v46/20261108_181906`.50 checks/full-capacity native CUDA/strict restore pass; live64-row AP cards and loss scalars verified. Fresh80epoch run ongoing; baseline AP0/0 is untrained. See report for measured memory/throughput and provenance.

### V47 ordered coordinate diffusion (2026-11-09)

Approved isolated coordinate-state/shared learned-codec variant of V46. Decoded2D MSE+geometry/EOS, coordinateDDIM eta0,64rows/capacity50, fresh EDPose frontend transfer and fresh codec/decoder. Four-TRAIN-image learnability gate blocks full80 if free-running counts/AP fail. [V47 report](reports/poseflow_v47.md) tracks verification, AP, refinement and cost separately; gate passed at500updates (four-image TRAIN pose98.80/bbox91.73); fresh full80 launched. Learned VAL outcomes pending.

2026-11-09: V47 full80 launch verified at `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v47/20261109_150341`, trainerPID3490732.55 checks, native full64/50people FP32/batch4accum2/restore and two-seed four-image learnability gate passed. Fresh initial hash matches verification; same fixed32VAL, separate TensorBoard stream, existing jobs/artifacts preserved. See V47 report for mixed refinement behavior and measured cost; fullVAL outcomes pending.

### V48: clean-prefix person births until EOS (2026-11-10)

Approved V47-coordinate-codec arm removes refinement and all prefix noise/time
conditioning. Parallel P+1 teacher-forced rows supervise only births/EOS; image-only
sampling fixes previous births and stops at EOS/capacity50. Effective birth
weights10/.5 replace the former task average.74 tests pass; native full-capacity
FP32 preflight/restore/evaluator verification and fresh full80 launch pending.
See [report](reports/poseflow_v48.md). No quality claim; compare matched-exposure
free-running AP, birth errors and measured cost separately. Existing jobs/TEST
preserved.

2026-11-10: V48 native FP32/capacity50/batch4accum2 passed (peak1.75GiB; exact resumed next loss/max state delta2.24e-8). Fresh full80 active at `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v48/20261110_095254`,PID4145233.560 archived source hashes/initial hash match verification;330 retained frontend tensors exact. TensorBoard V48-ordered-birth-only/. AP cards pinned,67 historical roots preserved. Learned outcomes pending.

### V48-AdaLN parallel image-attention conditioning (2026-11-10)

Approved second V48 run uses prefix-count AdaLN/gating only on image cross-attention.
Identity initialization matches common baseline weights/outputs/RNG; same clean
birth-only full80 recipe, no trained-checkpoint warmstart.81 tests pass; native
verification/parallel launch pending. Existing baseline preserved. Compare
matched-exposure AP and cost separately; see [report](reports/poseflow_v48_adaln.md).

2026-11-10: V48-AdaLN native preflight passed (FP32/50people/batch4accum2,peak1.79GiB,exact resumed loss/max delta1.49e-8). Fresh full80 active at `poseflow_v48_adaln/20261110_135143`,PID73156 beside baselinePID4145233. Common initial V48 weights exact;566 source hashes pinned. Same fixed32VAL, separate TensorBoard overlay, learned outcomes pending.


### V49 token-age diffusion forcing (2026-11-10)

Complete50-row/49birth randomized-source coordinate diffusion-forcing arm: per-token
noise conditioning, aggressive1->.1->.05 schedule, all-active supervision and
DDIM with EOS refinement tails. Fresh originalEDPose/full80 TRAIN/VAL recipe;
67focused/historical checks pass. Native preflight/launch results are recorded
in [the report](reports/poseflow_v49.md). No learned-quality claim from wiring.

### V49-balanced combined objective experiment (2026-11-11)

Approved fresh V49 variant keeps the model/sampler and combines robust geometry,
fixed TRAIN-gradient calibration, equal refinement-age buckets, box-first/
pose-later priorities and separately normalized firstEOS supervision. Model
initialization/RNG parity and42 focused/historical checks pass. Native calibration,
capacity/restore verification and full80 launch are next. Original V49 and other
jobs/artifacts preserved; this combined experiment cannot isolate individual
changes. See [exact objective and provenance](reports/poseflow_v49_balanced.md).

2026-11-11: V49-balanced full80 LAUNCH VERIFIED at`/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v49_balanced/20261111_012830`,PID445691.70 tests/native full-capacity FP32/32TRAIN calibration and strict resume pass; initial hash exactly matches V49,peak1.80GiB. Live epoch0/step120; same fixed32VAL and initial0AP. TensorBoard run`V49-balanced-token-age/.` live; existing jobs/artifacts preserved. Learned AP/refinement/count behavior pending.

2026-11-11: Original V49 and V49-balanced cancelled by explicit user request. Best/last checkpoint model/optimizer/scheduler metadata and SHA hashes verified; last safe steps82500/55000, artifacts preserved. Balanced variant improved count/box diagnostics but pose AP/refinement remained poor. No full80/all2000VAL result claimed; proposal-attention/residual successor is pending design review.

2026-11-11 — V50 approved successor to cancelled V49 experiments. Retains the rebalanced objective; adds proposal-attention identity selection and sigma-gated residual refinement. Native frontend/assignment parity and model checks precede a fresh full80 seed42 run. [V50 contract and outcomes](reports/poseflow_v50.md). No learned-performance claim from implementation checks.

2026-11-11 22:03: V50 full80 seed42 training active at`/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v50/20261111_215450` after79 checks/native capacity/restore/evaluator verification. Same fixed32VAL initial AP0/0; learned outcomes pending. Historical runs/checkpoints preserved.
