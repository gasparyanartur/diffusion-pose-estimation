# pose-graph-diff

Research code for image-conditioned, multi-person 2D human pose estimation on
[CrowdPose](https://github.com/Jeff-sjtu/CrowdPose). The repository explores graph
diffusion, coordinate flow, and latent diffusion with persistent hypothesis
memory. It includes a native DiffusionRegPose implementation, fixed-slot pose
baselines, and separate box-detection pipelines.

This is an experimental research repository. Model versions use different state
representations and checkpoint contracts. The [experiment diary](docs/experiments.md)
contains historical recipes, ablations, and investigation notes; the
[architecture guide](docs/architecture.md) describes the model internals.

## Models

| Family | Purpose | Entry point or configuration |
| --- | --- | --- |
| Fixed-slot baseline | Direct image-to-pose set prediction | `scripts/training/train_pose_refinement.py` |
| Graph pose | Joint box/keypoint diffusion with graph refinement | `scripts/training/train_graph_pipeline.py` |
| DiffusionRegPose | Native implementation with reference-checkpoint evaluation | `scripts/training/train_diffusionregpose.py` |
| PoseFlow | Coordinate-flow and latent-diffusion experiments | `scripts/training/train_poseflow.py`, `conf/benchmark_poseflow*.yaml` |
| Box detection | Separate box/objectness models | `scripts/training/train_box_diffusion.py` |
| Pose-and-box VAE | Image-free per-person geometry codec | `scripts/training/train_pose_vae.py` |

Recent PoseFlow experiments include V15 rollout training, V17 candidate-tied
hypothesis memory, V18 pre-image representation refinement, and V19 coordinate
refinement with tied memory. These are distinct experiments, without an implied
performance ranking. See the [diary](docs/experiments.md) for their commands and
[project conventions](docs/project.md) for coordinate and evaluation contracts.

## Image-free pose-and-box VAE

Train the 574,560-parameter geometry VAE with annotation JSONs alone:

```bash
venv/bin/python scripts/training/train_pose_vae.py
```

The default recipe uses CrowdPose **train only**, disjoint validation, seed 42,
batch 512 and 50 epochs. It reconstructs clean joints and independent boxes from
noisy/masked inputs, with a 32-dimensional Gaussian latent and KL warm-up. Image
dimensions are metadata; images and pretrained checkpoints are unnecessary.
The best validation checkpoint is evaluated on test once after training.

For a short wiring check without test scoring:

```bash
venv/bin/python scripts/training/train_pose_vae.py data.subset_frac=0.03 \
  training.max_epochs=2 training.test_after_fit=false
```

Run directories under `results/pose_vae/` include best/last checkpoints, resolved
configuration, source/annotation hashes, TensorBoard/CSV metrics, and geometry
grids. See [the codec report](docs/reports/pose_vae.md) for normalization, inference,
validation metrics, and the initial run. This codec is separate from PoseFlow.

The fidelity-first successor uses a deterministic codec and native AP gates:

```bash
venv/bin/python scripts/training/train_pose_autoencoder.py
```

It trains clean geometry for 100 epochs and selects the checkpoint maximizing
the lower of pose and box AP. Decoder-only latent-noise training runs for 50
epochs **only if both clean validation AP scores reach 97.5**. It freezes the
encoder and uses training-only latent statistics. This development command never
loads or scores test annotations. See [the autoencoder report](docs/reports/pose_autoencoder.md)
for the AP protocol, noise curves, acceptance policy and results. A wiring smoke
uses `data.subset_frac=0.03 training.max_epochs=2 run_robustness=false`.

The joint VAE follow-up starts from an accepted AE checkpoint and compares
posterior sampling, calibrated KL and joint masking:

```bash
PYTHONPATH=. venv/bin/python scripts/training/train_pose_vae_fidelity.py \
  initial_checkpoint=/absolute/path/to/accepted_ae.ckpt
```

See the [warm-started VAE report](docs/reports/pose_vae_fidelity.md) for the
eight-arm matrix, mean/sample fidelity and missing-joint/noise evaluations.

## Environment

The development stack uses Python 3.10, PyTorch 2.4.1 with CUDA 12.1,
Torchvision 0.19.1, MMCV 2.2.0, MMEngine 0.10.7, MMPose 1.3.2, and
MMDetection 3.3.0. CUDA extension builds require a complete toolkit with `nvcc`
and a compatible NVIDIA driver. The shared environment also uses the CrowdPose
API and Detectron2. Dependency snapshots are in [requirements.txt](requirements.txt)
and [requirements.lock.txt](requirements.lock.txt).

**A fresh clone needs additional setup.** The `external/`, `data/`, `checkpoints/`,
and `results/` directories are excluded from Git. Existing experiments assume
local external source trees, datasets, and pretrained weights.

The [shared environment script](scripts/utils/setup_shared_env.sh) documents the
development installation procedure, but currently resolves its root to `scripts/`
rather than the repository root. It also expects local `external/mmpose`,
`external/mmdet`, and `external/DiffusionRegPose` trees, including a local
MMDetection compatibility patch for MMCV 2.2.0. It is therefore not a working
one-command installer for a fresh public clone. The external patch and exact
source revisions must be supplied to reproduce that environment; installing the
requirements file alone does not provision those components.

Use an isolated environment. After the dependencies are available, install this
project from the repository root and run commands there:

```bash
python -m pip install --no-deps -e .
```

The commands below assume that environment is active. Optional DINO-based
experiments require their corresponding backbone weights as well.

## CrowdPose data and pretrained weights

Obtain images and annotations from the
[CrowdPose release](https://github.com/Jeff-sjtu/CrowdPose) under its applicable
terms. Place the images flat in `data/crowdpose/images/`. Training configurations
expect the following annotation layout:

```text
data/crowdpose/
├── images/
│   ├── 100000.jpg
│   └── ...
└── annotations/
    ├── crowdpose_train.json
    ├── crowdpose_val.json
    ├── crowdpose_trainval.json
    └── crowdpose_test.json
```

The [annotation helper](scripts/data/prepare_crowdpose.py) downloads the
OpenMMLab annotation archive and detector boxes:

```bash
PROJECT_DIR="$PWD" python -m scripts.data.prepare_crowdpose
```

The explicit `PROJECT_DIR` selects the repository root for this helper. The
archive uses `mmpose_crowdpose_{train,val,test}.json` filenames. Copy the split
files to the names above, and provide a combined train+validation annotation file
for configurations using `crowdpose_trainval.json`; the helper does not create
that merge. Alternatively, override `data.train_ann`, `data.val_ann`, and
`data.test_ann` to point to prepared files. Detector boxes are for top-down
baselines; the one-stage pose pipelines do not require prediction caches.

DiffusionRegPose and PoseFlow training presets initialize from an ED-Pose
checkpoint. The default path is
`external/DiffusionRegPose/pretrain_models/edpose_r50_crowdpose.pth`;
set `pipeline.pretrained_checkpoint` to use another location. Obtain pretrained
weights from their original authors and follow the applicable terms. Training
initialization weights and trained model-family checkpoints serve different roles.

## Training

Experiment-specific configurations and auxiliary objectives, including V19
evidence ranking and the gated dense hypothesis image read
(`benchmark_poseflow_v19_dense_read`), are documented in
[the experiment diary](docs/experiments.md).

V20 (`benchmark_poseflow_v20`) adds training-only GFPose guidance to the dense-read
model. Obtain the official U3D `best_model.pth` linked by the
[GFPose authors](https://github.com/Embracing/GFPose) and place it at
`checkpoints/gfpose/best_model.pth`, or set `pipeline.gfpose_checkpoint`.
The configuration pins its SHA256. V20 evaluation needs only its own student
checkpoint/configuration; it does not load GFPose. See the diary for the auxiliary
objective and its coordinate-transfer limitations.

Inspect a resolved configuration without starting training:

```bash
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v19 --cfg job
```

Train a selected experiment in its own output directory:

```bash
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v19 training.resume=null
```

For a small graph-pose wiring check after preparing the environment and data:

```bash
python -m scripts.training.train_graph_pipeline \
    data=crowdpose_set_prediction_debug \
    training.max_epochs=2 training.test_after_fit=false \
    hardware.accelerator=cpu hardware.devices=1 hardware.precision=32
```

This checks execution on a tiny subset; it does not measure benchmark quality.

Configuration uses [Hydra](https://hydra.cc/), with groups under `conf/`.
Override individual settings with `data.batch_size=4` or
`training.max_epochs=80`. Resume a compatible run with
`training.resume=path/to/run/checkpoints/last.ckpt` and its original configuration.
Outputs are controlled by `out_dir` and `run_name`; runs save the resolved config
in `.hydra/config.yaml` alongside checkpoints and logs.

## Evaluation

Evaluate a trained PoseFlow checkpoint with its saved run configuration:

```bash
python -m scripts.evaluation.evaluate_poseflow \
    --checkpoint path/to/run/checkpoints/last.ckpt \
    --config path/to/run/.hydra/config.yaml \
    --annotations data/crowdpose/annotations/crowdpose_test.json \
    --image-root data/crowdpose/images \
    --flow-sampling-steps 3 --flow-seed 42 \
    --output results/evaluation/predictions.json \
    --metrics results/evaluation/metrics.json
```

Use `--limit` for a subset diagnostic. Sampling-step semantics depend on the
version: earlier coordinate-flow models use Euler steps; latent-diffusion
families and V19 use DDIM calls. Keep the checkpoint and configuration paired.
The separate [DiffusionRegPose evaluator](scripts/evaluation/evaluate_diffusionregpose.py)
supports reference-checkpoint evaluation; use its `--help` for arguments.

For a frozen RTMPose missing-joint diagnostic using existing person boxes:

```bash
python -m scripts.evaluation.evaluate_joint_completion \
    --limit-images 256 --output results/joint_completion/my_pilot
```

The default input is CrowdPose VAL; `--limit-images 0` selects all VAL images.
The output directory must be new. Predictions preserve original annotations in
a separate cache and include per-joint metrics, independent flip consistency,
and review sheets. Missing joints have no reference-coordinate accuracy, and
the released CrowdPose teacher's VAL exposure makes these in-sample diagnostics.
See the [measured completion report](docs/reports/joint_completion.md).

The [DiffusionRegPose inspection notebook](notebooks/investigations/diffusionregpose_diffusion_inspection.ipynb)
compares 1/2/3 diffusion steps on 32 fixed test images with the reduced OTA
epoch-79 checkpoint. It includes best/worst galleries, each decoder layer's
references and predictions, query masks, and deformable image-attention samples.
Run it in the configured project environment with the saved checkpoint, resolved
config and CrowdPose data available. Its first cells expose these paths; later
runs reuse verified local caches under `tmp/diffusionregpose_diffusion_inspection/`.
Saved figures work in static viewers; image/call/group/head controls need a live
Jupyter kernel. These results are subset diagnostics.

CrowdPose metrics require `crowdposetools`. Native DiffusionRegPose/PoseFlow image
attention can use the vendored deformable-attention CUDA operator, built lazily
in PyTorch's extension cache. A differentiable PyTorch fallback is available;
record the execution path when comparing runtime.

When interpreting results:

- Several presets train on `trainval` while reporting validation on `val`.
  Those validation numbers measure in-sample fit; use the disjoint test split
  for generalization comparisons.
- Fixed32 quick tests and `--limit` evaluations are subset diagnostics.
  Quick-test noise fractions of 25% and 50% use GT-derived inputs; 100% starts
  from Gaussian noise.
- Compare metrics with the checkpoint, split/image coverage, resize policy,
  matcher, seed, sampling steps, and evaluator recorded. Smoke tests and
  GT-assisted diagnostics do not establish learned full-test AP.

## Repository guide

| Path | Contents |
| --- | --- |
| `conf/` | Hydra model, data, hardware, and training presets |
| `src/` | Data loading, models, training, and evaluation |
| `scripts/` | Training, evaluation, data, investigation, and utility entry points |
| `tests/` | Focused unit and regression tests |
| `docs/` | [Architecture](docs/architecture.md), [conventions](docs/project.md), and [experiment diary](docs/experiments.md) |
| `notebooks/`, `reports/` | Research investigations and recorded evidence |

Run focused tests from the configured environment with
`python -m pytest tests/<test_file>.py`. Follow [AGENTS.md](AGENTS.md) and
[STYLE.md](STYLE.md) when contributing changes.

## Acknowledgments and third-party code

This work builds on CrowdPose, ED-Pose, DiffusionRegPose, OpenMMLab, and DINOv3.
The vendored deformable-attention source retains its
[provenance and attribution](src/models/pipelines/pose_estimators/diffusionregpose/third_party/dinov3_msda/NOTICE.md)
and [DINOv3 license](src/models/pipelines/pose_estimators/diffusionregpose/third_party/dinov3_msda/DINOv3-LICENSE.md).
External code, datasets, and pretrained weights retain their respective terms.

V21 restores the reduced V1 detector recipe with 50 fixed hypothesis/candidate
pairs and hypothesis-derived initial boxes. Each decoder layer updates H first,
then geometry reads that updated H (`hypothesis_update_order: h_then_g`). It has
no 3D teacher. Train with
`venv/bin/python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v21`;
see the [experiment notes](docs/experiments.md#v21-v1-detector-with-fixed-slot-hypothesis-derived-proposals)
for its differences from V1 and evaluation limits.

V22 adds explicit timestep FiLM/AdaLN to the coordinate and H/geometry interaction
paths, retaining H-derived proposals and fixed candidate pairing. Train with
`venv/bin/python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v22`.
See the [experiment notes](docs/experiments.md#v22-explicit-timestep-conditioning-on-v21-h-first)
for the embedding-scale and gradient-boundary contract.
The current V22 preset uses `coordinate_batch_layout: per_image` to keep batched
coordinate attention isolated by image. Historical V22 configs without this field
restore their original layout.

V23 trains the corrected V22 architecture on a50/50 mixture of first-step loss and
second-step loss after a detached warm-up, with one supervised call per batch:
`venv/bin/python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v23`.

### Matched one-step h control (V24)

V24 tests adding image-conditioned h while preserving V1's proposal generator,
200-to-50 candidate selection and original geometry path. Both arms use corrected
per-image coordinate fusion and single-call training; the h branch starts as an
exact zero-gated residual. It only carries state between pose layers within a call.

```bash
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v24_control training.resume=null
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v24_h training.resume=null
```

The initial paired protocol uses training seed42 and80epochs. Keep
`training.seed` and `pipeline.deterministic_seed` equal in the two arms.
Additional seed repetitions are deferred until this pair warrants confirmation.
Compare final checkpoints with full CrowdPose test K1 AP under identical sampling
and evaluation; fixed32 charts are monitoring, not the experiment's final result.

### Matched h-to-DINO grid supervision (V25)

V25 adds a training auxiliary decoder from final person h to a7×7×384 DINO feature
grid. Targets come from a frozen DINOv3 ViT-S/16 image map at the pose-matched GT
box; existing pose assignments and pose losses are retained. Teacher inference is
training/loss-evaluation only, and ordinary pose sampling does not execute the
teacher or auxiliary decoder.

```bash
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v25_h_dino training.resume=null
```

The teacher snapshot must already be present in the Hugging Face cache configured
by `HF_HOME`. Model, revision and weights SHA256 are pinned in
`conf/pipeline/poseflow_v25.yaml`. Set `pipeline.h_dino_weight=0` to use the original
V24 loss path, or adjust the nonnegative weight (default1.0). V25 checkpoints
restore strictly with their saved configuration. A deliberate V24 h warm start
can use `model.load_v24_state(...)`; this imports model tensors and leaves only
the new decoder fresh, rather than resuming a V24 optimizer as V25.

### Joint person-latent diffusion pilot (V26)

V26 learns clean person representations from pixel joints and image features,
then trains a separate image-conditioned latent denoiser. It uses50 fixed proposals,
constructed positive/duplicate/negative labels and final-only direct supervision.
It has no query expansion, deformable attention, matching or cross-call memory.

```bash
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v26 training.resume=null
```

The loss-only ablation uses `--config-name benchmark_poseflow_v26_latent_only`.
It retains joint clean-encoder training and optimizes
`0.5 * clean_task + latent_MSE`; denoised task losses are monitoring-only.
The original V26 objective remains the default. The ablation has a separate
`poseflow_v26_latent_only` output namespace and starts from the same initialization.

The default pilot is FP32, batch4/accumulate2,80 epochs, seed42. Only compatible
image-extraction weights initialize from EDPose. Gaussian-start K4 DDIM is primary;
K1 is a diagnostic. Borrowed-pose negatives may accidentally match real people;
synthetic fallback noise defaults to5 resized pixels (`pipeline.fallback_noise_pixels`).
Implementation/smoke validation is not learned-performance evidence. See the V26
sections of `docs/project.md` and `docs/architecture.md` for the target/loss contract.

### OTA-supervised recovered predictions (V34)

```bash
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v34 training.resume=null
```

V34 retains V33's model and uses one Dynamic-K OTA assignment for recovered
pose/box/existence losses, clean-latent targets and DINO targets. Unlike V29,
the recovered task loss is optimized with weight0.5. Unmatched candidates receive
existence-zero supervision, no geometry/DINO loss, and nearest-clean-negative
latent targets. Clean supervision keeps its constructed correspondence. Both
clean-latent detach points remain; inference and postprocessing are unchanged.

### Box-only latent detector controls (V35)

```bash
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v35
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v35_spatial
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v35_spatial_latent_only
```

Both predict boxes/existence from box-conditioned clean latents, with OTA,
latent-MSE weight10 and clipping0.1. There are no joint/DINO losses or token
readout. The spatial variant adds global proposal sorting, denoiser-only fixed
Fourier query positions and spatial matching cost1. Box AP uses all50 queries
without pose NMS. See [V35 report](docs/reports/poseflow_v35_box_detectors.md).

The spatial latent-only control sets recovered box/existence task weight0;
clean task stays0.5 and matched latent MSE stays10. Recovered task values remain
monitoring-only. See [gradient-route report](docs/reports/poseflow_v35_spatial_latent_only.md).

### Structured person-token readout (V33)

```bash
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v33 training.resume=null
```

V33 replaces V32's task MLPs with17 tokens per person (14 joints, box, existence,
global), three within-person transformer blocks and final linear projections.
Joints remain direct image XY predictions, independent of predicted box coordinates.
Diffusion stays on50 person latents of width256; DINO supervision, loss weights,
image-gradient routing and clean-latent detaches remain as in V32. Readout depth
is configurable with `pipeline.pose_readout_layers` (default3).

V33 clipping ablations retain the same model and seed42/80-epoch recipe:

```bash
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v33_no_clip
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v33_no_clip_rescale
```

Both disable gradient clipping. The first keeps latent-MSE weight1; the second
uses weight10. Task/DINO weights and constructed correspondence stay unchanged.
See the [parallel-run report](docs/reports/poseflow_v33_no_clip_variants.md).

### DINO supervision on both latent branches (V32)

```bash
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v32 training.resume=null
```

V32 adds frozen DINOv3 person-region supervision to both clean and recovered
latents using one shared prediction head. It also restores clean-path gradients
into image extraction. Clean latents remain detached before corruption and as
latent-MSE targets. The added loss is
`0.5 * pipeline.dino_loss_weight * (DINO_clean + DINO_recovered)`, default weight1.
The pinned V25 DINOv3 ViT-S/16 snapshot must already be cached locally; the teacher
is excluded from checkpoints/optimizer and is unnecessary for inference.

### Denoised task supervision with detached clean targets (V31)

```bash
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v31 training.resume=null
```

V31 restores task gradients through the recovered readout, denoiser and live
image features. Clean latents remain detached before corruption and as latent-MSE
targets. The loss is `0.5 * clean_task + 0.5 * recovered_task + latent_MSE`.
It retains V28's clean-image stop-gradient, fixed correspondence and sampler.

### Decoder tolerance from recovered latents (V30)

```bash
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v30 training.resume=null
```

V30 adds task supervision to V28's shared readout on detached recovered latents:
`0.5 * clean_task + 0.5 * task(readout(stopgrad(recovered))) + latent_MSE`.
The recovered task loss trains only the readout heads; image extraction and the
denoiser retain their latent-MSE route. It keeps fixed correspondence, clean-image
stop-gradient, and the existing training/sampling recipe, with a fresh namespace.

### OTA person-latent targets (V29)

```bash
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v29 training.resume=null
```

V29 retains V28's model and latent-only objective. Existing Dynamic-K OTA assigns
recovered predictions to unique GT people; each matched slot targets that person's
detached original clean latent. Unmatched slots target the nearest clean negative
latent from the same constructed image batch. Clean reconstruction retains its
known correspondence. Denoised task losses are OTA-based monitoring only; no pose
loss gradients train the denoiser. This tests reassignment and background-target
policy together; it does not establish assignment as the cause of prior failures.

### Clean-branch image stop-gradient (V28)

```bash
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v28 training.resume=null
```

V28 branches from V26 latent-only. The clean latent encoder reads detached image
memory and positions; its task loss trains the clean encoder and readout without
updating image extraction. The denoiser reads live image features and trains image
extraction through latent recovery. Forward computation, four feature levels,
initialization, proposals, latent-only objective and Gaussian DDIM stay unchanged.
The shared image encoder can still change clean targets across updates.

### Coarse-to-fine person-latent attention (V27)

```bash
python -m scripts.training.train_poseflow --config-name benchmark_poseflow_v27 training.resume=null
```

Each clean/denoiser block reads one image scale, in stride32 →16 →8 order.
Image tokens are never concatenated across scales; no stride64 level or fusion
neck is added. V27 retains the latent-only objective, joint clean encoder,
50 proposals and K4/K1 DDIM evaluation. It is a separate checkpoint/config family
with fresh initialization and audited transfer of the first three image levels.

The shared experiment TensorBoard is available locally on
[port33329](http://127.0.0.1:33329/) and over
[Tailscale](http://lynx.tail481642.ts.net:33329/). Add new runs to this dashboard;
reuse `tmp/tensorboard_shared.sh` rather than starting a dashboard for each version.

V36 box-only latent controls use `benchmark_poseflow_v36_box` and
`benchmark_poseflow_v36_box_c`. Both retain the V35 spatial latent-only recipe,
but replace the clean transformer with an independent MLP + LayerNorm. The latter
adds the constructed existence label to clean box inputs only. Gaussian inference
requires only the image; neither variant supervises denoised task outputs.

`benchmark_poseflow_v36_box_normalized` runs the V36 box-only control with relative
box inputs directly (no /100). It retains the spatial latent-only objective and
fresh seed42 recipe; historical V36 configs keep their pixel input convention.

`benchmark_poseflow_v36_box_c_normalized` is the corresponding box+c control:
normalized geometry plus unscaled binary existence, only in clean encoding.

`benchmark_poseflow_v35_spatial_timed_cost` retains decoded box supervision and
ramps OTA's extra spatial-anchor weight from0 at t>=500 to1 at t0. Other V35/V36
configs retain constant spatial costs. This changes matching, not attention.

`benchmark_poseflow_v35_spatial_early_cost` mirrors the timed-cost ramp: strongest
spatial matching at high noise, zero at low noise, equal average weight to the
late ramp. It retains decoded supervision and the V35 spatial architecture.

`benchmark_poseflow_v35_spatial_early_cost_q200` runs the early-spatial V35 detector with200 queries on a10x20 grid. It retains decoded supervision, latent weight10 and clipping0.1. All200 predictions reach the unchanged bbox evaluator (maxDets100); there is no200-to50 query selection.


V37 (`conf/benchmark_poseflow_v37.yaml`) branches from V35-spatial and trains
with latent-distance OTA plus position/size/existence sigmoid alignment. It
retains the same inference model and bbox evaluator. See
[the V37 experiment report](docs/reports/poseflow_v37_latent_ota.md) for the
objective, verification and run provenance.

`conf/benchmark_poseflow_v37_reweight.yaml` is the V37 loss-weight control:
position/size/existence alignment1:1:1 and latentMSE100. See the
[V37 reweight report](docs/reports/poseflow_v37_reweight.md).

`conf/benchmark_poseflow_v38.yaml` runs V38 from the V37-reweight recipe:
a variational clean encoder replaces alignment. Sampled clean reconstruction,
detached posterior-mean diffusion/matching, latent MSE100 and prior KL0.1 are
trained jointly. Gaussian inference remains deterministic given its initial
noise. See the [V38 report](docs/reports/poseflow_v38_vae.md) for verification,
posterior RNG semantics and run provenance.


`conf/benchmark_poseflow_v39.yaml` branches from V35-OTA with exactly one
positive slot per person and permanent source pairing. OTA/Hungarian are absent.
Real slots receive box/latent reconstruction; fake slots receive classification
only. The V35 model, image attention and Gaussian DDIM sampler are unchanged.
See the [V39 report](docs/reports/poseflow_v39.md) for the
objective, diagnostics and run provenance.


`conf/benchmark_poseflow_v39_denoiser_grad.yaml` runs **V39-denoiser-grad**:
V39 with `detach_corruption_source=false`. Recovered decoded/latent losses can
train the clean encoder through the noisy input; the clean latent target stays
detached. Objective weights and inference are unchanged. See the
[V39-denoiser-grad report](docs/reports/poseflow_v39_denoiser_grad.md).


### V40 reference-conditioned box detector

`benchmark_poseflow_v40` adds ED-Pose-initialized encoder proposals to V35-OTA's
box latent model: top200 candidates, one Hungarian assignment before corruption,
and50 fixed reference-conditioned latent trajectories. The native decoder is not
used. Final boxes/existence come from the latent denoiser; geometry/latent losses
are positive-only and reference-conditioning gradients remain live.

Use `python scripts/training/train_poseflow.py --config-name benchmark_poseflow_v40`
for the standard training entry point. The recorded frozen launch additionally
runs held-out bbox-only fixed32 K1/K4 every epoch and full8000 K1/K4 at epoch79.
See [V40 details](docs/reports/poseflow_v40.md) for the exact objective, verification
and launch provenance.


### V40-EMA slow clean-code targets

`benchmark_poseflow_v40_ema` retains V40 inference and adds a training-only full
clean-code EMA teacher (decay0.999, updated after optimizer steps). Both corruption
and latent targets use the teacher; online clean/recovered detection losses keep
their original weights. Teacher and update count restore with the full pipeline;
model-only evaluation needs no teacher. SIGReg is a separate variant below.
See [V40-EMA details](docs/reports/poseflow_v40_ema.md).


### V40-SigReg single-encoder latent diffusion (experimental)

Train with `python scripts/training/train_poseflow.py --config-name benchmark_poseflow_v40_sigreg`.
This variant uses one clean encoder, live corruption and target gradients, and
matched-only SIGReg (weight 0.05), replacing the EMA teacher and final clean
LayerNorm. V40 proposals and inference remain unchanged.
No additional package or teacher checkpoint is required. The frozen launch adds
per-epoch fixed32 K1/K4 and full8000 afterepoch79.
See [V40-SigReg details](docs/reports/poseflow_v40_sigreg.md) for provenance,
normalization/configuration requirements and verification.

The original detached-target SigReg run diverged during epoch 3; epochs 1/2
have zero bbox AP. See the [failure diagnosis](docs/reports/poseflow_v40_sigreg_failure.md).

The current preset uses live latent targets. Historical configs without
`detach_latent_target` retain the original detached behavior. The two-epoch
[correction pilot](docs/reports/poseflow_v40_sigreg_live_target.md) completed
with finite training: fixed32 K1/K4 bbox AP 13.15/5.34 and held-out clean-code
RMS 0.492. The multistep AP gap remains. A fresh seed-42, 80-epoch
[corrected run](docs/reports/poseflow_v40_sigreg_live_target_run.md) later stopped
at its scale guard during epoch 13; artifacts are preserved.


### V41 coordinate box diffusion with JEPA/SIGReg

`python scripts/training/train_poseflow.py --config-name benchmark_poseflow_v41`

V41 retains V40's image-derived proposal boxes/features and predicts clean boxes
from Gaussian coordinate states, with output-based Hungarian matching and live
clean-embedding prediction targets. References stay fixed across DDIM steps;
features are re-encoded from coordinates each step. It uses 50 outputs and the
existing detection evaluator. See [V41 architecture and verification](docs/reports/poseflow_v41.md).

The fresh seed-42, 80-epoch V41 run is `/storage/outputs/pose-graph-diff/results/poseflow_v41/20261103_181429`; dashboard stream `V41`. Launch health is verified; learned AP is pending.


### V42 whole-pose baseline

`benchmark_poseflow_v42_uniform_t` is the corrected box-center baseline. It retains coordinate diffusion and
pose-estimation losses while encoding each noisy pose into one person vector.
Box-referenced deformable attention replaces joint query expansion, with a
learned full-pose reader. JEPA is deferred until baseline performance review.
The preset uses the existing 3E/3D 256-pixel control recipe; quick-test pose/box
AP is evaluated at1/4/8 sampling calls. See [the V42 report](docs/reports/poseflow_v42.md).

```bash
python scripts/training/train_diffusionregpose.py --config-name benchmark_poseflow_v42_uniform_t
```

Training samples an independent uniform timestep in `[0,999]` for each image,
then corrupts pose coordinates before encoding. Output-Hungarian supervision and
DDIM inference are retained. Historical `benchmark_poseflow_v42` and matched-target
runs used timestep zero only; they are not the requested diffusion-training
baseline. The failed matched-target run has been cancelled.

The V42 preset now uses a box-center-relative pose reader
(`pose_readout_mode: box_center_residual`). Historical configurations without
this field retain the absolute reader. The reader change preserves model state
keys; restore with the run's saved configuration. See the V42 report for the
bounded repair experiment and its evidence.


### V42 targets matched from noisy poses

```bash
python scripts/training/train_diffusionregpose.py --config-name benchmark_poseflow_v42_matched_target
```

This variant keeps repaired V42's model and inference, but assigns noisy input
poses to clean people once with Hungarian matching and reuses those targets for
all pose-refinement losses. It retains the original 80-epoch recipe and timestep-zero
training corruption. See [the experiment report](docs/reports/poseflow_v42_matched_target.md).

### V42 paired proposal sources and labels

```bash
python scripts/training/train_diffusionregpose.py --config-name benchmark_poseflow_v42_paired_proposal
```

`v42-paired-proposal` keeps score-based top-50 proposal selection in both training
and inference. Before pose injection, the native class/box Hungarian matcher
assigns selected references to valid GT people. Each matched slot receives that
person's clean pose, corrupted at an independent uniform per-image timestep
0..999, and keeps the same target for final and pose-auxiliary losses. Background
slots receive random coordinate padding and classification-only supervision.
Earlier box/proposal/DN losses retain native matching; Gaussian DDIM inference
is unchanged. Fresh seed-42 EDPose initialization, full 80 epochs, box-center
readout and fixed32 K1/K4/K8 evaluation are retained; JEPA remains deferred.
See [the implementation and launch report](docs/reports/poseflow_v42_paired_proposal.md).

### V42 inference with proposal-to-noisy-pose matching

The opt-in comparison matches current image proposal boxes to enclosing boxes
of the actual incoming noisy poses on every DDIM call, including the first.
It reuses the native Hungarian box matcher and preserves all slots. The same
permuted state is used for pose encoding and the DDIM residual/update. Model
weights, training assignment and native inference defaults are unchanged.

```bash
PYTHONPATH=/path/to/run/workspace python scripts/evaluation/evaluate_v42_proposal_pose_matching.py \
  --run-root /path/to/run --checkpoint /path/to/checkpoint.ckpt \
  --output tmp/v42_pose_matching --require-frozen-source
```

The entrypoint loads the original frozen model and a separately recorded sampler
extension, strictly restores weights, and compares native/matched K1/K4/K8 on the
run's saved fixed32 image IDs. Native AP must reproduce exactly. It writes
predictions, all assignment permutations, confidence summaries and source hashes.
This is an inference experiment; the training/inference association difference
and high-noise box ambiguity remain explicit limitations. See
[the comparison report](docs/reports/poseflow_v42_proposal_pose_inference.md).

The initial fixed32 comparison at paired checkpoints3 and6 lowered AP at all three sampling counts; native sampling remains the default. This option is retained as an experimental evaluation policy.

### V42 noisy-proposal pairing in training and inference

```bash
python scripts/training/train_diffusionregpose.py --config-name benchmark_poseflow_v42_noisy_proposal
```

`v42-noisy-proposal-paired` corrupts coordinates at uniform per-image t0..999,
then matches actual noisy poses to selected image proposals. Each noisy source
carries its original clean GT target or background label through the permutation.
Final and pose-auxiliary supervision keep those labels. Evaluation uses the same
association on every DDIM call, including the first Gaussian call; older configs
retain native inference. High-noise assignments may make target identity hard to
infer. Full80 training and fixed32 K1/K4/K8 monitoring are retained; JEPA deferred.
See [the experiment report](docs/reports/poseflow_v42_noisy_proposal.md).


### Frozen-codec LDM audit (V43)

Audit the selected High-KL pose-and-box VAE under actual detector augmentation:

```bash
PYTHONPATH=. venv/bin/python scripts/evaluation/audit_poseflow_v43_codec.py \
  --output results/poseflow_v43_codec_audit
```

This computes trainval-only posterior moments and replays the original codec
validation metrics. Use a fresh output directory;
`--limit-images 8 --skip-validation-replay --workers 0` is a subset smoke, whose statistics must not
initialize a full run. See [the V43 report](docs/reports/poseflow_v43_frozen_vae.md).

### V43 small-noise conditional recovery control

```bash
PYTHONPATH=. venv/bin/python scripts/training/train_poseflow_v43_small_noise.py
```

This bounded comparison caches16 TRAIN/16 VAL images and runs x0/v arms for2000
updates each with the image frontend and VAE frozen. It evaluates paired geometry
at0/1/2.5/5/10% latent noise; it is not a detector AP experiment. The preset pins the
reviewed parent checkpoint/hash; override both together for another parent. Only
best/last checkpoints are kept. Wider training remains gated on the tiny-set result.

```bash
PYTHONPATH=. venv/bin/python scripts/evaluation/evaluate_poseflow_v43_small_noise.py \
  --checkpoint /path/to/v/best.ckpt --cache /path/to/cache.pt \
  --output /path/to/restored_metrics.json
```

The first comparison completed: velocity preserves small-noise inputs, but both
arms miss the learned-recovery gate. The longer stage remains unstarted. See
[the small-noise control report](docs/reports/poseflow_v43_small_noise.md).

### V43 image-to-latent alignment baseline

```bash
PYTHONPATH=. venv/bin/python scripts/training/train_poseflow_v43_alignment.py
```

This image-only baseline predicts a clean pose-and-box code and decodes it through
the frozen High-KL VAE. It takes no noisy pose or box input. The preset trains for
10 epochs on 10,000 TRAIN images, validates on the disjoint 2,000 VAL images, and
keeps the best validation geometry checkpoint and last checkpoint. It uses a fresh
V43 prediction body and the existing EDPose frontend initialization. Existing
codec standardization retains its disclosed trainval geometry exposure.

Correct-versus-swapped image conditioning is monitored on 64 fixed VAL images;
both saved checkpoints receive full VAL AP and paired geometry evaluation after
training. TEST remains unopened: the unused eager DataModule test slot points to
VAL, and no test loop runs. This tests image-to-code alignment before further
diffusion training; it does not establish latent-noise robustness. See
[the alignment report](docs/reports/poseflow_v43_alignment.md).

### Frozen High-KL latent diffusion (V43)

The full80-epoch seed42 experiment launched on2026-11-05. It uses the audited full
trainval statistics above, a frozen32D pose-and-box VAE, V40/V40-EMA's shared image
conditioning, fixed top50 references and image-only confidence. Its initialization
is fresh ED-Pose plus a fresh denoiser, rather than the trained V40-EMA checkpoint.

```bash
PYTHONPATH=. venv/bin/python scripts/training/train_poseflow_v43.py \
  training.checkpoint.monitor=epoch training.checkpoint.mode=max
```

The checkpoint overrides retain all80 epochs plus last and avoid Lightning rejecting
`save_top_k=80` with no monitor. Full training checkpoints need about36GiB of disk
space; the current run is on the home filesystem.

The recipe is seed42/80epochs, batch4/accumulation2, FP32, historical detector crops
and trainval/test splits. `pipeline.latent_statistics_file=...` overrides the audited
statistics path; annotation/augmentation/codec mismatches are rejected. Checkpoints
embed the frozen codec, statistics and RNG, so restoring does not require the
original initialization assets.

```bash
PYTHONPATH=. venv/bin/python scripts/evaluation/evaluate_poseflow_v43.py \
  --checkpoint /path/to/last.ckpt --output results/v43_evaluation \
  --steps 1 4 8 --count 32
```

Omit `--count` for full8000-image test evaluation. Fixed32 test monitoring runs each
epoch; val loss overlaps training and is a fit diagnostic. The primary final
checkpoint is epoch79; codec reconstruction AP and short wiring smokes are separate
from trained detector AP. See [the V43 report](docs/reports/poseflow_v43_frozen_vae.md)
for the exact loss, source assignment, audit results and verification artifacts.

### V42 paired training with image-only confidence

```bash
python scripts/training/train_diffusionregpose.py --config-name benchmark_poseflow_v42_image_confidence
```

This variant adds a separate classifier on normalized selected image/time features
before pose injection. Its final/pose-aux scores use the same fixed paired GT
labels as geometry. Uniform t0..999, source construction, geometry losses and
native coordinate DDIM remain. Historical configs default to pose-conditioned
confidence; restore the new head strictly with its saved image-only mode/config.
Fresh full80 training uses fixed32 K1/K4/K8 monitoring; trained AP is pending.
See [the training report](docs/reports/poseflow_v42_image_confidence_training.md).

The first V43 image-alignment run was cancelled after proposal collapse. Its
recipe is retained for reproducibility, not as a validated recommendation. See
[the failure diagnosis](docs/reports/poseflow_v43_alignment_failure.md).

### V44: native human proposals + frozen High-KL latent diffusion

Train-only statistics and a separate latent-diffusion run (TRAIN/VAL; TEST unused):

```bash
PYTHONPATH=. venv/bin/python scripts/evaluation/compute_poseflow_v44_statistics.py --output tmp/v44/train_statistics.pt
PYTHONPATH=. venv/bin/python scripts/training/train_poseflow_v44.py pipeline.latent_statistics_file="$PWD/tmp/v44/train_statistics.pt"
```

V44 retains native proposal/box refinement and freezes the codec. Selected-slot
assignments and proposal conditioning are reused across latent diffusion steps.
TensorBoard uses the existing `quick-test/noise_var_100/steps_{1,4,8}/{pose_AP,bbox_AP}`
cards. Only best/last checkpoints are retained. See the [V44 report](docs/reports/poseflow_v44.md)
for the representation, losses, pretraining exposure and evaluation boundaries.

```bash
PYTHONPATH=. venv/bin/python scripts/evaluation/evaluate_poseflow_v44.py /absolute/run/seed42/checkpoints/epochNNN.ckpt --output tmp/v44/full_validation
```


V42 paired-image JEPA preset: `benchmark_poseflow_v42_jepa`. It adds an EMA
clean-coordinate representation target and shared-reader clean reconstruction
to the paired image-confidence baseline, retaining coordinate diffusion.
See [the objective, checkpoint and evaluation contracts](docs/reports/poseflow_v42_jepa.md).

The bounded V44 x0/velocity comparison uses an explicit trusted x0 warmstart:

```bash
PYTHONPATH=. venv/bin/python scripts/training/train_poseflow_v44_comparison.py arm=v source_checkpoint=/absolute/source.ckpt source_sha256=CHECKPOINT_SHA256
```

Use `arm=x0` for the matched control. Each arm runs10 epochs with fresh AdamW at
2e-5 (backbone1e-5), retaining best/last. Existing experiments remain separate.
The [comparison report](docs/reports/poseflow_v44_velocity.md) documents head
conversion, initialization differences and recovery/AP acceptance criteria.

The approved x0 rollout comparison uses the same frozen codec and fixed slot
assignments, changing only intermediate training-state construction:

```bash
PYTHONPATH=. venv/bin/python scripts/training/train_poseflow_v44_rollout.py arm=rollout source_checkpoint=/absolute/source.ckpt source_sha256=CHECKPOINT_SHA256
```

Use `arm=forward` for its matched control. Each arm runs2epochs with fresh
AdamW2e-5/backbone1e-5. Half the batches retain uniform diffusion training;
the remaining batches use the matched K4 timestep distribution. See the
[rollout report](docs/reports/poseflow_v44_rollout.md) for the detached warm-up,
restoration, logging and acceptance contracts.

### V45: progressive bidirectional person prediction

```bash
PYTHONPATH=. venv/bin/python scripts/training/train_poseflow_v45.py pipeline.latent_statistics_file="$PWD/tmp/v44/train_statistics.pt"
```

V45 grows a confidence-ranked prefix of50 native proposals, repeatedly applying
the same two decoder layers with bidirectional attention inside the active
prefix. Hidden states carry across stages; the frozen High-KL codec produces
every pose/box. Equal per-person intermediate supervision replaces diffusion.
TensorBoard uses `quick-test/progressive/stages_50/{pose_AP,bbox_AP}`, plus
training losses and birth/final errors by rank. Best/last retention and final
fullVAL evaluation are automatic. See the [V45 report](docs/reports/poseflow_v45.md)
for the full80 recipe, contracts and verification.


V45 task-pose presets: `benchmark_poseflow_v45_task_pose_control` and
`benchmark_poseflow_v45_task_pose_jepa`. They jointly learn an absolute-coordinate
pose codec while retaining coordinate diffusion and cached image context.
See [objectives, launch and evaluation](docs/reports/poseflow_v45_task_pose.md).
These are distinct from progressive-person V45.


### V46: ordered generation with parallel denoising rows

```bash
venv/bin/python scripts/training/train_poseflow_v46.py \
  pipeline.latent_statistics_file=tmp/v44/train_statistics.pt
```

Requires the original ED-Pose initializer, SHA-pinned High-KL codec and verified
TRAIN posterior-mean statistics used by V44. V46 sorts augmented targets into512
columns, batches64 independent noisy-prefix/PREDICT rows during training, and
generates/refines sequentially with whole-prefix re-noising at inference.
Capacity50; EOS stops additions while remaining rows refine. No proposals or
Hungarian matching. Fresh full80 TRAIN/VAL run, fixed32VAL selection, final
full2000VAL; TEST unopened. TensorBoard AP cards use
`quick-test/ordered_sequence/rows_64/{pose_AP,bbox_AP}`.

Restore with `PoseFlowV46Pipeline.load_from_checkpoint(path, weights_only=False)`;
restoration needs no initialization assets. Normal `model.predict(context,
image_wh, generator=...)` returns final geometry/confidence, valid masks and
sequence ranks; traces require `collect_trace=True`. See the
[V46 report](docs/reports/poseflow_v46.md) for objectives, sampling and limitations.


V46 evaluation also supports `model.predict(..., sampler="ddim")`: deterministic
DDIM for existing people and one-time forward noise for newborns. Legacy
re-noising remains the default. Compare both on the original fixed32VAL set:

```bash
venv/bin/python scripts/evaluation/compare_poseflow_v46_samplers.py \
  --run-root /path/to/v46/run --checkpoint /path/to/completed/last.ckpt \
  --output /path/to/fresh/comparison
```

No retraining or external initialization assets are required. The comparison
archives a frozen checkpoint, verifies original-sampler parity and writes paired
AP/refinement/trajectory metrics and TensorBoard events; see the V46 report.

The independent **V45 task-pose ordered JEPA** preset is
`benchmark_poseflow_v45_ordered`:50 learned rank slots with deterministic
512-column GT ordering and no Hungarian/DP assignment. It retains coordinate
diffusion and task-pose JEPA while replacing candidate selection with rank-query
image context. See [the report](docs/reports/poseflow_v45_ordered.md) for the
training command, full80 recipe and historical TEST-development exposure.

### V47 ordered coordinate diffusion

V47 uses a jointly learned shared person encoder/reader, coordinate-space corruption and DDIM, and decoded2D coordinate/geometry/EOS losses. It retains64 parallel training rows,512-column order and50-person capacity. Native verification and a two-seed four-image TRAIN memorization gate must pass before fresh full80 training; TEST stays unopened. See [architecture, verification and usage](docs/reports/poseflow_v47.md). TensorBoard AP tags are `quick-test/ordered_coordinate/rows_64/`; teacher-forced reconstruction and free-running detection are logged separately. Verification passed (55 checks/native CUDA/restore; four-image gate98.80pose/91.73bbox). Fresh full80 is running; VAL outcomes remain pending.

2026-11-09: V47 full80 launch verified at `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v47/20261109_150341`, trainerPID3490732.55 checks, native full64/50people FP32/batch4accum2/restore and two-seed four-image learnability gate passed. Fresh initial hash matches verification; same fixed32VAL, separate TensorBoard stream, existing jobs/artifacts preserved. See V47 report for mixed refinement behavior and measured cost; fullVAL outcomes pending.

### V48 clean-prefix ordered generation

V48 uses V47's learned coordinate codec with clean conditioning prefixes. It
predicts one person or EOS per step, keeps birth predictions fixed and stops
immediately at EOS/capacity50. Training uses parallel P+1 rows and birth/EOS-only
losses. See [V48 report](docs/reports/poseflow_v48.md) for contracts and provenance.

```bash
venv/bin/python -m scripts.training.verify_poseflow_v48 --output tmp/v48/verification
venv/bin/python -m scripts.training.train_poseflow_v48 verification_file="$PWD/tmp/v48/verification/verification.json"
```

The full80 seed42 launcher requires matching native verification, source/config/
input hashes and fresh initialization. Existing jobs/runs are preserved. AP cards
use `quick-test/ordered_birth_only/pose_AP` and `bbox_AP`; teacher-forced birth
errors are logged separately from free-running detection results.

#### Parallel V48-AdaLN variant

The separate `benchmark_poseflow_v48_adaln` run adds generation-step-conditioned
AdaLN and residual gating only to image cross-attention. Identity initialization
preserves V48 common weights/outputs and RNG; clean prefixes, birth/EOS losses
and the full80 recipe remain fixed. See [variant report](docs/reports/poseflow_v48_adaln.md).

```bash
venv/bin/python -m scripts.training.verify_poseflow_v48_adaln --output tmp/v48_adaln/verification --baseline-manifest /path/to/baseline/seed42/manifest.json
venv/bin/python -m scripts.training.train_poseflow_v48_adaln verification_file="$PWD/tmp/v48_adaln/verification/verification.json"
```

Baseline training continues unchanged. TensorBoard uses a separate V48-AdaLN run
with the same `quick-test/ordered_birth_only/` AP tags for direct overlays.


### Token-age diffusion forcing: input-builder iteration

An isolated helper now builds 50 parallel teacher-forced rows with randomly
permuted people, fixed source identities and per-token noise levels
`100% -> 10% -> 5% -> ...`. All active people have supervision masks, EOS leaves
a refinement tail, and row50 is reserved for cleanup (maximum49births).
Percentages denote Gaussian coefficients. This is not yet a trainable model or
an inference sampler. See [the interface and verification](docs/reports/token_age_diffusion_rows.md).

**V42 timestep-loss:** `benchmark_poseflow_v42_timestep_loss` retains the corrected
box-center/uniform-t model, native matching and sampler. Only geometry losses
change: box emphasis at high diffusion noise, pose emphasis near the clean end,
with native average loss strength preserved. See [the report](docs/reports/poseflow_v42_timestep_loss.md).


### V49 token-age diffusion forcing

V49 completes the50-row input builder with per-token noise-conditioned decoding,
source-paired supervision of all active people and coordinate DDIM. Newborns
start at pure noise; older people follow10%,5%,... Gaussian coefficients. EOS
stops births while remaining rows refine, with a49-person birth limit. Training
rows run in parallel; sampling carries noisy coordinate states sequentially.
Fresh full80/seed42 TRAIN/VAL training is gated on native FP32/batch4accum2
verification. See [contracts, usage and measurements](docs/reports/poseflow_v49.md).
TensorBoard AP: `quick-test/token_age_diffusion/rows_50/{pose_AP,bbox_AP}`.

### V49-balanced loss variant

`benchmark_poseflow_v49_balanced` retains the V49 model and sampler, adding robust
geometry errors, fixed TRAIN-only gradient calibration, equal refinement-age
buckets and calibrated box-first/pose-later loss priorities. First EOS is
supervised separately from the easy EOS tail. Fresh full80 training is gated by
native preflight and calibration provenance; historical jobs are preserved.
See [objective, verification and usage](docs/reports/poseflow_v49_balanced.md).

V49-balanced:70 checks/native full-capacity calibration and strict resume passed;
fresh seed42/full80 is running. TensorBoard run`V49-balanced-token-age/.` overlays
the original V49 with the same AP tags and fixed32VAL images.

### V50 proposal attention and gated residuals

V50 selects a sample-local proposal with attention before geometry prediction, retains detached proposal content as identity, and uses noise-gated residual refinement with an exact zero-noise bypass. It retains the robust/calibrated box–pose balance. See [V50 experiment contract](docs/reports/poseflow_v50.md).

```bash
python -m scripts.training.verify_poseflow_v50 --output tmp/v50/verification
python -m scripts.training.train_poseflow_v50 verification_file=tmp/v50/verification/verification.json
```

Full training requires matching native verification, runs80 epochs on TRAIN/VAL only, and logs AP under `quick-test/proposal_attention_residual/rows_50/`. Historical V49 checkpoints/runs are preserved separately.
