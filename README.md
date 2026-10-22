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

Recent PoseFlow experiments include V15 rollout training, V17 candidate-tied
hypothesis memory, V18 pre-image representation refinement, and V19 coordinate
refinement with tied memory. These are distinct experiments, without an implied
performance ranking. See the [diary](docs/experiments.md) for their commands and
[project conventions](docs/project.md) for coordinate and evaluation contracts.

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
