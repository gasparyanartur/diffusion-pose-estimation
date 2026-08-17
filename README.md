# pose-graph-diff

A toolkit for image-conditioned 2D human pose estimation and object detection
on [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose). The shared pose model is
always image → features → pose. The default is one-stage image-to-pose set
prediction; graph pose and box diffusion are explicit pipeline variants. Pure
box detection is a separate model family.

---

## Project layout

```
conf/           ← Hydra config groups (pipeline, data, training, probe, hardware)
data/           ← CrowdPose images and annotations
checkpoints/    ← model weights
external/       ← vendored dependencies (mmpose, mmdet) installed as editable packages
scripts/        ← categorized CLI entry points (data, training, evaluation, investigation, utilities)
src/            ← core library (data, models, training, evaluation)
notebooks/      ← investigations, visualizations, and legacy notebooks
results/        ← benchmark outputs, prediction caches, training runs
tests/          ← unit tests
```

---
## 1. Install the complete environment

The repository uses one environment for the root project and
`external/DiffusionRegPose`. The setup includes the mandatory CrowdPose APIs,
Detectron2, MMPose 1.3.2, and MMDetection 3.3.0.

Prerequisites:

- Python 3.10
- `pyenv` (recommended for installing the exact Python version)
- `uv`
- A complete CUDA toolkit with `nvcc`
- NVIDIA driver and a CUDA-capable GPU

If Python 3.10.20 is not already available, install and select it with pyenv:

```bash
pyenv install 3.10.20
pyenv local 3.10.20
```

Do not use the system Python directly. The setup script creates the project
environment from the selected Python 3.10 interpreter.

From the repository root, run:

```bash
./scripts/utils/setup_shared_env.sh
source venv/bin/activate
```

To test in a separate environment, pass its path:

```bash
./scripts/utils/setup_shared_env.sh /tmp/pose-graph-diff-venv
```

The script installs the tested Torch 2.4.1 + CUDA 12.1 wheel, MMCV 2.2.0,
MMEngine 0.10.7, local editable MMPose/MMDetection packages, CrowdPose,
Detectron2, and the DiffusionRegPose CUDA operator. It does not install the
legacy external requirements file wholesale because its broad Torch ranges can
replace the repository's pinned Torch pair.

### CUDA note

The preferred setup is a CUDA 12.1 toolkit matching the Torch wheel. On systems
with the available CUDA 13.0 toolkit, the script uses the validated build
workaround for the extension's strict PyTorch CUDA-version check. The extension
compiled and imported successfully in that configuration, but a matching
CUDA/Torch pair remains preferable.

The script uses `CUDA_HOME` when set. Otherwise it uses the toolkit detected by
PyTorch. To select a toolkit explicitly, set `CUDA_HOME` before running it.
It also registers the compiled DiffusionRegPose extension in the target
environment and links it against that environment's Torch libraries, so no
manual `PYTHONPATH` export or import-order workaround is required.

### MMPose and MMDetection

The checked-in `external/mmpose` and `external/mmdet` trees are installed
editable by the setup script. The local MMDetection tree contains the required
MMCV compatibility patch for the pinned `mmcv==2.2.0` stack.

---

## 2. Dataset setup — CrowdPose

### 2a. Images

Download the CrowdPose images from the [official release](https://github.com/Jeff-sjtu/CrowdPose)
and place them flat in `data/crowdpose/images/`:

```
data/crowdpose/images/107498.jpg
data/crowdpose/images/107499.jpg
...
```

### 2b. Annotations

Run the prepare script to download MMPose-formatted annotations and pre-generated detector boxes:

```bash
python scripts/data/prepare_crowdpose.py
```

---

## 3. One-stage pose smoke run

The default pose workflow does not require prediction caches or precomputed
DINO features. The smallest end-to-end smoke run uses the one-stage graph
diffusion pipeline and a tiny CrowdPose subset:

```bash
python scripts/training/train_graph_pipeline.py \
    data=crowdpose_set_prediction_debug \
    training.max_epochs=2 \
    hardware.accelerator=cpu hardware.devices=1 hardware.precision=32
```

The root configuration selects the CNN fixed-slot baseline pose pipeline by
default. The graph pipeline is an explicit one-stage variant selected by
`conf/benchmark_graph_pipeline.yaml`.

## 4. Data API

See [`src/data/datasets.py`](src/data/datasets.py) for `PoseDataset`,
`PredictionDataset`, `SceneGraphSampler`, and `prepare_pose_graph_batch`.

---

## 5. Training pose models

Config is managed by [Hydra](https://hydra.cc). All hyperparameters live in
`conf/` as composable YAML groups. Run from the project root so relative data
paths resolve correctly.

### One-stage pose models

The canonical runnable one-stage graph-pipeline command is
`scripts/training/train_graph_pipeline.py`:

```bash
python scripts/training/train_graph_pipeline.py
```

The root config defaults to the CNN fixed-slot baseline pose model:
`pipeline=baseline_pose` and `data=crowdpose_set_prediction`.

`scripts/training/train_pose_refinement.py` trains the canonical fixed-slot baseline
through `pipeline=baseline_pose`.

Checkpoints and logs are written to
`results/diffusion_training/<run-name>/<timestamp>/`.  
Hydra also saves the fully-resolved config to `.hydra/config.yaml` in that
directory for complete reproducibility.

### Selecting pipelines and overriding values

```bash
# Select the graph-pose pipeline
python scripts/training/train_graph_pipeline.py pipeline=graph_pose

# Select the tensor-native box-diffusion pipeline
python scripts/training/train_box_diffusion.py pipeline=box_diffusion

# Train the native DiffusionRegPose implementation
python scripts/training/train_diffusionregpose.py

# Reduced CPU smoke run
python scripts/training/train_diffusionregpose.py \
    pipeline.num_queries=4 pipeline.num_proposals=2 pipeline.num_group=2 \
    pipeline.num_encoder_layers=1 pipeline.num_decoder_layers=3 \
    pipeline.backbone_pretrained=false data.max_size=128 data.batch_size=1 \
    data.num_workers=0 training.max_epochs=1 \
    hardware.accelerator=cpu hardware.devices=1 hardware.precision=32

# Override hardware and training settings
python scripts/training/train_pose_refinement.py \
    training.max_epochs=100 hardware.accelerator=gpu hardware.devices=1
```

### Overriding individual values

Any leaf key can be overridden from the command line using dot notation:

```bash
python scripts/training/train_pose_refinement.py \
    run_name=exp_large \
    pipeline.lr=3e-4 \
    training.max_epochs=200 \
    pipeline.cosine_lr=true \
    hardware=gpu
```

### Resuming from a checkpoint

```bash
python scripts/training/train_pose_refinement.py \
    training.resume=results/pose_estimator_training/my_run/checkpoints/last.ckpt
```

Config groups (`pipeline`, `data`, `probe`, `hardware`) live in `conf/`.
To inspect the full resolved config without launching training:

```bash
python scripts/training/train_pose_refinement.py --cfg job
```

### Losses and metrics

The canonical pose pipelines use Hungarian matching with model-family-specific
pose, box, visibility, and diffusion losses. Evaluation reports CrowdPose AP
and AR metrics through `src.eval.metrics`; box diffusion reports COCO bbox AP.

---

## 7. Running benchmarks

### DiffusionRegPose evaluation

The native evaluator loads the original DiffusionRegPose checkpoint format,
performs deterministic DDIM keypoint sampling, writes CrowdPose-compatible
predictions, and can calculate AP/AP-E/AP-M/AP-H when `crowdposetools` is
installed:

```bash
python scripts/evaluation/evaluate_diffusionregpose.py \
    --checkpoint path/to/diffusionregpose_checkpoint.pth \
    --annotations data/crowdpose/annotations/crowdpose_test.json \
    --image-root data/crowdpose/images \
    --output results/diffusionregpose/test_results.json \
    --metrics results/diffusionregpose/metrics.json
```

Use `--limit` for a smoke run and `--sampling-timesteps` to select the DDIM
schedule. The default seed and one-step schedule match the reference
evaluation path used for checkpoint parity.

On CUDA, the evaluator lazily builds the repository's independent native
deformable-attention kernel in PyTorch's extension cache. This kernel is
required for bit-for-bit checkpoint parity with the reference CUDA operator;
if it cannot be built, the differentiable PyTorch sampler remains available as
a functional fallback.

```bash
# List available configurations
python scripts/evaluation/benchmark_mmpose.py --list

# Run a benchmark (checkpoint downloaded automatically on first run)
python scripts/evaluation/benchmark_mmpose.py <model-name>
```

Metrics are written to `results/<name>/metrics.yaml`.
See `python scripts/evaluation/benchmark_mmpose.py --help` for all options.

---

## 8. Results

Benchmark metrics are written under `results/`. Training outputs include
resolved Hydra configs, logs, checkpoints, and test results under the pipeline
run directory. See each script's `--help` output and its Hydra root config for
available options.
