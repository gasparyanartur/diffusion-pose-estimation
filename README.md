# pose-graph-diff

A toolkit for image-conditioned 2D human pose estimation and object detection
on [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose). The shared pose model is
always image → features → pose. The default is one-stage image-to-pose set
prediction; the legacy graph-diffusion refiner can optionally consume existing
detector keypoints. Pure box detection is a separate model family.

---

## Project layout

```
conf/           ← Hydra config groups (model, data, train, dino, hardware)
data/           ← CrowdPose images and annotations
checkpoints/    ← model weights
external/       ← vendored dependencies (mmpose, mmdet) installed as editable packages
scripts/        ← CLI entry points (training, benchmarking, data prep, inspection)
src/            ← core library (data, model, metrics, visualisation)
notebook/       ← experimental notebooks
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
./scripts/setup_shared_env.sh
source venv/bin/activate
```

To test in a separate environment, pass its path:

```bash
./scripts/setup_shared_env.sh /tmp/pose-graph-diff-venv
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
python scripts/prepare_crowdpose.py
```

---

## 3. One-stage pose smoke run

The default pose workflow does not require prediction caches or precomputed
DINO features. The smallest end-to-end smoke run uses the one-stage graph
diffusion pipeline and a tiny CrowdPose subset:

```bash
python scripts/train_graph_pipeline.py \
    data=crowdpose_set_prediction_debug \
    training_module.max_epochs=2 \
    training_module.val_ap_every_n_epochs=1 \
    hardware.accelerator=cpu hardware.devices=1 hardware.precision=32
```

The root configuration selects the CNN set-prediction pose model by default.
The graph pipeline is an explicit one-stage variant selected by
`conf/benchmark_graph_pipeline.yaml`.

## 4. Extracting prediction caches for the refiner

The legacy `graph_diffusion_refiner` model requires `.pkl` prediction caches for
in-training AP evaluation.

```bash
# List available models
python scripts/extract_predictions.py --list

# Extract predictions for a model
python scripts/extract_predictions.py <model-name>
```

Caches are written to `results/cache/<model-short-name>_predictions.pkl`.

---

## 5. Precomputing DINO features for the refiner

The refinement model uses precomputed [DINOv3 ConvNeXt-Small](https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m)
features for image conditioning. Features are extracted once and cached to disk
as `.pt` files — no GPU is needed at training time.

Three feature levels are saved per image:

| Key | Shape | Description |
|---|---|---|
| `spatial` | `(96, 56, 56)` | Stage-1 high-res spatial texture |
| `semantic` | `(768, 7, 7)` | Stage-4 semantic bottleneck |
| `global` | `(768,)` | Pooler output — per-image summary vector |

```bash
# Train split (required)
python scripts/precompute_dino.py \
    --ann-path  data/crowdpose/annotations/crowdpose_train.json \
    --img-root  data/crowdpose/images \
    --split     train \
    --model-name facebook/dinov3-convnext-small-pretrain-lvd1689m \
    --backbone-type convnext

# Val split (used during validation steps)
python scripts/precompute_dino.py \
    --ann-path  data/crowdpose/annotations/crowdpose_val.json \
    --img-root  data/crowdpose/images \
    --split     val \
    --model-name facebook/dinov3-convnext-small-pretrain-lvd1689m \
    --backbone-type convnext

# Test split (used for final evaluation)
python scripts/precompute_dino.py \
    --ann-path  data/crowdpose/annotations/crowdpose_test.json \
    --img-root  data/crowdpose/images \
    --split     test \
    --model-name facebook/dinov3-convnext-small-pretrain-lvd1689m \
    --backbone-type convnext
```

Caches land in
`results/cache/crowdpose/facebook_dinov3-convnext-small-pretrain-lvd1689m/<split>.pt`.
The paths are already wired into `conf/dino/dinov3_convnext_small.yaml`; no
config changes are needed after running the commands above.

> If a split cache is missing at training time, `DinoConditioner` logs a warning
> and continues — conditioning will be zero for images from that split.

---

## 6. Data API

See [`src/data.py`](src/data.py) for full documentation of `PoseDataset`,
`PredictionDataset`, `SceneGraphSampler`, and `prepare_pose_graph_batch`.

---

## 7. Training pose models

Config is managed by [Hydra](https://hydra.cc). All hyperparameters live in
`conf/` as composable YAML groups. Run from the project root so relative data
paths resolve correctly.

### One-stage pose models

The canonical runnable one-stage graph-pipeline command is
`scripts/train_graph_pipeline.py`:

```bash
python scripts/train_graph_pipeline.py
```

The root config also defaults to the CNN set-prediction pose model:
`pose_estimator=cnn_set_prediction`, `training_module=set_prediction`, and
`data=crowdpose_set_prediction`.

### Legacy graph-diffusion refiner

Use the refiner only when starting from existing detector keypoints:

```bash
python scripts/train_pose_refinement.py \
    pose_estimator=dino_diffusion_legacy \
    training_module=graph_diffusion_refiner \
    data=crowdpose
```

The refiner path preserves optional `initial_keypoints` support for debugging,
but it is not the default evaluation path.

`scripts/train_pose_refinement.py` is the historical refiner entry point. Use
it with a refinement-compatible configuration; do not run it with the
one-stage root defaults until its training-loop callbacks are migrated to the
set-prediction module.

Checkpoints and logs are written to
`results/diffusion_training/<run-name>/<timestamp>/`.  
Hydra also saves the fully-resolved config to `.hydra/config.yaml` in that
directory for complete reproducibility.

### Switching config groups

```bash
# Larger model (256-dim, 6 layers)
python scripts/train_pose_refinement.py model=large

# Single GPU with bfloat16
python scripts/train_pose_refinement.py hardware=gpu

# Enable DINOv3 ConvNeXt-Small image conditioning
python scripts/train_pose_refinement.py dino=dinov3_convnext_small
```

### Overriding individual values

Any leaf key can be overridden from the command line using dot notation:

```bash
python scripts/train_pose_refinement.py \
    run_name=exp_large \
    model=large \
    train.lr=3e-4 \
    train.max_epochs=200 \
    train.cosine_lr=true \
    hardware=gpu
```

### Resuming from a checkpoint

```bash
python scripts/train_pose_refinement.py \
    train.resume=results/diffusion_training/my_run/20260312_120000/checkpoints/last.ckpt
```

Config groups (`model`, `data`, `train`, `dino`, `hardware`) live in `conf/`.
To inspect the full resolved config without launching training:

```bash
python scripts/train_pose_refinement.py --cfg job
```

### Losses and metrics

The model is trained with a visibility-weighted per-joint MSE loss and a
visibility-weighted bone-length MSE loss. Evaluation reports
**MPJPE** (mean per-joint position error in pixels) and **PCK@20px**
(fraction of visible joints within 20 px of ground truth).

---

## 7. Sweep training

A sweep runs multiple training jobs sequentially, each with a different set of
hyperparameters. Results for every task land in their own isolated directory
so they can be compared side-by-side.

### Sweep file format

Create a plain-text `.sweep` file under `sweeps/`. Each non-blank,
non-comment line is one training run expressed as standard Hydra overrides:

```
# sweeps/my_sweep.sweep
model=base  train.lr=1e-3
model=base  train.lr=3e-4
model=large train.lr=1e-3  train.max_epochs=500
```

See `sweeps/example.sweep` for a ready-to-use example.

### Running a single task

Task IDs are 1-based and count only non-blank, non-comment lines:

```bash
# Run the first task in the sweep
bash scripts/run_sweep.sh scripts/train_pose_refinement.py sweeps/my_sweep.sweep 1
```

Output is written to `results/sweeps/<sweep_name>/task_<task_id>/`.  
Training and final test evaluation are handled by `train_pose_refinement.py`;
results are saved to `test_results.json` in that directory.

### Running all tasks sequentially

```bash
for i in $(seq 1 $(grep -cve '^\s*#\|^\s*$' sweeps/my_sweep.sweep)); do
    bash scripts/run_sweep.sh scripts/train_pose_refinement.py sweeps/my_sweep.sweep $i
done
```

---

## 8. Running a benchmark

```bash
# List available configurations
python scripts/benchmark_mmpose.py --list

# Run a benchmark (checkpoint downloaded automatically on first run)
python scripts/benchmark_mmpose.py <model-name>
```

Metrics are written to `results/<name>/metrics.yaml`.
See `python scripts/benchmark_mmpose.py --help` for all options.

---

## 9. Adding a new benchmark configuration

Open `src/configs.py` and add a new dict following the existing pattern, then
append it to `ALL_CONFIGS`:

```python
MY_MODEL_CROWDPOSE: dict = {
    "name": "my-model-crowdpose",          # used as the --list key
    "mmpose_dir": defaults.MMPOSE_DIR,
    "config_path": defaults.MMPOSE_DIR / "configs/..." / "my_model.py",
    "checkpoint": defaults.CHECKPOINTS_DIR / "my_model.pth",
    "checkpoint_url": "https://...",       # set to None to disable auto-download
    "data_root": defaults.DATA_ROOT,
    "ann_file": "crowdpose/annotations/crowdpose_test.json",
    "img_prefix": "crowdpose/images/",
    "metainfo_config": defaults.MMPOSE_DIR / "configs/_base_/datasets/crowdpose.py",
    "bbox_file": "crowdpose/annotations/det_for_crowd_test_0.1_0.5.json",
    # set bbox_file to None for bottom-up models (no person detector needed)
}

ALL_CONFIGS = [..., MY_MODEL_CROWDPOSE]
```

The new entry is immediately available via `--list` and as a positional argument
to both `benchmark_mmpose.py` and `extract_predictions.py`.

---

## 10. Results

Benchmark metrics are written to `results/<name>/metrics.yaml`.  
Training test-set results are written to `results/diffusion_training/<run>/<timestamp>/test_results.json`.  
Sweep task results are written to `results/sweeps/<sweep_name>/task_<task_id>/test_results.json`.
Sweep task results are written to `results/sweeps/<sweep_name>/task_<task_id>/test_results.json`.
