# pose-graph-diff

A graph-neural-network diffusion model for refining 2D human pose estimates on
[CrowdPose](https://github.com/Jeff-sjtu/CrowdPose), with a supporting toolkit
for running MMPose baselines, extracting prediction caches, and evaluating
refined results.

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

## 1. Prerequisites

### Python environment

Python 3.10 is required. [uv](https://github.com/astral-sh/uv) is recommended
for fast, reproducible installs.

```bash
uv venv venv --python 3.10
source venv/bin/activate
```

**Step 1 — PyTorch (CUDA 12.1):**

```bash
uv pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

**Step 2 — mmcv (pre-built wheel for CUDA 12.1 + PyTorch 2.4):**

```bash
uv pip install mmcv==2.2.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
```

**Step 3 — remaining dependencies:**

```bash
uv pip install -r requirements.txt
```

**Step 4 — local editable installs (mmdet and mmpose):**

```bash
uv pip install -e external/mmdet --no-build-isolation
uv pip install -e external/mmpose --no-build-isolation
```

> `--no-build-isolation` is required because mmdet and mmpose import torch at
> build time; without this flag uv's isolated build environment cannot see the
> already-installed torch.

Key runtime dependencies: `torch 2.4`, `torchvision 0.19`, `pytorch-lightning`,
`torch-geometric`, `mmengine`, `mmcv 2.2`, `transformers 5.x`, `diffusers`.

### MMPose and MMDet

Both are cloned into `external/` and installed as editable packages (see Step 4
above). The `external/mmdet` clone contains a one-line patch relaxing MMDet's
`mmcv < 2.2.0` assertion to `<= 2.2.0`, which is necessary for the
torch 2.4 + mmcv 2.2.0 stack.

```bash
git clone --depth 1 --branch v3.3.0 \
    https://github.com/open-mmlab/mmdetection.git external/mmdet
git clone https://github.com/open-mmlab/mmpose.git external/mmpose
```

> If you re-clone mmdet from upstream you will need to re-apply the patch
> to `external/mmdet/mmdet/__init__.py`:
> change `mmcv_version < digit_version(mmcv_maximum_version)` →
> `mmcv_version <= digit_version(mmcv_maximum_version)`.

### Environment variables

Source `.env` before running any script.  It sets `PROJECT_DIR` and
adds `external/mmpose` to `PYTHONPATH`:

```bash
source .env
```

> Add `source .env` to your shell rc or VS Code workspace settings to avoid
> repeating it every session.

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

## 3. Extracting prediction caches

Before training the refinement model you need `.pkl` prediction caches for
in-training AP evaluation.

```bash
# List available models
python scripts/extract_predictions.py --list

# Extract predictions for a model
python scripts/extract_predictions.py <model-name>
```

Caches are written to `results/cache/<model-short-name>_predictions.pkl`.

---

## 4. Precomputing DINO features

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

## 5. Data API

See [`src/data.py`](src/data.py) for full documentation of `PoseDataset`,
`PredictionDataset`, `SceneGraphSampler`, and `prepare_pose_graph_batch`.

---

## 6. Training the refinement model

Config is managed by [Hydra](https://hydra.cc). All hyperparameters live in
`conf/` as composable YAML groups. Run from the project root so relative data
paths resolve correctly.

### Quick start

```bash
python scripts/train_pose_refinement.py
```

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
bash scripts/run_sweep.sh sweeps/my_sweep.sweep 1
```

Output is written to `results/sweeps/<sweep_name>/task_<task_id>/`.  
Training and final test evaluation are handled by `train_pose_refinement.py`;
results are saved to `test_results.json` in that directory.

### Running all tasks sequentially

```bash
for i in $(seq 1 $(grep -cve '^\s*#\|^\s*$' sweeps/my_sweep.sweep)); do
    bash scripts/run_sweep.sh sweeps/my_sweep.sweep $i
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
