# pose-graph-diff

A graph-neural-network diffusion model for refining 2D human pose estimates on
[CrowdPose](https://github.com/Jeff-sjtu/CrowdPose), with a supporting toolkit
for running MMPose baselines, extracting prediction caches, and evaluating
refined results.

---

## Project layout

```
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

See `python scripts/extract_predictions.py --help` for all options.
Caches are written to `results/cache/<model-short-name>_predictions.pkl`.

---

## 4. Data API — `src/data.py`

### `PoseDataset`

GT-only dataset; one sample per image.  Images where every annotation has no
labeled keypoints are excluded.

```python
from src.data import PoseDataset

ds = PoseDataset.from_path("data/crowdpose/annotations/crowdpose_train.json")
sample = ds[0]
# sample["image_id"]   → int
# sample["img_path"]   → str
# sample["keypoints"]  → Tensor[N, 14, 2]   raw (x, y)
# sample["visibility"] → Tensor[N, 14]      0=absent 1=occluded 2=visible

loader = ds.create_dataloader(batch_size=32, shuffle=True, drop_last=True)
```

### `PredictionDataset`

Extends `PoseDataset` with model predictions loaded from a `.pkl` cache.
GT annotations (which have a ragged number of people per image) are deliberately
**not** returned by `__getitem__`; use `get_gt(image_id)` instead so
`default_collate` can stack predictions uniformly.

```python
from src.data import PredictionDataset

ds = PredictionDataset.from_path(
    "results/cache/rtmpose-m_crowdpose_predictions.pkl",
    ann_path="data/crowdpose/annotations/crowdpose_test.json",  # optional GT
)
sample = ds[0]
# sample["image_id"]             → int
# sample["img_path"]             → str
# sample["pred_keypoints"]       → Tensor[N, K, 2]
# sample["pred_keypoint_scores"] → Tensor[N, K]
# sample["bboxes"]               → Tensor[N, 4]   (top-down only)
# sample["bbox_scores"]          → Tensor[N]       (top-down only)

gt = ds.get_gt(sample["image_id"])
# gt["keypoints"]  → Tensor[N_gt, K, 2]
# gt["visibility"] → Tensor[N_gt, K]
# Returns None if no GT was loaded.

loader = ds.create_dataloader(batch_size=32, shuffle=False)
```

### `SceneGraphSampler`

Batch sampler used internally by both `create_dataloader` methods.  Groups
images by the **number of people** they contain so every batch has a uniform
`N`, enabling `default_collate` to stack tensors without padding.

- Shuffled mode interleaves batches from different group sizes globally.
- `drop_last=True` drops the final partial batch within each group.

### Graph utilities

`prepare_pose_graph_batch` is the main entry point for the model.  It takes
raw CrowdPose keypoints and returns a block-diagonal PyG graph ready for GNN
processing.

```python
from src.data import PoseDataset, prepare_pose_graph_batch

ds = PoseDataset.from_path("data/crowdpose/annotations/crowdpose_train.json", "data/crowdpose/images")
loader = ds.create_dataloader(batch_size=4)
batch = next(iter(loader))

# batch["keypoints"]  → FloatTensor[B, N, 14, 2]
# batch["visibility"] → FloatTensor[B, N, 14]   (0/1/2)
# Mutates batch in-place and also returns it.
prepare_pose_graph_batch(batch)  # optional: img_size=FloatTensor[B, 2]
```

What it does internally:

1. **Root node** — computes a virtual root as the mean of visible `left_hip`,
   `right_hip`, and `neck` anchors per person (falls back to centroid of all
   visible joints when none of those are visible).
2. **Canonical skeleton** — prepends the root as node 0, shifting original
   CrowdPose indices by +1 (`14 → 15` keypoints per person).
3. **Root-relative coordinates** — subtracts the root position so the root
   node is at `(0, 0)`.
4. **Normalisation** — divides all coordinates by `max(W, H)` per image
   (or 800 if `img_size` is not provided).
5. **Block-diagonal graph** — calls `create_graph_from_keypoints` to pack
   the `B` scene graphs into a single PyG-compatible structure.

Returned dict keys: `edge_index (2, B·N·E)`, `node_pos (B·N·K, 2)`,
`batch (B·N·K,)`, `batch_idx`, `pose_idx`, `kpt_type_idx`, `node_idx`,
`visibility (B, N, 15)`, `root_pos (B, N, 2)`, `scale (B,)`.

---

## 5. Training the refinement model

```bash
python scripts/train_pose_refinement.py
```

All paths and hyperparameters have sensible defaults. Checkpoints and logs are
written under `results/diffusion_training/<run-name>/<timestamp>/`.

See `python scripts/train_pose_refinement.py --help` for all options.

---

## 6. Running a benchmark

```bash
# List available configurations
python scripts/benchmark_mmpose.py --list

# Run a benchmark (checkpoint downloaded automatically on first run)
python scripts/benchmark_mmpose.py <model-name>
```

Metrics are written to `results/<name>/metrics.yaml`.
See `python scripts/benchmark_mmpose.py --help` for all options.

---

## 7. Adding a new benchmark configuration

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

## 8. Results

Results are written to `results/<name>/metrics.yaml`.
