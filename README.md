# pose-graph-diff

A graph-neural-network diffusion model for refining 2D human pose estimates on
[CrowdPose](https://github.com/Jeff-sjtu/CrowdPose), with a supporting toolkit
for running MMPose baselines, extracting prediction caches, and evaluating
refined results.

---

## Project layout

```
data/
    crowdpose/
        annotations/          ← downloaded by prepare_crowdpose.py
        images/               ← place dataset images here manually
checkpoints/                  ← model weights (auto-downloaded on first run)
external/
    mmpose/                   ← MMPose source clone ($MMPOSE)
results/
    cache/                    ← prediction pickle files (extract_predictions.py)
    <benchmark-name>/
        metrics.yaml          ← evaluation output
        work_dir/             ← mmengine logs
    diffusion_training/
        <run-name>/<timestamp>/
            checkpoints/      ← best + last .ckpt files
            hparams.yaml      ← all CLI flags for the run
            train.log
            tb_logs/          ← TensorBoard event files
            csv_logs/         ← per-step CSV
scripts/
    prepare_crowdpose.py      ← downloads CrowdPose annotations
    extract_predictions.py    ← run MMPose inference → .pkl cache
    train_pose_refinement.py  ← train the diffusion refinement model
    benchmark_mmpose.py       ← run an MMPose benchmark end-to-end
    inspect_predictions.py    ← CLI inspection helper
    example_refiner.py        ← minimal usage example
src/
    defaults.py               ← base directory constants (PROJECT_DIR, MMPOSE_DIR, …)
    configs.py                ← predefined MMPose configurations (REGISTRY)
    benchmark.py              ← shared MMPose runner logic
    data.py                   ← PoseDataset, PredictionDataset, SceneGraphSampler, graph utils
    model.py                  ← KeypointDiffusionModel, PoseDiffusionModule
    vis.py                    ← visualisation helpers
notebook/
    inspect_predictions.ipynb ← interactive prediction explorer
tests/
    test_data.py              ← unit tests for src/data.py
```

---

## 1. Prerequisites

### Python environment

Python 3.10 is required.  A virtual environment is recommended.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Key runtime dependencies: `torch`, `torchvision`, `pytorch-lightning`,
`torch-geometric`, `mmengine`, `mmcv`.

### MMPose

Clone into `external/` and install in editable mode:

```bash
git clone https://github.com/open-mmlab/mmpose.git external/mmpose
pip install -e external/mmpose
```

### Environment variables

Source `.env` before running any script.  It sets `PROJECT_DIR`, `MMPOSE`, and
adds MMPose to `PYTHONPATH`:

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

Run the prepare script to download MMPose-formatted annotations (including
pre-generated YOLOv3 detector boxes) from the OpenMMLab CDN:

```bash
python scripts/prepare_crowdpose.py
```

This downloads and extracts the following into `data/crowdpose/annotations/`:

| File | Split | Used by |
|------|-------|---------|
| `crowdpose_train.json` | train (10 000 images) | training |
| `crowdpose_val.json` | val (2 000 images) | training / validation |
| `crowdpose_test.json` | test (8 000 images) | top-down models, AP eval |
| `mmpose_crowdpose_test.json` | test | bottom-up models (DEKR) |
| `det_for_crowd_test_0.1_0.5.json` | test | pre-generated YOLOv3 boxes |

---

## 3. Extracting prediction caches

Before training the refinement model you need `.pkl` prediction caches that
the training script uses for in-training AP evaluation.  Caches are also
required to run the notebook.

```bash
# Extract RTMPose-m predictions on the test split (default)
source .env
python scripts/extract_predictions.py rtmpose-m_8xb64-210e_crowdpose-256x192

# Extract DEKR-HRNet-W32 predictions
source .env
python scripts/extract_predictions.py dekr-hrnet-w32_8xb10-300e_crowdpose-512x512

# Save to a custom path
source .env
python scripts/extract_predictions.py rtmpose-m_8xb64-210e_crowdpose-256x192 \
    --cache results/cache/my_rtmpose.pkl

# Force re-extraction (overwrite existing cache)
source .env
python scripts/extract_predictions.py rtmpose-m_8xb64-210e_crowdpose-256x192 --force

# List available models
source .env
python scripts/extract_predictions.py --list
```

Caches are written to `results/cache/<model-short-name>_predictions.pkl`.

> **Note — train / val splits:** the current REGISTRY configs point at the
> **test** split only.  Generating caches for train/val requires either a
> bottom-up model (no detector needed) or oracle-mode bbox files.  Add new
> `src/configs.py` entries pointing at `crowdpose_train.json` /
> `crowdpose_val.json` when needed.

### Pickle format

Each `.pkl` file is a dict with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `model_name` | `str` | Config key used to produce the cache |
| `architecture_type` | `"top-down"` \| `"bottom-up"` | Detector topology |
| `num_keypoints` | `int` | Number of keypoints per person (14 for CrowdPose) |
| `predictions` | `list[dict]` | One dict per image (see below) |

Each per-image dict contains `image_id`, `img_path`, `keypoints (N,K,2)`,
`keypoint_scores (N,K)`, and (top-down only) `bboxes (N,4)` and
`bbox_scores (N,)`.

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
from src.data import prepare_pose_graph_batch

# keypoints: FloatTensor[B, N, 14, 2]
# visibility: FloatTensor[B, N, 14]   (0/1/2)
# img_size: FloatTensor[B, 2]         (W, H) — optional, defaults to 800

graph = prepare_pose_graph_batch(keypoints, visibility, img_size=img_size)
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

### Quick start

```bash
python scripts/train_pose_refinement.py
```

All paths default to the project layout and model hyperparameters have sensible
defaults.  Checkpoints and logs are written under
`results/diffusion_training/<run-name>/<timestamp>/`.

### Custom experiment

```bash
python scripts/train_pose_refinement.py \
    --run-name exp_large \
    --batch-size 32 \
    --lr 3e-4 \
    --max-epochs 200 \
    --dim-hidden 256 --num-layers 6 \
    --devices 1
```

### Resume from checkpoint

```bash
python scripts/train_pose_refinement.py \
    --resume results/diffusion_training/my_run/20240101_120000/checkpoints/last.ckpt
```

### In-training AP evaluation

Pass prediction caches for the val and/or test splits to compute OKS-AP during
training.  The `--ap-eval-interval` flag controls how often the val cache is
evaluated (0 = disabled during training, always runs on test after training):

```bash
python scripts/train_pose_refinement.py \
    --val-pred-cache  results/cache/rtmpose-m_crowdpose_predictions.pkl \
    --test-pred-cache results/cache/rtmpose-m_crowdpose_predictions.pkl \
    --ap-eval-interval 5
```

### Key flags

| Group | Flag | Default | Description |
|-------|------|---------|-------------|
| **data** | `--data-root` | `data/` | Root for all dataset files |
| | `--train-ann` | `crowdpose/annotations/crowdpose_train.json` | Training annotation file (relative to `--data-root`) |
| | `--val-ann` | `crowdpose/annotations/crowdpose_val.json` | Validation annotation file |
| | `--test-ann` | `crowdpose/annotations/crowdpose_test.json` | Test annotation file |
| | `--val-pred-cache` | — | `.pkl` cache for in-training AP on val |
| | `--test-pred-cache` | — | `.pkl` cache for post-training AP on test |
| | `--ap-eval-interval` | `0` | Eval AP every N epochs (0 = disabled) |
| | `--num-workers` | `4` | DataLoader workers |
| **arch** | `--dim-hidden` | `128` | GNN hidden dim |
| | `--num-layers` | `4` | GNN depth |
| | `--num-train-steps` | `100` | Diffusion timesteps |
| | `--dim-pos/index/time/bone` | `128` | Embedding dims |
| **optimiser** | `--lr` | `1e-3` | Peak learning rate |
| | `--weight-decay` | `0.0` | AdamW weight decay |
| | `--cosine-lr` | off | Cosine annealing schedule |
| | `--inference-noise-frac` | `0.1` | DDIM starting noise fraction for AP eval |
| **training** | `--batch-size` | `256` | Images per batch |
| | `--max-epochs` | `100` | Maximum epochs |
| | `--grad-clip` | `1.0` | Gradient clipping norm (0 = disabled) |
| | `--early-stop-patience` | `0` | Epochs without improvement before stopping (0 = disabled) |
| | `--seed` | `42` | Global random seed |
| **hardware** | `--accelerator` | `auto` | `auto`, `gpu`, or `cpu` |
| | `--devices` | `auto` | e.g. `1`, `0,1`, or `auto` |
| | `--precision` | `32-true` | `32-true`, `16-mixed`, or `bf16-mixed` |
| **output** | `--run-name` | auto | Sub-directory name under `--out-dir` |
| | `--out-dir` | `results/diffusion_training/` | Root for checkpoints and logs |
| | `--resume` | — | Path to a `.ckpt` file to resume from |

---

## 6. Running a benchmark

### List available configurations

```bash
source .env
python scripts/benchmark_mmpose.py --list
```

### Run a benchmark

The checkpoint is downloaded automatically on first run.

```bash
source .env
python scripts/benchmark_mmpose.py rtmpose-m_8xb64-210e_crowdpose-256x192
```

Metrics are written to `results/<name>/metrics.yaml` when the run completes.

### Evaluation modes (top-down models)

| Mode | Flag | Description |
|------|------|-------------|
| Detector-box *(default)* | *(omit)* | Uses pre-generated YOLOv3 boxes — matches official model-zoo numbers |
| Oracle / GT-box | `--bbox-file none` | Uses ground-truth boxes — results will be higher than reported |

```bash
# Oracle mode
source .env
python scripts/benchmark_mmpose.py rtmpose-m_8xb64-210e_crowdpose-256x192 --bbox-file none
```

### Override individual fields

```bash
source .env
python scripts/benchmark_mmpose.py rtmpose-m_8xb64-210e_crowdpose-256x192 \
    --checkpoint checkpoints/my_finetuned.pth \
    --name my-finetune-run \
    --out results/my-finetune-run/metrics.yaml
```

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

Results are written to `results/<name>/metrics.yaml`.  Example:

```yaml
crowdpose/AP: 0.706
crowdpose/AP50: 0.843
crowdpose/AP75: 0.768
crowdpose/AP(E): 0.801
crowdpose/AP(M): 0.717
crowdpose/AP(H): 0.583
```
