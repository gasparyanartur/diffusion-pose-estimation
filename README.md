# pose-graph-diff

Benchmarking and experimentation toolkit for 2D human pose estimation, built on top of [MMPose](https://github.com/open-mmlab/mmpose).

---

## Project layout

```
data/
    crowdpose/
        annotations/     ← downloaded by prepare_crowdpose.py
        images/          ← place dataset images here manually
checkpoints/             ← model weights (auto-downloaded on first run)
external/
    mmpose/              ← MMPose source clone ($MMPOSE)
results/
    <benchmark-name>/
        metrics.yaml     ← evaluation output
        work_dir/        ← mmengine logs
scripts/
    benchmark_mmpose.py  ← main benchmark runner
    prepare_crowdpose.py ← downloads CrowdPose annotations
src/
    defaults.py          ← base directory constants ($PROJECT_DIR, $MMPOSE)
    configs.py           ← predefined benchmark configurations (REGISTRY)
    benchmark.py         ← shared runner logic
```

---

## 1. Prerequisites

**Python environment** — Python 3.10, with a virtual environment recommended.

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install mmengine mmcv
```

**MMPose** — clone into `external/`:

```bash
git clone https://github.com/open-mmlab/mmpose.git external/mmpose
pip install -e external/mmpose
```

**Environment variables** — source `.env` before running any script.  It sets
`PROJECT_DIR`, `MMPOSE`, and adds MMPose to `PYTHONPATH`:

```bash
source .env
```

> Add `source .env` to your shell rc or VS Code workspace settings to avoid
> running it manually every session.

---

## 2. Dataset setup — CrowdPose

### 2a. Images

Download the CrowdPose images from the [official release](https://github.com/Jeff-sjtu/CrowdPose) and place them flat in `data/crowdpose/images/`:

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

| File | Used by |
|------|---------|
| `crowdpose_test.json` | top-down models (RTMPose, HRNet, …) |
| `mmpose_crowdpose_test.json` | bottom-up models (DEKR, …) |
| `det_for_crowd_test_0.1_0.5.json` | pre-generated YOLOv3 boxes for top-down eval |
| `mmpose_crowdpose_train/val/trainval.json` | training |

---

## 3. Running a benchmark

### List available configurations

```bash
source .env
python scripts/benchmark_mmpose.py --list
```

Example output:

```
Registered configurations:
  dekr-hrnet-w32_8xb10-300e_crowdpose-512x512
  rtmpose-m_8xb64-210e_crowdpose-256x192
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
# oracle mode
python scripts/benchmark_mmpose.py rtmpose-m_8xb64-210e_crowdpose-256x192 --bbox-file none
```

### Override individual fields

Any value in the registered config can be overridden at the command line:

```bash
python scripts/benchmark_mmpose.py rtmpose-m_8xb64-210e_crowdpose-256x192 \
    --checkpoint checkpoints/my_finetuned.pth \
    --name my-finetune-run \
    --out results/my-finetune-run/metrics.yaml
```

---

## 4. Adding a new benchmark configuration

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
    # set bbox_file to None for bottom-up models (no person detector)
}

ALL_CONFIGS = [..., MY_MODEL_CROWDPOSE]
```

The new entry is immediately available via `--list` and as a positional argument to the benchmark script.

---

## 5. Results

Results are written to `results/<name>/metrics.yaml`.  Example:

```yaml
crowdpose/AP: 0.706
crowdpose/AP50: 0.843
crowdpose/AP75: 0.768
crowdpose/AP(E): 0.801
crowdpose/AP(M): 0.717
crowdpose/AP(H): 0.583
```
