"""
Predefined benchmark configurations.

Each entry is a plain dict with the keys consumed by src.benchmark.run_benchmark.
Add new dicts here to register additional model + dataset combinations.

Keys
----
mmpose_dir       : Path  – root of the MMPose source clone
config_path      : Path  – absolute path to the MMPose model config (.py)
checkpoint       : Path  – local checkpoint file (.pth)
checkpoint_url   : str | None – download URL; None to skip auto-download
data_root        : Path  – root directory for dataset files
ann_file         : str   – annotation file, relative to data_root
img_prefix       : str   – image directory, relative to data_root
metainfo_config  : Path  – absolute path to dataset metainfo config (.py)
bbox_file        : str | None – detector-box file, relative to data_root;
                               None → GT bounding boxes (oracle mode)
"""

from src import defaults

# DEKR  ×  HRNet-w32  ×  CrowdPose
DEKR_W32_CROWDPOSE_300: dict = {
    "name": "dekr-hrnet-w32_8xb10-300e_crowdpose-512x512",
    "mmpose_dir": defaults.MMPOSE_DIR,
    "config_path": (
        defaults.MMPOSE_DIR
        / "configs/body_2d_keypoint/dekr/crowdpose"
        / "dekr_hrnet-w32_8xb10-300e_crowdpose-512x512.py"
    ),
    "checkpoint": defaults.CHECKPOINTS_DIR / "dekr-hrnet-w32_crowdpose.pth",
    "checkpoint_url": (
        "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/dekr/crowdpose/"
        "dekr_hrnet-w32_8xb10-300e_crowdpose-512x512_147bae97-20221228.pth"
    ),
    "data_root": defaults.DATA_ROOT,
    "ann_file": "crowdpose/annotations/mmpose_crowdpose_test.json",
    "img_prefix": "crowdpose/images/",
    "metainfo_config": defaults.MMPOSE_DIR / "configs/_base_/datasets/crowdpose.py",
    "bbox_file": None,  # bottom-up: no person detector needed
}

# RTMPose-M  ×  CrowdPose
RTMPOSE_M_CROWDPOSE_210: dict = {
    "name": "rtmpose-m_8xb64-210e_crowdpose-256x192",
    "mmpose_dir": defaults.MMPOSE_DIR,
    "config_path": (
        defaults.MMPOSE_DIR
        / "configs/body_2d_keypoint/rtmpose/crowdpose"
        / "rtmpose-m_8xb64-210e_crowdpose-256x192.py"
    ),
    "checkpoint": defaults.CHECKPOINTS_DIR / "rtmpose-m_crowdpose.pth",
    "checkpoint_url": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
        "rtmpose-m_simcc-crowdpose_pt-aic-coco_210e-256x192-e6192cac_20230224.pth"
    ),
    "data_root": defaults.DATA_ROOT,
    "ann_file": "crowdpose/annotations/crowdpose_test.json",
    "img_prefix": "crowdpose/images/",
    "metainfo_config": defaults.MMPOSE_DIR / "configs/_base_/datasets/crowdpose.py",
    "bbox_file": "crowdpose/annotations/det_for_crowd_test_0.1_0.5.json",
}

ALL_CONFIGS = [DEKR_W32_CROWDPOSE_300, RTMPOSE_M_CROWDPOSE_210]
REGISTRY: dict[str, dict] = {
    cfg["name"]: cfg
    for cfg in ALL_CONFIGS
}
