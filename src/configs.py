"""Predefined benchmark configurations for src.benchmark.run_benchmark."""

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
