"""Dummy components for end-to-end pipeline validation.

  - :class:`DummyPoseDataset` — always emits one fixed 14-keypoint pose.
  - :class:`CenterPatchExtractor` — extracts a fixed-size center patch from an image.
  - :class:`CenterPatchEmbedder` — flattens that patch into per-node features.
  - :class:`ShiftDecoder` — predicts a single small (x, y) shift applied to all joints.

These are intentionally trivial; their purpose is to verify that the
extractor → embedder → decoder pipeline composes and runs end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from src.base_config import BaseConfig
from src.data.graph import PoseGraphBatch
from src.decoders.base import BaseDecoder, BaseDecoderConfig
from src.embedders.base import BaseFeatureEmbedder, BaseFeatureEmbedderConfig
from src.features.base import BaseFeatureExtractor, BaseFeatureExtractorConfig


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# 14 keypoints in pixel space, roughly representing a stick figure inside
# a 256×256 image. Order matches CROWDPOSE_JOINT_NAMES.
DUMMY_KEYPOINTS_PIXEL = torch.tensor(
    [
        [ 80, 100], [176, 100],   # shoulders
        [ 64, 140], [192, 140],   # elbows
        [ 56, 180], [200, 180],   # wrists
        [104, 160], [152, 160],   # hips
        [ 96, 200], [160, 200],   # knees
        [ 88, 240], [168, 240],   # ankles
        [128,  60],               # head
        [128,  90],               # neck
    ],
    dtype=torch.float32,
)


@dataclass
class DummyPoseSample:
    image_id: int
    img_path: str
    keypoints: Tensor      # (1, 14, 2)
    visibility: Tensor     # (1, 14)
    image: Tensor | None   # (3, H, W) or None


class DummyPoseDataset(Dataset):
    """Emits one fixed 14-keypoint pose every time."""

    def __init__(self, length: int = 16, image_size: int = 256) -> None:
        self.length = length
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img = torch.zeros(3, self.image_size, self.image_size)
        # Paint the keypoints onto the dummy image so center-patch sees something.
        for x, y in DUMMY_KEYPOINTS_PIXEL.long().tolist():
            x = max(0, min(self.image_size - 1, x))
            y = max(0, min(self.image_size - 1, y))
            img[:, y, x] = 1.0
        return {
            "image_id": idx,
            "img_path": f"<dummy:{idx}>",
            "keypoints": DUMMY_KEYPOINTS_PIXEL.unsqueeze(0).clone(),  # (1, 14, 2)
            "visibility": torch.full((1, 14), 2.0),
            "image": img,
            "image_size": (self.image_size, self.image_size),
        }


# ---------------------------------------------------------------------------
# Feature extractor: center-patch
# ---------------------------------------------------------------------------


class CenterPatchExtractor(BaseFeatureExtractor):
    """Extracts a fixed-size center patch from each image and pools to a vector."""

    def __init__(self, patch_size: int = 32, out_dim: int = 32) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.proj = nn.Linear(3 * patch_size * patch_size, out_dim)
        self._images: dict[int, Tensor] = {}  # populated externally for tests

    def register_image(self, image_id: int, image: Tensor) -> None:
        self._images[image_id] = image

    def forward(
        self,
        image: Tensor | None = None,
        image_ids: list[int] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if image_ids is None:
            raise ValueError("CenterPatchExtractor requires image_ids")
        patches = []
        for iid in image_ids:
            img = self._images[iid]                     # (3, H, W)
            _, H, W = img.shape
            ps = self.patch_size
            cy, cx = H // 2, W // 2
            patch = img[:, cy - ps // 2 : cy + ps // 2, cx - ps // 2 : cx + ps // 2]
            patches.append(patch)
        stacked = torch.stack(patches, dim=0)            # (B, 3, ps, ps)
        flat = stacked.flatten(1)
        global_vec = self.proj(flat)                     # (B, out_dim)
        return {"global": global_vec}


class CenterPatchExtractorConfig(BaseFeatureExtractorConfig):
    patch_size: int = 32
    out_dim: int = 32

    def build(self) -> CenterPatchExtractor:
        return CenterPatchExtractor(patch_size=self.patch_size, out_dim=self.out_dim)


# ---------------------------------------------------------------------------
# Embedder: broadcast global patch vector to every node
# ---------------------------------------------------------------------------


class BroadcastEmbedder(BaseFeatureEmbedder):
    """Broadcasts the patch's global vector to every (B, N, K) node."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        features: dict[str, Any],
        kpt_pixel: Tensor | None = None,
        img_hw: Tensor | None = None,
        joint_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        if kpt_pixel is None:
            raise ValueError("BroadcastEmbedder requires kpt_pixel")
        B, N, K, _ = kpt_pixel.shape
        global_vec = self.proj(features["global"])       # (B, out_dim)
        node_feats = global_vec.view(B, 1, 1, -1).expand(B, N, K, -1).contiguous()
        return {"node_feats": node_feats}


class BroadcastEmbedderConfig(BaseFeatureEmbedderConfig):
    in_dim: int = 32
    out_dim: int = 32

    def build(self) -> BroadcastEmbedder:
        return BroadcastEmbedder(in_dim=self.in_dim, out_dim=self.out_dim)


# ---------------------------------------------------------------------------
# Decoder: predict a single (x, y) shift applied to all joints
# ---------------------------------------------------------------------------


class ShiftDecoder(BaseDecoder):
    """Predicts a single small (dx, dy) shift from pooled embeddings.

    If ``initial_keypoints`` is None, returns zeros (the prior). Otherwise,
    returns ``initial_keypoints + shift`` broadcast across nodes.
    """

    def __init__(self, in_dim: int, max_shift: float = 0.05) -> None:
        super().__init__()
        self.max_shift = max_shift
        self.head = nn.Linear(in_dim, 2)
        self.head.weight.data.zero_()
        self.head.bias.data.zero_()

    def forward(
        self,
        embeddings: dict[str, Tensor],
        graph: PoseGraphBatch | None = None,
        initial_keypoints: Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        if graph is None:
            raise ValueError("ShiftDecoder requires graph")
        node_feats = embeddings["node_feats"]            # (B, N, K, D)
        # Pool to per-batch and predict one shift per batch element.
        pooled = node_feats.mean(dim=(1, 2))             # (B, D)
        shift = torch.tanh(self.head(pooled)) * self.max_shift  # (B, 2)

        # Broadcast the shift to every node in each batch.
        shift_per_node = shift[graph.node_batch]         # (M, 2)
        if initial_keypoints is None:
            kpts_flat = shift_per_node
        else:
            kpts_flat = initial_keypoints + shift_per_node
        keypoints = kpts_flat.view(graph.B, graph.N, graph.K, 2)
        return {
            "keypoints": keypoints,
            "scores": keypoints.new_ones(graph.B, graph.N),
            "keypoint_scores": keypoints.new_ones(graph.B, graph.N, graph.K),
        }


class ShiftDecoderConfig(BaseDecoderConfig):
    in_dim: int = 32
    max_shift: float = 0.05

    def build(self) -> ShiftDecoder:
        return ShiftDecoder(in_dim=self.in_dim, max_shift=self.max_shift)
