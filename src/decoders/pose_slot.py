"""Pose-slot decoder for set prediction.

Reads a global feature ``[B, D]`` and predicts a fixed set of ``N`` pose
slots, each with a keypoint set, an objectness score, and per-keypoint
visibility.

Output keys (see ``docs/assumptions.md``):
    keypoints       [B, N, K, 2]  canonical (-1, 1)
    scores          [B, N]        sigmoid(objectness logits) — per-slot
    scores_logits   [B, N]        raw objectness logits (for losses)
    keypoint_scores [B, N, K]     sigmoid(visibility logits) — per-keypoint
    vis_logits      [B, N, K]     raw visibility logits (for losses)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from src.data.graph import PoseGraphBatch
from src.decoders.base import BaseDecoder, BaseDecoderConfig


class PoseSlotDecoder(BaseDecoder):
    """MLP head producing ``N`` independent pose slots."""

    def __init__(
        self,
        in_dim: int,
        num_slots: int = 20,
        num_keypoints: int = 14,
        hidden_dim: int = 512,
        feature_key: str = "feature",
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.num_keypoints = num_keypoints
        self.feature_key = feature_key
        # per slot: 1 objectness + K*2 keypoints + K visibility
        out_dim = num_slots * (1 + num_keypoints * 2 + num_keypoints)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        embeddings: dict[str, Tensor],
        graph: PoseGraphBatch | None = None,         # noqa: ARG002
        initial_keypoints: Tensor | None = None,     # noqa: ARG002
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        feat = embeddings[self.feature_key]
        B = feat.shape[0]
        N, K = self.num_slots, self.num_keypoints
        out = self.mlp(feat)
        obj_logits = out[:, :N].contiguous()                           # [B, N]
        kpt_logits = out[:, N : N + N * K * 2].view(B, N, K, 2)        # [B, N, K, 2]
        vis_logits = out[:, N + N * K * 2 :].view(B, N, K)             # [B, N, K]

        # canonical (-1, 1) keypoints from sigmoid([0,1]) -> [-1, 1]
        keypoints = torch.sigmoid(kpt_logits) * 2.0 - 1.0
        return {
            "keypoints": keypoints,
            "scores": torch.sigmoid(obj_logits),
            "scores_logits": obj_logits,
            "keypoint_scores": torch.sigmoid(vis_logits),
            "vis_logits": vis_logits,
        }


class PoseSlotDecoderConfig(BaseDecoderConfig):
    in_dim: int = 256
    num_slots: int = 20
    num_keypoints: int = 14
    hidden_dim: int = 512
    feature_key: str = "feature"

    def build(self) -> PoseSlotDecoder:
        return PoseSlotDecoder(
            in_dim=self.in_dim,
            num_slots=self.num_slots,
            num_keypoints=self.num_keypoints,
            hidden_dim=self.hidden_dim,
            feature_key=self.feature_key,
        )
