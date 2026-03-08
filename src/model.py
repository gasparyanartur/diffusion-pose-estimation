"""Graph-based keypoint diffusion model.

Architecture overview
---------------------
Forward process
    Gaussian noise is added to root-relative, normalised keypoint positions.
    The noise scale is proportional to the normalised timestep t/T.

Backward process (denoising)
    A GNN (GAT) receives per-node features built from four sources:

    * position embedding  – linear projection of the noisy (x, y) position
    * keypoint-index embedding – learned embedding of the joint type (0…K-1)
    * timestep embedding  – sinusoidal embedding broadcast from batch to nodes
    * bone-vector embedding – edge bone vectors aggregated (mean) to nodes

    The GNN predicts a 2-D residual for every node (the denoised position).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch
import torch_geometric as geom
from torch import Tensor
from torch_geometric.utils import scatter

from src.data import prepare_pose_graph_batch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class KeypointDiffusionConfig:
    """Hyperparameters for :class:`KeypointDiffusionModel`."""

    name: str = "keypoint_diffusion"

    # Feature dimensions
    dim_pos: int = 128    # position-embedding output dim
    dim_index: int = 128  # keypoint-index embedding dim
    dim_time: int = 128   # timestep-embedding dim
    dim_bone: int = 128   # bone-vector embedding dim

    # GNN backbone
    dim_hidden: int = 128
    num_layers: int = 4

    # Diffusion schedule
    num_train_timesteps: int = 100


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class TimestepSinusoidalEmbedding(torch.nn.Module):
    """Sinusoidal embedding for a scalar integer timestep (batch of B values).

    Produces a (B, dim) float tensor whose columns are alternating cosine /
    sine functions of exponentially spaced frequencies — identical to the
    original DDPM / Transformer positional encoding.

    Args:
        dim: Output embedding dimension (must be even).
        max_period: Controls the minimum frequency (default 10 000).
    """

    def __init__(self, dim: int, max_period: int = 10_000) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.max_period = max_period

    def forward(self, timestep: Tensor) -> Tensor:
        """Args:
            timestep: LongTensor of shape (B,).

        Returns:
            FloatTensor of shape (B, dim).
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32, device=timestep.device)
            * np.log(self.max_period)
            / half_dim
        )  # (half_dim,)
        args = timestep[:, None].float() * freqs[None]   # (B, half_dim)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)


class KeypointDiffusionModel(torch.nn.Module):
    """Graph diffusion model over pose keypoints.

    Args:
        config: Model hyperparameters.
        num_keypoints: Number of keypoints per pose (default 15, canonical
            CrowdPose + virtual root node produced by
            :func:`src.data.prepare_pose_graph_batch`).
    """

    def __init__(
        self,
        config: KeypointDiffusionConfig,
        num_keypoints: int = 15,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_keypoints = num_keypoints

        # Per-node input projections
        self.position_embedding = torch.nn.Sequential(
            torch.nn.Linear(2, config.dim_pos),
            torch.nn.LayerNorm(config.dim_pos),
        )
        self.node_idx_embedding = torch.nn.Embedding(num_keypoints, config.dim_index)
        self.timestep_embedding = TimestepSinusoidalEmbedding(config.dim_time)
        self.bone_vector_embedding = torch.nn.Linear(2, config.dim_bone)

        # GNN backbone
        in_channels = config.dim_pos + config.dim_index + config.dim_time + config.dim_bone
        self.decoder = geom.nn.GAT(
            in_channels=in_channels,
            hidden_channels=config.dim_hidden,
            num_layers=config.num_layers,
            out_channels=2,
        )

    # ------------------------------------------------------------------
    # Forward / backward diffusion
    # ------------------------------------------------------------------

    def add_noise(
        self,
        node_pos: Tensor,
        timestep: Tensor,
        batch_idx: Tensor,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Add isotropic Gaussian noise scaled linearly with the timestep.

        Args:
            node_pos:  FloatTensor (B*N*K, 2) – clean node positions.
            timestep:  LongTensor  (B,)       – integer diffusion timestep.
            batch_idx: LongTensor  (B*N*K,)   – batch index per node (flat).

        Returns:
            noisy_node_pos: (B*N*K, 2)
            noise:          (B*N*K, 2)
        """
        noise = torch.randn_like(node_pos)                                     # (B*N*K, 2)
        noise_level = timestep.float() / self.config.num_train_timesteps       # (B,)
        noise_level = noise_level[batch_idx]                                   # (B*N*K,)
        noise = noise * noise_level[:, None]                                   # (B*N*K, 2)
        noisy_node_pos = node_pos + noise
        return noisy_node_pos, noise

    def backward_process(
        self,
        noisy_node_pos: Tensor,
        edge_index: Tensor,
        node_idx: Tensor,
        batch_idx: Tensor,
        timestep: Tensor,
        **kwargs,
    ) -> Tensor:
        """Predict the denoised position for every node.

        Args:
            noisy_node_pos: FloatTensor (B*N*K, 2)  – noisy node positions.
            edge_index:     LongTensor  (2, B*N*E)  – block-diagonal edge index.
            node_idx:       LongTensor  (B*N*K,)    – keypoint-type index per node.
            batch_idx:      LongTensor  (B*N*K,)    – batch index per node (flat).
            timestep:       LongTensor  (B,)        – diffusion timestep.

        Returns:
            FloatTensor (B*N*K, 2) – predicted clean (denoised) positions.
        """
        num_nodes = noisy_node_pos.shape[0]

        # 1. Position embedding
        pos_emb = self.position_embedding(noisy_node_pos)   # (B*N*K, dim_pos)

        # 2. Keypoint-index embedding
        idx_emb = self.node_idx_embedding(node_idx)         # (B*N*K, dim_index)

        # 3. Timestep embedding broadcast from batch level to node level
        timestep_emb = self.timestep_embedding(timestep)    # (B, dim_time)
        timestep_emb = timestep_emb[batch_idx]              # (B*N*K, dim_time)

        # 4. Bone-vector embedding aggregated back to nodes
        #    Bone vectors are naturally edge features; we mean-pool them per
        #    destination node so they become node features of the right size.
        edge_src = edge_index[0]                                                    # (B*N*E,)
        edge_dst = edge_index[1]                                                    # (B*N*E,)
        bone_vectors = noisy_node_pos[edge_dst] - noisy_node_pos[edge_src]          # (B*N*E, 2)
        bone_emb_edges = self.bone_vector_embedding(bone_vectors)                   # (B*N*E, dim_bone)
        bone_emb = scatter(                                                         # (B*N*K, dim_bone)
            bone_emb_edges, edge_dst, dim=0, dim_size=num_nodes, reduce="mean"
        )

        # 5. Concatenate all node features and run GNN
        x = torch.cat([pos_emb, idx_emb, timestep_emb, bone_emb], dim=-1)  # (B*N*K, in_channels)
        out = self.decoder(x, edge_index)                                   # (B*N*K, 2)
        return out


# ---------------------------------------------------------------------------
# Lightning training module
# ---------------------------------------------------------------------------


class PoseDiffusionModule(pl.LightningModule):
    """PyTorch-Lightning wrapper for :class:`KeypointDiffusionModel`.

    Training loop
    -------------
    For each batch:

    1. Build a canonical pose-graph with :func:`src.data.prepare_pose_graph_batch`.
    2. Sample a random diffusion timestep per image.
    3. Add noise to node positions proportional to the timestep.
    4. Run the backward process to predict the clean positions.
    5. Compute MSE loss between predictions and the clean positions.
    """

    def __init__(self, model: KeypointDiffusionModel) -> None:
        super().__init__()
        self.model = model

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        keypoints = batch["keypoints"]   # (B, N, K, 2)
        visibility = batch["visibility"] # (B, N, K)

        pose_graph_batch = prepare_pose_graph_batch(keypoints, visibility)

        B = keypoints.shape[0]
        node_pos   = pose_graph_batch["node_pos"]               # (B*N*K, 2)
        node_index = pose_graph_batch["pose_idx"].reshape(-1)   # (B*N*K,)
        batch_index = pose_graph_batch["batch"]                 # (B*N*K,) — flat PyG batch vector
        edge_index  = pose_graph_batch["edge_index"]            # (2, B*N*E)

        timestep = torch.randint(
            0,
            self.model.config.num_train_timesteps,
            (B,),
            device=node_pos.device,
        )  # (B,)

        noisy_node_pos, noise = self.model.add_noise(
            node_pos=node_pos,
            timestep=timestep,
            batch_idx=batch_index,
        )

        pred = self.model.backward_process(
            noisy_node_pos=noisy_node_pos,
            edge_index=edge_index,
            node_idx=node_index,
            batch_idx=batch_index,
            timestep=timestep,
        )

        loss = torch.nn.functional.mse_loss(pred, node_pos)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
