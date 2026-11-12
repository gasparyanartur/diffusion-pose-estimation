import torch
from torch import nn

from .model_v32 import PoseFlowV32Model


class PersonTokenDecoder(nn.Module):
    def __init__(self, width, *, joints, heads, feedforward_dim, layers=3):
        super().__init__()
        self.width = width
        self.token_count = joints+3
        self.expansion = nn.Linear(width, self.token_count*width, bias=False)
        self.roles = nn.Parameter(torch.empty(self.token_count, width))
        nn.init.normal_(self.roles, std=.02)
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(width, heads,
            dim_feedforward=feedforward_dim, dropout=0., activation='gelu',
            batch_first=True, norm_first=True) for _ in range(layers))
        self.norm = nn.LayerNorm(width)

    def forward(self, latents):
        # Flatten images and people, keeping each person's tokens in its own sequence.
        state = self.expansion(latents).reshape(-1, self.token_count, self.width)+self.roles
        for layer in self.layers:
            state = layer(state)
        return self.norm(state).reshape(*latents.shape[:-1], self.token_count, self.width)


class PoseFlowV33Model(PoseFlowV32Model):
    def __init__(self, *, pose_readout_layers=3, **kwargs):
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            self.pose_decoder = PersonTokenDecoder(self.hidden_dim, joints=self.num_body_points,
                heads=kwargs.get('num_heads', 8), feedforward_dim=kwargs.get('feedforward_dim', 2048),
                layers=pose_readout_layers)
            self.joint_head = nn.Linear(self.hidden_dim, 2)
            self.box_head = nn.Linear(self.hidden_dim, 4)
            self.existence_head = nn.Linear(self.hidden_dim, 1)

    def readout(self, latents):
        state = self.pose_decoder(latents)
        joints = self.joint_head(state[..., :self.num_body_points, :]).sigmoid()
        boxes = self.box_head(state[..., self.num_body_points, :]).sigmoid()
        existence = self.existence_head(state[..., self.num_body_points+1, :]).squeeze(-1)
        return dict(pred_latents=latents, existence_logits=existence,
            pred_logits=torch.stack((torch.full_like(existence, -20.), existence), -1),
            pred_boxes=boxes, joint_xy=joints,
            pred_keypoints=torch.cat((joints.flatten(-2), torch.ones_like(joints[..., 0])), -1))
