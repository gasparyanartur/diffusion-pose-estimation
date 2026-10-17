import torch
from torch import nn

from ..diffusionregpose.transformer import MLP


class V16StateCodec(nn.Module):
    box_role_count = 3
    latent_role_slices = [('confidence', slice(0, 1)), ('center', slice(1, 2)),
                          ('size', slice(2, 3)), ('joints', slice(3, None))]

    def __init__(self, *, width, joints):
        super().__init__()
        self.joints = joints
        self.existence_encoder = MLP(1, width, width, 2)
        self.center_encoder = MLP(2, width, width, 2)
        self.size_encoder = MLP(2, width, width, 2)
        self.joint_encoder = MLP(2, width, width, 2)
        self.roles = nn.Embedding(joints+3, width)
        self.clean_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.existence_head = MLP(width, width, 1, 3)
        self.center_head = MLP(width, width, 2, 3)
        self.size_head = MLP(width, width, 2, 3)
        self.point_head = MLP(width, width, 2, 3)

    def encode(self, targets):
        tokens = torch.cat((self.existence_encoder(targets['existence'])[:, :, None],
                            self.center_encoder(targets['boxes'][..., :2])[:, :, None],
                            self.size_encoder(targets['boxes'][..., 2:])[:, :, None],
                            self.joint_encoder(targets['joints'])), dim=2)
        return self.clean_norm(tokens+self.roles.weight[None, None])

    def decode(self, latents):
        points = self.point_head(latents[:, :, 3:]).sigmoid()
        return {
            'pred_logits': self.existence_head(latents[:, :, 0]),
            'pred_boxes': torch.cat((self.center_head(latents[:, :, 1]).sigmoid(),
                                    self.size_head(latents[:, :, 2]).sigmoid().clamp_min(1e-6)), -1),
            'pred_keypoints': torch.cat((points.flatten(-2), torch.ones_like(points[..., 0])), -1),
        }
