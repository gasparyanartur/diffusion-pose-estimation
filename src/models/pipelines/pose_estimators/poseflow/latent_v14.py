import torch
from torch import Tensor, nn

from ..diffusionregpose.transformer import MLP


class V14StateCodec(nn.Module):
    def __init__(self, *, width: int, joints: int):
        super().__init__()
        self.joints = joints
        self.box_encoder = MLP(4, width, width, 2)
        self.joint_encoder = MLP(2, width, width, 2)
        self.roles = nn.Embedding(joints + 1, width)
        self.clean_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.center_head = MLP(width, width, 2, 3)
        self.size_head = MLP(width, width, 2, 3)
        self.existence_head = MLP(width, width, 1, 3)
        self.point_head = MLP(width, width, 2, 3)

    def encode(self, targets: dict[str, Tensor]) -> Tensor:
        tokens = torch.cat((self.box_encoder(targets['boxes'])[:, :, None],
                            self.joint_encoder(targets['joints'])), dim=2)
        return self.clean_norm(tokens + self.roles.weight[None, None])

    def decode(self, latents: Tensor) -> dict[str, Tensor]:
        box_token = latents[:, :, 0]
        points = self.point_head(latents[:, :, 1:]).sigmoid()
        return {
            'pred_logits': self.existence_head(box_token),
            'pred_boxes': torch.cat((self.center_head(box_token).sigmoid(),
                                    self.size_head(box_token).sigmoid().clamp_min(1e-6)), -1),
            'pred_keypoints': torch.cat((points.flatten(-2), torch.ones_like(points[..., 0])), -1),
        }
