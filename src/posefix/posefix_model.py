import torch
import torch.nn as nn
import torchvision.models as models


class PoseFix(nn.Module):

    def __init__(self, num_joints=14):

        super().__init__()

        backbone = models.resnet50(weights=None)

        backbone.conv1 = nn.Conv2d(
            3 + num_joints,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        self.head = nn.Conv2d(2048, num_joints, 1)

    def forward(self, img, pose_heatmap):

        x = torch.cat([img, pose_heatmap], dim=1)

        feat = self.backbone(x)

        out = self.head(feat)

        return out