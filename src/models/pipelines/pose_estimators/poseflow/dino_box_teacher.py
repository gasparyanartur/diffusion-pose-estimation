from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.ops import roi_align


@dataclass
class BoxFeatureTargets:
    features: Tensor
    valid: Tensor


@torch.no_grad()
def pool_dino_boxes(
    feature_map: Tensor, boxes: Tensor, image_hw: tuple[int, int], *,
    grid_size: int = 7, patch_size: int = 16,
) -> BoxFeatureTargets:
    height, width = image_hw
    channels = feature_map.shape[0]
    features = feature_map.new_zeros(boxes.shape[0], grid_size * grid_size, channels)
    boxes = boxes.detach().to(feature_map)
    finite = torch.isfinite(boxes).all(-1)
    center, size = boxes[:, :2], boxes[:, 2:]
    lower = (center - size / 2).clamp(0, 1)
    upper = (center + size / 2).clamp(0, 1)
    valid = finite & (size > 0).all(-1) & (upper > lower).all(-1)
    if valid.any():
        scale = boxes.new_tensor([width, height]) / patch_size
        coordinates = torch.cat((lower[valid] * scale, upper[valid] * scale), dim=-1)
        pooled = roi_align(feature_map[None], [coordinates], output_size=grid_size,
                           spatial_scale=1., sampling_ratio=2, aligned=True)
        features[valid] = pooled.flatten(2).transpose(1, 2)
        if not torch.isfinite(features[valid]).all():
            raise ValueError('DINO teacher produced non-finite valid region features')
    return BoxFeatureTargets(features, valid)


@torch.no_grad()
def build_box_targets(
    network, images: Tensor, image_hw: Tensor, boxes: list[Tensor], *,
    grid_size: int = 7, patch_size: int = 16, skip_tokens: int = 5,
) -> list[BoxFeatureTargets]:
    if len(boxes) != images.shape[0] or image_hw.shape != (images.shape[0], 2):
        raise ValueError('Teacher image sizes and target boxes must match the batch')
    results = []
    network.eval()
    with torch.autocast(device_type=images.device.type, enabled=False):
        for index, image_boxes in enumerate(boxes):
            height, width = (int(value) for value in image_hw[index])
            if min(height, width) < 1 or height > images.shape[-2] or width > images.shape[-1]:
                raise ValueError('Invalid unpadded DINO image size')
            if not len(image_boxes):
                channels = getattr(getattr(network, 'config', None), 'hidden_size', 384)
                results.append(BoxFeatureTargets(
                    images.new_zeros((0, grid_size * grid_size, channels), dtype=torch.float32),
                    torch.empty(0, dtype=torch.bool, device=images.device)))
                continue
            # Native batches are already ImageNet-normalized. Pad with normalized zero.
            image = images[index:index + 1, :, :height, :width].float()
            image = F.pad(image, (0, (-width) % patch_size, 0, (-height) % patch_size))
            rows, columns = image.shape[-2] // patch_size, image.shape[-1] // patch_size
            tokens = network(image).last_hidden_state[:, skip_tokens:]
            if tokens.shape[1] != rows * columns:
                raise ValueError('DINO patch tokens do not match the padded image grid')
            spatial = tokens[0].transpose(0, 1).reshape(-1, rows, columns)
            results.append(pool_dino_boxes(spatial, image_boxes, (height, width),
                                           grid_size=grid_size, patch_size=patch_size))
    return results


class FrozenDinoBoxTeacher:
    def __init__(self, model_name: str, revision: str, expected_sha256: str, *, grid_size: int = 7):
        from huggingface_hub import snapshot_download
        from transformers import AutoModel

        snapshot = Path(snapshot_download(model_name, revision=revision, local_files_only=True))
        weights = snapshot / 'model.safetensors'
        checksum = hashlib.sha256()
        with weights.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                checksum.update(chunk)
        if checksum.hexdigest() != expected_sha256:
            raise ValueError('DINO checkpoint SHA256 does not match the configured teacher')
        with torch.random.fork_rng(devices=[]):
            self.network = AutoModel.from_pretrained(snapshot, local_files_only=True)
        self.network.eval().requires_grad_(False)
        self.patch_size = self.network.config.patch_size
        self.skip_tokens = 1 + self.network.config.num_register_tokens
        if self.patch_size != 16 or self.network.config.hidden_size != 384:
            raise ValueError('V25 requires the pinned DINOv3 ViT-S/16 teacher')
        self.grid_size = grid_size
        self.sha256 = expected_sha256

    def targets(self, images: Tensor, image_hw: Tensor, boxes: list[Tensor]) -> list[BoxFeatureTargets]:
        self.network.to(device=images.device, dtype=torch.float32).eval()
        return build_box_targets(self.network, images, image_hw, boxes,
                                 grid_size=self.grid_size, patch_size=self.patch_size,
                                 skip_tokens=self.skip_tokens)
