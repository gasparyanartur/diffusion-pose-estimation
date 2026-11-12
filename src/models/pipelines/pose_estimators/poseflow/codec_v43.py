from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from src.models.pipelines.pose_vae.fidelity import FidelityPoseVAE


CODEC_CHECKPOINT = '/storage/outputs/pose-graph-diff/results/pose_vae_comparison/20261104_233335/sample_kl_high_clean/checkpoints/epoch099-score97.2048.ckpt'
CODEC_SHA256 = '7168bffc7452f83e5e7c70b062d55ff867d2cb7d1eb0932cd1339cdc9e0ee951'
GEOMETRY_FIELDS = ('keypoints_px', 'boxes_cxcywh_px', 'joint_mask', 'image_wh')


class PosteriorMoments:
    def __init__(self, latent_dim):
        self.count = 0
        self.total = torch.zeros(latent_dim, dtype=torch.float64)
        self.second = torch.zeros_like(self.total)

    def update(self, mean, logvar):
        if mean.ndim != 2 or mean.shape != logvar.shape or mean.shape[-1] != len(self.total):
            raise ValueError('Posterior dimensions disagree with moments')
        mean, logvar = mean.detach().double().cpu(), logvar.detach().double().cpu()
        second = mean.square()+logvar.exp()
        if not torch.isfinite(mean).all() or not torch.isfinite(second).all():
            raise ValueError('Posterior moments must be finite')
        self.count += len(mean)
        self.total += mean.sum(0)
        self.second += second.sum(0)

    def finalize(self):
        if not self.count:
            raise ValueError('Cannot calculate moments of an empty posterior')
        mean = self.total/self.count
        return mean, (self.second/self.count-mean.square()).clamp_min(1e-12).sqrt()


class FrozenPoseCodec(nn.Module):
    def __init__(self, model, mean=None, std=None):
        super().__init__()
        self.model = model.requires_grad_(False).eval()
        self.latent_dim = model.latent_dim
        self.register_buffer('latent_mean', torch.zeros(self.latent_dim))
        self.register_buffer('latent_std', torch.ones(self.latent_dim))
        self.register_buffer('statistics_ready', torch.tensor(False))
        if mean is not None or std is not None:
            self.set_statistics(mean, std)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def set_statistics(self, mean, std):
        if (mean is None or std is None or mean.shape != self.latent_mean.shape
                or std.shape != mean.shape or not torch.isfinite(mean).all()
                or not torch.isfinite(std).all() or (std <= 0).any()):
            raise ValueError('Expected finite latent-dimension means and positive standard deviations')
        self.latent_mean.copy_(mean.to(self.latent_mean))
        self.latent_std.copy_(std.to(self.latent_std).clamp_min(1e-6))
        self.statistics_ready.fill_(True)

    def _require_statistics(self):
        if not self.statistics_ready:
            raise RuntimeError('Frozen codec latent statistics have not been initialized')

    @torch.no_grad()
    def encode_posterior(self, geometry):
        return self.model.encode(**{name: geometry[name] for name in GEOMETRY_FIELDS})

    @torch.no_grad()
    def sample_latents(self, geometry, *, generator):
        self._require_statistics()
        mean, logvar = self.encode_posterior(geometry)
        noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype,
                            device=generator.device).to(mean.device)
        return self.standardize(mean+(.5*logvar).exp()*noise)

    def standardize(self, latent):
        self._require_statistics()
        return (latent-self.latent_mean)/self.latent_std

    def unstandardize(self, latent):
        self._require_statistics()
        return self.latent_mean+self.latent_std*latent

    def decode_latents(self, latent, image_wh):
        return self.model.decode(self.unstandardize(latent), image_wh)


def load_frozen_codec(checkpoint=CODEC_CHECKPOINT, *, expected_sha256=CODEC_SHA256,
                      width=256, latent_dim=32):
    checkpoint = Path(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError('Frozen codec checkpoint hash mismatch')
    record = torch.load(checkpoint, map_location='cpu', weights_only=False)
    parameters = record['hyper_parameters']
    if parameters['width'] != width or parameters['latent_dim'] != latent_dim:
        raise ValueError('Frozen codec checkpoint dimension mismatch')
    with torch.random.fork_rng(devices=[]):
        model = FidelityPoseVAE(width=width, latent_dim=latent_dim)
    state = {name.removeprefix('model.'): value for name, value in record['state_dict'].items()
             if name.startswith('model.')}
    model.load_state_dict(state, strict=True)
    return FrozenPoseCodec(model)


def geometry_from_batch(batch):
    # B images, P padded people; output rows retain their [image, person] indices.
    boxes = batch['gt_boxes_pixel']
    labeled = batch['gt_labeled'].bool()
    image_wh = batch['img_hw'].flip(-1).to(boxes)
    if not torch.isfinite(image_wh).all() or (image_wh <= 0).any():
        raise ValueError('Expected finite positive unpadded image dimensions')
    valid = batch['person_mask'].bool() & batch['gt_box_labeled'].bool() & labeled.any(-1)
    valid &= torch.isfinite(boxes).all(-1) & (boxes[..., 2:] > 0).all(-1)
    locations = valid.nonzero()
    selected_boxes = boxes[valid]
    geometry = dict(keypoints_px=((batch['gt_keypoints']+1)/2*image_wh[:, None, None])[valid],
        boxes_cxcywh_px=torch.cat((selected_boxes[:, :2]+selected_boxes[:, 2:]/2, selected_boxes[:, 2:]), -1),
        joint_mask=labeled[valid], image_wh=image_wh[locations[:, 0]])
    if 'image_id' in batch:
        geometry['image_id'] = batch['image_id'][locations[:, 0]]
    return geometry, locations
