import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .audit_v43 import AugmentedGeometryDataset, collate_audit
from .codec_v43 import CODEC_SHA256, load_frozen_codec


class MeanCodeMoments:
    def __init__(self, dimensions):
        self.count = 0
        self.mean = torch.zeros(dimensions, dtype=torch.float64)
        self.squared_deviations = torch.zeros_like(self.mean)

    def update(self, codes):
        codes = codes.detach().double().cpu()
        if codes.ndim != 2 or codes.shape[1:] != self.mean.shape or not torch.isfinite(codes).all():
            raise ValueError('Expected finite posterior mean codes')
        if not len(codes):
            return
        count = self.count + len(codes)
        delta = codes.mean(0) - self.mean
        self.squared_deviations += ((codes - codes.mean(0)).square().sum(0)
            + delta.square() * self.count * len(codes) / count)
        self.mean += delta * len(codes) / count
        self.count = count

    def finalize(self):
        if not self.count:
            raise ValueError('No TRAIN posterior means')
        return self.mean, (self.squared_deviations / self.count).sqrt().clamp_min(1e-6)


@torch.no_grad()
def compute_train_statistics(data_config, output, *, device='cuda', workers=4, batch_size=32):
    annotation = (Path(data_config.root) / data_config.train_ann).resolve()
    if annotation.name != 'crowdpose_train.json':
        raise ValueError('V44 statistics require TRAIN only')
    codec = load_frozen_codec().to(device)
    dataset = AugmentedGeometryDataset(annotation, Path(data_config.root) / data_config.img_dir, data_config, seed=42)
    moments = MeanCodeMoments(codec.latent_dim)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False, collate_fn=collate_audit)
    images = 0
    for index, batch in enumerate(loader):
        geometry = {name: value.to(device) for name, value in batch['augmented'].items()}
        mean, _ = codec.encode_posterior(geometry)
        moments.update(mean)
        images += min(batch_size, len(dataset) - images)
        if index % 25 == 0:
            print(f'V44 TRAIN statistics: {images}/{len(dataset)} images, {moments.count} people', flush=True)
    mean, std = moments.finalize()
    record = dict(mean=mean, std=std, people=moments.count, images=images,
        split='train', posterior_mode='mean', complete=images == len(dataset),
        annotation_path=str(annotation), annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
        codec_sha256=CODEC_SHA256, data_config=data_config.model_dump(), seed=42,
        augmentation='one deterministic augmented view per TRAIN image, seed + image_id')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, output)
    return record
