from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.data_module import _external_eval_transform, _external_train_transform
from src.data.datasets import PoseDataset, SetPoseDataset, set_pose_collate
from src.data.pose_geometry import PoseGeometryDataset, normalize_geometry, restore_geometry
from src.models.pipelines.pose_autoencoder.evaluation import (
    CodecAPEvaluator, EVALUATION_SEEDS, export_reconstructions, paired_metrics,
)
from src.models.pipelines.pose_autoencoder.experiment import verify_ceilings
from src.models.pipelines.pose_vae.fidelity_experiment import collect_posterior, decode_cpu, error_slices

from .codec_v43 import CODEC_SHA256, GEOMETRY_FIELDS, PosteriorMoments, geometry_from_batch, load_frozen_codec


class AugmentedGeometryDataset(Dataset):
    def __init__(self, annotation_path, image_root, config, *, seed=42, limit=None):
        base = PoseDataset.from_path(annotation_path, img_root=image_root)
        self.annotation_ids = {}
        for row in json.loads(Path(annotation_path).read_text())['annotations']:
            if not row.get('iscrowd', 0) and row.get('num_keypoints', 0) > 0:
                self.annotation_ids.setdefault(row['image_id'], []).append(row['id'])
        base.samples.sort(key=lambda sample: sample['image_id'])
        if limit is not None:
            base.samples = base.samples[:limit]
        self.dataset = SetPoseDataset(base)
        self.config, self.seed = config, seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample, config = self.dataset[index], self.config
        image = sample['image'].float()/255
        if image.shape[0] == 1:
            image = image.expand(3, -1, -1)
        arguments = (image, sample['gt_keypoints'], sample['gt_labeled'], sample['gt_visible'],
                     sample['gt_boxes_pixel'], sample['gt_box_labeled'])
        augmented = _external_train_transform(*arguments,
            scales=config.train_resize_scales, max_size=config.train_max_size,
            scales2_resize=config.train_crop_resize_scales, scales2_crop=config.train_crop_size,
            random_flip=config.train_random_flip, random_crop=config.train_random_crop,
            rng=random.Random(self.seed+sample['image_id']))
        uncropped = _external_eval_transform(*arguments,
            short_side=config.eval_resize_scale, max_size=config.eval_max_size)
        result = {}
        for name, transformed in (('uncropped', uncropped), ('augmented', augmented)):
            if name == 'augmented':
                transformed = (transformed[0], transformed[1], *transformed[3:])
            _, points, labels, visible, boxes, box_labels, image_hw, scales = transformed
            packed = set_pose_collate([dict(sample, gt_keypoints=points, gt_labeled=labels,
                gt_visible=visible, gt_boxes_pixel=boxes, gt_box_labeled=box_labels,
                img_hw=image_hw, person_scale=scales)])
            geometry, locations = geometry_from_batch(packed)
            geometry['annotation_id'] = torch.tensor(self.annotation_ids[sample['image_id']])[locations[:, 1]]
            result[name] = geometry
        return result


def concatenate_geometry(rows):
    return {name: torch.cat([row[name] for row in rows]) for name in rows[0]}


def collate_audit(samples):
    return {name: concatenate_geometry([sample[name] for sample in samples]) for name in samples[0]}


@torch.no_grad()
def compute_codec_audit(codec, batches):
    device = next(codec.parameters()).device
    moments = PosteriorMoments(codec.latent_dim)
    collections = {name: dict(target=[], mean=[], logvar=[]) for name in ('uncropped', 'augmented')}
    for index, batch in enumerate(batches):
        for name, target in batch.items():
            if not len(target['image_wh']):
                continue
            geometry = {key: value.to(device) for key, value in target.items()}
            mean, logvar = codec.encode_posterior(geometry)
            if name == 'augmented':
                moments.update(mean, logvar)
            collection = collections[name]
            collection['target'].append({key: value.cpu() for key, value in target.items()})
            collection['mean'].append(mean.cpu())
            collection['logvar'].append(logvar.cpu())
        if index % 100 == 0:
            print(f'V43 codec audit: {index+1} batches', flush=True)
    center, scale = moments.finalize()
    statistics = dict(mean=center, std=scale, people=moments.count)
    report, examples = {}, {}
    for name, collection in collections.items():
        if not collection['target']:
            raise ValueError(f'Empty {name} audit targets')
        target = concatenate_geometry(collection['target'])
        mean, logvar = torch.cat(collection['mean']), torch.cat(collection['logvar'])
        relative, placement = normalize_geometry(**{key: target[key] for key in GEOMETRY_FIELDS})
        restored = restore_geometry(relative, placement, target['image_wh'])
        joint_error = (restored['keypoints_px']-target['keypoints_px'])[target['joint_mask']].abs().max()
        box_error = (restored['boxes_cxcywh_px']-target['boxes_cxcywh_px']).abs().max()
        maximum_error = float(torch.maximum(joint_error, box_error))
        if maximum_error > 1e-3:
            raise RuntimeError(f'Geometry normalization round trip failed: {maximum_error}')
        prediction = decode_cpu(codec.model, mean, target, device)
        noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(1042))
        sampled = decode_cpu(codec.model, mean+(.5*logvar).exp()*noise, target, device)
        digest = hashlib.sha256()
        for key, value in sorted(target.items()):
            digest.update(key.encode())
            digest.update(value.contiguous().numpy().tobytes())
        report[name] = dict(people=len(mean), normalization_max_error=maximum_error,
            transformed_target_sha256=digest.hexdigest(), mean=paired_metrics(prediction, target),
            sampled=paired_metrics(sampled, target), error_slices=error_slices(prediction, target))
        examples[name] = dict(target=target, prediction=prediction, sampled=sampled)
    statistics['transformed_target_sha256'] = report['augmented']['transformed_target_sha256']
    return report, statistics, examples


@torch.no_grad()
def replay_codec_validation(codec, annotation_path, *, batch_size=512):
    dataset = PoseGeometryDataset(annotation_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    evaluator = CodecAPEvaluator(annotation_path, [int(row['annotation_id']) for row in dataset.records])
    ceilings = verify_ceilings(evaluator, loader)
    device = next(codec.parameters()).device
    mean, logvar, target = collect_posterior(codec.model, loader, device)
    result = dict(images=len(dataset.image_ids), people=len(dataset), ceilings=ceilings,
                  mean=evaluator.evaluate(decode_cpu(codec.model, mean, target, device), target), samples=[])
    for seed in EVALUATION_SEEDS:
        noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(seed))
        result['samples'].append(dict(seed=seed, **evaluator.evaluate(
            decode_cpu(codec.model, mean+(.5*logvar).exp()*noise, target, device), target)))
    return result


def run_codec_audit(config, output, *, device='cpu', workers=4, batch_size=32, seed=42, limit=None,
                    replay_validation=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(config.root).resolve()
    annotation_path = root/config.train_ann
    codec = load_frozen_codec().to(device)
    dataset = AugmentedGeometryDataset(annotation_path, root/config.img_dir, config, seed=seed, limit=limit)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False,
        collate_fn=collate_audit, generator=torch.Generator().manual_seed(seed))
    report, statistics, examples = compute_codec_audit(codec, loader)
    provenance = dict(codec_sha256=CODEC_SHA256, seed=seed, transform_seed='seed + image_id',
        images=len(dataset), complete_trainval=limit is None, source='actual augmented trainval targets',
        annotation_path=str(annotation_path), annotation_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        data_config=config.model_dump(), device=str(device), torch_version=str(torch.__version__))
    statistics.update(provenance)
    torch.save(statistics, output/'latent_statistics.pt')
    report.update(provenance)
    if replay_validation:
        report['codec_validation'] = replay_codec_validation(codec, root/config.val_ann)
    for name, example in examples.items():
        export_reconstructions(output/f'{name}_mean.png', example['prediction'], example['target'])
        export_reconstructions(output/f'{name}_sampled.png', example['sampled'], example['target'])
    sources = (Path(__file__), Path(__file__).with_name('codec_v43.py'),
               Path('src/data/data_module.py'), Path('src/models/pipelines/pose_vae/fidelity.py'),
               Path('src/data/pose_geometry.py'))
    report['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    report['test_access'] = False
    (output/'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
