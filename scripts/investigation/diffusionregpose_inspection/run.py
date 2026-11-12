from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

from omegaconf import OmegaConf
import torch

from scripts.evaluation.evaluate_diffusionregpose import build_evaluation_model, extract_model_state_dict
from scripts.investigation.diffusionregpose_inspection.analysis import (
    evaluate_records, postprocess_with_queries, rank_images,
)
from scripts.investigation.diffusionregpose_inspection.trace import capture_sampling, cpu_copy
from src.data.datasets import PoseDataset
from src.data.io import read_image
from src.models.pipelines.pose_estimators.diffusionregpose import DiffusionRegPosePipelineConfig
from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import (
    crowdpose_result_records, resize_crowdpose_eval_image,
)

DEFAULT_RUN = ('results/diffusionregpose_development_compatible_256_keep_layers_ota/'
               'diffusionregpose_development_compatible_256_keep_layers_ota/20260912_224837')


@dataclass
class Settings:
    checkpoint: str = DEFAULT_RUN + '/checkpoints/epoch079.ckpt'
    config: str = DEFAULT_RUN + '/.hydra/config.yaml'
    annotations: str = 'data/crowdpose/annotations/crowdpose_test.json'
    image_root: str = 'data/crowdpose/images'
    output: str = 'tmp/diffusionregpose_diffusion_inspection'
    sample_count: int = 32
    subset_seed: int = 42
    proposal_seed: int = 1729
    shortest_side: int = 256
    max_size: int = 427
    steps: tuple = (1, 2, 3)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def experiment_manifest(settings):
    source_root = Path(__file__).resolve().parents[3]
    sources = [*Path(__file__).parent.glob('*.py'),
               *source_root.glob('src/models/pipelines/pose_estimators/diffusionregpose/*.py')]
    return {'settings': json.loads(json.dumps(asdict(settings))),
            'checkpoint_sha256': file_hash(settings.checkpoint),
            'config_sha256': file_hash(settings.config),
            'annotations_sha256': file_hash(settings.annotations),
            'source_sha256': {str(path.relative_to(source_root)): file_hash(path) for path in sorted(sources)},
            'torch_version': str(torch.__version__)}


def load_model(settings):
    resolved = OmegaConf.load(settings.config)
    config = DiffusionRegPosePipelineConfig.from_hydra_config(resolved.pipeline)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = build_evaluation_model(config, sampling_timesteps=1)
    checkpoint = torch.load(settings.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(extract_model_state_dict(checkpoint), strict=True)
    return model.to(settings.device).eval(), int(checkpoint['epoch'])


def prepare_image(sample, settings):
    return resize_crowdpose_eval_image(
        read_image(sample['img_path']).float().div(255),
        shortest_side=settings.shortest_side, max_size=settings.max_size,
    )[None].to(settings.device)


def sample_model(model, image, settings, image_id, steps):
    model.sampling_timesteps = steps
    generator = torch.Generator(device=settings.device).manual_seed(settings.proposal_seed + image_id)
    return model.sample_keypoints(image, generator=generator)


def assert_same_predictions(left, right):
    for key in ('pred_logits', 'pred_boxes', 'pred_keypoints', 'diffusion_keypoints'):
        torch.testing.assert_close(left[key].cpu(), right[key].cpu(), rtol=0, atol=0)


def run_inspection(settings, *, force=False):
    if settings.sample_count < 6:
        raise ValueError('at least six images are needed for best/worst galleries')
    output_dir = Path(settings.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = experiment_manifest(settings)
    cache_path = output_dir / 'inspection.pt'
    if cache_path.exists() and not force:
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        if cached['manifest'] == manifest:
            print('Loaded verified matching inspection cache.')
            return cached
        print('Settings or source changed; rebuilding inspection cache.')

    dataset = PoseDataset.from_path(settings.annotations, img_root=settings.image_root)
    ordered = sorted(dataset.samples, key=lambda sample: sample['image_id'])
    samples = random.Random(settings.subset_seed).sample(ordered, settings.sample_count)
    image_ids = [sample['image_id'] for sample in samples]
    (output_dir / 'manifest.json').write_text(json.dumps({**manifest, 'image_ids': image_ids}, indent=2))
    model, epoch = load_model(settings)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    sweep, metrics, rankings, rows = {}, {}, {}, []
    with torch.no_grad():
        for steps in settings.steps:
            records = []
            for index, sample in enumerate(samples):
                image = prepare_image(sample, settings)
                prediction = sample_model(model, image, settings, sample['image_id'], steps)
                result = postprocess_with_queries(prediction, sample['img_hw'], image.shape[-2:])
                records.extend(crowdpose_result_records(sample['image_id'], [result]))
                sweep[(sample['image_id'], steps)] = {
                    'prediction': cpu_copy({key: prediction[key] for key in
                                           ('pred_logits', 'pred_boxes', 'pred_keypoints', 'diffusion_keypoints')}),
                    'result': cpu_copy(result), 'resized_hw': tuple(image.shape[-2:]),
                }
                print(f'S={steps}: {index + 1}/{len(samples)} image={sample["image_id"]}', flush=True)
            (output_dir / f'predictions_s{steps}.json').write_text(json.dumps(records))
            aggregate, image_rows = evaluate_records(settings.annotations, records, image_ids)
            metrics[steps] = aggregate
            rankings[steps] = rank_images(image_rows)
            rows.extend([{**row, 'steps': steps} for row in image_rows])
            print(f'S={steps}: {aggregate}; {rankings[steps]}', flush=True)

        selected = sorted({image_id for ranking in rankings.values()
                           for cohort in ('best', 'worst') for image_id in ranking[cohort]})
        trace_paths = {}
        for sample in samples:
            image_id = sample['image_id']
            if image_id not in selected:
                continue
            image = prepare_image(sample, settings)
            for steps in settings.steps:
                with capture_sampling(model) as calls:
                    replay = sample_model(model, image, settings, image_id, steps)
                assert_same_predictions(sweep[(image_id, steps)]['prediction'], replay)
                assert len(calls) == steps
                assert len(calls[0]['layers']) == model.transformer.decoder.num_layers
                trace_path = output_dir / f'trace_{image_id}_s{steps}.pt'
                torch.save(calls, trace_path)
                trace_paths[(image_id, steps)] = str(trace_path)
                print(f'Trace image={image_id} S={steps}: exact final-output parity', flush=True)

    from src.models.pipelines.pose_estimators.diffusionregpose import transformer
    cache = {'manifest': manifest, 'epoch': epoch, 'samples': samples, 'image_ids': image_ids,
             'sweep': sweep, 'metrics': metrics, 'rankings': rankings, 'rows': rows,
             'trace_paths': trace_paths, 'selected': selected,
             'decoder_layers': model.transformer.decoder.num_layers,
             'num_box_layers': model.transformer.decoder.num_box_decoder_layers,
             'num_groups': model.transformer.decoder.num_group,
             'num_heads': model.transformer.decoder.num_heads,
             'native_cuda_kernel_loaded': transformer._native_msda is not None,
             'verification': {'selected_replay_exact': True, 'evaluated_settings': len(sweep)}}
    torch.save(cache, cache_path)
    (output_dir / 'summary.json').write_text(json.dumps(
        {key: cache[key] for key in ('image_ids', 'metrics', 'rankings', 'rows', 'verification')}, indent=2))
    del model
    if settings.device.startswith('cuda'):
        torch.cuda.empty_cache()
    return cache


def load_trace(cache, image_id, steps):
    return torch.load(cache['trace_paths'][(image_id, steps)], map_location='cpu', weights_only=False)
