"""Compare native V42 DDIM with per-call proposal-to-incoming-pose matching."""

import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf

from src.data import DataConfig
from src.data.datasets import PoseDataset
from src.models.pipelines.pose_estimators.diffusionregpose.training import DiffusionRegPosePipelineConfig
from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import crowdpose_result_records, postprocess_crowdpose
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v13 import evaluate_records, evaluation_context
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v21 import v21_test_batches
from scripts.evaluation.evaluate_diffusionregpose import extract_model_state_dict


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_sampler(path):
    # Load the new policy alongside a frozen model package without modifying that package.
    name = 'src.models.pipelines.pose_estimators.poseflow.proposal_pose_comparison_policy'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.sample_v42_with_proposal_pose_matching


def evaluate(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'report.json').exists():
        raise FileExistsError(f'Comparison report already exists: {output}')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.run_root.resolve()
    config_path = root / 'seed42/.hydra/config.yaml'
    config = OmegaConf.load(config_path)
    settings = DiffusionRegPosePipelineConfig.from_hydra_config(config.pipeline)
    settings = settings.model_copy(update={'backbone_pretrained': False, 'sampling_timesteps': 1})
    pipeline = settings.build(max_epochs=config.training.max_epochs, load_pretrained_checkpoint=False)
    model, matcher = pipeline.model, pipeline.criterion.matcher
    if type(matcher).__name__ != 'DiffusionRegPoseMatcher':
        raise ValueError('This comparison requires the native one-to-one Hungarian matcher')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    epoch, step = checkpoint['epoch'], checkpoint['global_step']
    model.load_state_dict(extract_model_state_dict(checkpoint), strict=True)
    del checkpoint, pipeline
    gc.collect()
    source = Path(inspect.getfile(type(model))).resolve()
    verified_sources = 0
    if args.require_frozen_source:
        if root / 'workspace' not in source.parents:
            raise ValueError('Use PYTHONPATH=<run-root>/workspace to load the original frozen model')
        manifest = json.loads((root / 'manifest.json').read_text())
        for name, expected in manifest['source_hashes'].items():
            if digest(root / 'workspace' / name) != expected:
                raise ValueError(f'Frozen source hash mismatch: {name}')
        verified_sources = len(manifest['source_hashes'])
    model.to(args.device).eval()
    matcher.to(args.device).eval()
    sample = load_sampler(args.sampler_source.resolve())
    baseline_path = args.baseline_metrics or root / f'seed42/quick_test/epoch{epoch:03d}/metrics.json'
    baseline = json.loads(baseline_path.read_text())
    if baseline['epoch'] != epoch or baseline['global_step'] != step:
        raise ValueError('Saved baseline must come from the selected checkpoint epoch/step')
    image_ids = baseline['image_ids']
    data = DataConfig.from_hydra_config(config.data).build(pin_memory=False)
    dataset = PoseDataset.from_path(data.test_ann_path, img_root=data.img_dir, num_keypoints=data.num_keypoints)
    records = {f'{policy}_K{steps}': [] for policy in ('native', 'matched') for steps in args.steps}
    scores = {name: [] for name in records}
    assignments = []
    original_steps = model.sampling_timesteps
    try:
        with evaluation_context(model):
            for index, batch in enumerate(v21_test_batches(data, dataset, image_ids, args.device)):
                image = batch['image']
                image_id = int(batch['image_id'][0])
                for policy in ('native', 'matched'):
                    for steps in args.steps:
                        model.sampling_timesteps = steps
                        name = f'{policy}_K{steps}'
                        image_assignments = []
                        result = sample(model, image, matcher=matcher, padding_mask=batch.get('padding_mask'),
                            generator=torch.Generator(device=image.device).manual_seed(settings.quick_eval_seed + image_id),
                            enabled=policy == 'matched',
                            assignment_observer=lambda time, permutation: image_assignments.append(
                                dict(time=time, permutation=permutation.cpu().tolist())))
                        if not all(torch.isfinite(result[key]).all() for key in ('pred_logits', 'pred_boxes', 'pred_keypoints', 'diffusion_keypoints')):
                            raise ValueError(f'Nonfinite {name} predictions for image {image_id}')
                        if policy == 'matched':
                            if len(image_assignments) != steps:
                                raise ValueError('Every denoising call must execute proposal/state matching')
                            assignments.append(dict(image_id=image_id, condition=name, calls=image_assignments))
                        probabilities = result['pred_logits'].sigmoid()[..., 1]
                        scores[name].append(dict(mean_person_probability=float(probabilities.mean()),
                                                queries_above_half=int((probabilities > .5).sum())))
                        postprocessed = postprocess_crowdpose(result,
                            batch.get('original_img_hw', batch['img_hw']).to(image),
                            num_body_points=model.transformer.decoder.num_body_points,
                            nms_target_sizes=batch['img_hw'].to(image))
                        records[name].extend(crowdpose_result_records(image_id, postprocessed))
                save(output / 'progress.json', dict(phase='sampling', images=index + 1, total=len(image_ids), epoch=epoch, step=step))
                if (index + 1) % 8 == 0:
                    print('IMAGES', index + 1, flush=True)
    finally:
        model.sampling_timesteps = original_steps
    report = dict(epoch=epoch, global_step=step, checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=digest(args.checkpoint), config=str(config_path), config_sha256=digest(config_path),
        frozen_model_source=str(source), frozen_source_files_verified=verified_sources,
        sampler_source=str(args.sampler_source.resolve()), sampler_sha256=digest(args.sampler_source),
        evaluator_source=str(Path(__file__).resolve()), evaluator_sha256=digest(__file__),
        images=len(image_ids), image_ids=image_ids, seed=settings.quick_eval_seed,
        resize=[data.eval_resize_scale, data.eval_max_size], annotations=str(data.test_ann_path),
        annotations_sha256=digest(data.test_ann_path), baseline_metrics=str(baseline_path),
        baseline_metrics_sha256=digest(baseline_path), matcher=type(matcher).__name__, matching_weights=asdict(matcher.weights),
        matching='Selected current proposal boxes to enclosing boxes of actual incoming noisy joints at every call, including the first; all slots retained',
        state_order='Same matched state used for pose encoding, noise reconstruction and DDIM update',
        assignment_GT=False, scope='fixed32 frozen checkpoint comparison; inference-only change; not full-test results', conditions={})
    for name, predictions in records.items():
        path = output / f'{name}_predictions.json'
        path.write_text(json.dumps(predictions, allow_nan=False))
        report['conditions'][name] = dict(predictions=str(path),
            metrics=evaluate_records(predictions, data.test_ann_path, image_ids),
            confidence={key: sum(row[key] for row in scores[name]) / len(scores[name]) for key in scores[name][0]})
        print('METRICS', name, json.dumps(report['conditions'][name]['metrics']), flush=True)
    report['native_baselines_exact'] = all(
        abs(value - report['conditions'][f'native_K{steps}']['metrics'][key]) < 1e-12
        for steps in args.steps for key, value in baseline['conditions'][f'noise_var_100/steps_{steps}']['metrics'].items())
    save(output / 'assignments.json', assignments)
    save(output / 'report.json', report)
    if not report['native_baselines_exact']:
        raise ValueError('Native baseline metrics failed exact replay; do not interpret the comparison')
    save(output / 'progress.json', dict(phase='complete', images=len(image_ids), epoch=epoch, step=step))
    print('PASS_NATIVE_BASELINES_EXACT', flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-metrics', type=Path)
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--require-frozen-source', action='store_true')
    parser.add_argument('--sampler-source', type=Path, default=Path(__file__).resolve().parents[2]
                        / 'src/models/pipelines/pose_estimators/poseflow/sampling_v42.py')
    return parser.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
