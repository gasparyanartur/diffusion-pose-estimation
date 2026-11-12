"""Compare V46 re-noising and DDIM on one frozen checkpoint and fixed32 VAL."""

import argparse
from copy import copy, deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter

from src.data import DataConfig
from src.data.datasets import PoseDataset
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v13 import evaluation_context
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v21 import v21_test_batches
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v46 import run_v46_evaluation, tensorboard_metrics
from src.models.pipelines.pose_estimators.poseflow.training_v46 import PoseFlowV46Pipeline


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_digest(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def freeze_checkpoint(source, destination):
    for _ in range(3):
        before = source.stat()
        shutil.copy2(source, destination)
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns):
            checksum = digest(destination)
            if digest(source) == checksum:
                return checksum
    raise RuntimeError('Checkpoint changed during snapshot; retry between epoch saves')


def verify_legacy(model, run_root, data, image_ids, *, seed):
    source = run_root/'workspace/src/models/pipelines/pose_estimators/poseflow/model_v46.py'
    specification = importlib.util.spec_from_file_location(
        'src.models.pipelines.pose_estimators.poseflow._original_v46', source)
    frozen_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(frozen_module)
    original = deepcopy(model)
    original.__class__ = frozen_module.PoseFlowV46Model
    dataset = PoseDataset.from_path(data.val_ann_path, img_root=data.img_dir, num_keypoints=14)
    evaluation_data = copy(data)
    evaluation_data.trainer = SimpleNamespace(training=False)
    maximum = 0.
    with evaluation_context(model), evaluation_context(original):
        for batch in v21_test_batches(evaluation_data, dataset, image_ids[:3], next(model.parameters()).device):
            context = model.prepare_context(batch['image'], batch.get('padding_mask'))
            sizes = batch['img_hw'].flip(-1).to(batch['image'])
            image_id = int(batch['image_id'][0])
            generators = [torch.Generator().manual_seed(seed+image_id) for _ in range(2)]
            old = original.predict(context, sizes, generator=generators[0], collect_trace=True)
            new = model.predict(context, sizes, generator=generators[1], collect_trace=True, sampler='renoise')
            for key in ('pred_latents', 'pred_boxes', 'pred_keypoints', 'pred_logits', 'person_counts', 'stop_reason'):
                torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
            for left, right in zip(old['trace'], new['trace']):
                torch.testing.assert_close(left['pred_latents'], right['pred_latents'], rtol=0, atol=0)
            assert torch.equal(generators[0].get_state(), generators[1].get_state())
    del original
    return dict(images=min(3, len(image_ids)), maximum_difference=maximum,
                rng_exact=True, original_source_sha256=digest(source))


def log_report(report, directory, epoch):
    with SummaryWriter(str(directory)) as writer:
        for name, value in tensorboard_metrics(report).items():
            writer.add_scalar(name, value, epoch)
        common = f'quick-test/sampler_comparison/rows_{report["rows"]}'
        for name, value in report['metrics'].items():
            writer.add_scalar(f'{common}/{name}', value, epoch)
        for name in ('count_mae', 'mean_eos_row', 'inference_images_per_second', 'peak_cuda_allocated_bytes'):
            writer.add_scalar(f'{common}/{name}', report[name], epoch)
        for name, value in report['refinement'].items():
            writer.add_scalar(f'sampler-comparison/{name}', value, epoch)
        for row in range(1, report['rows']+1):
            for metric in ('latent_mse', 'geometry'):
                name = f'trajectory/row_{row:02d}/rank_01/{metric}'
                if name in report.get('trajectory', {}):
                    writer.add_scalar(f'trajectory/rank_01/{metric}', report['trajectory'][name], row)
        writer.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Comparison output must be a fresh namespace')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pl.seed_everything(args.seed, workers=True)
    checkpoint = args.output/'checkpoint.ckpt'
    checksum = freeze_checkpoint(args.checkpoint, checkpoint)
    record = torch.load(checkpoint, map_location='cpu', weights_only=False)
    epoch = record['epoch']
    manifest = json.loads((args.run_root/'seed42/manifest.json').read_text())
    config = DataConfig.from_hydra_config(OmegaConf.create(manifest['config']['data']))
    if config.test_ann != config.val_ann:
        raise ValueError('Unused eager test slot must point to VAL')
    data = config.model_copy(update=dict(num_workers=0,
        root=str(args.run_root.resolve()/'workspace/data'))).build()
    subset = args.run_root/'seed42/quick_test/subset.json'
    ids = json.loads(subset.read_text())['image_ids']
    if len(ids) != 32:
        raise ValueError('Expected original fixed32 VAL manifest')
    source_files = {str(path): digest(path) for folder in ('src', 'conf', 'scripts')
                    for path in Path(folder).rglob('*') if path.suffix in ('.py', '.yaml')}
    for folder in ('src', 'conf', 'scripts'):
        shutil.copytree(folder, args.output/'sources'/folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    inputs = dict(checkpoint=checksum, annotations=digest(data.val_ann_path), subset=digest(subset))
    (args.output/'manifest.json').write_text(json.dumps(dict(time=datetime.now().isoformat(),
        checkpoint_source=str(args.checkpoint.resolve()), frozen_checkpoint=str(checkpoint.resolve()),
        epoch=epoch, global_step=record['global_step'], inputs=inputs, sources=source_files,
        seed=args.seed, rows=record['v46_contract']['rows'], image_ids=ids,
        training_contract=record['v46_contract'], codec_provenance=record['v46_codec_provenance'],
        policy='Frozen FP32 fixed32 VAL comparison; no optimizer updates, no TEST, eta0 existing tokens, paired birth noise'), indent=2)+'\n')
    module = PoseFlowV46Pipeline.load_from_checkpoint(checkpoint, map_location='cpu', weights_only=False).to(args.device).eval()
    del record
    before, codec_before = model_digest(module.model), model_digest(module.model.codec)
    legacy = verify_legacy(module.model, args.run_root, data, ids, seed=args.seed)
    print('LEGACY_PARITY', json.dumps(legacy), flush=True)
    reports = {}
    for sampler in ('renoise', 'ddim'):
        arm = args.output/sampler
        arm.mkdir()
        shutil.copy2(subset, arm/'subset.json')
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        report = run_v46_evaluation(module.model, data, arm, epoch=epoch, count=32, seed=args.seed,
                                    sampler=sampler, collect_trajectory=True)
        assert report['image_ids'] == ids
        assert model_digest(module.model) == before and model_digest(module.model.codec) == codec_before
        reports[sampler] = report
        log_report(report, arm/'tb_logs', epoch)
        print('SAMPLER_RESULT', sampler, json.dumps(dict(metrics=report['metrics'], count_mae=report['count_mae'],
              first_person={k:v for k,v in report['refinement'].items() if 'generated/rank_01/' in k})), flush=True)
    for label in ('birth_latent_mse', 'birth_geometry'):
        key = f'generated/rank_01/{label}'
        assert abs(reports['renoise']['refinement'][key]-reports['ddim']['refinement'][key]) < 1e-6
    comparison = dict(checkpoint_sha256=checksum, epoch=epoch, images=32, image_ids=ids,
        legacy_parity=legacy, model_unchanged=True, codec_unchanged=True, initial_model_sha256=before,
        reports={key:dict(metrics=value['metrics'], count_mae=value['count_mae'], mean_eos_row=value['mean_eos_row'],
            inference_images_per_second=value['inference_images_per_second'], peak_cuda_allocated_bytes=value['peak_cuda_allocated_bytes'],
            refinement=value['refinement'], refinement_counts=value['refinement_counts'], trajectory=value['trajectory'],
            trajectory_counts=value['trajectory_counts']) for key,value in reports.items()},
        scope='Paired frozen fixed32VAL sampler diagnostic; canonical rank errors, no causal claim about training or fullVAL')
    (args.output/'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False)+'\n')
    print('COMPARISON_COMPLETE', str(args.output/'comparison.json'), flush=True)


if __name__ == '__main__':
    main()
