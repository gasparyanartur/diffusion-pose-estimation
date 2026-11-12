#!/usr/bin/env python3
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter

from src.data import DataConfig
from src.models.pipelines.pose_estimators.poseflow.cache_control_v43 import build_context_cache
from src.models.pipelines.pose_estimators.poseflow.control_v43 import (
    configure_cached_model, control_step, evaluate_cached, frozen_state_digest,
    make_control_checkpoint, restore_control_checkpoint, tiny_set_gate,
)
from src.models.pipelines.pose_estimators.poseflow.training_v43 import PoseFlowV43Pipeline


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save_checkpoint(path, record):
    temporary = path.with_suffix('.part')
    torch.save(record, temporary)
    temporary.replace(path)


def log_evaluation(writer, split, metrics, step):
    for sigma, result in metrics.items():
        writer.add_scalar(f'{split}/sigma_{sigma}/foreground_latent_mse', result['foreground_latent_mse'], step)
        for label in ('clean_codec', 'raw_noisy', 'denoised'):
            for name, value in result[label].items():
                writer.add_scalar(f'{split}/sigma_{sigma}/{label}/{name}', value, step)


def run_arm(parent, config, cache, cache_sha256, root, mode):
    pl.seed_everything(config.seed, workers=True)
    specification = parent.config.model_copy(update=dict(prediction_type=mode,
        noise_policy='small_noise', posterior_mode='mean', center_min_side_px=config.center_min_side_px))
    model = specification.create_model().to(config.device)
    state = parent.model.state_dict()
    if mode == 'v':
        state = {name: value for name, value in state.items() if not name.startswith('latent_output.')}
        missing = model.load_state_dict(state, strict=False)
        assert set(missing.missing_keys) == {'latent_output.weight', 'latent_output.bias'} and not missing.unexpected_keys
        assert not model.latent_output.weight.any() and not model.latent_output.bias.any()
    else:
        model.load_state_dict(state, strict=True)
    optimizer = torch.optim.AdamW(configure_cached_model(model), lr=config.lr, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed+10000)
    frozen = frozen_state_digest(model)
    directory = root/mode
    directory.mkdir()
    writer = SummaryWriter(str(directory/'tb_logs'))
    writer.add_text('scope', 'Cached 16 TRAIN / 16 VAL conditional denoising control. Frozen parent has prior trainval exposure. '
        'Known source codes and fixed references, not Gaussian-start detection or learned detector AP.')
    settings = dict(pipeline=specification.model_dump(), control=OmegaConf.to_container(config, resolve=True),
                    codec_provenance=parent.codec_provenance)
    initial_record = make_control_checkpoint(model, optimizer, generator, step=0, cache_sha256=cache_sha256, settings=settings)
    initial_frozen_hash = frozen_state_digest(model)
    # Exercise the same restore path used by the evaluation entrypoint before updates.
    assert restore_control_checkpoint(initial_record, model, optimizer, generator, cache_sha256=cache_sha256) == 0
    del initial_record
    started, best, evaluations = time.monotonic(), float('inf'), []
    try:
        for step in range(config.max_updates+1):
            if step:
                metrics = control_step(model, optimizer, cache['train'], generator, batch_size=config.batch_size,
                    noise_levels=config.noise_levels, geometry_weight=config.geometry_weight,
                    center_min_side_px=config.center_min_side_px, grad_clip=config.grad_clip)
                if step % config.log_every == 0 or step == 1:
                    for name, value in metrics.items():
                        if name != 'sample':
                            writer.add_scalar(f'train/{name}', value, step)
                    status = dict(phase='training', arm=mode, step=step, maximum_updates=config.max_updates,
                        pid=os.getpid(), elapsed_seconds=time.monotonic()-started, **metrics)
                    write_json(root/'status.json', status)
                    print('PROGRESS', json.dumps(status), flush=True)
                    writer.flush()
            if step % config.evaluate_every == 0 or step == config.max_updates:
                assert frozen_state_digest(model) == frozen
                metrics = {split: evaluate_cached(model, rows, sigmas=config.evaluation_sigmas,
                    seeds=config.evaluation_seeds, batch_size=config.batch_size) for split, rows in cache.items()}
                gate = tiny_set_gate(metrics['train'])
                summary = dict(step=step, train_gate=gate, metrics=metrics)
                evaluations.append(summary)
                write_json(directory/f'evaluation_step{step:04d}.json', summary)
                for split in cache:
                    log_evaluation(writer, split, metrics[split], step)
                score = sum(metrics['train'][str(float(sigma))]['foreground_latent_mse']
                            for sigma in config.noise_levels)/len(config.noise_levels)
                record = make_control_checkpoint(model, optimizer, generator, step=step,
                    cache_sha256=cache_sha256, settings=settings)
                save_checkpoint(directory/'last.ckpt', record)
                if score < best:
                    best = score
                    save_checkpoint(directory/'best.ckpt', record)
                    write_json(directory/'best.json', dict(step=step, score=score, gate=gate))
                del record
                print('EVALUATION', json.dumps(dict(arm=mode, step=step, train_gate=gate, train_mse=score)), flush=True)
                writer.flush()
        assert frozen_state_digest(model) == frozen == initial_frozen_hash
        assert digest(root/'cache.pt') == cache_sha256
        result = dict(arm=mode, updates=config.max_updates, final_gate=evaluations[-1]['train_gate'],
            best=json.loads((directory/'best.json').read_text()), frozen_state_exact=True,
            checkpoint_sha256={name: digest(directory/name) for name in ('best.ckpt', 'last.ckpt')},
            evaluation_steps=[row['step'] for row in evaluations])
        write_json(directory/'result.json', result)
        return result
    finally:
        writer.close()
        del model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@hydra.main(version_base='1.3', config_path='../../conf', config_name='benchmark_poseflow_v43_small_noise')
def main(config):
    if not 0 < config.max_updates <= 2000 or config.geometry_weight != 0:
        raise ValueError('This entrypoint is the approved latent-only control, capped at2000 updates')
    if set(config.arms) != {'x0', 'v'} or list(config.noise_levels) != [.01, .025, .05]:
        raise ValueError('The controlled comparison requires x0/v and the reviewed noise grid')
    root = Path(HydraConfig.get().runtime.output_dir)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pl.seed_everything(config.seed, workers=True)
    if digest(config.parent_checkpoint) != config.parent_sha256:
        raise ValueError('Parent checkpoint hash mismatch')
    parent = PoseFlowV43Pipeline.load_from_checkpoint(config.parent_checkpoint, weights_only=False).to(config.device).eval()
    data = DataConfig.from_hydra_config(config.data).build()
    cache, split_manifest = build_context_cache(parent, data, images_per_split=config.images_per_split, seed=config.seed)
    torch.save(cache, root/'cache.pt')
    parent.cpu()
    cache_sha256 = digest(root/'cache.pt')
    paths = [path for folder in ('src', 'conf') for path in Path(folder).rglob('*') if path.suffix in ('.py', '.yaml')]
    paths.extend((Path(__file__), Path('scripts/evaluation/evaluate_poseflow_v43_small_noise.py')))
    manifest = dict(parent_checkpoint=str(config.parent_checkpoint), parent_sha256=config.parent_sha256,
        cache_sha256=cache_sha256, splits=split_manifest, config=OmegaConf.to_container(config, resolve=True),
        source_sha256={str(path): digest(path) for path in paths},
        annotations={str(path): digest(path) for path in (data.train_ann_path, data.val_ann_path)},
        scope='Known-person low-noise control; frozen components have prior trainval exposure. No detector AP claim.',
        initialization='Same pinned parent denoiser body; x0 retains parent head; new velocity head is zero initialized',
        posterior='fixed standardized posterior means; independent private train and paired evaluation noise',
        retention='best TRAIN error and last only; no per-epoch checkpoint history')
    write_json(root/'manifest.json', manifest)
    results = []
    try:
        for mode in config.arms:
            results.append(run_arm(parent, config, cache, cache_sha256, root, mode))
        write_json(root/'results.json', dict(arms=results, next_stage_authorized=False,
            scope=manifest['scope'], both_completed=True))
        write_json(root/'status.json', dict(phase='complete', arms=results, pid=os.getpid()))
    except BaseException as error:
        write_json(root/'status.json', dict(phase='failed', error=repr(error), pid=os.getpid()))
        raise


if __name__ == '__main__':
    main()
