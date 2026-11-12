from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.pose_geometry import normalize_geometry, restore_geometry
from src.models.pipelines.pose_vae.training import PoseGeometryDataModule
from src.training import TrainingConfig, TrainingRunner
from src.training.callbacks import TrustedCheckpointIO
from .evaluation import (CodecAPEvaluator, EVALUATION_SEEDS, NOISE_LEVELS,
                         clean_gate, perturb_latents, robust_gain)
from .training import (GEOMETRY_FIELDS, TARGET_FIELDS, PoseAutoencoderConfig,
                       PoseAutoencoderPipeline, concatenate_batches, encode_batch)


@torch.inference_mode()
def latent_statistics(model, loader, device):
    model.to(device).eval()
    total = torch.zeros(model.latent_dim, dtype=torch.float64)
    squares, count = torch.zeros_like(total), 0
    for batch in loader:
        latent = encode_batch(model, {name:value.to(device) for name,value in batch.items()}).double().cpu()
        total += latent.sum(0)
        squares += latent.square().sum(0)
        count += len(latent)
    mean = total/count
    std = (squares/count-mean.square()).clamp_min(0).sqrt()
    return mean.float(), std.float()


@torch.inference_mode()
def collect_outputs(model, loader, device):
    model.to(device).eval()
    targets, predictions, latents = [], [], []
    for batch in loader:
        batch = {name:value.to(device) for name,value in batch.items()}
        latent = encode_batch(model, batch)
        predictions.append({name:value.cpu() for name,value in model.decode(latent, batch['image_wh']).items()})
        targets.append({name:batch[name].cpu() for name in TARGET_FIELDS})
        latents.append(latent.cpu())
    return concatenate_batches(predictions), concatenate_batches(targets), torch.cat(latents)


@torch.inference_mode()
def noise_curve(model, target, latents, std, evaluator, device):
    model.to(device).eval()
    rows, clean = [], None
    for seed in EVALUATION_SEEDS:
        for fraction in NOISE_LEVELS:
            if fraction == 0 and clean is not None:
                metrics = clean
            else:
                latent = perturb_latents(latents.to(device), std, fraction, seed=seed)
                prediction = {name:value.cpu() for name,value in model.decode(latent, target['image_wh'].to(device)).items()}
                metrics = evaluator.evaluate(prediction, target)
                if fraction == 0:
                    clean = metrics
            rows.append(dict(seed=seed, noise_fraction=fraction, **metrics))
    return rows


def encoder_hash(model):
    digest = hashlib.sha256()
    for name,value in model.state_dict().items():
        if name.startswith(('encoder.', 'latent_head.')):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def verify_ceilings(evaluator, loader):
    target = concatenate_batches([{name:batch[name] for name in TARGET_FIELDS} for batch in loader])
    relative, placement = normalize_geometry(**{name:target[name] for name in GEOMETRY_FIELDS})
    results = dict(gt_copy=evaluator.evaluate(target, target),
                   normalization=evaluator.evaluate(restore_geometry(relative, placement, target['image_wh']), target))
    for metrics in results.values():
        if min(metrics['pose_AP'], metrics['bbox_AP']) < 99.999:
            raise RuntimeError('Native codec AP ceiling is below 100; training must not start')
    return results


def write_manifest(directory, config, data, module, source_root):
    sources = [path for folder in ('src', 'conf') for path in (source_root/folder).rglob('*')
               if path.suffix in ('.py', '.yaml')]
    sources.extend((source_root/'scripts/training/train_pose_autoencoder.py', source_root/'pyproject.toml'))
    manifest = dict(parameters=sum(parameter.numel() for parameter in module.parameters()),
                    trainable_parameters=sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad),
                    python=platform.python_version(), torch=torch.__version__, source_root=str(source_root),
                    phase=config.pipeline.phase, subset_frac=data.subset_frac,
                    people={name:len(dataset) for name,dataset in data.datasets.items()},
                    train_val_image_overlap=len(data.datasets['train'].image_ids & data.datasets['val'].image_ids),
                    annotations={name:dict(path=str(data.paths[name].resolve()),
                                          sha256=hashlib.sha256(data.paths[name].read_bytes()).hexdigest())
                                 for name in ('train', 'val')},
                    source_sha256={str(path.relative_to(source_root)):hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in sorted(sources)})
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2))
    OmegaConf.save(config, directory/'resolved.yaml', resolve=True)


def fit_phase(config, directory, data, source_root, *, initial=None, statistics=None):
    directory.mkdir(parents=True, exist_ok=True)
    runtime = TrainingConfig.from_hydra_config(config.training)
    if runtime.test_after_fit or runtime.skip_training:
        raise ValueError('This development experiment permits training/validation only; test must remain untouched')
    runner = TrainingRunner(runtime, directory)
    runner.seed_everything()
    val_ids = None
    if data.subset_frac < 1:
        count = max(1, int(len(data.datasets['val'])*data.subset_frac))
        val_ids = [int(record['annotation_id']) for record in data.datasets['val'].records[:count]]
    module = PoseAutoencoderConfig.from_hydra_config(config.pipeline).build(
        max_epochs=runtime.max_epochs, val_ann=str(data.paths['val']), val_annotation_ids=val_ids,
        artifact_directory=str(directory/'validation'))
    if initial is not None:
        module.model.load_state_dict(initial.state_dict(), strict=True)
    if statistics is not None:
        module.set_latent_statistics(*statistics)
    if module.hparams.phase == 'robust' and (initial is None or statistics is None):
        raise ValueError('Robust phase requires a gated clean model and training-only latent statistics')
    write_manifest(directory, config, data, module, source_root)
    runner.logger().info('Phase=%s parameters=%d epochs=%d', module.hparams.phase,
                         sum(parameter.numel() for parameter in module.parameters()), runtime.max_epochs)
    result = runner.run(module, data, max_epochs=runtime.max_epochs, plugins=[TrustedCheckpointIO()])
    best = PoseAutoencoderPipeline.load_from_checkpoint(result.checkpoint, weights_only=False)
    summary = dict(checkpoint=result.checkpoint, completed_epochs=runtime.max_epochs)
    (directory/'fit_complete.json').write_text(json.dumps(summary, indent=2))
    return best, result.checkpoint


def run_experiment(config, directory: Path, source_root: Path):
    if config.pipeline.phase != 'clean':
        raise ValueError('Start with the clean preset; the robust preset is applied internally after the AP gate')
    directory.mkdir(parents=True, exist_ok=True)
    data = PoseGeometryDataModule(**OmegaConf.to_container(config.data, resolve=True))
    data.setup('fit')
    val_ids = None
    if data.subset_frac < 1:
        val_ids = [int(record['annotation_id']) for record in data.datasets['val'].records[
            :max(1, int(len(data.datasets['val'])*data.subset_frac))]]
    evaluator = CodecAPEvaluator(data.paths['val'], val_ids)
    (directory/'ceilings.json').write_text(json.dumps(verify_ceilings(evaluator, data.val_dataloader()), indent=2))
    device = 'cuda' if torch.cuda.is_available() and config.hardware.accelerator != 'cpu' else 'cpu'
    clean, clean_path = fit_phase(config, directory/'clean', data, source_root)
    prediction, target, latents = collect_outputs(clean.model, data.val_dataloader(), device)
    clean_metrics = evaluator.evaluate(prediction, target)
    train_loader = DataLoader(data.train_dataloader().dataset, batch_size=data.batch_size, shuffle=False)
    mean, std = latent_statistics(clean.model, train_loader, device)
    statistics = dict(mean=mean, std=std, source='train only, unflipped records, population standard deviation',
                      people=len(train_loader.dataset), encoder_sha256=encoder_hash(clean.model))
    torch.save(statistics, directory/'latent_statistics.pt')
    (directory/'latent_statistics.json').write_text(json.dumps(
        dict(statistics, mean=mean.tolist(), std=std.tolist()), indent=2))
    control_curve = noise_curve(clean.model, target, latents, std, evaluator, device)
    (directory/'clean_noise_curve.json').write_text(json.dumps(control_curve, indent=2))
    train_prediction, train_target, _ = collect_outputs(clean.model, train_loader, device)
    train_ids = None if data.subset_frac == 1 else train_target['annotation_id'].tolist()
    train_metrics = CodecAPEvaluator(data.paths['train'], train_ids).evaluate(train_prediction, train_target)
    summary = dict(clean_checkpoint=clean_path, clean_validation=clean_metrics, clean_training=train_metrics,
                   clean_gate=clean_gate(clean_metrics, config.pipeline.gate_ap),
                   robust_started=False, robust_accepted=False, accepted_checkpoint=None,
                   integration_ready=False, test_evaluated=False)
    (directory/'clean_assessment.json').write_text(json.dumps(summary, indent=2))
    if not summary['clean_gate'] or not config.run_robustness or data.subset_frac < 1:
        summary['reason'] = ('clean AP gate failed; no robustness training' if not summary['clean_gate']
                             else 'robustness training disabled or subset smoke')
        summary['accepted_checkpoint'] = clean_path if summary['clean_gate'] else None
        (directory/'completion.json').write_text(json.dumps(summary, indent=2))
        return summary
    robust_overrides = OmegaConf.load(source_root/'conf/benchmark_pose_autoencoder_robust.yaml')
    robust_config = OmegaConf.merge(config, {name:robust_overrides[name] for name in ('pipeline', 'training')})
    original_hash = encoder_hash(clean.model)
    robust, robust_path = fit_phase(robust_config, directory/'robust', data, source_root,
                                     initial=clean.model, statistics=(mean, std))
    if encoder_hash(robust.model) != original_hash:
        raise RuntimeError('Frozen encoder changed during robustness training')
    robust_prediction, robust_target, robust_latents = collect_outputs(robust.model, data.val_dataloader(), device)
    robust_clean_metrics = evaluator.evaluate(robust_prediction, robust_target)
    robust_curve = noise_curve(robust.model, robust_target, robust_latents, std, evaluator, device)
    (directory/'robust_noise_curve.json').write_text(json.dumps(robust_curve, indent=2))
    control05 = [row for row in control_curve if row['noise_fraction'] == .05]
    robust05 = [row for row in robust_curve if row['noise_fraction'] == .05]
    accepted = clean_gate(robust_clean_metrics, config.pipeline.gate_ap) and robust_gain(control05, robust05)
    summary.update(robust_started=True, robust_checkpoint=robust_path, robust_validation=robust_clean_metrics,
                   robust_accepted=bool(accepted), accepted_checkpoint=robust_path if accepted else clean_path,
                   encoder_sha256=original_hash, reason='robust accepted' if accepted else 'retain clean control')
    (directory/'completion.json').write_text(json.dumps(summary, indent=2))
    return summary
