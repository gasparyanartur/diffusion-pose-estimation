"""Append native V21 test diagnostics for an existing training run."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

from omegaconf import OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter

from scripts.utils.quick_test_logging import comparison_tag
from src.data import DataConfig
from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v21 import run_v21_quick_test


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def evaluate_checkpoint(run, checkpoint_path, device):
    config_path = run / '.hydra/config.yaml'
    resolved = OmegaConf.load(config_path)
    config = PoseFlowPipelineConfig.from_hydra_config(resolved.pipeline)
    if not config.quick_eval_enabled:
        raise ValueError('V21 quick evaluation is disabled in the saved configuration')
    before = checkpoint_path.stat()
    checkpoint_hash = file_hash(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    after = checkpoint_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Checkpoint changed while being read; retry after its writer finishes')
    epoch, step = int(checkpoint['epoch']), int(checkpoint['global_step'])
    metrics_path = run / 'quick_test' / f'epoch{epoch:03d}' / 'metrics.json'
    if metrics_path.exists():
        report = json.loads(metrics_path.read_text())
        if report['global_step'] != step:
            raise ValueError('Existing quick-test report has a different checkpoint step')
        source_hash = report.get('source', {}).get('checkpoint_sha256')
        if source_hash is not None and source_hash != checkpoint_hash:
            raise ValueError('Existing quick-test report has different checkpoint weights')
    else:
        model = config.model_copy(update={'backbone_pretrained': False,
                                          'pretrained_checkpoint': None})._build_poseflow_model()
        model.load_state_dict({key.removeprefix('model.'): value
                               for key, value in checkpoint['state_dict'].items()
                               if key.startswith('model.')}, strict=True)
        del checkpoint
        model.to(device).eval()
        data = DataConfig.from_hydra_config(resolved.data).build(pin_memory=device.type == 'cuda')
        report = run_v21_quick_test(model, data, config, run / 'quick_test', epoch=epoch,
            global_step=step, source={'kind': 'checkpoint sidecar', 'checkpoint': str(checkpoint_path),
                'checkpoint_sha256': checkpoint_hash, 'config_sha256': file_hash(config_path)})
    # A completion marker follows the flush, so interrupted logging can be retried.
    marker = metrics_path.with_name('tensorboard_logged.json')
    if not marker.exists():
        with SummaryWriter(str(run / 'tb_logs'), filename_suffix='.v21_quick_test') as writer:
            for condition, row in report['conditions'].items():
                for metric, value in row['metrics'].items():
                    if value >= 0:
                        writer.add_scalar(comparison_tag(f'quick-test/{condition}/{metric}'), value, step)
            writer.flush()
        marker.write_text(json.dumps({'epoch': epoch, 'global_step': step,
                                      'checkpoint_sha256': checkpoint_hash}) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error('--poll-seconds must be positive')
    run = args.run.resolve()
    root = run / 'quick_test'
    root.mkdir(exist_ok=True)
    lock = (root / 'watcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(2)
    # Match the existing FP32 training run and record these flags in every report.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    max_epochs = int(OmegaConf.load(run / '.hydra/config.yaml').training.max_epochs)
    while True:
        checkpoints = sorted((run / 'checkpoints').glob('epoch[0-9][0-9][0-9].ckpt'))
        if checkpoints:
            latest = checkpoints[-1]
            epoch = int(latest.stem.removeprefix('epoch'))
            marker = root / f'epoch{epoch:03d}' / 'tensorboard_logged.json'
            if not marker.exists():
                # The trainer writes checkpoints in place; avoid files still being written.
                if time.time() - latest.stat().st_mtime < 10:
                    time.sleep(10)
                    continue
                report = evaluate_checkpoint(run, latest, device)
                print(json.dumps({'epoch': report['epoch'], 'global_step': report['global_step'],
                    'pose_AP': {name: row['metrics']['pose_AP']
                                for name, row in report['conditions'].items()}}), flush=True)
            if args.once or epoch >= max_epochs - 1:
                return
        elif args.once:
            raise FileNotFoundError('No completed epoch checkpoint found')
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
