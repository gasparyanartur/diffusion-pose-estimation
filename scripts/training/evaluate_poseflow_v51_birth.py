import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

from omegaconf import OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter

from src.data import DataConfig
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v51 import run_v51_evaluation, birth_tensorboard_metrics
from src.models.pipelines.pose_estimators.poseflow.training_v51 import PoseFlowV51Pipeline


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def verify_final_parity(report, original):
    for name in ('image_ids', 'seed', 'resize', 'detections'):
        if report[name] != original[name]:
            raise ValueError(f'Final evaluation parity failed: {name}')
    differences = {}
    for name, expected in original['metrics'].items():
        actual = report['metrics'][name]
        if not math.isfinite(actual) or abs(actual-expected) > 1e-8:
            raise ValueError(f'Final evaluation parity failed: {name}: {actual} != {expected}')
        differences[name] = actual-expected
    return dict(passed=True, absolute_tolerance=1e-8, metric_differences=differences)


def evaluate_checkpoint(checkpoint, run, output):
    snapshot = output/'checkpoint.ckpt'
    before = checkpoint.stat()
    shutil.copyfile(checkpoint, snapshot)
    after = checkpoint.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        snapshot.unlink()
        return None
    record = torch.load(snapshot, map_location='cpu', weights_only=False)
    epoch = int(record['epoch'])
    trainer_report = run/f'seed42/quick_test/epoch{epoch:03d}/metrics.json'
    destination = output/f'epoch{epoch:03d}'
    if not trainer_report.exists() or (destination/'provenance.json').exists():
        snapshot.unlink()
        return None
    original = json.loads(trainer_report.read_text())
    module = PoseFlowV51Pipeline(**record['hyper_parameters'])
    module.on_load_checkpoint(record)
    module.load_state_dict(record['state_dict'], strict=True)
    step = int(record['global_step'])
    del record
    manifest = json.loads((run/'seed42/manifest.json').read_text())
    data_settings = dict(manifest['config']['data'])
    if not Path(data_settings['root']).is_absolute():
        data_settings['root'] = str(run/'workspace'/data_settings['root'])
    configuration = DataConfig.from_hydra_config(OmegaConf.create(data_settings))
    if configuration.test_ann != configuration.val_ann:
        raise ValueError('Unused test slot must point to VAL')
    data = configuration.build()
    expected_subset = json.loads((run/'seed42/quick_test/subset.json').read_text())
    subset = output/'subset.json'
    if subset.exists() and json.loads(subset.read_text()) != expected_subset:
        raise ValueError('Diagnostic fixed32 VAL subset changed')
    subset.write_text(json.dumps(expected_subset, indent=2)+'\n')
    module.cuda().eval()
    report = run_v51_evaluation(module.model, data, output, epoch=epoch,
        count=len(expected_subset['image_ids']), seed=original['seed'])
    parity = verify_final_parity(report, original)
    sources = {str(path): file_digest(path) for directory in ('src', 'scripts', 'conf')
               for path in Path(directory).rglob('*') if path.suffix in ('.py', '.yaml')}
    provenance = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_digest(snapshot),
        epoch=epoch, global_step=step, final_parity=parity, sources=sources,
        inputs={str(data.val_ann_path): file_digest(data.val_ann_path)},
        trainer_report_sha256=file_digest(trainer_report),
        scope='Same-trajectory birth and final geometry; fixed32 VAL; trainer untouched; TEST unopened')
    for stage, metrics in (('birth', birth_tensorboard_metrics(report)),
            ('final', {f'quick-test/{name}': value for name, value in report['metrics'].items()})):
        with SummaryWriter(str(output/'tb_logs'/stage)) as writer:
            for name, value in metrics.items():
                writer.add_scalar(name, value, step)
            writer.flush()
    (destination/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    snapshot.unlink()
    print(json.dumps(dict(epoch=epoch, step=step, birth=report['birth_metrics'],
        final=report['metrics'], parity=parity)), flush=True)
    del module, data
    gc.collect()
    torch.cuda.empty_cache()
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = output/'status.json'
    processed = set()
    try:
        while True:
            checkpoints = sorted((arguments.run_root/'seed42/checkpoints').glob('epoch*.ckpt'))
            last = arguments.run_root/'seed42/checkpoints/last.ckpt'
            if last.exists():
                checkpoints.append(last)
            status.write_text(json.dumps(dict(phase='evaluating', pid=os.getpid()))+'\n')
            for checkpoint in checkpoints:
                try:
                    information = checkpoint.stat()
                    identity = (str(checkpoint), information.st_size, information.st_mtime_ns)
                    if identity in processed:
                        continue
                    evaluate_checkpoint(checkpoint, arguments.run_root, output)
                    processed.add(identity)
                except FileNotFoundError:
                    # Retention may remove an older checkpoint before it is copied.
                    continue
            training = json.loads((arguments.run_root/'seed42/status.json').read_text())
            if not arguments.watch or training['phase'] in ('complete', 'failed'):
                status.write_text(json.dumps(dict(phase='complete', pid=os.getpid(), trainer=training['phase']))+'\n')
                break
            if not Path(f'/proc/{training["pid"]}').exists():
                raise RuntimeError('Trainer exited without a terminal status')
            status.write_text(json.dumps(dict(phase='waiting_for_checkpoint', pid=os.getpid()))+'\n')
            time.sleep(30)
    except BaseException as error:
        status.write_text(json.dumps(dict(phase='failed', pid=os.getpid(), error=repr(error)))+'\n')
        raise


if __name__ == '__main__':
    main()
