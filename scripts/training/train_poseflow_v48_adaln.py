import json
import os
import time
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch

from src.data import DataConfig
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v48 import run_v48_evaluation, tensorboard_metrics
from src.models.pipelines.pose_estimators.poseflow.training_v48_adaln import (
    PoseFlowV48AdaLNPipeline as PoseFlowV48Pipeline, PoseFlowV48AdaLNPipelineConfig as PoseFlowV48PipelineConfig)
from src.training import TrainingConfig, TrainingRunner, TrainingSpec
from src.training.callbacks import TrustedCheckpointIO
from src.utils import resolve_checkpoint
from scripts.training.train_poseflow_v46 import digest, model_digest, split_manifest


def source_hashes():
    return {str(path): digest(path) for folder in ('src', 'conf', 'scripts')
            for path in Path(folder).rglob('*') if path.suffix in ('.py', '.yaml')}


def validate_verification(path, config, initial_hash):
    if not path:
        raise ValueError('V48 full training requires completed native preflight and evaluator verification')
    report = json.loads(Path(path).read_text())
    if (not report.get('passed') or not report.get('native_preflight', {}).get('passed')
            or report.get('evaluator_smoke', {}).get('images') != 2):
        raise ValueError('V48 verification did not pass; full training is blocked')
    if (not report.get('common_V48_initialization_verified')
            or not report.get('native_preflight', {}).get('adaln_gradients_verified')):
        raise ValueError('V48-AdaLN conditioning verification is incomplete')
    if report['initial_model_sha256'] != initial_hash or report['sources'] != source_hashes():
        raise ValueError('V48 initialization/source differs from verified implementation')
    if report['pipeline_config'] != OmegaConf.to_container(config.pipeline, resolve=True):
        raise ValueError('V48 pipeline differs from verification')
    if report['data_config'] != OmegaConf.to_container(config.data, resolve=True):
        raise ValueError('V48 data differs from verification')
    if any(digest(filename) != checksum for filename, checksum in report['inputs'].items()):
        raise ValueError('V48 input assets changed after verification')
    return report


class V48Progress(pl.Callback):
    def __init__(self, output):
        self.output, self.last_step = output, -1

    def on_train_start(self, trainer, module):
        self.started, self.start_step = time.perf_counter(), trainer.global_step

    def on_fit_start(self, trainer, module):
        if not module.config.quick_eval_enabled or trainer.global_step:
            return
        report = run_v48_evaluation(module.model, trainer.datamodule, self.output/'quick_test', epoch=-1,
            count=module.config.quick_eval_images, seed=module.config.seed)
        for logger in trainer.loggers:
            logger.log_metrics(tensorboard_metrics(report), step=0)
            if hasattr(logger.experiment, 'flush'):
                logger.experiment.flush()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        step = trainer.global_step
        if not step or step == self.last_step or step % 10:
            return
        self.last_step = step
        elapsed = time.perf_counter()-self.started
        throughput = (step-self.start_step)*trainer.datamodule.batch_size*trainer.accumulate_grad_batches/max(elapsed, 1e-9)
        measured = {'train/images_per_second': throughput,
                    'train/peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
        status = dict(phase='training', pid=os.getpid(), epoch=trainer.current_epoch, step=step,
            max_epochs=trainer.max_epochs, loss=float(outputs['loss'].detach()),
            lr=trainer.optimizers[0].param_groups[0]['lr'], **measured)
        for logger in trainer.loggers:
            logger.log_metrics(measured, step=step)
            if hasattr(logger.experiment, 'flush'):
                logger.experiment.flush()
        (self.output/'status.json').write_text(json.dumps(status, allow_nan=False)+'\n')
        print('V48_ADALN_PROGRESS', json.dumps(status), flush=True)


@hydra.main(version_base='1.3', config_path='../../conf', config_name='benchmark_poseflow_v48_adaln')
def main(config):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(HydraConfig.get().runtime.output_dir)
    training = TrainingConfig.from_hydra_config(config.training)
    if (training.test_after_fit or training.skip_training or training.max_epochs != 80
            or str(training.precision) not in ('32', '32-true')
            or training.accumulate_grad_batches != 2 or config.data.batch_size != 4):
        raise ValueError('V48 full run requires FP32/full80/batch4/accum2 and TRAIN/VAL only')
    if training.resume:
        training = training.model_copy(update=dict(resume=resolve_checkpoint(training.resume)))
    runner = TrainingRunner(training, output)
    runner.seed_everything()
    data_config = DataConfig.from_hydra_config(config.data)
    if data_config.test_ann != data_config.val_ann:
        raise ValueError('Unused eager test slot must point to VAL')
    data = data_config.build(pin_memory=True)
    specification = PoseFlowV48PipelineConfig.from_hydra_config(config.pipeline)
    module = specification.build(max_epochs=80, load_pretrained_checkpoint=not bool(training.resume))
    initial_hash = model_digest(module.model)
    verification = None if training.resume else validate_verification(config.verification_file, config, initial_hash)
    splits = split_manifest(data)
    if splits['train_images'] != 10000 or splits['val_images'] != 2000:
        raise ValueError('V48 requires original TRAIN10k/VAL2k split')
    expected = config.get('expected_initial_model_sha256')
    if expected and not training.resume and initial_hash != expected:
        raise ValueError('V48 initialization differs from expected hash')
    inputs = [data.train_ann_path, data.val_ann_path]
    if not training.resume:
        inputs.extend((specification.pretrained_checkpoint, config.verification_file))
    manifest = dict(config=OmegaConf.to_container(config, resolve=True), splits=splits,
        parameters=sum(parameter.numel() for parameter in module.parameters()),
        trainable_parameters=sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad),
        initialization=module.model.initialization_report, initial_model_sha256=initial_hash,
        sources=source_hashes(), inputs={str(path): digest(path) for path in inputs if path},
        contract=module.checkpoint_contract(), verification=config.verification_file,
        exposure='V48-AdaLN training uses TRAIN only; VAL for model selection. ED-Pose pretraining exposure retained; TEST unopened.',
        retention='best min(pose_AP,bbox_AP) on fixed32VAL and last; selected checkpoint full2000VAL')
    output.mkdir(parents=True, exist_ok=True)
    manifest_name = f'resume_manifest_{time.time_ns()}.json' if training.resume else 'manifest.json'
    (output/manifest_name).write_text(json.dumps(manifest, indent=2)+'\n')
    if verification:
        subset = dict(seed=specification.seed, requested_images=32, annotations=str(data.val_ann_path),
                      image_ids=verification['fixed_val_image_ids'])
        (output/'quick_test').mkdir(exist_ok=True)
        (output/'quick_test/subset.json').write_text(json.dumps(subset, indent=2)+'\n')
    try:
        (output/'status.json').write_text(json.dumps(dict(phase='starting', pid=os.getpid()))+'\n')
        result = runner.run(module, data, max_epochs=80,
            spec=TrainingSpec(plugins=[TrustedCheckpointIO()], callbacks=[V48Progress(output)]))
        if config.final_evaluation:
            device = module.trainer.strategy.root_device
            module.cpu()
            torch.cuda.empty_cache()
            restored = PoseFlowV48Pipeline.load_from_checkpoint(result.checkpoint, weights_only=False).to(device).eval()
            record = torch.load(result.checkpoint, map_location='cpu', weights_only=False)
            report = run_v48_evaluation(restored.model, data, output/'full_validation', epoch=record['epoch'],
                count=None, seed=specification.seed)
            report.update(checkpoint=result.checkpoint, checkpoint_sha256=digest(result.checkpoint))
            (output/'final_results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        (output/'status.json').write_text(json.dumps(dict(phase='complete', pid=os.getpid()))+'\n')
    except BaseException as error:
        (output/'status.json').write_text(json.dumps(dict(phase='failed', pid=os.getpid(), error=repr(error)))+'\n')
        raise


if __name__ == '__main__':
    main()
