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
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v53 import run_v53_evaluation, tensorboard_metrics
from src.models.pipelines.pose_estimators.poseflow.training_v53 import PoseFlowV53Pipeline, PoseFlowV53PipelineConfig
from src.models.pipelines.pose_estimators.poseflow.calibration_v53 import fixed_scales
from src.training import TrainingConfig, TrainingRunner, TrainingSpec
from src.training.callbacks import TrustedCheckpointIO
from src.utils import resolve_checkpoint
from scripts.training.train_poseflow_v46 import digest, model_digest, split_manifest


def source_hashes():
    return {str(path): digest(path) for folder in ('src', 'conf', 'scripts')
            for path in Path(folder).rglob('*') if path.suffix in ('.py', '.yaml', '.cpp', '.cu', '.h', '.cuh')}


def validate_verification(path, config, initial_hash):
    if not path:
        raise ValueError('V53 full training requires completed native preflight and evaluator verification')
    report = json.loads(Path(path).read_text())
    if (not report.get('passed') or not report.get('native_preflight', {}).get('passed')
            or report.get('evaluator_smoke', {}).get('images') != 2):
        raise ValueError('V53 verification did not pass; full training is blocked')
    if report['initial_model_sha256'] != initial_hash or report['sources'] != source_hashes():
        raise ValueError('V53 initialization/source differs from verified implementation')
    if report['pipeline_config'] != OmegaConf.to_container(config.pipeline, resolve=True):
        raise ValueError('V53 pipeline differs from verification')
    if report['data_config'] != OmegaConf.to_container(config.data, resolve=True):
        raise ValueError('V53 data differs from verification')
    if any(digest(filename) != checksum for filename, checksum in report['inputs'].items()):
        raise ValueError('V53 input assets changed after verification')
    calibration = report.get('calibration', {})
    if (len(calibration.get('norms', [])) != config.pipeline.calibration_batches
            or calibration.get('model_sha256_before') != initial_hash
            or calibration.get('model_sha256_after') != initial_hash):
        raise ValueError('V53 calibration provenance differs from initialization')
    if calibration.get('scales') != fixed_scales(calibration['norms']).tolist():
        raise ValueError('V53 calibration scales differ from measured norms')
    return report


class V53Progress(pl.Callback):
    def __init__(self, output, *, evaluate_on_resume=False):
        self.output, self.last_step = output, -1
        self.evaluate_on_resume = evaluate_on_resume

    def on_train_start(self, trainer, module):
        self.started, self.start_step = time.perf_counter(), trainer.global_step
        if self.evaluate_on_resume and trainer.global_step and module.config.quick_eval_enabled:
            report = run_v53_evaluation(module.model, trainer.datamodule, self.output/'quick_test',
                epoch=trainer.current_epoch, count=module.config.quick_eval_images, seed=module.config.seed)
            module.log_detection_report(report, step=trainer.global_step)

    def on_fit_start(self, trainer, module):
        if not module.config.quick_eval_enabled or trainer.global_step or trainer.ckpt_path:
            return
        report = run_v53_evaluation(module.model, trainer.datamodule, self.output/'quick_test', epoch=-1,
            count=module.config.quick_eval_images, seed=module.config.seed)
        module.log_detection_report(report, step=0)

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
        print('V53_PROGRESS', json.dumps(status), flush=True)


@hydra.main(version_base='1.3', config_path='../../conf', config_name='benchmark_poseflow_v53')
def main(config):
    run_training(config)


def run_training(config, *, pipeline_config_class=PoseFlowV53PipelineConfig,
                 pipeline_class=PoseFlowV53Pipeline, validation=validate_verification, initialize=None):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from src.models.pipelines.pose_estimators.diffusionregpose.transformer import _load_native_msda
    backend = _load_native_msda()
    if backend is None or not hasattr(backend, 'backward'):
        raise RuntimeError('V53 training requires native CUDA deformable attention; fallback is forbidden')
    output = Path(HydraConfig.get().runtime.output_dir)
    training = TrainingConfig.from_hydra_config(config.training)
    if (training.test_after_fit or training.skip_training or training.max_epochs != 80
            or str(training.precision) not in ('32', '32-true')
            or training.accumulate_grad_batches != 2 or config.data.batch_size != 4):
        raise ValueError('V53 full run requires FP32/full80/batch4/accum2 and explicit final evaluation')
    if training.resume:
        training = training.model_copy(update=dict(resume=resolve_checkpoint(training.resume)))
    runner = TrainingRunner(training, output)
    runner.seed_everything()
    data_config = DataConfig.from_hydra_config(config.data)
    if Path(data_config.test_ann).name != 'crowdpose_test.json':
        raise ValueError('Quick-test requires official crowdpose_test.json, not a VAL alias')
    data = data_config.build(pin_memory=True)
    specification = pipeline_config_class.from_hydra_config(config.pipeline)
    module = specification.build(max_epochs=80, load_pretrained_checkpoint=not bool(training.resume))
    initial_hash = model_digest(module.model)
    verification = None if training.resume else validation(config.verification_file, config, initial_hash)
    if verification and verification['contract'] != module.checkpoint_contract():
        raise ValueError('V53 verification contract differs from model')
    if verification:
        if initialize is not None:
            initialize(module, verification)
        module.set_calibration(verification['calibration']['norms'], verification['calibration'])
    splits = split_manifest(data)
    if splits['train_images'] != 10000 or splits['val_images'] != 2000:
        raise ValueError('V53 requires original TRAIN10k/VAL2k split')
    expected = config.get('expected_initial_model_sha256')
    if expected and not training.resume and initial_hash != expected:
        raise ValueError('V53 initialization differs from expected hash')
    test_annotations = json.loads(Path(data.test_ann_path).read_text())
    test_ids = {row['id'] for row in test_annotations['images']}
    trainval_ids = {row['id'] for path in (data.train_ann_path, data.val_ann_path)
                    for row in json.loads(Path(path).read_text())['images']}
    if len(test_ids) != 8000 or test_ids & trainval_ids:
        raise ValueError('TEST must contain 8000 images disjoint from TRAIN and VAL')
    splits.update(test_images=len(test_ids), test_trainval_overlap=0)
    inputs = [data.train_ann_path, data.val_ann_path, data.test_ann_path]
    if not training.resume:
        inputs.extend((specification.pretrained_checkpoint, config.verification_file))
    manifest = dict(config=OmegaConf.to_container(config, resolve=True), splits=splits,
        parameters=sum(parameter.numel() for parameter in module.parameters()),
        trainable_parameters=sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad),
        initialization=module.model.initialization_report, initial_model_sha256=initial_hash,
        sources=source_hashes(), inputs={str(path): digest(path) for path in inputs if path},
        contract=module.checkpoint_contract(), calibration=module.calibration_provenance, verification=config.verification_file,
        exposure='Training uses TRAIN only; VAL losses are potentially pretrained-exposed diagnostics. '
            'Fixed32 TEST is used for development/model selection, not an untouched holdout.',
        retention='best min(pose_AP,bbox_AP) on fixed32 TEST and last; selected checkpoint full8000 TEST')
    if training.resume:
        record = torch.load(training.resume, map_location='cpu', weights_only=False)
        manifest['resume'] = dict(checkpoint=training.resume, checkpoint_sha256=digest(training.resume),
            epoch=record['epoch'], global_step=record['global_step'],
            selection='New checkpoint directory resets prior split selection scores; optimizer/scheduler/RNG restored')
        del record
    output.mkdir(parents=True, exist_ok=True)
    if initialize is not None and verification:
        manifest['prefix_bank'] = module.prefix_bank_provenance
    manifest_name = f'resume_manifest_{time.time_ns()}.json' if training.resume else 'manifest.json'
    (output/manifest_name).write_text(json.dumps(manifest, indent=2)+'\n')
    try:
        (output/'status.json').write_text(json.dumps(dict(phase='starting', pid=os.getpid()))+'\n')
        result = runner.run(module, data, max_epochs=80,
            spec=TrainingSpec(plugins=[TrustedCheckpointIO()], callbacks=[V53Progress(output,
                evaluate_on_resume=config.get('evaluate_on_resume', False))]))
        if config.final_evaluation:
            device = module.trainer.strategy.root_device
            module.cpu()
            torch.cuda.empty_cache()
            restored = pipeline_class.load_from_checkpoint(result.checkpoint, weights_only=False).to(device).eval()
            record = torch.load(result.checkpoint, map_location='cpu', weights_only=False)
            report = run_v53_evaluation(restored.model, data, output/'full_test', epoch=record['epoch'],
                count=None, seed=specification.seed)
            report.update(checkpoint=result.checkpoint, checkpoint_sha256=digest(result.checkpoint))
            (output/'final_results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        (output/'status.json').write_text(json.dumps(dict(phase='complete', pid=os.getpid()))+'\n')
    except BaseException as error:
        (output/'status.json').write_text(json.dumps(dict(phase='failed', pid=os.getpid(), error=repr(error)))+'\n')
        raise


if __name__ == '__main__':
    main()
