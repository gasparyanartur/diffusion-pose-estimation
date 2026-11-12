import hashlib
import json
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch

from src.data import DataConfig
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v44 import run_v44_evaluation
from src.models.pipelines.pose_estimators.poseflow.training_v44 import PoseFlowV44Pipeline, PoseFlowV44PipelineConfig
from src.models.pipelines.pose_estimators.poseflow.training_v43 import validate_statistics_data
from src.training import TrainingConfig, TrainingRunner, TrainingSpec
from src.training.callbacks import TrustedCheckpointIO
from src.utils import resolve_checkpoint


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_digest(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


class V44Progress(pl.Callback):
    def __init__(self, output):
        self.output, self.last_step = output, -1

    def on_fit_start(self, trainer, module):
        if not module.config.quick_eval_enabled or trainer.global_step:
            return
        report = run_v44_evaluation(module.model, trainer.datamodule, self.output / 'quick_test', epoch=-1,
            count=module.config.quick_eval_images, steps=module.config.quick_eval_steps, seed=module.config.seed,
            levels=module.config.small_noise_levels)
        metrics = {f'quick-test/{condition}/{name}': value for condition, row in report['conditions'].items()
            for name, value in row['metrics'].items()}
        metrics.update({f'quick-test/{name}': value for name, value in report['small_noise'].items()})
        for logger in trainer.loggers:
            logger.log_metrics(metrics, step=0)
            if hasattr(logger.experiment, 'flush'):
                logger.experiment.flush()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        step = trainer.global_step
        if not step or step == self.last_step or step % 25:
            return
        self.last_step = step
        status = dict(phase='training', pid=os.getpid(), epoch=trainer.current_epoch, step=step,
            max_epochs=trainer.max_epochs, loss=float(outputs['loss'].detach()),
            lr=trainer.optimizers[0].param_groups[0]['lr'])
        (self.output / 'status.json').write_text(json.dumps(status, allow_nan=False) + '\n')
        print('V44_PROGRESS', json.dumps(status), flush=True)


def split_manifest(data):
    train = json.loads(Path(data.train_ann_path).read_text())
    val = json.loads(Path(data.val_ann_path).read_text())
    train_ids = {row['id'] for row in train['images']}
    val_ids = {row['id'] for row in val['images']}
    if not train_ids or not val_ids or train_ids & val_ids:
        raise ValueError('V44 requires nonempty disjoint TRAIN/VAL')
    return dict(train_images=len(train_ids), val_images=len(val_ids), overlap=0)


@hydra.main(version_base='1.3', config_path='../../conf', config_name='benchmark_poseflow_v44')
def main(config):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(HydraConfig.get().runtime.output_dir)
    training = TrainingConfig.from_hydra_config(config.training)
    if training.test_after_fit or training.skip_training:
        raise ValueError('V44 training uses TRAIN/VAL only')
    if training.resume:
        training = training.model_copy(update=dict(resume=resolve_checkpoint(training.resume)))
    runner = TrainingRunner(training, output)
    runner.seed_everything()
    data_config = DataConfig.from_hydra_config(config.data)
    if data_config.test_ann != data_config.val_ann:
        raise ValueError('Unused eager test slot must point to VAL')
    data = data_config.build(pin_memory=True)
    specification = PoseFlowV44PipelineConfig.from_hydra_config(config.pipeline)
    module = specification.build(max_epochs=training.max_epochs, load_pretrained_checkpoint=not bool(training.resume))
    initial_hash = model_digest(module.model)
    expected = config.get('expected_initial_model_sha256')
    if expected and not training.resume and initial_hash != expected:
        raise ValueError('V44 initialization differs from verified preflight')
    if not training.resume:
        validate_statistics_data(module.codec_provenance, data_config)
    files = [path for folder in ('src', 'conf', 'scripts') for path in Path(folder).rglob('*') if path.suffix in ('.py', '.yaml')]
    inputs = [data.train_ann_path, data.val_ann_path]
    if not training.resume:
        inputs.extend(path for path in (specification.codec_checkpoint, specification.latent_statistics_file,
            specification.pretrained_checkpoint) if path)
    manifest = dict(config=OmegaConf.to_container(config, resolve=True), splits=split_manifest(data),
        parameters=sum(parameter.numel() for parameter in module.parameters()),
        trainable_parameters=sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad),
        initialization=module.model.initialization_report, initial_model_sha256=initial_hash,
        codec_provenance=module.codec_provenance,
        sources={str(path): digest(path) for path in files}, inputs={str(path): digest(path) for path in inputs},
        exposure='Codec and ED-Pose are pretrained and may have prior trainval exposure; V44 training and statistics use TRAIN only. '
            'VAL monitors model selection; TEST remains unopened. VAL is not independent pretrained-detector evidence.',
        retention='best max-over-K min(pose_AP,bbox_AP) on fixed32 VAL and last; final full2000 VAL K1/K4',
        contract='frozen posterior-mean 32D codec; native cached time-independent frontend; fixed selected-slot assignment')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    try:
        (output / 'status.json').write_text(json.dumps(dict(phase='starting', pid=os.getpid())) + '\n')
        result = runner.run(module, data, max_epochs=training.max_epochs,
            spec=TrainingSpec(plugins=[TrustedCheckpointIO()], callbacks=[V44Progress(output)]))
        if config.final_evaluation:
            device = module.trainer.strategy.root_device
            module.cpu()
            torch.cuda.empty_cache()
            restored = PoseFlowV44Pipeline.load_from_checkpoint(result.checkpoint, weights_only=False).to(device).eval()
            record = torch.load(result.checkpoint, map_location='cpu', weights_only=False)
            report = run_v44_evaluation(restored.model, data, output / 'full_validation', epoch=record['epoch'],
                count=None, steps=(1, 4), seed=specification.seed)
            report.update(checkpoint=result.checkpoint, checkpoint_sha256=digest(result.checkpoint))
            (output / 'final_results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        (output / 'status.json').write_text(json.dumps(dict(phase='complete', pid=os.getpid())) + '\n')
    except BaseException as error:
        (output / 'status.json').write_text(json.dumps(dict(phase='failed', pid=os.getpid(), error=repr(error))) + '\n')
        raise


if __name__ == '__main__':
    main()
