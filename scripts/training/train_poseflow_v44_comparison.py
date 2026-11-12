import hashlib
import json
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch

from src.data import DataConfig
from src.models.pipelines.pose_estimators.poseflow.comparison_v44 import warmstart_comparison
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v44 import run_v44_evaluation
from src.models.pipelines.pose_estimators.poseflow.training_v44 import PoseFlowV44Pipeline
from src.models.pipelines.pose_estimators.poseflow.training_v43 import validate_statistics_data
from src.training import TrainingConfig, TrainingRunner, TrainingSpec
from src.training.callbacks import TrustedCheckpointIO
from scripts.training.train_poseflow_v44 import V44Progress, digest, model_digest, split_manifest


class ComparisonProgress(V44Progress):
    def __init__(self, output, source_step):
        super().__init__(output)
        self.source_step = source_step
        self.first_batches = []

    def on_fit_start(self, trainer, module):
        super().on_fit_start(trainer, module)
        for logger in trainer.loggers:
            logger.log_metrics({'comparison/queued': 0.}, step=trainer.global_step)

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if len(self.first_batches) >= 8:
            return
        tensors = {name: batch[name] for name in ('image_id', 'img_hw', 'gt_keypoints', 'gt_labeled', 'gt_boxes_pixel')}
        value = hashlib.sha256()
        for name, tensor in tensors.items():
            value.update(name.encode())
            value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        self.first_batches.append(dict(image_ids=batch['image_id'].tolist(), geometry_sha256=value.hexdigest()))
        (self.output / 'first_batches.json').write_text(json.dumps(self.first_batches, indent=2) + '\n')

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, module, outputs, batch, batch_idx)
        module.log('comparison/effective_global_step', float(self.source_step + trainer.global_step), on_step=True)


@hydra.main(version_base='1.3', config_path='../../conf', config_name='benchmark_poseflow_v44_comparison')
def main(config):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(HydraConfig.get().runtime.output_dir)
    training = TrainingConfig.from_hydra_config(config.training)
    if training.resume or training.skip_training or training.test_after_fit:
        raise ValueError('Comparison entrypoint starts fresh bounded TRAIN/VAL arms; use explicit checkpoint restoration for resume')
    source = Path(config.source_checkpoint)
    if not config.source_sha256 or digest(source) != config.source_sha256:
        raise ValueError('Comparison source checkpoint SHA256 mismatch')
    record = torch.load(source, map_location='cpu', weights_only=False)
    runner = TrainingRunner(training, output)
    runner.seed_everything()
    module = warmstart_comparison(record, prediction_type=config.arm, max_epochs=training.max_epochs)
    initial_hash = model_digest(module.model)
    if config.expected_initial_model_sha256 and initial_hash != config.expected_initial_model_sha256:
        raise ValueError('Comparison initialization differs from preflight')
    source_step, source_epoch = record['global_step'], record['epoch']
    del record
    # Model construction must not change the paired loader/DN/dropout random stream.
    runner.seed_everything()
    data_config = DataConfig.from_hydra_config(config.data)
    if data_config.test_ann != data_config.val_ann:
        raise ValueError('Unused eager test slot must point to VAL')
    validate_statistics_data(module.codec_provenance, data_config)
    data = data_config.build(pin_memory=True)
    files = [path for folder in ('src', 'conf', 'scripts') for path in Path(folder).rglob('*') if path.suffix in ('.py', '.yaml')]
    manifest = dict(config=OmegaConf.to_container(config, resolve=True), splits=split_manifest(data),
        prediction_type=config.arm, source_checkpoint=str(source), source_sha256=config.source_sha256,
        source_epoch=source_epoch, source_global_step=source_step, initial_model_sha256=initial_hash,
        pipeline=module.config.model_dump(), codec_provenance=module.codec_provenance,
        parameters=sum(value.numel() for value in module.parameters()),
        source_files={str(path): digest(path) for path in files},
        inputs={str(path): digest(path) for path in (data.train_ann_path, data.val_ann_path)},
        initialization='Identical source weights; velocity heads explicitly negated; initial function differs and is evaluated',
        optimizer='Fresh AdamW in both arms; lr2e-5/backbone1e-5, no milestone within10 epochs',
        pairing='Global/data seed42 after construction; source private diffusion RNG copied; initial augmented batches hashed',
        exposure='Frozen codec/ED-Pose prior trainval exposure disclosed; TRAIN-only continuation, fixed32 VAL model selection; no TEST',
        budget='10 epochs / 12500 updates at batch4 accumulation2; best and last only',
        final_evaluation='Selected checkpoint full2000VAL K1/K4; no automatic further training')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    try:
        (output / 'status.json').write_text(json.dumps(dict(phase='starting', pid=os.getpid(), arm=config.arm)) + '\n')
        result = runner.run(module, data, max_epochs=training.max_epochs,
            spec=TrainingSpec(plugins=[TrustedCheckpointIO()], callbacks=[ComparisonProgress(output, source_step)]))
        if config.final_evaluation:
            device = module.trainer.strategy.root_device
            module.cpu()
            for optimizer in module.trainer.optimizers:
                optimizer.state.clear()
            torch.cuda.empty_cache()
            restored = PoseFlowV44Pipeline.load_from_checkpoint(result.checkpoint, weights_only=False).to(device).eval()
            record = torch.load(result.checkpoint, map_location='cpu', weights_only=False)
            report = run_v44_evaluation(restored.model, data, output / 'full_validation', epoch=record['epoch'],
                count=None, steps=(1, 4), seed=restored.config.seed, levels=restored.config.small_noise_levels)
            report.update(checkpoint=result.checkpoint, checkpoint_sha256=digest(result.checkpoint),
                source_global_step=source_step, continuation_steps=record['global_step'])
            (output / 'final_results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        (output / 'status.json').write_text(json.dumps(dict(phase='complete', pid=os.getpid(), arm=config.arm)) + '\n')
    except BaseException as error:
        (output / 'status.json').write_text(json.dumps(dict(phase='failed', pid=os.getpid(), arm=config.arm, error=repr(error))) + '\n')
        raise


if __name__ == '__main__':
    main()
