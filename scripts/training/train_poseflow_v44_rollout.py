import json
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch

from src.data import DataConfig
from src.models.pipelines.pose_estimators.poseflow.rollout_v44 import PoseFlowV44RolloutPipeline, warmstart_rollout
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v44 import run_v44_evaluation
from src.models.pipelines.pose_estimators.poseflow.training_v43 import validate_statistics_data
from src.models.pipelines.pose_estimators.poseflow.trajectory_v44 import evaluate_trajectory_gap
from src.training import TrainingConfig, TrainingRunner, TrainingSpec
from src.training.callbacks import TrustedCheckpointIO
from scripts.training.train_poseflow_v44 import digest, model_digest, split_manifest
from scripts.training.train_poseflow_v44_comparison import ComparisonProgress


@hydra.main(version_base='1.3', config_path='../../conf', config_name='benchmark_poseflow_v44_rollout')
def main(config):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(HydraConfig.get().runtime.output_dir)
    training = TrainingConfig.from_hydra_config(config.training)
    if training.resume or training.skip_training or training.test_after_fit:
        raise ValueError('Rollout comparison requires a fresh bounded TRAIN/VAL arm')
    source = Path(config.source_checkpoint)
    if not config.source_sha256 or digest(source) != config.source_sha256:
        raise ValueError('Rollout source checkpoint SHA256 mismatch')
    runner = TrainingRunner(training, output)
    runner.seed_everything()
    record = torch.load(source, map_location='cpu', weights_only=False)
    module = warmstart_rollout(record, state_policy=config.arm, max_epochs=training.max_epochs)
    source_step = record['global_step']
    initial_hash = model_digest(module.model)
    if config.expected_initial_model_sha256 and initial_hash != config.expected_initial_model_sha256:
        raise ValueError('Rollout initial weights differ from the shared verified source')
    runner.seed_everything()
    data_config = DataConfig.from_hydra_config(config.data)
    if data_config.test_ann != data_config.val_ann:
        raise ValueError('Unused eager test slot must point to VAL')
    validate_statistics_data(module.codec_provenance, data_config)
    data = data_config.build(pin_memory=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(config=OmegaConf.to_container(config, resolve=True), splits=split_manifest(data),
        state_policy=module.training_state_policy(), source_checkpoint=str(source), source_sha256=config.source_sha256,
        source_epoch=record['epoch'], source_global_step=source_step, initial_model_sha256=initial_hash,
        pipeline=module.config.model_dump(), codec_provenance=module.codec_provenance,
        parameters=sum(parameter.numel() for parameter in module.parameters()),
        inputs={str(path): digest(path) for path in (data.train_ann_path, data.val_ann_path)},
        source_files={str(path): digest(path) for folder in ('src', 'conf', 'scripts')
            for path in Path(folder).rglob('*') if path.suffix in ('.py', '.yaml')},
        pairing='seed42 after identical initialization; first eight augmented batches hashed; private diffusion RNG copied',
        exposure='Frozen codec/ED-Pose prior trainval exposure retained; TRAIN-only optimization, no TEST',
        budget='2epochs/2500updates each; batch4/accum2; freshAdamW2e-5/backbone1e-5; no LR drop',
        difference='Only trajectory training input: forward-corrupted target versus detached generated state',
        final_evaluation='selected full2000VAL K1/K4; final matched-age fixed32 K4 gap; no automatic follow-up')
    del record
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    try:
        (output/'status.json').write_text(json.dumps(dict(phase='starting', pid=os.getpid(), arm=config.arm))+'\n')
        result = runner.run(module, data, max_epochs=training.max_epochs,
            spec=TrainingSpec(plugins=[TrustedCheckpointIO()], callbacks=[ComparisonProgress(output, source_step)]))
        if config.final_evaluation:
            device = module.trainer.strategy.root_device
            module.cpu()
            for optimizer in module.trainer.optimizers:
                optimizer.state.clear()
            torch.cuda.empty_cache()
            restored = PoseFlowV44RolloutPipeline.load_from_checkpoint(result.checkpoint, weights_only=False).to(device).eval()
            record = torch.load(result.checkpoint, map_location='cpu', weights_only=False)
            report = run_v44_evaluation(restored.model, data, output/'full_validation', epoch=record['epoch'],
                count=None, steps=(1, 4), seed=restored.config.seed, levels=restored.config.small_noise_levels)
            report.update(checkpoint=result.checkpoint, checkpoint_sha256=digest(result.checkpoint),
                continuation_steps=record['global_step'])
            del restored, record
            torch.cuda.empty_cache()
            last = output/'checkpoints/last.ckpt'
            restored = PoseFlowV44RolloutPipeline.load_from_checkpoint(last, weights_only=False).to(device).eval()
            ids = json.loads((output/'quick_test/subset.json').read_text())['image_ids']
            report['last_checkpoint_trajectory'] = evaluate_trajectory_gap(restored, data, output/'last_trajectory', ids)
            report['last_checkpoint_sha256'] = digest(last)
            (output/'final_results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        (output/'status.json').write_text(json.dumps(dict(phase='complete', pid=os.getpid(), arm=config.arm))+'\n')
    except BaseException as error:
        (output/'status.json').write_text(json.dumps(dict(phase='failed', pid=os.getpid(), arm=config.arm, error=repr(error)))+'\n')
        raise


if __name__ == '__main__':
    main()
