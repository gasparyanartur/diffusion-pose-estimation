#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch

from src.data import DataConfig
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v43 import run_v43_evaluation
from src.models.pipelines.pose_estimators.poseflow.training_v43 import PoseFlowV43PipelineConfig, validate_statistics_data
from src.training import TrainingConfig, TrainingRunner, TrainingSpec
from src.training.callbacks import TrustedCheckpointIO
from src.utils import resolve_checkpoint


@hydra.main(version_base='1.3', config_path='../../conf', config_name='benchmark_poseflow_v43_frozen_vae')
def main(config):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(HydraConfig.get().runtime.output_dir)
    training = TrainingConfig.from_hydra_config(config.training)
    if training.skip_training or config.get('skip_training', False):
        raise ValueError('Use the evaluation entrypoint scripts/evaluation/evaluate_poseflow_v43.py for evaluation-only runs')
    if training.resume:
        training = training.model_copy(update=dict(resume=resolve_checkpoint(training.resume)))
    runner = TrainingRunner(training, output)
    runner.seed_everything()
    data_config = DataConfig.from_hydra_config(config.data)
    data = data_config.build(pin_memory=training.accelerator in ('gpu', 'cuda'))
    specification = PoseFlowV43PipelineConfig.from_hydra_config(config.pipeline)
    module = specification.build(max_epochs=training.max_epochs, load_pretrained_checkpoint=not bool(training.resume))
    if not training.resume:
        validate_statistics_data(module.codec_provenance, data_config)
    files = [path for folder in ('src', 'conf') for path in Path(folder).rglob('*') if path.suffix in ('.py', '.yaml')]
    files.extend((Path(__file__), Path('scripts/evaluation/evaluate_poseflow_v43.py')))
    inputs = [data.train_ann_path, data.val_ann_path, data.test_ann_path]
    if not training.resume:
        inputs.extend(Path(path) for path in (specification.codec_checkpoint, specification.latent_statistics_file,
                                              specification.pretrained_checkpoint) if path)
    manifest = dict(config=OmegaConf.to_container(config, resolve=True), pipeline=specification.model_dump(),
        parameters=sum(parameter.numel() for parameter in module.parameters()),
        trainable_parameters=sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad),
        initialization=module.model.initialization_report, codec_provenance=module.codec_provenance,
        sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        inputs={str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in inputs})
    output.mkdir(parents=True, exist_ok=True)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    runner.run(module, data, max_epochs=training.max_epochs, spec=TrainingSpec(plugins=[TrustedCheckpointIO()]))
    if config.final_evaluation:
        module.to(module.trainer.strategy.root_device)
        run_v43_evaluation(module.model, data, output/'full_test', epoch=training.max_epochs-1,
            global_step=module.global_step, count=None, steps=specification.quick_eval_steps, seed=specification.seed)


if __name__ == '__main__':
    main()
