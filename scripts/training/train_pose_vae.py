#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.models.pipelines.pose_vae.training import PoseGeometryDataModule, PoseVAEConfig
from src.training import TrainingConfig, TrainingRunner
from src.training.callbacks import TrustedCheckpointIO


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_provenance(directory: Path, config, data, module):
    root = Path(__file__).resolve().parents[2]
    sources = [path for folder in ('src', 'conf') for path in (root/folder).rglob('*')
               if path.suffix in ('.py', '.yaml')]
    sources.extend((Path(__file__).resolve(), root/'pyproject.toml'))
    annotation_hashes = {split: dict(path=str(path.resolve()), sha256=file_hash(path))
                         for split, path in data.paths.items()}
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
                    source_root=str(root), python=platform.python_version(), torch=torch.__version__,
                    cuda=torch.version.cuda, parameters=sum(parameter.numel() for parameter in module.parameters()),
                    annotations=annotation_hashes,
                    fit_people={split: len(dataset) for split, dataset in data.datasets.items()},
                    train_val_image_overlap=len(data.datasets['train'].image_ids & data.datasets['val'].image_ids),
                    subset_frac=data.subset_frac,
                    source_sha256={str(path.relative_to(root)): file_hash(path) for path in sorted(sources)})
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2))
    OmegaConf.save(config, directory/'resolved.yaml', resolve=True)


@hydra.main(version_base='1.3', config_path='../../conf', config_name='benchmark_pose_vae')
def main(config: DictConfig):
    directory = Path(HydraConfig.get().runtime.output_dir)
    runtime = TrainingConfig.from_hydra_config(config.training)
    runner = TrainingRunner(runtime, directory)
    logger = runner.logger()
    runner.seed_everything()
    data = PoseGeometryDataModule(**OmegaConf.to_container(config.data, resolve=True))
    data.setup('fit')
    module = PoseVAEConfig.from_hydra_config(config.pipeline).build(
        max_epochs=runtime.max_epochs, artifact_directory=str(directory/'artifacts'))
    write_provenance(directory, config, data, module)
    logger.info('Pose VAE parameters=%d; train=%d, val=%d; no image decoding',
                sum(parameter.numel() for parameter in module.parameters()),
                len(data.datasets['train']), len(data.datasets['val']))
    result = runner.run(module, data, max_epochs=runtime.max_epochs, plugins=[TrustedCheckpointIO()])
    summary = dict(checkpoint=result.checkpoint, test_results=str(result.result_path) if result.result_path else None,
                   metrics=result.metrics, completed_epochs=runner.config.max_epochs)
    (directory/'completion.json').write_text(json.dumps(summary, indent=2))
    logger.info('Complete: %s', json.dumps(summary))


if __name__ == '__main__':
    main()
