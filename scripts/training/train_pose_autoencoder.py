#!/usr/bin/env python3
from pathlib import Path
import json

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.models.pipelines.pose_autoencoder.experiment import run_experiment


@hydra.main(version_base='1.3', config_path='../../conf', config_name='benchmark_pose_autoencoder')
def main(config: DictConfig):
    directory = Path(HydraConfig.get().runtime.output_dir)
    summary = run_experiment(config, directory, Path(__file__).resolve().parents[2])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
