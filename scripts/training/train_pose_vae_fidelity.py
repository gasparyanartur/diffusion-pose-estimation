#!/usr/bin/env python3
import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.models.pipelines.pose_vae.fidelity_experiment import run_fidelity_experiment


@hydra.main(version_base='1.3', config_path='../../conf', config_name='benchmark_pose_vae_fidelity')
def main(config: DictConfig):
    summary = run_fidelity_experiment(config, Path(HydraConfig.get().runtime.output_dir),
                                       Path(__file__).resolve().parents[2])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
