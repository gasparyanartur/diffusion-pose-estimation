#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from src.data import DataConfig
from src.models.pipelines.pose_estimators.poseflow.audit_v43 import run_codec_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--limit-images', type=int)
    parser.add_argument('--skip-validation-replay', action='store_true')
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_poseflow_v40')
    data = DataConfig.from_hydra_config(config.data)
    result = run_codec_audit(data, arguments.output, device=arguments.device,
        workers=arguments.workers, batch_size=arguments.batch_size, limit=arguments.limit_images,
        replay_validation=not arguments.skip_validation_replay)
    print(json.dumps({name: result[name] for name in ('images', 'complete_trainval', 'uncropped', 'augmented')}, indent=2))


if __name__ == '__main__':
    main()
