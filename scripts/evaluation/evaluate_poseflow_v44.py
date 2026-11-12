import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.data import DataConfig
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v44 import run_v44_evaluation
from src.models.pipelines.pose_estimators.poseflow.training_v44 import PoseFlowV44Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--count', type=int)
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Recover the exact run's data configuration, not a possibly changed repository preset.
    manifest = json.loads((arguments.checkpoint.parent.parent / 'manifest.json').read_text())
    data_config = DataConfig.from_hydra_config(OmegaConf.create(manifest['config']['data']))
    data = data_config.build()
    record = torch.load(arguments.checkpoint, map_location='cpu', weights_only=False)
    module = PoseFlowV44Pipeline.load_from_checkpoint(arguments.checkpoint, weights_only=False).to(arguments.device).eval()
    result = run_v44_evaluation(module.model, data, arguments.output, epoch=record['epoch'],
        count=arguments.count, steps=(1, 4), seed=module.config.seed)
    print(json.dumps({name: row['metrics'] for name, row in result['conditions'].items()}, indent=2))


if __name__ == '__main__':
    main()
