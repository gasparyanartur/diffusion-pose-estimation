#!/usr/bin/env python3
import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
import torch

from src.data import DataConfig
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v43 import run_v43_evaluation
from src.models.pipelines.pose_estimators.poseflow.training_v43 import PoseFlowV43Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--count', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', nargs='+', type=int, default=[1, 4, 8])
    arguments = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    module = PoseFlowV43Pipeline.load_from_checkpoint(arguments.checkpoint, weights_only=False).to(arguments.device).eval()
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_poseflow_v43_frozen_vae')
    data = DataConfig.from_hydra_config(config.data).build()
    state = torch.load(arguments.checkpoint, map_location='cpu', weights_only=False)
    run_v43_evaluation(module.model, data, arguments.output, epoch=state['epoch'], global_step=state['global_step'],
        count=arguments.count, steps=tuple(arguments.steps), seed=arguments.seed)


if __name__ == '__main__':
    main()
