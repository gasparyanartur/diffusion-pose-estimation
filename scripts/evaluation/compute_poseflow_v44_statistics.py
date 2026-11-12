import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
import torch

from src.data import DataConfig
from src.models.pipelines.pose_estimators.poseflow.statistics_v44 import compute_train_statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    with initialize_config_dir(version_base=None, config_dir=str(Path('conf').resolve())):
        config = compose(config_name='benchmark_poseflow_v44')
    result = compute_train_statistics(DataConfig.from_hydra_config(config.data), args.output, device=args.device)
    print(f'V44 TRAIN statistics complete: {result["images"]} images, {result["people"]} people', flush=True)


if __name__ == '__main__':
    main()
