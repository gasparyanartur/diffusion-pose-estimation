from pathlib import Path

import hydra
import torch

from scripts.training.train_poseflow_v53 import digest, run_training, validate_verification
from src.models.pipelines.pose_estimators.poseflow.training_v54 import PoseFlowV54Pipeline, PoseFlowV54PipelineConfig


def validate_v54_verification(path, config, initial_hash):
    report = validate_verification(path, config, initial_hash)
    if (report.get('contract', {}).get('architecture') != 'V54' or not report.get('prefix_bank')
            or digest(config.residual_bank_file) != report.get('residual_bank_file_sha256')):
        raise ValueError('V54 empirical prefix bank differs from verification')
    return report


def initialize_residual_bank(module, verification):
    bank = torch.load(verification['residual_bank_file'], map_location='cpu', weights_only=False)
    module.set_residual_bank(bank['residuals'], bank['joint_mask'], bank['provenance'])
    if module.prefix_bank_provenance != verification['prefix_bank']:
        raise ValueError('V54 restored residual bank differs from verified provenance')


@hydra.main(version_base='1.3', config_path='../../conf', config_name='benchmark_poseflow_v54')
def main(config):
    def initialize(module, report):
        archived = dict(report, residual_bank_file=str(Path(config.residual_bank_file).resolve()))
        initialize_residual_bank(module, archived)
    run_training(config, pipeline_config_class=PoseFlowV54PipelineConfig, pipeline_class=PoseFlowV54Pipeline,
                 validation=validate_v54_verification, initialize=initialize)


if __name__ == '__main__':
    main()
