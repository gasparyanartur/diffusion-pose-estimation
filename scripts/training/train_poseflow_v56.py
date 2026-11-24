from pathlib import Path

import hydra

from scripts.training.train_poseflow_v53 import digest, run_training, validate_verification
from scripts.training.train_poseflow_v54 import initialize_residual_bank
from src.models.pipelines.pose_estimators.poseflow.training_v56 import PoseFlowV56Pipeline, PoseFlowV56PipelineConfig


def validate_v56_verification(path, config, initial_hash):
    report = validate_verification(path, config, initial_hash)
    if (report.get('contract', {}).get('architecture') != 'V56' or not report.get('prefix_bank')
            or not report.get('gradient_routes', {}).get('passed')
            or not report.get('conditional_parity', {}).get('passed')
            or digest(config.residual_bank_file) != report.get('residual_bank_file_sha256')):
        raise ValueError('V56 full-depth/gradient/bank verification differs from configuration')
    return report


@hydra.main(version_base='1.3', config_path='../../conf', config_name='benchmark_poseflow_v56')
def main(config):
    def initialize(module, report):
        initialize_residual_bank(module, dict(report, residual_bank_file=str(Path(config.residual_bank_file).resolve())))
    run_training(config, pipeline_config_class=PoseFlowV56PipelineConfig, pipeline_class=PoseFlowV56Pipeline,
                 validation=validate_v56_verification, initialize=initialize)


if __name__ == '__main__':
    main()
