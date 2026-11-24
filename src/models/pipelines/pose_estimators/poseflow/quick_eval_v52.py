import json
from pathlib import Path

from .quick_eval_v50 import run_v50_evaluation, tensorboard_metrics


def run_v52_evaluation(model, data, root, *, epoch, count=32, seed=42):
    report = run_v50_evaluation(model, data, root, epoch=epoch, count=count, seed=seed,
        collect_birth=True, collect_first_eos=True)
    report.update(architecture='V52', readout='joint_clean_box_and_center_relative_pose;256D_features;no_codec')
    (Path(root)/f'epoch{epoch:03d}'/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
