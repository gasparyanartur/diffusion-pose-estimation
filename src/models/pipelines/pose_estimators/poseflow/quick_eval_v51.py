import json
from pathlib import Path

from .quick_eval_v50 import run_v50_evaluation, tensorboard_metrics as previous_metrics


def run_v51_evaluation(*args, **kwargs):
    kwargs.setdefault('collect_birth', True)
    report = run_v50_evaluation(*args, **kwargs)
    report['architecture'] = 'V51'
    root = Path(args[2] if len(args) > 2 else kwargs['root'])
    (root/f"epoch{kwargs['epoch']:03d}"/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def tensorboard_metrics(report):
    return previous_metrics(report)


def birth_tensorboard_metrics(report):
    return {f'quick-test/{name}': value for name, value in report['birth_metrics'].items()}
