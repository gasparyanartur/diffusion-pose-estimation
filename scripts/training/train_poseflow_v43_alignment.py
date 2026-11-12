#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch

from src.data import DataConfig
from src.models.pipelines.pose_estimators.poseflow.alignment_v43 import run_alignment_evaluation
from src.models.pipelines.pose_estimators.poseflow.training_v43 import PoseFlowV43Pipeline, PoseFlowV43PipelineConfig
from src.training import TrainingConfig, TrainingRunner, TrainingSpec
from src.training.callbacks import TrustedCheckpointIO
from src.utils import resolve_checkpoint


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_digest(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


class AlignmentProgress(pl.Callback):
    def __init__(self, output):
        self.output, self.last_step = output, -1

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        step = trainer.global_step
        if not step or step == self.last_step or step % 50:
            return
        self.last_step = step
        status = dict(phase='training', pid=os.getpid(), epoch=trainer.current_epoch,
            step=step, max_epochs=trainer.max_epochs,
            loss=float(outputs['loss'].detach()), lr=trainer.optimizers[0].param_groups[0]['lr'])
        (self.output/'status.json').write_text(json.dumps(status, allow_nan=False)+'\n')
        print('ALIGNMENT_PROGRESS', json.dumps(status), flush=True)


def alignment_splits(train_path, val_path):
    train = json.loads(Path(train_path).read_text())
    val = json.loads(Path(val_path).read_text())
    train_ids = {image['id'] for image in train['images']}
    val_ids = {image['id'] for image in val['images']}
    if not train_ids or not val_ids or train_ids & val_ids:
        raise ValueError('Image alignment requires nonempty disjoint TRAIN/VAL images')
    return dict(train_images=len(train_ids), val_images=len(val_ids), overlap=0,
        train_annotation_sha256=digest(train_path), val_annotation_sha256=digest(val_path))


@hydra.main(version_base='1.3', config_path='../../conf', config_name='benchmark_poseflow_v43_alignment')
def main(config):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(HydraConfig.get().runtime.output_dir)
    training = TrainingConfig.from_hydra_config(config.training)
    if training.test_after_fit or training.skip_training:
        raise ValueError('Image alignment is TRAIN/VAL only; no test or evaluation-only mode')
    if training.resume:
        training = training.model_copy(update=dict(resume=resolve_checkpoint(training.resume)))
    runner = TrainingRunner(training, output)
    runner.seed_everything()
    data_config = DataConfig.from_hydra_config(config.data)
    if data_config.test_ann != data_config.val_ann:
        raise ValueError('Unused eager test slot must point to VAL, leaving TEST unopened')
    data = data_config.build(pin_memory=training.accelerator in ('gpu', 'cuda'))
    splits = alignment_splits(data.train_ann_path, data.val_ann_path)
    specification = PoseFlowV43PipelineConfig.from_hydra_config(config.pipeline)
    if specification.training_task != 'image_alignment':
        raise ValueError('Expected image-only alignment task')
    module = specification.build(max_epochs=training.max_epochs, load_pretrained_checkpoint=not bool(training.resume))
    initial_digest = model_digest(module.model)
    if config.expected_initial_model_sha256 and not training.resume and initial_digest != config.expected_initial_model_sha256:
        raise ValueError('Initial model differs from verified preflight initialization')
    files = [path for folder in ('src', 'conf') for path in Path(folder).rglob('*') if path.suffix in ('.py', '.yaml')]
    files.append(Path(__file__))
    inputs = [data.train_ann_path, data.val_ann_path]
    if not training.resume:
        inputs.extend(Path(path) for path in (specification.codec_checkpoint, specification.latent_statistics_file,
            specification.pretrained_checkpoint) if path)
    manifest = dict(config=OmegaConf.to_container(config, resolve=True), splits=splits,
        parameters=sum(parameter.numel() for parameter in module.parameters()),
        trainable_parameters=sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad),
        initial_model_sha256=initial_digest,
        initialization=module.model.initialization_report, codec_provenance=module.codec_provenance,
        sources={str(path): digest(path) for path in files}, inputs={str(path): digest(path) for path in inputs},
        training_task='image_alignment',
        initialization_scope='Fresh seed42 V43 prediction body, existing pretrained EDPose frontend; no V43 parent warmstart',
        exposure='TRAIN/VAL disjoint for this stage. Fixed codec standardization uses prior augmented trainval geometry; '
            'pretrained image frontend may have prior trainval exposure. No independent detector-test claim.',
        retention='Best full-VAL decoded geometry loss and last only',
        decision='Held-out paired geometry and native AP with correct-versus-wrong image conditioning; '
            'no downstream diffusion stage is launched automatically')
    output.mkdir(parents=True, exist_ok=True)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    try:
        (output/'status.json').write_text(json.dumps(dict(phase='training', max_epochs=training.max_epochs))+'\n')
        result = runner.run(module, data, max_epochs=training.max_epochs,
            spec=TrainingSpec(plugins=[TrustedCheckpointIO()], callbacks=[AlignmentProgress(output)]))
        if config.final_evaluation:
            paths = dict(best=Path(result.checkpoint), last=output/'checkpoints/last.ckpt')
            final = {}
            device = module.trainer.strategy.root_device
            module.cpu()
            torch.cuda.empty_cache()
            for label, path in paths.items():
                (output/'status.json').write_text(json.dumps(dict(phase='full_validation', checkpoint=label))+'\n')
                restored = PoseFlowV43Pipeline.load_from_checkpoint(path, weights_only=False).to(device).eval()
                record = torch.load(path, map_location='cpu', weights_only=False)
                final[label] = run_alignment_evaluation(restored.model, data, output/'full_validation'/label,
                    epoch=record['epoch'], count=None, matcher=restored.matcher)
                final[label].update(checkpoint=str(path), checkpoint_sha256=digest(path))
                del restored, record
                torch.cuda.empty_cache()
            (output/'final_results.json').write_text(json.dumps(final, indent=2, allow_nan=False)+'\n')
        (output/'status.json').write_text(json.dumps(dict(phase='complete', next_stage_launched=False))+'\n')
    except BaseException as error:
        (output/'status.json').write_text(json.dumps(dict(phase='failed', error=repr(error)))+'\n')
        raise


if __name__ == '__main__':
    main()
