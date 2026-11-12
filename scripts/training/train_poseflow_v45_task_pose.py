import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from src.data import DataConfig
from src.models.pipelines.pose_estimators.poseflow.training_v45_task_pose import PoseFlowV45TaskPosePipelineConfig
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v42 import run_v42_evaluation
from src.training import TrainingConfig, TrainingRunner, TrainingSpec
from src.training.callbacks import TrustedCheckpointIO


def digest(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):hasher.update(block)
    return hasher.hexdigest()


def save(path,value):
    temporary=path.with_suffix('.part')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


class Progress(pl.Callback):
    def __init__(self,root):
        self.root=root
        self.started=time.monotonic()
        self.status=dict(phase='starting',pid=os.getpid(),global_step=0)
        self.time_counts=torch.zeros(1000,dtype=torch.long)
    def update(self,**values):
        self.status.update(values,updated_unix=time.time())
        save(self.root/'status.json',self.status)
    def on_fit_start(self,trainer,module):
        assert trainer.max_epochs==80 and trainer.accumulate_grad_batches==2 and trainer.gradient_clip_val==.1
        assert module.max_epochs==80 and module.hparams['training_timestep_policy']=='uniform'
        assert module.v42_config.pose_readout_mode=='absolute'
        assert type(module).__name__=='PoseFlowV45TaskPosePipeline'
        assert module.v42_config.ema_decay==.999 and not module.teacher.training
        assert module.hparams['target_assignment']=='proposal_hungarian_fixed'
        assert module.hparams['confidence_mode']=='image_only' and module.model.confidence_mode=='image_only'
        self.model=module.model
        module.model.image_confidence_head.register_forward_hook(self.observe_image_confidence)
        module.model.register_forward_pre_hook(self.observe_timesteps)
        module.model.register_forward_hook(self.observe_pairing)
        self.original_q_sample=module.model.q_sample
        module.model.q_sample=self.observe_corruption
        self.paired_batches=0
        self.paired_people=0
        self.update(phase='training',epochs=80,seed=42,training_timestep_policy='uniform',target_assignment='proposal_hungarian_fixed',confidence_mode='image_only',jepa_weight=module.v42_config.jepa_weight)
    def observe_timesteps(self,model,args):
        if not model.training:return
        self.current_image_scores=None
        timesteps=args[2].detach().cpu()
        assert timesteps.dtype==torch.long and timesteps.shape==(args[0].shape[0],)
        assert ((timesteps>=0)&(timesteps<1000)).all()
        self.time_counts+=torch.bincount(timesteps,minlength=1000)
        active=self.time_counts.nonzero().flatten()
        self.status['timestep_observations']=dict(images=int(self.time_counts.sum()),
            minimum=int(active.min()),maximum=int(active.max()),unique=int(active.numel()),
            mean=float((self.time_counts*torch.arange(1000)).sum()/self.time_counts.sum()),
            bins_100=self.time_counts.reshape(10,100).sum(1).tolist())
    def observe_image_confidence(self,head,args,output):
        if not self.model.training:return
        assert self.current_image_scores is None,'Image scores must be computed once per forward'
        self.current_image_scores=output.detach()
    def observe_corruption(self,clean,timesteps,noise=None):
        if self.model.training:assert self.current_image_scores is not None,'Image scores must precede corruption/injection' 
        self.last_source=(clean.detach(),timesteps.detach())
        return self.original_q_sample(clean,timesteps,noise)
    def observe_pairing(self,model,args,output):
        if not model.training:return
        clean,timesteps=self.last_source
        assert torch.equal(timesteps,args[2])
        targets=args[3]
        assert torch.equal(output['pred_logits'],self.current_image_scores)
        for auxiliary in output['aux_outputs'][model.transformer.decoder.num_box_decoder_layers:]:
            assert torch.equal(auxiliary['pred_logits'],self.current_image_scores)
        matches=output['pose_target_matches']
        people_count=0
        for image,(queries,people) in enumerate(matches):
            expected=(targets[image]['keypoints'][people,:28]*2-1)*model.scale
            assert torch.equal(clean[image,queries],expected),'Paired pose source differs from its loss target'
            assert len(people)==len(targets[image]['labels'])
            assert len(queries.unique())==len(queries) and len(people.unique())==len(people)
            people_count+=len(people)
        self.paired_batches+=1
        self.paired_people+=people_count
        self.status['pairing_observations']=dict(batches=self.paired_batches,matched_sources=self.paired_people,
            source_target_mismatches=0,complete_target_coverage=True,confidence_before_pose_verified=True)
    def on_train_epoch_start(self,trainer,module):
        self.update(phase='training',epoch=trainer.current_epoch)
    def on_before_backward(self,trainer,module,loss):
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite V45 task-pose loss')
    def on_before_optimizer_step(self,trainer,module,optimizer):
        groups={}
        assert all(parameter.grad is None for parameter in module.teacher.parameters())
        for name,parameter in module.model.named_parameters():
            if parameter.grad is None:continue
            if not torch.isfinite(parameter.grad).all():raise FloatingPointError(f'Nonfinite gradient: {name}')
            groups.setdefault(name.split('.')[0],[]).append(parameter.grad.detach().double().square().sum())
        self.status['gradient_norms']={name:float(torch.stack(values).sum().sqrt()) for name,values in groups.items()}
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        if batch_idx and (batch_idx+1)%100:return
        loss=outputs['loss'] if isinstance(outputs,dict) else outputs
        assert int(module.ema_updates)==trainer.global_step
        self.update(phase='training',epoch=trainer.current_epoch,global_step=trainer.global_step,ema_updates=int(module.ema_updates),
            batch=batch_idx+1,batches_per_epoch=trainer.num_training_batches,
            batch_loss=float(loss)*trainer.accumulate_grad_batches,
            elapsed_seconds=time.monotonic()-self.started,
            peak_gpu_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
        print('PROGRESS',json.dumps(self.status),flush=True)
        for logger in trainer.loggers:
            if hasattr(logger.experiment,'flush'):logger.experiment.flush()
    def on_validation_epoch_start(self,trainer,module):
        self.update(phase='validation_and_fixed32',epoch=trainer.current_epoch,global_step=trainer.global_step)
    def on_validation_end(self,trainer,module):
        path=self.root/'seed42/quick_test'/f'epoch{trainer.current_epoch:03d}'/'metrics.json'
        if path.exists():
            report=json.loads(path.read_text())
            self.update(last_fixed32={name:row['metrics'] for name,row in report['conditions'].items()})
            print('EPOCH_RESULT',json.dumps(self.status['last_fixed32']),flush=True)
    def on_exception(self,trainer,module,error):
        path=self.root/'seed42/checkpoints/interrupted.ckpt'
        try:trainer.save_checkpoint(path)
        except Exception as checkpoint_error:self.status['checkpoint_error']=repr(checkpoint_error)
        self.update(phase='cancelled' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error),checkpoint=str(path))


def main(root):
    root=root.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    for name,expected in manifest['source_hashes'].items():assert digest(Path(manifest['workspace'])/name)==expected,name
    for name,expected in manifest['input_hashes'].items():assert digest(name)==expected,name
    config=OmegaConf.load(root/'seed42/.hydra/config.yaml')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    training=TrainingConfig.from_hydra_config(config.training)
    runner=TrainingRunner(training,root/'seed42')
    runner.logger().info('V45 task-defined pose latent; control/JEPA matched objective; uniform t0..999; proposal-Hungarian sources/fixed labels; pre-pose classifier; complete-pose online/teacher reconstruction + optional JEPA; coordinate diffusion; frozen source %s',manifest['workspace'])
    runner.seed_everything()
    data=DataConfig.from_hydra_config(config.data).build(pin_memory=False)
    settings=PoseFlowV45TaskPosePipelineConfig.from_hydra_config(config.pipeline)
    module=settings.build(max_epochs=80)
    initial=hashlib.sha256()
    for name,value in module.model.state_dict().items():
        initial.update(name.encode());initial.update(value.cpu().contiguous().numpy().tobytes())
    assert initial.hexdigest()==manifest['initial_model_sha256']
    assert settings.quick_eval_steps==(1,4,8) and settings.quick_eval_images==32
    assert settings.training_timestep_policy=='uniform' and config.training.resume is None
    assert training.limit_val_batches==1.0, 'Full validation losses must remain enabled'
    progress=Progress(root)
    result=runner.run(module,data,max_epochs=80,spec=TrainingSpec(plugins=[TrustedCheckpointIO()]),callbacks=[progress])
    checkpoint=root/'seed42/checkpoints/epoch079.ckpt'
    assert checkpoint.exists(),checkpoint
    progress.update(phase='full_test_K1_K4_K8',checkpoint=str(checkpoint))
    global_step=module.global_step
    module.cpu();del module,data,runner;gc.collect();torch.cuda.empty_cache()
    # The native evaluator supports this config; avoid the Euler-only model builder.
    from scripts.evaluation.evaluate_diffusionregpose import build_evaluation_model, extract_model_state_dict
    model=build_evaluation_model(settings,sampling_timesteps=1)
    model.load_state_dict(extract_model_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)),strict=True)
    model.cuda().eval()
    data=DataConfig.from_hydra_config(config.data).build(pin_memory=False)
    report=run_v42_evaluation(model,data,settings,root/'seed42/full_test',epoch=79,global_step=global_step,count=None,sampler_description=model.sampler_kind)
    assert report['images']==8000 and report['confidence_mode']=='image_only'
    report.update(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),config_sha256=digest(root/'seed42/.hydra/config.yaml'))
    save(root/'seed42/full_test/metrics.json',report)
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(root/'seed42/tb_logs')) as writer:
        for row in report['conditions'].values():
            for name,value in row['metrics'].items():writer.add_scalar(f"full-test/steps_{row['steps']}/{name}",value,global_step)
    progress.update(phase='complete',full_test=report['conditions'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
