from copy import deepcopy
from pathlib import Path

import torch
from pydantic import Field

from .balance_v51 import GradientBalance, clip_families, gradient_families
from .losses_v49_balanced import TASK_NAMES
from .model_v51 import PoseFlowV51Model
from .training_v49 import PoseFlowV49Pipeline
from .training_v50 import PoseFlowV50Pipeline, PoseFlowV50PipelineConfig


class PoseFlowV51PipelineConfig(PoseFlowV50PipelineConfig):
    calibration_seed: int = 42051
    gradient_probe_interval: int = Field(default=100, ge=1)

    def create_model(self, native=None):
        from ..diffusionregpose.model import DiffusionRegPoseModel
        from .training_v44 import NATIVE_MODEL_FIELDS
        config = self.native_config()
        if native is None:
            native = DiffusionRegPoseModel(backbone_pretrained=False,
                separate_expansion_pose_head=config.pretrained_keep_decoder_layers is not None,
                **{name:getattr(config,name) for name in NATIVE_MODEL_FIELDS})
        return PoseFlowV51Model(native=native, rows=self.rows, capacity=self.capacity, scale=self.scale,
            hidden_dim=config.hidden_dim, num_heads=config.num_heads, feedforward_dim=config.feedforward_dim,
            codec_width=self.codec_width, latent_dim=self.latent_dim, activation_checkpointing=self.activation_checkpointing)

    def build(self, *, max_epochs=80, load_pretrained_checkpoint=True, **kwargs):
        base = super().build(max_epochs=max_epochs,load_pretrained_checkpoint=load_pretrained_checkpoint,**kwargs)
        return PoseFlowV51Pipeline(config=self,model=base.model,max_epochs=max_epochs)


class PoseFlowV51Pipeline(PoseFlowV50Pipeline):
    def __init__(self, *, config, model=None, max_epochs=80):
        specification = config if isinstance(config,PoseFlowV51PipelineConfig) else PoseFlowV51PipelineConfig(**config)
        super().__init__(config=specification,model=model,max_epochs=max_epochs)
        self.balance = GradientBalance(4)
        self.register_buffer('optimization_updates',torch.zeros((),dtype=torch.long))
        self.register_buffer('last_probe_update',torch.full((),-1,dtype=torch.long))

    def probe_groups(self):
        return (tuple(self.model.person_codec.encoder.parameters()), tuple(self.model.latent_input.parameters()),
                tuple(self.model.layers[0].parameters()), tuple(self.model.layers[1].parameters()))

    def measure_gradients(self, losses):
        groups = self.probe_groups()
        parameters = tuple(parameter for group in groups for parameter in group)
        rows = []
        for name in TASK_NAMES:
            gradients = torch.autograd.grad(losses[f'{name}_contribution'], parameters,retain_graph=True,allow_unused=True)
            offset, norms = 0, []
            for group in groups:
                values = gradients[offset:offset+len(group)]
                squared = sum(gradient.detach().float().square().sum() for gradient in values if gradient is not None)
                count = sum(parameter.numel() for parameter in group)
                norms.append(torch.as_tensor(squared,device=self.gradient_scales.device).div(count).sqrt())
                offset += len(group)
            rows.append(torch.stack(norms))
        return torch.stack(rows)

    def set_gradient_calibration(self, norms, provenance):
        if self.calibration_ready or len(norms) != self.config.calibration_batches:
            raise ValueError('Invalid V51 initial gradient calibration')
        for values in norms:
            self.balance.update(values)
        self.calibration_ready.fill_(True)
        self.calibration_samples.fill_(len(norms))
        self.calibration_provenance = dict(deepcopy(provenance),norms=deepcopy(norms),task_names=list(TASK_NAMES),
            initial_weights=self.balance.weights.tolist(),scales=[1.,1.,1.,1.])

    def _apply_balance(self, losses):
        result = dict(losses)
        for index,name in enumerate(TASK_NAMES):
            result[f'unweighted/{name}_contribution'] = losses[f'{name}_contribution']
            result[f'{name}_contribution'] = losses[f'{name}_contribution']*self.balance.weights[index].detach().clone()
            result[f'gradient_weight/{name}'] = self.balance.weights[index].detach().clone()
            for probe,label in enumerate(('encoder','projection','decoder0','decoder1')):
                result[f'gradient_ema/{name}/{label}'] = self.balance.ema[index,probe].detach().clone()
        result['total'] = sum(result[f'{name}_contribution'] for name in TASK_NAMES)+sum(result[name] for name in ('eos','selection','frontend'))
        for name in ('prefix_rms','time_rms_before_cap','time_rms_after_cap'):
            if hasattr(self.model,name):
                result[f'conditioning/{name}'] = getattr(self.model,name)
        return result

    def _losses_for_batch(self, batch, *, training=True, batch_idx=0):
        if not self.calibration_ready:
            raise RuntimeError('V51 requires TRAIN-only gradient calibration')
        generator = None if training else torch.Generator().manual_seed(self.config.seed+batch_idx)
        losses = self._forward_losses(batch,generator,torch.ones_like(self.gradient_scales))
        step = int(self.optimization_updates)
        if training and step % self.config.gradient_probe_interval == 0 and int(self.last_probe_update) != step:
            self.balance.update(self.measure_gradients(losses))
            self.last_probe_update.fill_(step)
        return self._apply_balance(losses)

    def on_before_optimizer_step(self, optimizer):
        norms = [parameter.grad.detach().norm() for parameter in self.parameters() if parameter.grad is not None]
        if norms and not torch.isfinite(torch.stack(norms)).all():
            raise FloatingPointError('Nonfinite V51 gradients')
        self.optimization_updates.add_(1)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        if gradient_clip_val is not None and abs(float(gradient_clip_val)-.1) > 1e-8:
            raise ValueError('V51 combined clipping contract requires .1')
        values = clip_families(gradient_families(self.model))
        if self._trainer is not None:
            self.log_dict({f'train/gradients/{name}':value for name,value in values.items()},on_step=True,on_epoch=False)
            time_weight = self.model.time_embedding[-1].weight.detach()
            self.log('train/gradients/time_output_weight_rms',time_weight.square().mean().sqrt())
        return values

    def optimizer_step(self, *args, **kwargs):
        probes = dict(encoder=self.model.person_codec.encoder[0].weight,
            decoder=self.model.layers[0].self_attention.in_proj_weight,
            selector=self.model.selector.key.weight, refinement=self.model.residual_heads[1].weight,
            time=self.model.time_embedding[-1].weight)
        before = {name:parameter.detach().clone() for name,parameter in probes.items()}
        super().optimizer_step(*args, **kwargs)
        if self._trainer is not None:
            self.log_dict({f'train/updates/{name}_relative':(parameter.detach()-before[name]).norm()/before[name].norm().clamp_min(1e-12)
                           for name,parameter in probes.items()},on_step=True,on_epoch=False)

    def on_validation_epoch_end(self):
        if not self.config.quick_eval_enabled or self.trainer.sanity_checking:
            return
        from .quick_eval_v51 import run_v51_evaluation, tensorboard_metrics
        callback = self.trainer.checkpoint_callback
        root = Path(callback.dirpath).parent if callback else Path(self.trainer.default_root_dir)
        report = run_v51_evaluation(self.model,self.trainer.datamodule,root/'quick_test',epoch=self.current_epoch,
            count=self.config.quick_eval_images,seed=self.config.seed)
        self.log_dict(tensorboard_metrics(report))
        self.log('val/gradient_balanced_selection_AP',min(report['metrics']['pose_AP'],report['metrics']['bbox_AP']))

    def checkpoint_contract(self):
        contract = super().checkpoint_contract()
        contract.update(architecture='V51',variant='protected_balanced_clean_anchor_v1',
            identity='V50_detached_geometry_identity;selector_reads_detached_geometry_prefix;candidate_keys_trainable',
            residual='separate_fresh_clean_latent_anchor_heads_copied_from_birth_heads;shared_coordinate_reader;gate=min(sigma/.1,1);linear_XY_and_log_size_interpolation;exact_sigma0_identity',
            time_conditioning='per_token_sinusoidal_sigma999_MLP;additive_RMS_cap1',
            calibration=dict(tasks=list(TASK_NAMES),batches=self.config.calibration_batches,seed=self.config.calibration_seed,
                probes=['coordinate_encoder','coordinate_projection','decoder0','decoder1'],metric='per_block_task_RMS_shares_geometric_mean',
                ema=.95,weights=[.1,10.],prior=[.125,.375,1/3,1/6],interval=self.config.gradient_probe_interval),
            gradient_clipping='after_accumulation;disjoint_native_selector_geometry;each_.1/sqrt3;combined<=.1')
        return contract

    def on_load_checkpoint(self, checkpoint):
        provenance = checkpoint.get('v49_balanced_calibration')
        state = checkpoint['state_dict']
        if (not provenance or provenance.get('task_names') != list(TASK_NAMES)
                or len(provenance.get('norms',[])) != self.config.calibration_batches
                or not state.get('calibration_ready',False)
                or int(state.get('calibration_samples',0)) != self.config.calibration_batches
                or not torch.equal(state['gradient_scales'].cpu(),torch.ones(4))
                or state['balance.ema'].shape != (4,4) or not torch.isfinite(state['balance.ema']).all()
                or (state['balance.ema'] < 0).any()
                or int(state['balance.measurements']) < self.config.calibration_batches
                or not torch.equal(state['balance.prior'].cpu(),torch.tensor([.125,.375,1/3,1/6]))
                or state['balance.weights'].shape != (4,) or not torch.isfinite(state['balance.weights']).all()
                or (state['balance.weights'] < .1).any() or (state['balance.weights'] > 10).any()):
            raise ValueError('Invalid V51 gradient calibration/state')
        initial = GradientBalance(4)
        for norms in provenance['norms']:
            initial.update(norms)
        expected_weights = torch.tensor(provenance.get('initial_weights',[]),dtype=initial.weights.dtype)
        if expected_weights.shape != (4,) or not torch.allclose(initial.weights,expected_weights,rtol=2e-6,atol=2e-6):
            raise ValueError('Corrupt V51 calibration measurements')
        PoseFlowV49Pipeline.on_load_checkpoint(self,checkpoint)
        self.calibration_provenance = deepcopy(provenance)
