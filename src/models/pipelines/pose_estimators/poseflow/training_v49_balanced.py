from copy import deepcopy

from pydantic import Field
import torch

from .calibration_v49_balanced import fixed_scales
from .codec_v43 import geometry_from_batch
from .losses_v49_balanced import TASK_NAMES, row_losses
from .training_v49 import PoseFlowV49Pipeline, PoseFlowV49PipelineConfig


class PoseFlowV49BalancedPipelineConfig(PoseFlowV49PipelineConfig):
    calibration_batches: int = Field(default=32, ge=1)
    calibration_seed: int = 42049

    def build(self, *, max_epochs=80, load_pretrained_checkpoint=True, **kwargs):
        base = super().build(max_epochs=max_epochs, load_pretrained_checkpoint=load_pretrained_checkpoint, **kwargs)
        return PoseFlowV49BalancedPipeline(config=self, model=base.model, max_epochs=max_epochs)


class PoseFlowV49BalancedPipeline(PoseFlowV49Pipeline):
    def __init__(self, *, config, model=None, max_epochs=80):
        specification = config if isinstance(config, PoseFlowV49BalancedPipelineConfig) else PoseFlowV49BalancedPipelineConfig(**config)
        super().__init__(config=specification, model=model, max_epochs=max_epochs)
        self.register_buffer('gradient_scales', torch.ones(4))
        self.register_buffer('calibration_ready', torch.tensor(False))
        self.register_buffer('calibration_samples', torch.tensor(0, dtype=torch.long))
        self.calibration_provenance = None

    def set_calibration(self, norms, provenance):
        if self.calibration_ready:
            raise RuntimeError('V49 gradient calibration is already frozen')
        if len(norms) != self.config.calibration_batches:
            raise ValueError('V49 calibration batch count differs from contract')
        scales = fixed_scales(norms)
        self.gradient_scales.copy_(scales.to(self.gradient_scales))
        self.calibration_ready.fill_(True)
        self.calibration_samples.fill_(len(norms))
        self.calibration_provenance = dict(deepcopy(provenance), norms=deepcopy(norms), scales=scales.tolist(),
                                           task_names=list(TASK_NAMES))

    def _forward_losses(self, batch, generator, scales):
        geometry, locations = geometry_from_batch(batch)
        rows, sources = self.training_rows(batch, geometry, locations, generator=generator)
        context = self.model.prepare_context(batch['image'], batch.get('padding_mask'))
        output = self.model.forward_rows(context, **{name: rows[name] for name in self.model.row_input_fields})
        losses = row_losses(output, rows, sources, geometry, batch['img_hw'].flip(-1), scales=scales,
            coordinate_weight=self.config.coordinate_loss_weight, geometry_weight=self.config.geometry_loss_weight)
        losses['debug/people'] = sources['counts'].sum().to(losses['total'])
        return losses

    def calibration_losses(self, batch, batch_idx):
        generator = torch.Generator().manual_seed(self.config.calibration_seed+batch_idx)
        return self._forward_losses(batch, generator, torch.ones_like(self.gradient_scales))

    def _losses_for_batch(self, batch, *, training=True, batch_idx=0):
        if not self.calibration_ready:
            raise RuntimeError('V49-balanced requires frozen TRAIN gradient calibration')
        generator = None if training else torch.Generator().manual_seed(self.config.seed+batch_idx)
        losses = self._forward_losses(batch, generator, self.gradient_scales)
        for name, scale in zip(TASK_NAMES, self.gradient_scales):
            losses[f'calibration_scale_{name}'] = scale.detach()
        return losses

    def checkpoint_contract(self):
        contract = super().checkpoint_contract()
        contract.update(architecture='V49-balanced', variant='robust_gradient_age_boxpose_v1',
            supervision='both_layers;per_person;birth_half;refine_half;present_age_buckets_1_2_older_equal',
            objectives=dict(pose='10*masked_joint_coordinate_mean+.5*radial_Huber_OKS_exponent',
                box='10*four_box_coordinate_mean+.5*(Huber_relative_center+Huber_log_size+1-GIoU)',
                huber='r^2_if_abs(r)<=1_else_2abs(r)-1',
                priorities=dict(birth_box_pose=[.75,.25], age1_box_pose=[.5,.5], older_box_pose=[.25,.75]),
                eos='per_image_present_group_renormalization;person.5_firstEOS.25_tailEOS.25'),
            calibration=dict(tasks=list(TASK_NAMES), batches=self.config.calibration_batches,
                seed=self.config.calibration_seed, source='augmented_TRAIN_at_fresh_initialization',
                rule='fixed_median_shared_decoder_gradient_L2;floor1e-8;overall_scale1'))
        return contract

    def on_save_checkpoint(self, checkpoint):
        if not self.calibration_ready:
            raise ValueError('Cannot save uncalibrated V49-balanced checkpoint')
        super().on_save_checkpoint(checkpoint)
        checkpoint['v49_balanced_calibration'] = deepcopy(self.calibration_provenance)

    def on_load_checkpoint(self, checkpoint):
        provenance = checkpoint.get('v49_balanced_calibration')
        state = checkpoint['state_dict']
        if (not provenance or provenance.get('task_names') != list(TASK_NAMES)
                or len(provenance.get('norms', [])) != self.config.calibration_batches
                or not state.get('calibration_ready', False)
                or int(state.get('calibration_samples', 0)) != self.config.calibration_batches):
            raise ValueError('Invalid V49-balanced calibration provenance')
        expected = fixed_scales(provenance['norms'])
        if (not torch.equal(state['gradient_scales'].cpu(), expected)
                or provenance.get('scales') != expected.tolist()):
            raise ValueError('Corrupted V49-balanced calibration scales')
        super().on_load_checkpoint(checkpoint)
        self.calibration_provenance = deepcopy(provenance)
