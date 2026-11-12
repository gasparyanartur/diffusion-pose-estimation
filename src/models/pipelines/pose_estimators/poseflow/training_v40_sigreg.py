from typing import ClassVar

import torch
from pydantic import Field

from .latent_v40_sigreg import latent_distribution_metrics, sigreg_penalty
from .model_v40_sigreg import PoseFlowV40SigRegModel
from .training_v40 import PoseFlowV40Pipeline, PoseFlowV40PipelineConfig


class PoseFlowV40SigRegPipeline(PoseFlowV40Pipeline):
    def __init__(self, model, *, config, **kwargs):
        super().__init__(model, config=config, **kwargs)
        generator = torch.Generator().manual_seed(config.sigreg_seed)
        self.register_buffer('sigreg_rng_state', generator.get_state())

    def _training_output_options(self):
        return dict(detach_corruption_source=self.v26_config.detach_corruption_source)

    def _latent_reconstruction_loss(self, clean, recovered, latent_target, targets):
        if self.v26_config.detach_latent_target:
            return super()._latent_reconstruction_loss(clean, recovered, latent_target, targets)
        positive = targets['positive']
        prediction = recovered['pred_latents']
        difference = (prediction[positive]-clean['pred_latents'][positive]).square()
        return difference.sum()/(positive.sum().clamp_min(1)*prediction.shape[-1])

    def _clean_regularization(self, clean, targets, *, training, batch_idx):
        codes = clean['pred_latents'][targets['positive']]
        config = self.v26_config
        enabled = config.sigreg_weight > 0 and len(codes) >= 2
        self._diagnostics['sigreg/count'] = codes.new_tensor(len(codes))
        self._diagnostics['sigreg/skipped'] = codes.new_tensor(not enabled)
        self._diagnostics['latent_target_detached'] = codes.new_tensor(config.detach_latent_target)
        for name, value in latent_distribution_metrics(codes).items():
            self._diagnostics[f'latents/clean/{name}'] = value
        if not enabled:
            raw = codes.float().sum()*0
        else:
            generator = torch.Generator()
            if training:
                generator.set_state(self.sigreg_rng_state.cpu())
            else:
                generator.manual_seed(config.sigreg_seed+batch_idx)
            directions = torch.randn(self.model.hidden_dim, config.sigreg_projections, generator=generator)
            directions /= directions.norm(dim=0)
            raw = sigreg_penalty(codes, directions.to(codes.device),
                                knots=config.sigreg_knots, t_max=config.sigreg_t_max)
            if training:
                self.sigreg_rng_state.copy_(generator.get_state().to(self.sigreg_rng_state.device))
        weighted = config.sigreg_weight*raw
        self._diagnostics['sigreg/raw'] = raw.detach()
        self._diagnostics['sigreg/weighted'] = weighted.detach()
        return dict(sigreg=raw, total=weighted)

    @torch.no_grad()
    def _record_reference_diagnostics(self, candidates, references, targets, proposal_targets,
                                      recovered, clean, times, losses, latent_target=None):
        super()._record_reference_diagnostics(candidates, references, targets, proposal_targets,
            recovered, clean, times, losses, latent_target=latent_target)
        for name, value in latent_distribution_metrics(recovered['pred_latents'][targets['positive']]).items():
            self._diagnostics[f'latents/recovered/{name}'] = value


class PoseFlowV40SigRegPipelineConfig(PoseFlowV40PipelineConfig):
    pipeline_class: ClassVar[type] = PoseFlowV40SigRegPipeline
    model_class: ClassVar[type] = PoseFlowV40SigRegModel
    detach_corruption_source: bool = False
    detach_latent_target: bool = True
    sigreg_weight: float = Field(default=.05, ge=0, allow_inf_nan=False)
    sigreg_projections: int = Field(default=256, ge=1)
    sigreg_knots: int = Field(default=17, ge=2)
    sigreg_t_max: float = Field(default=3., gt=0, allow_inf_nan=False)
    sigreg_seed: int = Field(default=42, ge=0)
