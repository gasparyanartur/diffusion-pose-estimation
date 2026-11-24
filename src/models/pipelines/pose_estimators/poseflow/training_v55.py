import math

import torch

from ..diffusionregpose.model import DiffusionRegPoseModel
from .model_v55 import PoseFlowV55Model
from .balance_v51 import clip_families_with_budgets, gradient_families
from .training_v44 import NATIVE_MODEL_FIELDS
from .training_v54 import PoseFlowV54Pipeline, PoseFlowV54PipelineConfig


class PoseFlowV55PipelineConfig(PoseFlowV54PipelineConfig):
    def create_model(self, native=None):
        config = self.native_config()
        if native is None:
            native = DiffusionRegPoseModel(backbone_pretrained=False,
                separate_expansion_pose_head=config.pretrained_keep_decoder_layers is not None,
                **{name: getattr(config, name) for name in NATIVE_MODEL_FIELDS})
        return PoseFlowV55Model(native=native, capacity=self.capacity, activation_checkpointing=self.activation_checkpointing)

    def build(self, *, max_epochs=80, **kwargs):
        original = super().build(max_epochs=max_epochs, **kwargs)
        return PoseFlowV55Pipeline(config=self, model=original.model, max_epochs=max_epochs)


class PoseFlowV55Pipeline(PoseFlowV54Pipeline):
    def __init__(self, *, config, model=None, max_epochs=80):
        config = config if isinstance(config, PoseFlowV55PipelineConfig) else PoseFlowV55PipelineConfig(**config)
        super().__init__(config=config, model=model, max_epochs=max_epochs)
        self.model.architecture = 'V55'

    def gradient_clipping_families(self):
        groups = gradient_families(self.model)
        context_ids = {id(parameter) for parameter in self.model.proposal_context.parameters()}
        geometry = groups.pop('geometry')
        groups['shared_context'] = [parameter for parameter in geometry if id(parameter) in context_ids]
        groups['private_geometry'] = [parameter for parameter in geometry if id(parameter) not in context_ids]
        return groups

    def gradient_clipping_budgets(self):
        return dict(native=.1/math.sqrt(3), selector=.1/math.sqrt(3),
                    shared_context=.1/math.sqrt(6), private_geometry=.1/math.sqrt(6))

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        if gradient_clip_val is not None and abs(float(gradient_clip_val)-.1) > 1e-8:
            raise ValueError('V55 requires combined clipping bound .1')
        values = clip_families_with_budgets(self.gradient_clipping_families(), self.gradient_clipping_budgets())
        for measurement in ('preclip', 'postclip'):
            context = values[f'shared_context/{measurement}'].to(self.gradient_scales)
            private = values[f'private_geometry/{measurement}'].to(self.gradient_scales)
            values[f'geometry/{measurement}'] = torch.sqrt(context.square()+private.square())
        if self._trainer is not None:
            self.log_dict({f'train/gradients/{name}': value for name, value in values.items()},
                          on_step=True, on_epoch=False)
        return values

    def checkpoint_contract(self):
        contract = super().checkpoint_contract()
        contract.update(version=2, architecture='V55',
            proposal_context=dict(width=self.model.hidden_dim, heads=self.model.num_heads, layers=1,
                dropout=0., attention='proposal_and_learned_EOS_queries_to_individual_previous_people',
                normalization='preLN_queries_and_keys_values;unnormalized_feature_residual',
                empty='learned_null_key_value_only_when_no_previous_people',
                masks='PERSON_only;row_independent;used_candidates_excluded_from_selection',
                correction='zero_output_projection;selected_corrected_feature_before_native_expansion',
                eos_initialization='unit_scale_query_matching_proposal_residuals_before_score_LN',
                scoring='private_candidate_and_EOS_MLP;conditional_CE_and_marginal_EOS_BCE',
                detach='raw_proposal_and_prefix_inputs;corrected_output_differentiable'),
            gradients=dict(policy='disjoint_native_selector_shared_context_private_geometry',
                budgets=self.gradient_clipping_budgets(), combined_bound=.1,
                timing='after_accumulation_before_AdamW',
                routes='classification_and_geometry_train_context;raw_context_inputs_detached'),
            initialization='original_EDPose_encoder2_4_6_box2_pose4_6_and_expansion_heads;fresh_prefix_context_and_selector;zero_context_output_projection')
        return contract

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        for key in ('contract', 'initialization', 'calibration', 'global_rng', 'prefix_bank'):
            checkpoint[f'v55_{key}'] = checkpoint.pop(f'v54_{key}')

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('v55_contract') != self.checkpoint_contract():
            raise ValueError('Checkpoint does not satisfy V55 contract')
        translated = dict(checkpoint)
        for key in ('contract', 'initialization', 'calibration', 'global_rng', 'prefix_bank'):
            translated[f'v54_{key}'] = checkpoint.get(f'v55_{key}')
        super().on_load_checkpoint(translated)
