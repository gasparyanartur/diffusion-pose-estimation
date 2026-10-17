import copy

import torch
from torch import nn

from ..diffusionregpose.transformer import MLP, sine_embedding_for_position
from .coordinate_v19 import CoordinateState
from .latent_v12 import ddim_update
from .model_v14 import PoseFlowV14Model, tensor_rms
from .model_v17 import V17AttentionUpdate, attention_entropy


class CoordinateDecoderLayer(nn.Module):
    def __init__(self, original, box_head, joint_head, extent_head, class_head):
        super().__init__()
        self.block = copy.deepcopy(original)
        self.box_head = copy.deepcopy(box_head)
        self.joint_head = copy.deepcopy(joint_head)
        self.extent_head = copy.deepcopy(extent_head)
        self.class_head = nn.Linear(class_head.in_features, 1)
        with torch.no_grad():
            person_class = min(1, class_head.out_features-1)
            self.class_head.weight.copy_(class_head.weight[person_class:person_class+1])
            self.class_head.bias.copy_(class_head.bias[person_class:person_class+1])

    def forward(self, features, hypotheses, references, positions, image, *, collect_weights=False):
        # B = images, N = candidates, T = box+joint roles, D = feature width.
        batch, candidates, roles, width = features.shape
        ratios = torch.cat((image.valid_ratios, image.valid_ratios), -1)
        scaled = references.flatten(1, 2)[:, :, None]*ratios[:, None]
        crossed = self.block.cross_attn((features+positions).flatten(1, 2), scaled,
            image.memory, image.spatial_shapes, image.level_start_index, image.padding_mask)
        features = self.block.norm1(features+self.block.dropout1(crossed.reshape_as(features)))
        tokens = torch.cat((features, hypotheses[:, :, None]), 2).flatten(0, 1).transpose(0, 1)
        position = torch.cat((positions, torch.zeros_like(hypotheses[:, :, None])), 2)
        position = position.flatten(0, 1).transpose(0, 1)
        attended, weights = self.block.self_attn(tokens+position, tokens+position, tokens,
            need_weights=collect_weights, average_attn_weights=False)
        tokens = self.block.norm2(tokens+self.block.dropout2(attended))
        update = self.block.linear2(self.block.dropout3(torch.relu(self.block.linear1(tokens))))
        tokens = self.block.norm3(tokens+self.block.dropout4(update))
        tokens = tokens.transpose(0, 1).reshape(batch, candidates, roles+1, width)
        return tokens[:, :, :-1], tokens[:, :, -1], weights


class PoseFlowV19Model(PoseFlowV14Model):
    @property
    def num_state_tokens(self):
        return self.num_body_points+1

    def __init__(self, image_model, *, sampling_steps=3):
        nn.Module.__init__(self)
        decoder = image_model.transformer.decoder
        original = decoder.layers[0]
        self.hidden_dim = decoder.d_model
        self.num_body_points = decoder.num_body_points
        self.num_proposals = self.num_queries = self.num_hypotheses = image_model.num_proposals
        self.num_feature_levels = image_model.num_feature_levels
        self.flow_sampling_steps = sampling_steps
        self.flow_renewal = self.flow_ensemble = False
        self.flow_renewal_score_threshold = .5
        self.codec = CoordinateState(self.num_body_points)
        self.backbone, self.input_proj = image_model.backbone, image_model.input_proj
        self.encoder, self.level_embed = image_model.transformer.encoder, image_model.transformer.level_embed
        self.time_embedding = nn.Sequential(image_model.transformer.time_mlp, image_model.transformer.block_time_mlp)
        width = self.hidden_dim
        self.box_embedding = nn.Parameter(torch.zeros(1, 1, 1, width))
        self.joint_embeddings = copy.deepcopy(decoder.keypoint_embed)
        self.joint_extents = copy.deepcopy(decoder.hw)
        self.reference_position = copy.deepcopy(decoder.ref_point_head)
        self.readout_norm = copy.deepcopy(decoder.norm)
        self.box_conditioning = MLP(4, width, width, 2)
        self.joint_conditioning = MLP(2, width, width, 2)
        self.hypothesis_slots = nn.Parameter(torch.randn(1, self.num_hypotheses, width)*.02)
        self.hypothesis_attention = V17AttentionUpdate(width, original.self_attn.num_heads,
            original.linear1.out_features, original.dropout1.p)
        self.layers = nn.ModuleList()
        head_sources = []
        for index, layer in enumerate(decoder.layers):
            pose_index = max(index-decoder.num_box_decoder_layers, 0)
            self.layers.append(CoordinateDecoderLayer(layer, image_model.bbox_embed[index],
                image_model.pose_embed[pose_index], image_model.pose_hw_embed[pose_index],
                image_model.class_embed[index]))
            head_sources.append({'layer': index, 'box_head': index, 'joint_and_extent_head': pose_index})
        prefixes = {
            'backbone.': 'backbone.', 'input_proj.': 'input_proj.',
            'encoder.': 'transformer.encoder.', 'level_embed': 'transformer.level_embed',
            'time_embedding.0.': 'transformer.time_mlp.',
            'time_embedding.1.': 'transformer.block_time_mlp.',
            'joint_embeddings.': 'transformer.decoder.keypoint_embed.',
            'joint_extents.': 'transformer.decoder.hw.',
            'reference_position.': 'transformer.decoder.ref_point_head.',
            'readout_norm.': 'transformer.decoder.norm.',
        }
        for row in head_sources:
            index, pose = row['layer'], row['joint_and_extent_head']
            prefixes.update({f'layers.{index}.block.': f'transformer.decoder.layers.{index}.',
                f'layers.{index}.box_head.': f'bbox_embed.{index}.',
                f'layers.{index}.joint_head.': f'pose_embed.{pose}.',
                f'layers.{index}.extent_head.': f'pose_hw_embed.{pose}.',
                f'layers.{index}.class_head.': f'class_embed.{index}.'})
        state = self.state_dict()
        self.transfer_sources = {name:source+name[len(target):]
            for name in state for target,source in prefixes.items() if name.startswith(target)}
        copied = {name:state[name] for name in self.transfer_sources}
        total = sum(value.numel() for value in state.values())
        self.initialization_report = {
            'copied_source_tensors': len(copied),
            'copied_source_elements': sum(value.numel() for value in copied.values()),
            'total_state_tensors': len(state), 'total_state_elements': total,
            'copied_source_fraction': sum(value.numel() for value in copied.values())/total,
            'fresh_components': ['box_conditioning', 'joint_conditioning', 'box_embedding',
                                 'hypothesis_slots', 'hypothesis_attention'],
            'decoder_head_sources': head_sources, 'person_class_row': min(1, image_model.num_classes-1),
            'attribution': 'Copied from initialized source model; checkpoint coverage must be audited separately.',
        }

    def _references(self, coordinates, extents):
        geometry = self.codec.decode(coordinates)
        points = geometry['pred_keypoints'][..., :2*self.num_body_points]
        joints = points.reshape(*coordinates.shape[:2], self.num_body_points, 2)
        return torch.cat((geometry['pred_boxes'][:, :, None], torch.cat((joints, extents), -1)), 2)

    def _positions(self, references, image):
        ratios = torch.cat((image.valid_ratios[:, 0], image.valid_ratios[:, 0]), -1)
        scaled = references*ratios[:, None, None]
        return self.reference_position(sine_embedding_for_position(scaled,
            num_pos_feats=self.hidden_dim//2))

    def denoise(self, noisy_latents, timesteps, encoded_image, *, hypothesis_memory=None,
                collect_diagnostics=False):
        coordinates = noisy_latents
        if coordinates.ndim != 3:
            raise ValueError('V19 coordinate state must have shape [batch,candidates,coordinates]')
        expected = (coordinates.shape[0], self.num_proposals, self.codec.width)
        if coordinates.shape != expected or timesteps.shape != (expected[0],):
            raise ValueError('V19 coordinate/timestep shapes disagree with configuration')
        if (not torch.isfinite(coordinates).all() or not torch.isfinite(timesteps).all()
                or ((timesteps < 0) | (timesteps > 1)).any()):
            raise ValueError('V19 needs finite coordinate logits and times in [0,1]')
        memory = self.hypothesis_slots.expand(expected[0], -1, -1) if hypothesis_memory is None else hypothesis_memory
        if (memory.shape != (expected[0], self.num_proposals, self.hidden_dim)
                or memory.device != coordinates.device or memory.dtype != coordinates.dtype
                or not torch.isfinite(memory).all()):
            raise ValueError('V19 hypothesis memory must match batch/candidates/device/dtype')
        geometry = self.codec.decode(coordinates)
        points = geometry['pred_keypoints'][..., :2*self.num_body_points].reshape(*expected[:2], self.num_body_points, 2)
        box = self.box_conditioning(geometry['pred_boxes'])[:, :, None]+self.box_embedding
        joints = self.joint_conditioning(points)+self.joint_embeddings.weight[None, None, :self.num_body_points]
        features = torch.cat((box, joints), 2)+self.time_embedding(timesteps)[:, None, None]
        extents = self.joint_extents.weight[:self.num_body_points].sigmoid()[None, None]*geometry['pred_boxes'][:, :, None, 2:]
        predictions, diagnostics = [], []
        for index, layer in enumerate(self.layers):
            previous_memory, previous_coordinates = memory, coordinates
            memory, bank_weights = self.hypothesis_attention(memory, memory, collect_weights=collect_diagnostics)
            references = self._references(coordinates, extents)
            features, memory, local_weights = layer(features, memory, references,
                self._positions(references, encoded_image), encoded_image, collect_weights=collect_diagnostics)
            readout = self.readout_norm(features)
            delta = torch.cat((layer.box_head(readout[:, :, 0]),
                               layer.joint_head(readout[:, :, 1:]).flatten(-2)), -1)
            coordinates = coordinates+delta
            extents = (torch.logit(extents.clamp(self.codec.epsilon, 1-self.codec.epsilon))
                       +layer.extent_head(readout[:, :, 1:])).sigmoid()
            predictions.append({'pred_coordinates': coordinates, 'pred_latents': coordinates,
                'pred_logits': layer.class_head(readout[:, :, 0]), **self.codec.decode(coordinates)})
            if collect_diagnostics:
                diagnostics.append({'layer': index, 'coordinate_update_rms': tensor_rms(delta),
                    'reference_coordinate_rms': tensor_rms(previous_coordinates),
                    'hypothesis_rms': tensor_rms(memory), 'hypothesis_update_rms': tensor_rms(memory-previous_memory),
                    'hypothesis_slot_variance': memory.detach().float().var(1, unbiased=False).mean(),
                    'hypothesis_attention_entropy_mean_over_heads': attention_entropy(bank_weights),
                    'local_attention_entropy_mean_over_heads': attention_entropy(local_weights),
                    'local_hypothesis_read_mass': local_weights[:, :, :-1, -1].detach().mean(),
                    'local_hypothesis_write_mass': local_weights[:, :, -1, :-1].detach().sum(-1).mean(),
                    'joint_candidate_std': predictions[-1]['pred_keypoints'][..., :2*self.num_body_points].detach().std(1, unbiased=False).mean(),
                    'box_candidate_std': predictions[-1]['pred_boxes'].detach().std(1, unbiased=False).mean(),
                    'sampling_box_size_mean': references[:, :, 0, 2:].detach().mean()})
        return {**predictions[-1], 'aux_outputs': predictions[:-1],
                'hypothesis_memory': memory, 'latent_diagnostics': diagnostics}

    @torch.no_grad()
    def sample_keypoints(self, images, *, padding_mask=None, num_proposals=None, generator=None,
                         collect_step_diagnostics=False, initial_latents=None, start_time=1.,
                         sampling_steps=None, encoded_image=None, persist_hypotheses=True):
        steps = self.flow_sampling_steps if sampling_steps is None else sampling_steps
        if (num_proposals is not None and num_proposals != self.num_proposals
                or not isinstance(steps, int) or steps < 1 or not 0 < start_time <= 1):
            raise ValueError('Invalid V19 sampling configuration')
        shape = (images.shape[0], self.num_proposals, self.codec.width)
        if initial_latents is None:
            if start_time != 1:
                raise ValueError('Partial-noise sampling requires explicit coordinate state')
            state = torch.randn(shape, device=images.device, dtype=images.dtype, generator=generator)
        else:
            if initial_latents.shape != shape or initial_latents.device != images.device or initial_latents.dtype != images.dtype:
                raise ValueError('V19 initial coordinate state has invalid shape/device/dtype')
            state = initial_latents.clone()
        image = self.encode_image(images, padding_mask) if encoded_image is None else encoded_image
        memory, diagnostics = None, []
        for index in range(steps):
            times = images.new_full((images.shape[0],), start_time*(1-index/steps))
            next_times = images.new_full(times.shape, start_time*(1-(index+1)/steps))
            output = self.denoise(state, times, image, hypothesis_memory=memory,
                                  collect_diagnostics=collect_step_diagnostics)
            updated = ddim_update(state, output['pred_coordinates'], times, next_times)
            memory = output['hypothesis_memory'] if persist_hypotheses else None
            if collect_step_diagnostics:
                diagnostics.append({'time': float(times[0]), 'latent_state': updated.clone(),
                    'update_rms': tensor_rms(updated-state), 'layers': output['latent_diagnostics'],
                    'predictions': [{key:stage[key] for key in ('pred_logits','pred_boxes','pred_keypoints')}
                                    for stage in output['aux_outputs']+[output]]})
            state = updated
        return {**output, 'coordinate_state': state, 'latent_state': state,
                'latent_step_diagnostics': diagnostics}
