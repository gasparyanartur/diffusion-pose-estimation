import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..diffusionregpose.model import DiffusionRegPoseModel
from .human_context import NativeHumanContext


class PoseFlowV45Model(NativeHumanContext, DiffusionRegPoseModel):
    sampler_kind = 'progressive bidirectional prefix; shared two-layer decoder; image-only'

    def __init__(self, *, codec, native_state=None, activation_checkpointing=True, **kwargs):
        super().__init__(**kwargs)
        if native_state is not None:
            self.load_state_dict(native_state, strict=True)
        decoder = self.transformer.decoder
        if decoder.num_layers - decoder.num_box_decoder_layers != 2:
            raise ValueError('V45 requires exactly two shared person decoder layers')
        if self.num_proposals != decoder.num_group or self.num_proposals > self.num_queries:
            raise ValueError('V45 requires groups == proposals <= queries')
        self.codec = codec.requires_grad_(False).eval()
        self.latent_dim = codec.latent_dim
        self.num_body_points = decoder.num_body_points
        self.activation_checkpointing = activation_checkpointing
        self.predict_token = nn.Parameter(torch.zeros(decoder.d_model))
        self.latent_heads = nn.ModuleList(nn.Linear(decoder.d_model, self.latent_dim) for _ in range(2))
        for name in ('pose_embed', 'pose_hw_embed', 'expansion_pose_embed'):
            delattr(self, name)
            delattr(decoder, name)
        for name in ('hw', 'keypoint_embed', 'emb_Q', 'emb_K', 'emb_V', 'emb_SQ', 'emb_SK', 'emb_SV'):
            delattr(decoder, name)
        self.initialization_report = {}

    def _stage(self, features, references, memory, inputs):
        outputs = []
        decoder = self.transformer.decoder
        for layer in decoder.layers[decoder.num_box_decoder_layers:]:
            features = self._decoder_layer(layer, features, references, memory, inputs)
            outputs.append(features)
        return tuple(outputs)

    def latent_predictions(self, context):
        # Features <B,N,D>; carried hidden states <active_people,B,D>.
        features = context['features']
        if features.ndim != 3 or features.shape[1] != self.num_proposals:
            raise ValueError('V45 context disagrees with proposal capacity')
        hidden = features.new_empty(0, features.shape[0], features.shape[2])
        for stage in range(1, self.num_proposals + 1):
            introduced = features[:, stage-1] + self.predict_token
            hidden = torch.cat((hidden, introduced[None]), 0)
            references = context['boxes'][:, :stage].transpose(0, 1)
            if self.training and torch.is_grad_enabled() and self.activation_checkpointing:
                first, hidden = checkpoint(self._stage, hidden, references, context['memory'],
                    context['inputs'], use_reentrant=False, preserve_rng_state=True)
            else:
                first, hidden = self._stage(hidden, references, context['memory'], context['inputs'])
            for layer, state in enumerate((first, hidden)):
                latent = self.latent_heads[layer](self.transformer.decoder.norm(state).transpose(0, 1))
                yield stage, layer + 1, latent

    def decode_prediction(self, latent, context, image_wh):
        count = latent.shape[1]
        if image_wh.shape != (len(latent), 2) or not torch.isfinite(image_wh).all() or (image_wh <= 0).any():
            raise ValueError('Expected finite positive image sizes shaped [B,2]')
        sizes = image_wh[:, None].expand(-1, count, -1).to(latent)
        geometry = self.codec.decode_latents(latent, sizes)
        points = geometry['keypoints_px'] / sizes[..., None, :]
        logits = context['existence_logits'][:, :count]
        return dict(geometry, pred_latents=latent, existence_logits=logits,
            pred_logits=torch.stack((torch.full_like(logits, -20.), logits), -1),
            pred_boxes=geometry['boxes_cxcywh_px'] / sizes.repeat(1, 1, 2),
            pred_keypoints=torch.cat((points.flatten(-2), torch.ones_like(points[..., 0])), -1),
            reference_indices=context['indices'][:, :count], reference_boxes=context['boxes'][:, :count])

    def predict(self, context, image_wh, collect_trace=False):
        trace = []
        for stage, layer, latent in self.latent_predictions(context):
            if collect_trace:
                output = self.decode_prediction(latent, context, image_wh)
                trace.append(dict(stage=stage, layer=layer, **output))
        if not collect_trace:
            output = self.decode_prediction(latent, context, image_wh)
        else:
            output = dict(output, trace=trace)
        return output

    def forward(self, images, *, image_wh, padding_mask=None, collect_trace=False):
        return self.predict(self.prepare_context(images, padding_mask), image_wh, collect_trace=collect_trace)

    def sample_keypoints(self, images, *, image_wh, padding_mask=None, collect_trace=False):
        return self.forward(images, image_wh=image_wh, padding_mask=padding_mask, collect_trace=collect_trace)
