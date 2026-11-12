import torch
from torch import nn

from ..diffusionregpose.transformer import inverse_sigmoid
from .model_v45_task_pose import PoseFlowV45TaskPoseModel


class OrderedTaskPoseModel(PoseFlowV45TaskPoseModel):
    sampler_kind = 'ordered50 task-pose JEPA; coordinate cosine DDIM eta0; cached rank context'
    confidence_description = 'image-only canonical rank confidence; fixed across sampling'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        width = self.transformer.decoder.d_model
        self.rank_features = nn.Embedding(self.num_proposals, width)
        self.rank_references = nn.Embedding(self.num_proposals, 4)
        with torch.no_grad():
            boxes = torch.full((self.num_proposals, 4), .2)
            boxes[:, 0] = torch.linspace(.05, .95, self.num_proposals)
            boxes[:, 1] = .5
            self.rank_references.weight.copy_(inverse_sigmoid(boxes))
        # Candidate/DN modules are retained for native weight compatibility, but unused.
        for name in ('enc_out_class_embed', 'enc_out_bbox_embed', 'label_enc'):
            if hasattr(self, name):
                getattr(self, name).requires_grad_(False)
        for name in ('enc_output', 'enc_output_norm', 'enc_out_bbox_embed', 'enc_out_class_embed', 'tgt_embed'):
            if hasattr(self.transformer, name):
                getattr(self.transformer, name).requires_grad_(False)

    def prepare_context(self, images, padding_mask=None, *, targets=None):
        if padding_mask is None:
            padding_mask = torch.zeros(len(images), *images.shape[2:], dtype=torch.bool, device=images.device)
        inputs = self.prepare_transformer_inputs(images, padding_mask)
        memory = self.encode_features(inputs, timesteps=None)
        memory = self.decoder_feature_memory(inputs, memory)
        decoder = self.transformer.decoder
        features = self.rank_features.weight[:, None].expand(-1, len(images), -1)
        references = self.rank_references.weight.sigmoid()[:, None].expand(-1, len(images), -1)
        frontend = []
        for index, layer in enumerate(decoder.layers[:decoder.num_box_decoder_layers]):
            features = self._decoder_layer(layer, features, references, memory, inputs)
            normalized = decoder.norm(features).transpose(0, 1)
            boxes = (self.bbox_embed[index](normalized) + inverse_sigmoid(references.transpose(0, 1))).sigmoid()
            logits = self.class_embed[index](normalized)
            frontend.append(dict(pred_logits=logits, pred_boxes=boxes))
            references = boxes.detach().transpose(0, 1)
        features = features.transpose(0, 1)
        scores = self.image_confidence_head(decoder.norm(features))
        return dict(inputs=inputs, memory=memory, features=features, boxes=boxes.detach(),
            indices=torch.arange(self.num_proposals, device=images.device)[None].expand(len(images), -1),
            pred_logits=scores, image_logits=scores, frontend_outputs=frontend)

    def forward(self, images, diffusion_keypoints, timesteps, targets=None, padding_mask=None, *, pose_input_builder=None):
        context = self.prepare_context(images, padding_mask)
        matches = None
        if pose_input_builder is not None:
            diffusion_keypoints, matches = pose_input_builder(context['pred_logits'], context['boxes'], timesteps, targets)
        output = self.denoise_coordinates(diffusion_keypoints, timesteps, context)
        points = self.transformer.decoder.num_body_points
        frontend = [dict(row, pred_keypoints=row['pred_boxes'].new_zeros(*row['pred_boxes'].shape[:2], 3*points))
                    for row in context['frontend_outputs']]
        output['aux_outputs'] = frontend + output['aux_outputs']
        if matches is not None:
            output['pose_target_matches'] = matches
        return output
