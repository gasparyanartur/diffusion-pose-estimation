import torch
from torch import nn

from .latent_v16 import V16StateCodec
from .model_v15 import PoseFlowV15Model, V15EvidenceUpdate


class V16EvidenceUpdate(V15EvidenceUpdate):
    def __init__(self, width, heads, levels, points, joints, feedforward, dropout):
        super().__init__(width, heads, levels, points, joints, feedforward, dropout)
        self.joints = joints
        self.roles = nn.Embedding(joints+3, width)

    def forward(self, representations, geometry, image):
        batch, groups, roles, width = representations.shape
        boxes = geometry['pred_boxes']
        joints = geometry['pred_keypoints'][..., :2*self.joints].reshape(batch, groups, self.joints, 2)
        box_positions = self.box_position(boxes)[:, :, None].expand(-1, -1, 3, -1)
        position = torch.cat((box_positions, self.joint_position(joints)), 2)
        queries = representations+position+self.roles.weight[None, None]
        ratios = image.valid_ratios[:, None]
        box_references = boxes[:, :, None].expand(-1, -1, 3, -1).flatten(1, 2)[:, :, None]
        box_references = box_references*torch.cat((ratios, ratios), -1)
        joint_references = joints.flatten(1, 2)[:, :, None]*ratios
        arguments = (image.memory, image.spatial_shapes, image.level_start_index, image.padding_mask)
        box_message = self.image_attention(queries[:, :, :3].flatten(1, 2), box_references, *arguments)
        joint_message = self.image_attention(queries[:, :, 3:].flatten(1, 2), joint_references, *arguments)
        message = torch.cat((box_message.reshape(batch, groups, 3, width),
                             joint_message.reshape(batch, groups, self.joints, width)), 2)
        updated = self.norm(representations+self.dropout(message))
        return self.ffn_norm(updated+self.dropout(self.ffn(updated)))


class PoseFlowV16Model(PoseFlowV15Model):
    def __init__(self, image_model, *, sampling_steps=3):
        super().__init__(image_model, sampling_steps=sampling_steps)
        original = image_model.transformer.decoder.layers[0]
        self.codec = V16StateCodec(width=self.hidden_dim, joints=self.num_body_points)
        self.state_roles = nn.Embedding(self.num_state_tokens, self.hidden_dim)
        self.evidence_extractor = V16EvidenceUpdate(self.hidden_dim, original.self_attn.num_heads,
            self.num_feature_levels, original.cross_attn.num_points, self.num_body_points,
            original.linear1.out_features, original.dropout1.p)
        state = self.state_dict()
        total_elements = sum(value.numel() for value in state.values())
        self.initialization_report.update(total_state_tensors=len(state),
            total_state_elements=total_elements,
            retained_image_element_fraction=self.initialization_report['retained_image_elements']/total_elements,
            state_roles=['confidence', 'center', 'size', 'joints'])
