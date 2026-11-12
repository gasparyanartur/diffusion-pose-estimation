import torch

from ..diffusionregpose.transformer import inverse_sigmoid, sine_embedding_for_position


class NativeHumanContext:
    def _decoder_layer(self, layer, features, references, memory, inputs, mask=None):
        scaled = references[:, :, None] * torch.cat((inputs.valid_ratios, inputs.valid_ratios), -1)[None]
        position = self.transformer.decoder.ref_point_head(sine_embedding_for_position(
            scaled[:, :, 0], num_pos_feats=self.transformer.decoder.d_model // 2))
        return layer(features, tgt_query_pos=position, tgt_reference_points=scaled,
            memory=memory, memory_level_start_index=inputs.level_start_index,
            memory_spatial_shapes=inputs.spatial_shapes, memory_key_padding_mask=inputs.padding_mask,
            self_attn_mask=mask)

    def prepare_context(self, images, padding_mask=None, *, targets=None):
        if padding_mask is None:
            padding_mask = torch.zeros(images.shape[0], *images.shape[2:], dtype=torch.bool, device=images.device)
        inputs = self.prepare_transformer_inputs(images, padding_mask)
        memory = self.encode_features(inputs, timesteps=None)
        features, reference_logits, encoder_logits, encoder_boxes = self.select_two_stage_queries(memory, inputs)
        memory = self.decoder_feature_memory(inputs, memory)
        decoder = self.transformer.decoder
        dn_state, dn_count, mask = None, 0, None
        if self.training and targets is not None:
            dn_features, dn_references, dn_state = self._prepare_denoising_queries(targets, images.device)
            dn_count = decoder.num_dn
            features = torch.cat((dn_features, features), 1)
            reference_logits = torch.cat((dn_references, reference_logits), 1)
            mask = self.training_box_attention_mask(targets, len(images), images.device)
        features = features.transpose(0, 1)  # <queries, batch, width>
        references = reference_logits.sigmoid().transpose(0, 1)
        frontend, dn_outputs = [], []
        for index, layer in enumerate(decoder.layers[:decoder.num_box_decoder_layers]):
            features = self._decoder_layer(layer, features, references, memory, inputs, mask)
            normalized = decoder.norm(features).transpose(0, 1)
            boxes = (self.bbox_embed[index](normalized) + inverse_sigmoid(references.transpose(0, 1))).sigmoid()
            logits = self.class_embed[index](normalized)
            frontend.append(dict(pred_logits=logits[:, dn_count:], pred_boxes=boxes[:, dn_count:]))
            if dn_count:
                dn_outputs.append(dict(pred_logits=logits[:, :dn_count], pred_boxes=boxes[:, :dn_count]))
            references = (self.bbox_embed[index](features) + inverse_sigmoid(references)).sigmoid().detach()
        selection_logits = self.class_embed[decoder.num_box_decoder_layers - 1](features[dn_count:])
        selected = selection_logits.max(-1).values.topk(self.num_proposals, dim=0).indices
        selected_features = features[dn_count:].gather(0, selected[..., None].expand(-1, -1, decoder.d_model))
        selected_boxes = references[dn_count:].gather(0, selected[..., None].expand(-1, -1, 4))
        selected_logits = selection_logits.gather(0, selected[..., None].expand(-1, -1, self.num_classes))
        context = dict(inputs=inputs, memory=memory, features=selected_features.transpose(0, 1),
            boxes=selected_boxes.transpose(0, 1), indices=selected.transpose(0, 1),
            pred_logits=selected_logits.transpose(0, 1),
            frontend_outputs=frontend, encoder_output=dict(pred_logits=encoder_logits, pred_boxes=encoder_boxes))
        context['existence_logits'] = context['pred_logits'][..., 1]
        if dn_state is not None:
            context['dn_output'] = dict(dn_class_input=dn_state['labels'], dn_bbox_input=dn_state['boxes'],
                dn_class_pred=dn_outputs[-1]['pred_logits'], dn_bbox_pred=dn_outputs[-1]['pred_boxes'])
        return context

