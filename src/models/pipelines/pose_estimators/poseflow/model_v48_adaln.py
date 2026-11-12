import torch
from torch import nn
from torch.nn import functional as F

from .model_v47 import CachedRowLayer
from .model_v48 import PoseFlowV48Model


class StepConditionedImageLayer(CachedRowLayer):
    def __init__(self, original):
        width = original.self_attention.embed_dim
        super().__init__(width, original.heads, original.feedforward[0].out_features)
        self.load_state_dict(original.state_dict(), strict=True)
        self.image_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 3*width))
        nn.init.zeros_(self.image_modulation[-1].weight)
        nn.init.zeros_(self.image_modulation[-1].bias)

    def forward(self, state, padding_mask, image_keys, image_values, image_mask, step_embedding):
        # State <B,R,Q,W>; each row's generation-step embedding <B,R,W>.
        batch, rows, queries, width = state.shape
        query = self.norms[0](state).reshape(batch*rows, queries, width)
        update = self.self_attention(query, query, query,
            key_padding_mask=padding_mask.reshape(batch*rows, queries), need_weights=False)[0]
        state = state+update.reshape_as(state)
        scale, shift, gate = self.image_modulation(step_embedding).chunk(3, dim=-1)
        query = self.norms[1](state)*(1+scale[..., None, :])+shift[..., None, :]
        query = query.reshape(batch, rows*queries, width)
        query_weight = self.image_attention.in_proj_weight[:width]
        query_bias = self.image_attention.in_proj_bias[:width]
        query = self._heads(F.linear(query, query_weight, query_bias))
        update = F.scaled_dot_product_attention(query, image_keys, image_values,
            attn_mask=~image_mask[:, None, None, :], dropout_p=0.)
        update = update.transpose(1, 2).flatten(-2)
        update = self.image_attention.out_proj(update).reshape_as(state)
        state = state+(1+gate[..., None, :])*update
        state = state+self.feedforward(self.norms[2](state))
        return state.masked_fill(padding_mask[..., None], 0.)


class PoseFlowV48AdaLNModel(PoseFlowV48Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Keep V48's common initializer and subsequent augmentation RNG unchanged.
        with torch.random.fork_rng(devices=[]):
            self.generation_step_embedding = nn.Embedding(self.capacity+1, self.hidden_dim)
            self.layers = nn.ModuleList(StepConditionedImageLayer(layer) for layer in self.layers)

    def _decode_rows(self, state, padding_mask, image_mask, *projected):
        steps = (~padding_mask).sum(-1)-1
        step_embedding = self.generation_step_embedding(steps)
        latents, coordinates, eos = [], [], []
        for index, (layer, norm, head, eos_head) in enumerate(zip(
                self.layers, self.readout_norms, self.latent_heads, self.eos_heads)):
            state = layer(state, padding_mask, projected[2*index], projected[2*index+1], image_mask, step_embedding)
            readout = norm(state)
            latent = head(readout)
            latents.append(latent.masked_fill(padding_mask[..., None], 0.))
            coordinates.append(self.person_codec.decode(latent).masked_fill(padding_mask[..., None], 0.))
            eos.append(eos_head(readout).squeeze(-1).masked_fill(padding_mask, 0.))
        return torch.stack(coordinates), torch.stack(latents), torch.stack(eos)
