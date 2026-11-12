import torch
from torch import Tensor, nn

from ..diffusionregpose.backbone import DiffusionRegPoseResNetBackbone, PositionEmbeddingSineHW
from ..diffusionregpose.diffusion import DiffusionRegPoseDiffusion
from ..diffusionregpose.model import DiffusionRegPoseModel, _BackboneJoiner
from ..diffusionregpose.transformer import MLP, SinusoidalPositionEmbeddings, build_input_projections
from ..diffusionregpose.evaluation import postprocess_crowdpose


class LatentImageBlock(nn.Module):
    def __init__(self, width: int, heads: int, feedforward_dim: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.)
        self.image_attention = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.)
        self.norms = nn.ModuleList(nn.LayerNorm(width, elementwise_affine=False) for _ in range(3))
        self.modulation = nn.Linear(width, width*6)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        self.feedforward = nn.Sequential(nn.Linear(width, feedforward_dim), nn.GELU(),
                                         nn.Linear(feedforward_dim, width))

    def forward(self, state: Tensor, time_features: Tensor, encoded_image, query_position=None) -> Tensor:
        memory, position, padding_mask = encoded_image
        modulation = self.modulation(time_features).chunk(6, -1)
        def conditioned(index):
            scale, shift = modulation[2*index:2*index+2]
            return self.norms[index](state)*(1+scale[:, None])+shift[:, None]
        query = conditioned(0)
        positioned = query if query_position is None else query+query_position
        state = state+self.self_attention(positioned, positioned, query, need_weights=False)[0]
        query = conditioned(1)
        positioned = query if query_position is None else query+query_position
        state = state+self.image_attention(positioned, memory+position, memory,
            key_padding_mask=padding_mask, need_weights=False)[0]
        return state+self.feedforward(conditioned(2))


class PoseFlowV26Model(DiffusionRegPoseDiffusion):
    sampler_kind = 'latent DDIM eta0'
    flow_renewal = False
    flow_ensemble = False
    flow_renewal_score_threshold = .5

    def __init__(self, *, backbone_pretrained=True, timesteps=1000, sampling_steps=4,
                 num_proposals=50, num_body_points=14, hidden_dim=256,
                 num_heads=8, feedforward_dim=2048, num_feature_levels=4,
                 freeze_backbone_stem_and_layer1=True, freeze_backbone=False):
        super().__init__(timesteps=timesteps, sampling_timesteps=sampling_steps,
                         scale=1., ddim_sampling_eta=0.)
        if hidden_dim % 32 or hidden_dim % num_heads:
            raise ValueError('V26 width must be divisible by32 and the head count')
        self.num_proposals = self.num_queries = num_proposals
        self.num_body_points = num_body_points
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.backbone = _BackboneJoiner(DiffusionRegPoseResNetBackbone(
            pretrained=backbone_pretrained, freeze_stem_and_layer1=freeze_backbone_stem_and_layer1),
            PositionEmbeddingSineHW(hidden_dim//2, temperature_h=20, temperature_w=20))
        if freeze_backbone:
            self.backbone.requires_grad_(False)
        self.input_proj = build_input_projections((512, 1024, 2048),
            hidden_dim=hidden_dim, num_levels=num_feature_levels)
        for projection in self.input_proj:
            nn.init.xavier_uniform_(projection[0].weight)
            nn.init.zeros_(projection[0].bias)
        self.level_embed = nn.Parameter(torch.empty(num_feature_levels, hidden_dim))
        nn.init.normal_(self.level_embed)
        self.input_mlp = nn.Sequential(nn.Linear(num_body_points*2, hidden_dim), nn.GELU(),
                                       nn.Linear(hidden_dim, hidden_dim))
        self.time_embedding = nn.Sequential(SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*4), nn.GELU(), nn.Linear(hidden_dim*4, hidden_dim))
        self.clean_layers = nn.ModuleList(LatentImageBlock(hidden_dim, num_heads, feedforward_dim) for _ in range(3))
        self.denoiser_layers = nn.ModuleList(LatentImageBlock(hidden_dim, num_heads, feedforward_dim) for _ in range(3))
        self.clean_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.denoiser_projection = nn.Linear(hidden_dim, hidden_dim)
        self.box_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.joint_head = MLP(hidden_dim, hidden_dim, num_body_points*2, 3)
        self.existence_head = MLP(hidden_dim, hidden_dim, 1, 3)
        self.initialization_report = {}

    @property
    def flow_sampling_steps(self):
        return self.sampling_timesteps

    @flow_sampling_steps.setter
    def flow_sampling_steps(self, value):
        if not 1 <= value <= self.num_timesteps:
            raise ValueError('V26 sampling steps must fit the diffusion schedule')
        self.sampling_timesteps = int(value)

    def encode_image(self, images: Tensor, padding_mask: Tensor | None = None):
        if padding_mask is None:
            padding_mask = torch.zeros(images.shape[0], *images.shape[-2:], dtype=torch.bool, device=images.device)
        inputs = DiffusionRegPoseModel.prepare_transformer_inputs(self, images, padding_mask)
        if inputs.padding_mask.all(-1).any():
            raise ValueError('V26 image requires valid feature tokens')
        offsets = self.level_embed.repeat_interleave(inputs.spatial_shapes.prod(-1), dim=0)[None]
        return inputs.memory, inputs.position+offsets, inputs.padding_mask

    def readout(self, latents: Tensor) -> dict[str, Tensor]:
        joints = self.joint_head(latents).sigmoid().reshape(*latents.shape[:2], self.num_body_points, 2)
        existence = self.existence_head(latents).squeeze(-1)
        return dict(pred_latents=latents, existence_logits=existence,
            pred_logits=torch.stack((torch.full_like(existence, -20.), existence), -1),
            pred_boxes=self.box_head(latents).sigmoid(), joint_xy=joints,
            pred_keypoints=torch.cat((joints.flatten(-2), torch.ones_like(joints[..., 0])), -1))

    def encode_clean(self, input_pixels: Tensor, encoded_image):
        state = self.input_mlp(input_pixels/100.)
        times = state.new_zeros(state.shape[0])
        time_features = self.time_embedding(times)
        for layer in self.clean_layers:
            state = layer(state, time_features, encoded_image)
        return self.readout(self.clean_norm(state))

    def denoise(self, noisy_latents: Tensor, timesteps: Tensor, encoded_image):
        expected = (encoded_image[0].shape[0], self.num_proposals, self.hidden_dim)
        if noisy_latents.shape != expected or timesteps.shape != (expected[0],):
            raise ValueError('V26 latent or timestep shape disagrees with the fixed proposal set')
        if ((timesteps < 0) | (timesteps >= self.num_timesteps)).any():
            raise ValueError('V26 timestep outside diffusion schedule')
        state = noisy_latents
        time_features = self.time_embedding(timesteps.to(state.dtype))
        for layer in self.denoiser_layers:
            state = layer(state, time_features, encoded_image)
        return self.readout(self.denoiser_projection(state))

    def forward(self, images, noisy_latents, timesteps, padding_mask=None):
        return self.denoise(noisy_latents, timesteps, self.encode_image(images, padding_mask))

    def training_outputs(self, images, input_pixels, *, timesteps, noise, padding_mask=None,
                         detach_corruption_source=True):
        encoded = self.encode_image(images, padding_mask)
        clean = self.encode_clean(input_pixels, encoded)
        source = clean['pred_latents'].detach() if detach_corruption_source else clean['pred_latents']
        noisy = self.q_sample(source, timesteps, noise)
        denoised = self.denoise(noisy, timesteps, encoded)
        return clean, denoised, noisy, timesteps

    def q_sample(self, x_start, timesteps, noise=None):
        return super().q_sample(x_start, timesteps, noise).to(x_start.dtype)

    @torch.no_grad()
    def sample_keypoints(self, images, *, padding_mask=None, generator=None, num_proposals=None,
                         collect_step_diagnostics=False):
        if num_proposals is not None and num_proposals != self.num_proposals:
            raise ValueError('V26 preserves its configured proposal budget')
        encoded = self.encode_image(images, padding_mask)
        state = torch.randn(images.shape[0], self.num_proposals, self.hidden_dim,
                            device=images.device, dtype=images.dtype, generator=generator)
        diagnostics = []
        for time, next_time in self.ddim_timestep_pairs():
            times = torch.full((len(images),), time, device=images.device, dtype=torch.long)
            output = self.denoise(state, times, encoded)
            noise = self.predict_noise_from_start(state, times, output['pred_latents']).to(state.dtype)
            state = self.ddim_step(output['pred_latents'], noise, time, next_time,
                                   noise=torch.zeros_like(state))
            if collect_step_diagnostics:
                diagnostics.append(dict(time=time, next_time=next_time,
                    latent_rms=state.square().mean().sqrt().item()))
        return dict(output, latent_state=state, latent_step_diagnostics=diagnostics)

    def postprocess(self, output, target_sizes, *, nms_target_sizes=None):
        return postprocess_crowdpose(output, target_sizes, num_body_points=self.num_body_points,
                                    nms_target_sizes=nms_target_sizes)
