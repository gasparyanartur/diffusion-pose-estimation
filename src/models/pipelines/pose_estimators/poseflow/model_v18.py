from torch import nn

from .model_v17 import PoseFlowV17Model


class PreImageRefinement(nn.Module):
    def __init__(self, width, feedforward, dropout):
        super().__init__()
        self.input_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, feedforward), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(feedforward, width))
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(width)

    def forward(self, representations):
        normalized = self.input_norm(representations)
        return self.output_norm(normalized + self.dropout(self.ffn(normalized)))


class PoseFlowV18Model(PoseFlowV17Model):
    def __init__(self, image_model, *, sampling_steps=3):
        super().__init__(image_model, sampling_steps=sampling_steps)
        original = image_model.transformer.decoder.layers[0]
        self.pre_image_updates = nn.ModuleList([
            PreImageRefinement(self.hidden_dim, original.linear1.out_features, original.dropout1.p)
            for _ in self.layers
        ])
        state = self.state_dict()
        total = sum(value.numel() for value in state.values())
        self.initialization_report.update(
            fresh_components=self.initialization_report['fresh_components'] + ['pre_image_updates'],
            total_state_tensors=len(state), total_state_elements=total,
            retained_image_element_fraction=self.initialization_report['retained_image_elements']/total)

    def refine_before_image(self, index, representations):
        return self.pre_image_updates[index](representations)
