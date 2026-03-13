import torch

class PoseFixRefiner:

    def __init__(self, model):

        self.model = model.eval()

    @torch.no_grad()
    def refine(self, img, heatmaps):

        refined = self.model(img, heatmaps)

        return refined