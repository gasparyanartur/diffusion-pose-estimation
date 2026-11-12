from .model_v26 import PoseFlowV26Model


class PoseFlowV28Model(PoseFlowV26Model):
    def encode_clean(self, input_pixels, encoded_image):
        memory, position, padding_mask = encoded_image
        clean_image = (memory.detach(), position.detach(), padding_mask)
        return super().encode_clean(input_pixels, clean_image)
