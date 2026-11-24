"""Run one local detector and Sapiens2 session for a batch of source images."""
import argparse
import json
import math
import sys

DETECTOR = "facebook/detr-resnet-101-dc5"
DET_REV = "96317ca979e231bd960cb3cac31328e0165a3e94"
POSE = "facebook/sapiens2-pose-0.8b"
POSE_REV = "76f06b9854d04bb1e59e0624de39b1e8921b12aa"
# Meta's keypoints308.py at 7e5bae8. The HF config uses generic LABEL_n names.
CROWDPOSE_INDICES = (5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14, None, 69)


def map_joints(points, scores, width, height):
    joints, confidences = [], []
    for index in CROWDPOSE_INDICES:
        joint = confidence = None
        if index is not None:
            x, y = points[index]
            confidence = scores[index] if math.isfinite(scores[index]) else None
            if confidence is not None and confidence > 0 and 0 <= x < width and 0 <= y < height:
                joint = [x, y, 2]
        joints.append(joint)
        confidences.append(confidence)
    return joints, confidences


class Predictor:
    def __init__(self, confidence=0.3):
        import torch
        from transformers import AutoImageProcessor, DetrForObjectDetection, Sapiens2ForPoseEstimation

        if not torch.cuda.is_available():
            raise RuntimeError("Sapiens2 requires an available CUDA GPU.")
        self.confidence = confidence
        self.det_processor = AutoImageProcessor.from_pretrained(DETECTOR, revision=DET_REV, local_files_only=True)
        self.detector = DetrForObjectDetection.from_pretrained(
            DETECTOR, revision=DET_REV, local_files_only=True).eval().cuda()
        self.pose_processor = AutoImageProcessor.from_pretrained(POSE, revision=POSE_REV, local_files_only=True)
        self.pose = Sapiens2ForPoseEstimation.from_pretrained(
            POSE, revision=POSE_REV, local_files_only=True, dtype=torch.float16).eval().cuda()

    def predict(self, path):
        import torch
        from PIL import Image
        from torchvision.ops import clip_boxes_to_image, nms

        with Image.open(path) as file:
            image = file.convert("RGB")
        width, height = image.size
        with torch.inference_mode():
            inputs = self.det_processor(images=image, return_tensors="pt").to("cuda")
            detection = self.det_processor.post_process_object_detection(
                self.detector(**inputs), threshold=self.confidence, target_sizes=[(height, width)])[0]
            person_label = self.detector.config.label2id["person"]
            mask = detection["labels"] == person_label
            boxes = clip_boxes_to_image(detection["boxes"][mask], (height, width))
            scores = detection["scores"][mask]
            people = []
            for idx in nms(boxes, scores, 0.3).tolist():
                x, y, right, bottom = boxes[idx].tolist()
                bbox = [x, y, right - x, bottom - y]
                if min(bbox[2:]) <= 0:
                    continue
                inputs = self.pose_processor(image, boxes=[[bbox]], return_tensors="pt").to("cuda", self.pose.dtype)
                result = self.pose_processor.post_process_pose_estimation(self.pose(**inputs), boxes=[[bbox]])[0][0]
                joints, joint_scores = map_joints(result["keypoints"].tolist(), result["scores"].tolist(), width, height)
                people.append(dict(bbox=bbox, joints=joints, joint_scores=joint_scores,
                                   detection_score=scores[idx].item()))
            return people


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Cache the pinned checkpoints before offline use")
    if parser.parse_args().download:
        from huggingface_hub import snapshot_download
        for repo, revision in ((DETECTOR, DET_REV), (POSE, POSE_REV)):
            snapshot_download(repo, revision=revision, allow_patterns=["config.json", "preprocessor_config.json",
                                                                       "model.safetensors"])
        return
    request = json.load(sys.stdin)
    paths = request["paths"] if isinstance(request, dict) else request
    options = request["options"] if isinstance(request, dict) else {}
    try:
        predictor = Predictor(**options)
    except Exception as error:
        print(json.dumps({"fatal": str(error)}), flush=True)
        return
    for index, path in enumerate(paths):
        try:
            response = dict(index=index, people=predictor.predict(path))
            print(json.dumps(response, allow_nan=False), flush=True)
        except Exception as error:
            print(json.dumps(dict(index=index, error=str(error))), flush=True)


if __name__ == "__main__":
    main()
