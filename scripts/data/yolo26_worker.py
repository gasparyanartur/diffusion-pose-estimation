"""Run local YOLO26 box or pose proposals for a CrowdPose batch."""

from contextlib import redirect_stdout
import hashlib
import json
from math import isfinite
from pathlib import Path
import sys

COCO_LIMBS = tuple(range(5, 17))


def map_yolo_joints(points, width, height, threshold):
    if len(points) != 17 or any(len(point) != 3 for point in points):
        raise ValueError("A YOLO pose checkpoint must provide COCO-17 x/y/confidence keypoints.")
    joints, scores = [], []
    for index in COCO_LIMBS:
        x, y, score = points[index]
        scores.append(score if isfinite(score) else None)
        valid = isfinite(score) and score > 0 and score >= threshold and 0 <= x < width and 0 <= y < height
        joints.append([x, y, 2] if valid else None)
    return joints + [None, None], scores + [None, None]


class Predictor:
    def __init__(self, options):
        import torch
        import ultralytics

        self.options = options
        checkpoint = Path(options["checkpoint"])
        if not checkpoint.is_file():
            raise ValueError(f"Local checkpoint {checkpoint.name} is unavailable.")
        with checkpoint.open("rb") as file:
            digest = hashlib.sha256()
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        self.model = ultralytics.YOLO(str(checkpoint))
        if self.model.task != options["task"]:
            raise ValueError(f"{checkpoint.name} is a {self.model.task} checkpoint, expected {options['task']}.")
        if self.model.task == "pose" and tuple(self.model.model.yaml.get("kpt_shape") or ()) != (17, 3):
            raise ValueError("A YOLO pose checkpoint must use the COCO-17 layout.")
        self.person_classes = [i for i, name in self.model.names.items() if str(name).casefold() == "person"]
        if not self.person_classes:
            raise ValueError("The checkpoint has no person class.")
        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.end_to_end = bool(getattr(self.model.model, "end2end", False))
        self.nms_iou = 0.7
        self.source = dict(checkpoint_sha256=digest.hexdigest(), ultralytics_version=ultralytics.__version__,
                           device="cuda:0" if self.device == "0" else "cpu",
                           postprocessing="end-to-end" if self.end_to_end else "NMS")
        if not self.end_to_end:
            self.source["nms_iou"] = self.nms_iou

    def predict(self, path):
        import numpy as np
        from PIL import Image

        with Image.open(path) as file:
            image = np.asarray(file.convert("RGB"))[:, :, ::-1].copy()
        height, width = image.shape[:2]
        result = self.model.predict(source=image, device=self.device, imgsz=self.options["imgsz"],
                                    conf=self.options["confidence"], classes=self.person_classes,
                                    end2end=self.end_to_end, iou=self.nms_iou, verbose=False)[0]
        if not len(result.boxes):
            return []
        boxes = result.boxes.xyxy.tolist()
        confidences = result.boxes.conf.tolist()
        pose = self.model.task == "pose"
        points = result.keypoints.data.tolist() if pose and result.keypoints is not None else []
        if pose and len(points) != len(boxes):
            raise ValueError("YOLO returned different numbers of boxes and poses.")
        people = []
        for index, ((x, y, right, bottom), score) in enumerate(zip(boxes, confidences)):
            if not all(map(isfinite, (x, y, right, bottom, score))) or right <= x or bottom <= y:
                continue
            if pose:
                joints, joint_scores = map_yolo_joints(points[index], width, height, self.options["joint_confidence"])
            else:
                joints, joint_scores = [None] * 14, [None] * 14
            people.append(dict(bbox=[x, y, right - x, bottom - y], joints=joints,
                               detection_score=score, joint_scores=joint_scores))
        return people


def main():
    request = json.load(sys.stdin)
    try:
        with redirect_stdout(sys.stderr):
            predictor = Predictor(request["options"])
        print(json.dumps({"source": predictor.source}), flush=True)
    except Exception as error:
        print(json.dumps({"fatal": str(error)}), flush=True)
        return
    for index, path in enumerate(request["paths"]):
        try:
            with redirect_stdout(sys.stderr):
                people = predictor.predict(path)
            print(json.dumps(dict(index=index, people=people), allow_nan=False), flush=True)
        except Exception as error:
            print(json.dumps(dict(index=index, error=str(error))), flush=True)


if __name__ == "__main__":
    main()
