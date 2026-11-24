"""Batch proposal import for the local annotation editor."""
import json
import os
import re
import subprocess
from copy import deepcopy
from pathlib import Path
from threading import Lock, Thread

from scripts.data.manual_annotations import Conflict
from scripts.data.sapiens2_worker import DETECTOR, DET_REV, POSE, POSE_REV

SOURCE = dict(model=POSE, revision=POSE_REV, detector=DETECTOR, detector_revision=DET_REV,
              detector_threshold=0.3, nms_iou=0.3)


def predict_batch(python, paths, *, worker_name="sapiens2_worker.py", options=None, source=None, env=None):
    if not python.is_file():
        raise ValueError("The selected worker runtime is unavailable. Check the Autolabel setup in the README.")
    worker = Path(__file__).with_name(worker_name)
    with subprocess.Popen([str(python), str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          env=dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1", **(env or {})),
                          text=True) as process:
        try:
            json.dump(paths if options is None else {"paths": paths, "options": options}, process.stdin)
            process.stdin.close()
            for line in process.stdout:
                event = json.loads(line)
                if "fatal" in event:
                    raise RuntimeError(event["fatal"])
                if "source" in event:
                    if source is not None:
                        source.update(event["source"])
                    continue
                yield event["index"], event.get("people", []), event.get("error")
            if process.wait():
                raise RuntimeError(f"Autolabel worker exited with code {process.returncode}.")
        finally:
            if process.poll() is None:
                process.terminate()


def probability(value, name):
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a number from 0 to 1.")
    return value


class AutolabelProfiles:
    def __init__(self, sapiens_python, yolo_python, weights_dir, config=None):
        self.sapiens_python = sapiens_python.absolute()
        self.yolo_python = yolo_python.absolute()
        self.weights_dir = weights_dir.resolve()
        self.models = {
            "yolo26-boxes": self.yolo_profile("YOLO26 Boxes", "detect", weights_dir / "yolo26n.pt"),
            "yolo26-pose": self.yolo_profile("YOLO26 Pose", "pose", weights_dir / "yolo26n-pose.pt"),
        }
        if config is not None:
            custom = json.loads(config.read_text(encoding="utf-8"))
            if not isinstance(custom, dict) or any(not isinstance(profile, dict) for profile in custom.values()):
                raise ValueError("Autolabel profile config must map IDs to profile objects.")
            for ident, profile in custom.items():
                if (not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", ident)
                        or ident == "sapiens2" or ident in self.models):
                    raise ValueError(f"Invalid or reserved profile ID: {ident}.")
                checkpoint = config.parent / profile["checkpoint"]
                self.models[ident] = self.yolo_profile(profile["label"], profile["task"], checkpoint,
                                                       profile.get("imgsz", 640), profile.get("confidence", 0.25),
                                                       profile.get("joint_confidence", 0.25))

    @staticmethod
    def yolo_profile(label, task, checkpoint, imgsz=640, confidence=0.25, joint_confidence=0.25):
        if not isinstance(label, str) or not 1 <= len(label.strip()) <= 100 or task not in ("detect", "pose"):
            raise ValueError("YOLO profiles need a label and a detect or pose task.")
        if type(imgsz) is not int or imgsz < 32 or imgsz > 2048 or imgsz % 32:
            raise ValueError("Profile image size must be a multiple of 32 from 32 to 2048.")
        if checkpoint.suffix.lower() != ".pt":
            raise ValueError("YOLO profiles require a local .pt checkpoint.")
        return dict(label=label.strip(), task=task, checkpoint=checkpoint.resolve(), imgsz=imgsz,
                    confidence=probability(confidence, "confidence"),
                    joint_confidence=probability(joint_confidence, "joint_confidence"))

    def catalog(self):
        return [
            dict(id="sapiens2", label="Sapiens2", geometry="Boxes and poses", confidence=0.3,
                 available=self.sapiens_python.is_file()),
            *[dict(id=ident, label=model["label"], geometry="Boxes and poses" if model["task"] == "pose" else "Boxes",
                   confidence=model["confidence"],
                   joint_confidence=model["joint_confidence"] if model["task"] == "pose" else None,
                   available=self.yolo_python.is_file() and model["checkpoint"].is_file())
              for ident, model in self.models.items()],
        ]

    def resolve(self, ident, settings):
        if not isinstance(settings, dict):
            raise ValueError("Expected autolabel settings.")
        if ident is None or ident == "sapiens2":
            if set(settings) - {"confidence"}:
                raise ValueError("Unsupported Sapiens2 setting.")
            threshold = probability(settings.get("confidence", SOURCE["detector_threshold"]), "confidence")
            source = dict(SOURCE, profile="sapiens2", label="Sapiens2", detector_threshold=threshold)
            return (lambda paths: predict_batch(self.sapiens_python, paths, options={"confidence": threshold})), source
        if not isinstance(ident, str) or ident not in self.models:
            raise ValueError("Unknown autolabel profile.")
        model = self.models[ident]
        pose = model["task"] == "pose"
        allowed = {"confidence", "joint_confidence"} if pose else {"confidence"}
        if set(settings) - allowed:
            raise ValueError("Unsupported YOLO setting.")
        options = {key: model[key] for key in ("task", "imgsz")}
        options.update({key: probability(settings.get(key, model[key]), key) for key in allowed})
        checkpoint = model["checkpoint"]
        source = dict(options, model=checkpoint.name, label=model["label"], profile=ident)
        return (lambda paths: predict_batch(
            self.yolo_python, paths, worker_name="yolo26_worker.py", source=source,
            options=dict(options, checkpoint=str(checkpoint)),
            env={"YOLO_OFFLINE": "true", "YOLO_CONFIG_DIR": str(self.weights_dir / "runtime")},
        )), source


class BatchLabels:
    def __init__(self, sets, image_path, predict):
        self.sets = sets
        self.image_path = image_path
        self.predict = predict
        self.lock = Lock()
        self.jobs = {}
        self.thread = None

    def status(self, ident):
        with self.lock:
            return deepcopy(self.jobs.get(ident, dict(running=False, items=[])))

    def start(self, ident, revision, selected, profile=None):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise Conflict("A batch is already running. Wait for it to finish.")
            record = self.sets.read(ident)
            if revision != record["revision"]:
                raise Conflict("This set changed. Reopen Autolabel to refresh.")
            entries = {(i["split"], i["image_id"]): i for i in record["images"]}
            if not isinstance(selected, list) or not selected:
                raise ValueError("Select at least one image.")
            items, work, seen = [], [], set()
            for row in selected:
                key = row["split"], row["image_id"]
                if type(key[1]) is not int or key not in entries or key in seen:
                    raise ValueError("Choose distinct images from this set.")
                seen.add(key)
                entry = entries[key]
                item = dict(split=key[0], image_id=key[1], status="queued", detail="")
                items.append(item)
                if entry["reviewed"] or entry["people"]:
                    item.update(status="skipped", detail="Already labeled or reviewed")
                    continue
                try:
                    path = self.image_path(*key)
                except (ValueError, OSError) as error:
                    item.update(status="failed", detail=str(error))
                    continue
                work.append((item, str(path), entry.get("edit", 0)))
            predict, source = profile if profile is not None else (self.predict, SOURCE)
            job = self.jobs[ident] = dict(running=bool(work), items=items, label=source.get("label", "Sapiens2"))
            if work:
                self.thread = Thread(target=self.run, args=(ident, job, work, predict, source), daemon=True)
                self.thread.start()
            return deepcopy(job)

    def run(self, ident, job, work, predict, source):
        def mark(item, status, detail=""):
            with self.lock:
                item.update(status=status, detail=detail)
        failure = "Worker stopped before this image finished."
        try:
            mark(work[0][0], "processing")
            for index, proposals, error in predict([path for _, path, _ in work]):
                item, _, expected = work[index]
                if error is not None:
                    mark(item, "failed", error)
                else:
                    try:
                        self.sets.seed(ident, item["split"], item["image_id"], expected, proposals, source)
                    except Conflict as exc:
                        mark(item, "skipped", str(exc))
                    except (OSError, ValueError) as exc:
                        mark(item, "failed", str(exc))
                    else:
                        count = len(proposals)
                        mark(item, "saved" if count else "no people", f"{count} detected")
                if index + 1 < len(work):
                    mark(work[index + 1][0], "processing")
        except Exception as error:
            failure = str(error) or type(error).__name__
        finally:
            with self.lock:
                for item, _, _ in work:
                    if item["status"] in {"queued", "processing"}:
                        item.update(status="failed", detail=failure)
                job["running"] = False
