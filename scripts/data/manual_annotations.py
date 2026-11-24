"""Local, independently revisioned CrowdPose annotation sets."""
from contextlib import contextmanager

import fcntl
import json
import math
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4


UNDO_LIMIT = 20


class Conflict(ValueError):
    pass


def valid_numbers(values, count):
    return (isinstance(values, list) and len(values) == count and
            all(type(value) in (int, float) and math.isfinite(value) for value in values))


def validate_people(people, existing):
    if not isinstance(people, list):
        raise ValueError("Expected a person list.")
    known = {person["id"] for person in existing}
    for person in people:
        if not isinstance(person, dict) or person.get("id") not in known:
            raise ValueError("Invalid or duplicate person.")
        known.remove(person["id"])
        box = person["bbox"]
        if box is not None and (not valid_numbers(box, 4) or min(box[2:]) <= 0):
            raise ValueError("Box width and height must be positive.")
        if not isinstance(person["joints"], list) or len(person["joints"]) != 14:
            raise ValueError("Expected 14 joint slots.")
        for joint in person["joints"]:
            if joint is None:
                continue
            if (not valid_numbers(joint, 3) or joint[2] not in (0, 1, 2)
                    or (joint[2] == 0 and joint[:2] != [0, 0])):
                raise ValueError("Invalid joint coordinates or visibility.")


def write_json_atomic(path, record):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        try:
            json.dump(record, tmp, allow_nan=False)
            tmp.flush()
            os.fsync(tmp.fileno())
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


class AnnotationSets:
    def __init__(self, directory, known_image, ground_truth=None):
        self.directory = Path(directory)
        self.known_image = known_image
        self.ground_truth = ground_truth

    def path(self, ident):
        if not isinstance(ident, str) or not re.fullmatch("[a-f0-9]{32}", ident):
            raise ValueError("Invalid annotation set ID.")
        return self.directory / f"{ident}.json"

    def read(self, ident):
        return json.loads(self.path(ident).read_text(encoding="utf-8"))

    def catalog(self):
        sets = []
        for path in sorted(self.directory.glob("*.json")):
            record = self.read(path.stem)
            sets.append({"id": record["id"], "name": record["name"]})
        return sorted(sets, key=lambda item: item["name"].casefold())

    @contextmanager
    def locked(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def apply(self, command):
        with self.locked():
            if command["action"] == "create":
                name = command["name"]
                if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
                    raise ValueError("Enter a set name of 1–100 characters.")
                if any(s["name"].casefold() == name.strip().casefold() for s in self.catalog()):
                    raise ValueError("That set name already exists.")
                record = dict(version=1, id=uuid4().hex, name=name.strip(), revision=0, images=[])
            else:
                record = self.read(command["id"])
                if record["revision"] != command["revision"]:
                    raise Conflict("Another tab changed this set. Reload before saving.")
                self.update(record, command)
            self.persist(record)
        return record

    def persist(self, record):
        record["revision"] += 1
        write_json_atomic(self.path(record["id"]), record)

    def seed(self, ident, split, image_id, expected_edit, proposals, source):
        """Compare and import this image while holding the same lock as manual saves."""
        with self.locked():
            record = self.read(ident)
            entry = next(i for i in record["images"] if (i["split"], i["image_id"]) == (split, image_id))
            if entry["people"] or entry["reviewed"] or entry.get("edit", 0) != expected_edit:
                raise Conflict("Image changed while queued.")
            people, originals = [], []
            for item in proposals:
                person = dict(id=uuid4().hex, bbox=item["bbox"], joints=item["joints"])
                people.append(person)
                originals.append(dict(item, id=person["id"]))
            validate_people(people, people)
            entry.update(people=people, autolabel=dict(source, people=originals), edit=expected_edit+1)
            if "undo" in record:
                record["undo"] = [snapshot for snapshot in record["undo"]
                                  if (snapshot["split"], snapshot["image_id"]) != (split, image_id)]
            self.persist(record)

    def copy_source(self, source_id, split, image_id):
        if not isinstance(split, str) or type(image_id) is not int or not self.known_image(split, image_id):
            raise ValueError("Unknown source image.")
        if source_id == "original":
            if self.ground_truth is None:
                raise ValueError("Original annotations are unavailable.")
            people = []
            for annotation in self.ground_truth(split, image_id):
                if annotation.get("iscrowd"):
                    continue
                points = annotation.get("keypoints", [])
                joints = [points[i:i + 3] for i in range(0, 42, 3)] if points else [None] * 14
                people.append(dict(id=annotation["id"], bbox=annotation.get("bbox"), joints=joints))
            return dict(revision=None, people=people)
        source = self.read(source_id)
        image = next((i for i in source["images"] if (i["split"], i["image_id"]) == (split, image_id)), None)
        return dict(revision=source["revision"], people=image["people"] if image else [])

    def copy_people(self, target_id, entry, command):
        source_id = command["source"]
        if source_id == target_id:
            raise ValueError("Choose a different source set.")
        source = self.copy_source(source_id, entry["split"], entry["image_id"])
        if source["revision"] != command.get("source_revision"):
            raise Conflict("Source set changed. Reopen the copy dialog.")
        selected = command["people"]
        boxes, joints = command["boxes"], command["joints"]
        if type(boxes) is not bool or type(joints) is not bool or not (boxes or joints):
            raise ValueError("Choose boxes, joints, or both.")
        if not isinstance(selected, list) or not selected or any(type(ident) not in (int, str) for ident in selected):
            raise ValueError("Choose at least one source person.")
        by_id = {person["id"]: person for person in source["people"]}
        if len(set(selected)) != len(selected) or any(ident not in by_id for ident in selected):
            raise ValueError("Choose distinct people from the source image.")

        copies = []
        for ident in selected:
            person = by_id[ident]
            box = person["bbox"] if boxes else None
            keypoints = person["joints"] if joints else [None] * 14
            if box is None and not any(slot is not None for slot in keypoints):
                continue
            copies.append(dict(id=uuid4().hex, bbox=box, joints=keypoints,
                               copied_from=dict(set_id=source_id, revision=source["revision"], person_id=ident)))
        if not copies:
            raise ValueError("No selected person has the requested geometry.")
        validate_people(copies, copies)
        return [*entry["people"], *copies]

    def update(self, record, command):
        entries = {(i["split"], i["image_id"]): i for i in record["images"]}
        if command["action"] == "add":
            for item in command["images"]:
                split, ident = item["split"], item["image_id"]
                if type(ident) is not int or not self.known_image(split, ident):
                    raise ValueError(f"Unknown image {split}/{ident}.")
                if (split, ident) not in entries:
                    entry = dict(split=split, image_id=ident, reviewed=False, people=[])
                    entries[split, ident] = entry
                    record["images"].append(entry)
            return
        if command["action"] == "undo":
            history = record.get("undo", [])
            if not history:
                raise ValueError("Nothing to undo.")
            snapshot = history.pop()
            entry = entries[snapshot["split"], snapshot["image_id"]]
            entry.update(people=snapshot["people"], reviewed=snapshot["reviewed"], edit=entry.get("edit", 0) + 1)
            return

        entry = entries.get((command["split"], command["image_id"]))
        if entry is None:
            raise ValueError("Image is outside this set.")
        if command["action"] == "person":
            people = [*entry["people"], dict(id=uuid4().hex, bbox=None, joints=[None] * 14)]
            reviewed = False
        elif command["action"] == "copy":
            people = self.copy_people(record["id"], entry, command)
            reviewed = False
        elif command["action"] == "save":
            people, reviewed = command["people"], command["reviewed"]
            validate_people(people, entry["people"])
            if type(reviewed) is not bool:
                raise ValueError("Invalid review status.")
        else:
            raise ValueError("Unknown annotation action.")

        if people == entry["people"] and reviewed == entry["reviewed"]:
            return
        history = record.setdefault("undo", [])
        history.append({key: entry[key] for key in ("split", "image_id", "people", "reviewed")})
        del history[:-UNDO_LIMIT]
        entry.update(people=people, reviewed=reviewed, edit=entry.get("edit", 0) + 1)
