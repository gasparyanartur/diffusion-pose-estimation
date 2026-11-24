"""Source-label statistics and independent human audit records."""
import fcntl
import json
from itertools import combinations
from math import dist, isfinite, sqrt

from scripts.data.manual_annotations import valid_numbers, write_json_atomic
from src.data.skeleton import CROWDPOSE_JOINT_NAMES

OVERLAP_IOU = 0.20
SMALL_HEIGHT = 64
DUPLICATE_RADIUS = 3
JOINT_COUNT = len(CROWDPOSE_JOINT_NAMES)
TAGS = {
    "missing_person": "Missing person",
    "extra_person": "Extra person",
    "wrong_owner": "Wrong owner",
    "misplaced_joint": "Misplaced joint",
    "visibility_error": "Visibility error",
    "bad_box": "Bad box",
    "clean": "Reviewed clean",
}
COHORTS = {"": "Unassigned", "random": "Random audit", "picked": "Hand-picked"}


def corners(box):
    if valid_numbers(box, 4) and min(box[2:]) > 0:
        x, y, width, height = box
        return x, y, x + width, y + height
    return None


def box_iou(first, second):
    x0, y0, x1, y1 = first
    a0, b0, a1, b1 = second
    intersection = max(0, min(x1, a1) - max(x0, a0)) * max(0, min(y1, b1) - max(y0, b0))
    return intersection / ((x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - intersection)


def labeled_joints(annotation):
    points = annotation.get("keypoints") or []
    result = []
    for start in range(0, JOINT_COUNT * 3, 3):
        joint = points[start:start + 3]
        result.append(joint if valid_numbers(joint, 3) and joint[2] in (1, 2) else None)
    return result


def image_stats(image, annotations):
    people = [person for person in annotations if not person.get("iscrowd")]
    boxes = {person["id"]: box for person in people if (box := corners(person.get("bbox")))}
    heights = [y1 - y0 for _, y0, _, y1 in boxes.values()]
    width, height = image.get("width"), image.get("height")
    edge = None
    if type(width) in (int, float) and type(height) in (int, float) and min(width, height) > 0:
        edge = sum(x0 <= 0 or y0 <= 0 or x1 >= width or y1 >= height for x0, y0, x1, y1 in boxes.values())
    overlap_people, max_iou = set(), None
    for (first, first_box), (second, second_box) in combinations(boxes.items(), 2):
        iou = box_iou(first_box, second_box)
        max_iou = iou if max_iou is None else max(max_iou, iou)
        if iou >= OVERLAP_IOU:
            overlap_people.update((first, second))

    missing, occluded = [0] * JOINT_COUNT, [0] * JOINT_COUNT
    seen = [[] for _ in range(JOINT_COUNT)]
    duplicate = outside = None
    box_only = 0
    for person in people:
        ident = person["id"]
        joints = labeled_joints(person)
        box = boxes.get(ident)
        box_only += box is not None and not any(joints)
        for index, joint in enumerate(joints):
            if joint is None:
                missing[index] += 1
                continue
            x, y, visibility = joint
            occluded[index] += visibility == 1
            if duplicate is None:
                for other, point in seen[index]:
                    if dist((x, y), point) <= DUPLICATE_RADIUS:
                        duplicate = dict(ids=[other, ident], joint=index)
                        break
                seen[index].append((ident, (x, y)))
            if outside is None and box and visibility == 2:
                x0, y0, x1, y1 = box
                offset = dist((x, y), (min(max(x, x0), x1), min(max(y, y0), y1)))
                if offset > max(3, 0.10 * sqrt((x1 - x0) * (y1 - y0))):
                    outside = dict(ids=[ident], joint=index)

    crowd_index = image.get("crowdIndex")
    if type(crowd_index) not in (int, float) or not isfinite(crowd_index):
        crowd_index = None
    return dict(image_id=image["id"], crowd_index=crowd_index, people=len(people),
                box_only=box_only, max_iou=max_iou, overlapping=len(overlap_people),
                min_height=min(heights, default=None), small=sum(h < SMALL_HEIGHT for h in heights),
                edge=edge, missing=missing, occluded=occluded, duplicate=duplicate, outside=outside)


class AuditStore:
    def __init__(self, directory):
        self.directory = directory

    def read(self, split, image_id):
        path = self.directory / split / f"{image_id}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, NotADirectoryError):
            return dict(cohort="", reviewed=False, tags=[])

    def for_split(self, split):
        return {path.stem: self.read(split, path.stem)
                for path in (self.directory / split).glob("*.json")}

    def apply(self, split, image_id, change):
        folder = self.directory / split
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            record = self.read(split, image_id)
            change(record)
            write_json_atomic(folder / f"{image_id}.json", record)
            return record


def review_update(record, command, annotations):
    action = command["action"]
    if action == "cohort" and command["value"] in COHORTS:
        record["cohort"] = command["value"]
        return
    if action == "reviewed" and type(command["value"]) is bool:
        record["reviewed"] = command["value"]
        if not record["reviewed"]:
            record["tags"] = [tag for tag in record["tags"] if tag["kind"] != "clean"]
        return
    if action not in {"tag", "remove"}:
        raise ValueError("Unknown audit update.")
    kind, person, joint = (command[key] for key in ("kind", "person", "joint"))
    person_ids = {item["id"] for item in annotations if not item.get("iscrowd")}
    if kind not in TAGS or (person is not None and (type(person) is not int or person not in person_ids)):
        raise ValueError("Unknown tag or source person.")
    if joint is not None and (person is None or type(joint) is not int or not 0 <= joint < JOINT_COUNT):
        raise ValueError("Choose a source person and a valid joint.")
    if kind == "clean" and person is not None:
        raise ValueError("Reviewed clean applies to the whole image.")
    tag = dict(kind=kind, person=person, joint=joint)
    if action == "remove":
        if tag in record["tags"]:
            record["tags"].remove(tag)
            record["reviewed"] = False
    elif tag not in record["tags"]:
        if kind == "clean" and record["tags"]:
            raise ValueError("Remove the issue tags before marking this image clean.")
        if kind != "clean":
            record["tags"] = [item for item in record["tags"] if item["kind"] != "clean"]
        record["tags"].append(tag)
        record["reviewed"] = kind == "clean"
