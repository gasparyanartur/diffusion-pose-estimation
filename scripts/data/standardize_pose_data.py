"""Create COCO interchange copies from local CrowdPose or COCO pose labels."""

import argparse
import hashlib
import json
from math import isfinite
from os.path import relpath
from pathlib import Path

from scripts.data.manual_annotations import valid_numbers, write_json_atomic

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA = "pose_graph_diff"
FORMAT_VERSION = 1


def index_by_id(document, section):
    indexed = {}
    for item in document[section]:
        ident = item["id"]
        if type(ident) is not int:
            raise ValueError(f"{section} IDs must be integers.")
        if ident in indexed:
            raise ValueError(f"{section} has duplicate ID {ident}.")
        indexed[ident] = item
    return indexed


def normalize_category(category, skeleton_base):
    if "keypoints" not in category and "skeleton" not in category:
        return category
    names = category["keypoints"]
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"Category {category['id']} has invalid keypoint names.")
    if len(set(names)) != len(names):
        raise ValueError(f"Category {category['id']} has duplicate keypoint names.")
    edges = category["skeleton"]
    if not isinstance(edges, list) or any(
        not isinstance(edge, list) or len(edge) != 2 or
        any(type(i) is not int or not skeleton_base <= i < skeleton_base + len(names) for i in edge)
        for edge in edges
    ):
        raise ValueError(f"Category {category['id']} has an invalid skeleton.")
    return {**category, "skeleton": [[i - skeleton_base + 1 for i in edge] for edge in edges]}


def validate_image(image, licenses):
    ident = image["id"]
    filename = image["file_name"]
    if (not isinstance(filename, str) or not filename or Path(filename).is_absolute()
            or ".." in Path(filename).parts):
        raise ValueError(f"Image {ident} has an invalid relative file_name.")
    if any(type(image[axis]) is not int or image[axis] <= 0 for axis in ("width", "height")):
        raise ValueError(f"Image {ident} has invalid dimensions.")
    if "license" in image and (type(image["license"]) is not int or image["license"] not in licenses):
        raise ValueError(f"Image {ident} references an unknown license.")


def validate_segmentation(segmentation, image, ident):
    if isinstance(segmentation, list):
        valid = all(isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0
                    and valid_numbers(polygon, len(polygon)) for polygon in segmentation)
    elif isinstance(segmentation, dict):
        dimensions = segmentation.get("size")
        counts = segmentation.get("counts")
        valid = valid_numbers(dimensions, 2) and all(type(axis) is int for axis in dimensions)
        valid = valid and dimensions == [image["height"], image["width"]]
        if isinstance(counts, list):
            valid = (valid and all(type(n) is int and n >= 0 for n in counts)
                     and sum(counts) == image["height"] * image["width"])
        else:
            valid = valid and isinstance(counts, str) and bool(counts) and counts.isascii()
    else:
        valid = False
    if not valid:
        raise ValueError(f"Annotation {ident} has invalid segmentation.")


def referenced_record(annotation, field, index):
    value = annotation[field]
    if type(value) is not int or value not in index:
        raise ValueError(f"Annotation {annotation['id']} references an unknown {field}.")
    return index[value]


def normalize_annotation(annotation, image, category):
    ident = annotation["id"]
    box = annotation["bbox"]
    if not valid_numbers(box, 4) or min(box[2:]) <= 0:
        raise ValueError(f"Annotation {ident} has an invalid bbox.")
    if type(annotation["iscrowd"]) is not int or annotation["iscrowd"] not in (0, 1):
        raise ValueError(f"Annotation {ident} has invalid iscrowd.")
    if "segmentation" in annotation:
        validate_segmentation(annotation["segmentation"], image, ident)
    if METADATA in annotation:
        raise ValueError(f"Annotation {ident} already has reserved {METADATA} metadata.")

    normalized = dict(annotation)
    derived = {}
    if "area" not in annotation:
        normalized["area"] = box[2] * box[3]
        derived["area_origin"] = "bbox"
    area = normalized["area"]
    if type(area) not in (int, float) or not isfinite(area) or area <= 0:
        raise ValueError(f"Annotation {ident} has invalid area.")

    if "keypoints" in annotation:
        points = annotation["keypoints"]
        if not valid_numbers(points, 3 * len(category.get("keypoints", []))):
            raise ValueError(f"Annotation {ident} has an invalid keypoints array.")
        visibility = points[2::3]
        if any(type(v) is not int or v not in (0, 1, 2) or (v == 0 and any(points[i:i+2]))
               for i, v in zip(range(0, len(points), 3), visibility)):
            raise ValueError(f"Annotation {ident} has invalid keypoints visibility or coordinates.")
        count = sum(v > 0 for v in visibility)
        if "num_keypoints" in annotation and annotation["num_keypoints"] != count:
            derived["source_num_keypoints"] = annotation["num_keypoints"]
        normalized["num_keypoints"] = count
    elif category.get("keypoints") or "num_keypoints" in annotation:
        raise ValueError(f"Annotation {ident} is missing its keypoints array.")
    if derived:
        normalized[METADATA] = derived
    return normalized


def normalize_coco(document, *, dataset, split, source_sha256, skeleton_base):
    """Normalize and validate one split without changing the source records."""
    if skeleton_base not in (0, 1):
        raise ValueError("Source skeleton indexing must start at 0 or 1.")
    info = dict(document.get("info", {}))
    if METADATA in info:
        raise ValueError(f"Source already has reserved {METADATA} metadata.")
    info[METADATA] = {
        "version": FORMAT_VERSION, "dataset": dataset, "split": split, "source_sha256": source_sha256,
    }
    licenses = document.get("licenses", [])
    license_index = index_by_id({"licenses": licenses}, "licenses")
    images = index_by_id(document, "images")
    for image in images.values():
        validate_image(image, license_index)
    categories = {ident: normalize_category(category, skeleton_base)
                  for ident, category in index_by_id(document, "categories").items()}
    annotations = [
        normalize_annotation(annotation, referenced_record(annotation, "image_id", images),
                             referenced_record(annotation, "category_id", categories))
        for annotation in index_by_id(document, "annotations").values()
    ]
    return {**document, "info": info, "licenses": licenses, "categories": list(categories.values()),
            "annotations": annotations}


def convert_dataset(dataset, root, output):
    if dataset == "crowdpose":
        split_files = {split: f"crowdpose_{split}.json" for split in ("train", "val", "test")}
        image_root = root / "images"
        skeleton_base = 0
    elif dataset == "coco":
        split_files = {"val": "person_keypoints_val2017.json"}
        image_root = root / "val2017"
        skeleton_base = 1
    else:
        raise ValueError(f"Unknown dataset {dataset!r}.")
    if not image_root.is_dir():
        raise ValueError(f"Missing image root: {image_root}")

    destination = output / dataset
    annotation_dir = destination / "annotations"
    documents = {}
    manifest = {"format": "coco", "version": FORMAT_VERSION, "dataset": dataset, "splits": {}}
    for split, filename in split_files.items():
        path = root / "annotations" / filename
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        try:
            document = normalize_coco(json.loads(raw), dataset=dataset, split=split,
                                      source_sha256=digest, skeleton_base=skeleton_base)
            for image in document["images"]:
                if not (image_root / image["file_name"]).is_file():
                    raise ValueError(f"Missing image file: {image['file_name']}")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{path}: {error}") from error
        documents[split] = document
        manifest["splits"][split] = {
            "annotations": f"annotations/{split}.json",
            "image_root": relpath(image_root.resolve(), destination.resolve()),
            "source_annotations": relpath(path.resolve(), destination.resolve()),
            "source_sha256": digest,
            "image_count": len(document["images"]),
            "annotation_count": len(document["annotations"]),
        }

    annotation_dir.mkdir(parents=True, exist_ok=True)
    for split, document in documents.items():
        write_json_atomic(annotation_dir / f"{split}.json", document)
    manifest_path = destination / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("crowdpose", "coco"))
    parser.add_argument("--root", required=True, type=Path, help="Existing image and annotation directory")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/standardized")
    args = parser.parse_args()
    try:
        print(convert_dataset(args.dataset, args.root, args.output))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Could not standardize dataset: {error}\n")


if __name__ == "__main__":
    main()
