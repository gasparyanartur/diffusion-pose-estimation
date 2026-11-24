import copy
import json

import pytest
from pycocotools.coco import COCO

from scripts.data.standardize_pose_data import convert_dataset, normalize_coco


@pytest.fixture
def source():
    return {
        "info": {"description": "Example labels"},
        "licenses": [{"id": 1, "name": "Fixture license", "url": "https://example.org/licenses"}],
        "categories": [{"id": 1, "name": "person", "keypoints": ["shoulder", "wrist"], "skeleton": [[0, 1]]}],
        "images": [{"id": 20, "width": 20, "height": 30, "file_name": "20.jpg", "license": 1,
                    "crowdIndex": 0.42, "coco_url": "https://example.org/20.jpg"}],
        "annotations": [{"id": 4, "image_id": 20, "category_id": 1, "iscrowd": 0, "bbox": [0, 0, 5, 10],
                         "keypoints": [1, 2, 2, 3, 4, 1], "num_keypoints": 1,
                         "segmentation": [[0, 0, 0, 5, 5, 5, 5, 0]]},
                        {"id": 5, "image_id": 20, "category_id": 1, "iscrowd": 1, "bbox": [0, 0, 3, 4],
                         "keypoints": [0] * 6, "num_keypoints": 0, "area": 12,
                         "segmentation": {"size": [30, 20], "counts": [0, 600]}}],
    }


def normalize(source):
    return normalize_coco(source, dataset="crowdpose", split="val", source_sha256="123", skeleton_base=0)


def test_normalization_preserves_metadata_and_masks(source):
    before = copy.deepcopy(source)
    document = normalize(source)
    assert source == before
    assert document["images"] == source["images"]
    assert document["licenses"] == source["licenses"]
    assert document["info"]["pose_graph_diff"]["source_sha256"] == "123"
    assert document["categories"][0]["skeleton"] == [[1, 2]]
    assert document["annotations"][0]["area"] == 50
    assert document["annotations"][0]["num_keypoints"] == 2
    assert document["annotations"][0]["pose_graph_diff"] == {"area_origin": "bbox", "source_num_keypoints": 1}
    assert document["annotations"][1] == source["annotations"][1]
    assert document["annotations"][0]["segmentation"] == source["annotations"][0]["segmentation"]


@pytest.mark.parametrize(("section", "index", "field", "value", "message"), [
    ("annotations", 1, "id", 4, "duplicate"),
    ("annotations", 0, "image_id", 999, "image_id"),
    ("annotations", 0, "bbox", [0, 0, 0, 5], "bbox"),
    ("annotations", 0, "keypoints", [1, 2, 2], "keypoints"),
    ("annotations", 1, "segmentation", {"size": [30, 20], "counts": [5]}, "segmentation"),
    ("categories", 0, "skeleton", [[0, 2]], "skeleton"),
    ("images", 0, "license", 3, "license"),
])
def test_invalid_source_is_reported(source, section, index, field, value, message):
    source[section][index][field] = value
    with pytest.raises(ValueError, match=message):
        normalize(source)


def test_cli_layout_and_output_are_repeatable(source, tmp_path):
    root = tmp_path / "source"
    source["categories"][0]["skeleton"] = [[1, 2]]
    (root / "annotations").mkdir(parents=True)
    image_root = root / "val2017"
    image_root.mkdir()
    (image_root / "20.jpg").touch()
    source_path = root / "annotations" / "person_keypoints_val2017.json"
    source_path.write_text(json.dumps(source))

    manifest_path = convert_dataset("coco", root, tmp_path / "standard")
    manifest = json.loads(manifest_path.read_text())
    split = manifest["splits"]["val"]
    labels = manifest_path.parent / split["annotations"]
    assert (manifest_path.parent / split["image_root"]).resolve() == image_root
    indexed = COCO(str(labels))
    assert indexed.getImgIds() == [20]
    assert indexed.getAnnIds() == [4, 5]
    assert indexed.loadCats([1])[0]["skeleton"] == [[1, 2]]
    files = (manifest_path, labels, source_path)
    before = [path.read_bytes() for path in files]
    convert_dataset("coco", root, tmp_path / "standard")
    assert [path.read_bytes() for path in files] == before
