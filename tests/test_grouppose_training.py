import hashlib
import json

import pytest
import torch

from scripts.training.train_grouppose import save_checkpoint, validate_protocol


def test_protocol_rejects_train_test_overlap(tmp_path):
    annotations = tmp_path / "json"
    annotations.mkdir()
    train = annotations / "crowdpose_trainval.json"
    test = annotations / "crowdpose_test.json"
    manifest = tmp_path / "subset.json"
    train.write_text(json.dumps({"images": [{"id": image_id} for image_id in range(12000)]}))
    test.write_text(json.dumps({"images": [{"id": image_id} for image_id in range(12000, 20000)]}))
    manifest.write_text(json.dumps({
        "seed": 42, "annotation_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
        "image_ids": list(range(12000, 12032)),
    }))
    assert validate_protocol(tmp_path, manifest) == (train, test)

    train.write_text(json.dumps({"images": [{"id": image_id} for image_id in range(1, 12001)]}))
    with pytest.raises(ValueError, match="disjoint"):
        validate_protocol(tmp_path, manifest)


def test_epoch_checkpoint_keeps_the_previous_best(tmp_path):
    save_checkpoint(tmp_path, {"epoch": 1}, is_best=True)
    first_best = (tmp_path / "best.pth").read_bytes()
    save_checkpoint(tmp_path, {"epoch": 2}, is_best=False)
    assert (tmp_path / "best.pth").read_bytes() == first_best
    assert torch.load(tmp_path / "last.pth", weights_only=True) == {"epoch": 2}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["best.pth", "last.pth"]
