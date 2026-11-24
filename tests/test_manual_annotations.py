from copy import deepcopy

import pytest

from scripts.data.manual_annotations import AnnotationSets, Conflict


def test_partial_labels_are_isolated_and_survive_restart(tmp_path):
    def known(split, ident):
        return split == "val" and ident == 9

    store = AnnotationSets(tmp_path, known)
    def create(name):
        return store.apply({"action": "create", "name": name})
    def update(record, action, **payload):
        return store.apply(dict(id=record["id"], revision=record["revision"], action=action, **payload))

    first, second = create("A"), create("B")
    item = {"split": "val", "image_id": 9}
    queued = update(first, "add", images=[item, item])
    assert len(queued["images"]) == 1
    with pytest.raises(Conflict):
        update(first, "add", images=[item])
    result = update(queued, "person", **item)
    person = result["images"][0]["people"][0]
    person["joints"][:3] = [[3, 4, 2], [5, 6, 1], [0, 0, 0]]
    person["bbox"] = [-1, 0.5, 8, 7]
    saved = update(result, "save", people=[person], reviewed=False, **item)
    assert AnnotationSets(tmp_path, known).read(first["id"]) == saved
    assert store.read(second["id"])["images"] == []
    person = deepcopy(person)
    person["joints"][0][2] = 4
    with pytest.raises(ValueError):
        update(saved, "save", people=[person], reviewed=False, **item)
    with pytest.raises(ValueError):
        update(saved, "add", images=[{"split":"val", "image_id":77}])
    assert store.read(first["id"]) == saved
    empty = update(second, "add", images=[item])
    assert update(empty, "save", people=[], reviewed=True, **item)["images"][0]["reviewed"]


def test_undo_restores_people_and_review_across_images(tmp_path):
    store = AnnotationSets(tmp_path, lambda split, ident: True)
    record = store.apply({"action": "create", "name": "undo"})
    images = [{"split": "val", "image_id": 1}, {"split": "test", "image_id": 2}]

    def apply(record, action, **options):
        return store.apply(dict(id=record["id"], revision=record["revision"], action=action, **options))

    record = apply(record, "add", images=images)
    record = apply(record, "person", **images[0])
    saved_person = deepcopy(record["images"][0]["people"][0])
    saved_person["joints"][4] = [8, 9, 1]
    saved_person["bbox"] = [2, 3, 6, 7]
    record = apply(record, "save", people=[saved_person], reviewed=True, **images[0])
    record = apply(record, "save", people=[], reviewed=False, **images[0])
    record = apply(record, "save", people=[], reviewed=True, **images[1])
    assert len(record["undo"]) == 4
    noop = apply(record, "save", people=[], reviewed=True, **images[1])
    assert noop["undo"] == record["undo"]
    with pytest.raises(Conflict):
        apply(record, "undo")
    store = AnnotationSets(tmp_path, lambda split, ident: True)
    restored = apply(noop, "undo")
    assert not restored["images"][1]["reviewed"]
    assert restored["images"][1]["edit"] > noop["images"][1]["edit"]
    restored = apply(restored, "undo")
    assert restored["images"][0]["people"] == [saved_person]
    assert restored["images"][0]["reviewed"]
    assert len(restored["undo"]) == 2


def test_undo_cap_and_autolabel_invalidation(tmp_path):
    store = AnnotationSets(tmp_path, lambda split, ident: True)
    record = store.apply({"action": "create", "name": "cap"})
    ident = record["id"]
    first = {"split": "val", "image_id": 1}
    second = {"split": "val", "image_id": 2}

    def apply(action, **options):
        return store.apply(dict(id=ident, revision=store.read(ident)["revision"], action=action, **options))

    apply("add", images=[first, second])
    for index in range(22):
        reviewed = index % 2 == 0
        record = apply("save", people=[], reviewed=reviewed, **first)
    assert len(record["undo"]) == 20
    record = apply("person", **second)
    store.seed(ident, "val", 1, record["images"][0]["edit"],
               [{"bbox": [0, 0, 20, 20], "joints": [None] * 14}], {"model": "fixture"})
    record = apply("undo")
    assert not record["images"][1]["people"]
    assert record["images"][0]["autolabel"]["model"] == "fixture"
    assert record["images"][0]["people"]
    assert record["undo"] == []
    with pytest.raises(ValueError, match="Nothing to undo"):
        apply("undo")


def test_copy_source_people_from_same_image_and_undo(tmp_path):
    gt = [
        {"id": 17, "iscrowd": 0, "bbox": [10, 12, 20, 24],
         "keypoints": [1, 2, 2, 3, 4, 1] + [0] * 36},
        {"id": 18, "iscrowd": 1, "bbox": [10, 12, 20, 24], "keypoints": [0] * 42},
    ]
    store = AnnotationSets(tmp_path, lambda split, ident: split == "val" and ident in (1, 2),
                           lambda split, ident: gt if ident == 1 else [])
    placement = {"split": "val", "image_id": 1}

    def apply(record, action, **options):
        return store.apply(dict(id=record["id"], revision=record["revision"], action=action, **options))

    source = store.apply({"action": "create", "name": "Source"})
    source = apply(source, "add", images=[placement])
    source = apply(source, "person", **placement)
    original = dict(source["images"][0]["people"][0], bbox=[10, 10, 30, 30])
    source = apply(source, "save", **placement, people=[original], reviewed=False)
    target = store.apply({"action": "create", "name": "Target"})
    target = apply(target, "add", images=[placement])
    target = apply(target, "person", **placement)
    before = target["images"][0]["people"]

    assert [p["id"] for p in store.copy_source("original", "val", 1)["people"]] == [17]
    target = apply(target, "copy", **placement, source="original", source_revision=None,
                   people=[17], boxes=True, joints=True)
    copy = target["images"][0]["people"][-1]
    assert copy["id"] != before[0]["id"]
    assert copy["joints"][:3] == [[1, 2, 2], [3, 4, 1], [0, 0, 0]]
    assert copy["copied_from"]["person_id"] == 17
    target = apply(target, "undo")
    assert target["images"][0]["people"] == before

    with pytest.raises(Conflict):
        apply(target, "copy", **placement, source=source["id"], source_revision=0,
              people=[original["id"]], boxes=True, joints=False)
    with pytest.raises(ValueError, match="different source set"):
        apply(target, "copy", **placement, source=target["id"], source_revision=target["revision"],
              people=[before[0]["id"]], boxes=True, joints=False)
    assert store.copy_source(source["id"], "val", 2)["people"] == []
    target = apply(target, "copy", **placement, source=source["id"], source_revision=source["revision"],
                   people=[original["id"]], boxes=True, joints=False)
    copy = target["images"][0]["people"][-1]
    assert copy["id"] != original["id"] and copy["joints"] == [None] * 14
    assert copy["bbox"] == original["bbox"]
    assert store.read(source["id"]) == source
