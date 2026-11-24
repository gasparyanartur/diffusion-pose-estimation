import pytest

from scripts.data.crowdpose_audit import AuditStore, image_stats, review_update


def test_source_stats_distinguish_overlap_visibility_and_conflicts():
    image = dict(id=2, width=100, height=100, crowdIndex=0.7)
    def person(ident, bbox, head, wrist=None, **kwargs):
        joints = [0] * 42
        joints[36:39] = head
        if wrist:
            joints[12:15] = wrist
        return dict(id=ident, bbox=bbox, keypoints=joints, **kwargs)

    people = [
        person(1, [0, 0, 40, 40], [30, 30, 2], [60, 15, 2]),
        person(2, [10, 0, 40, 40], [31, 31, 1]),
        dict(id=3, bbox=[90, 90, 5, 5]),
        dict(id=4, bbox=[0, 0, 100, 100], iscrowd=1),
    ]
    stats = image_stats(image, people)
    assert stats["people"] == 3 and stats["box_only"] == 1
    assert stats["max_iou"] == pytest.approx(0.6) and stats["overlapping"] == 2
    assert stats["min_height"] == 5 and stats["small"] == 3 and stats["edge"] == 2
    assert stats["missing"][4] == 2 and stats["occluded"][12] == 1
    assert stats["duplicate"] == {"ids": [1, 2], "joint": 12}
    assert stats["outside"] == {"ids": [1], "joint": 4}
    assert image_stats(image, [])["min_height"] is None


def test_audit_updates_are_idempotent_and_persist(tmp_path):
    root = tmp_path / "audit"
    store = AuditStore(root)
    people = [dict(id=1), dict(id=2, iscrowd=1)]
    update = dict(action="tag", kind="wrong_owner", person=1, joint=12)
    def apply(command):
        return store.apply("val", 9, lambda review: review_update(review, command, people))
    apply(update)
    record = apply(update)
    assert len(record["tags"]) == 1 and not record["reviewed"]
    for invalid in (dict(update, person=2), dict(update, person=None), dict(update, joint=14)):
        with pytest.raises(ValueError):
            apply(invalid)
    apply(dict(action="cohort", value="random"))
    apply(dict(action="reviewed", value=True))
    assert AuditStore(root).read("val", 9)["cohort"] == "random"
    assert "9" in AuditStore(root).for_split("val")
    clean = dict(action="tag", kind="clean", person=None, joint=None)
    with pytest.raises(ValueError):
        apply(clean)
    apply(dict(update, action="remove"))
    apply(clean)
    record = store.read("val", 9)
    assert record["reviewed"] and record["tags"] == [dict(kind="clean", person=None, joint=None)]
    record = apply(update)
    assert record["tags"] == [dict(kind="wrong_owner", person=1, joint=12)] and not record["reviewed"]
    assert apply(dict(update, action="remove"))["tags"] == []
