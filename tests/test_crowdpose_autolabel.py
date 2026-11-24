import pytest

from scripts.data.manual_annotations import AnnotationSets, Conflict
from scripts.data.crowdpose_autolabel import BatchLabels
from scripts.data.sapiens2_worker import map_joints


def test_joint_mapping_uses_wrist_and_neck_landmarks():
    points = [[i, i + 1] for i in range(308)]
    points[6] = [-10, 5]
    joints, scores = map_joints(points, [0.75] * 308, 100, 100)
    assert joints == [[5, 6, 2], None, [7, 8, 2], [8, 9, 2], [62, 63, 2], [41, 42, 2],
                      [9, 10, 2], [10, 11, 2], [11, 12, 2], [12, 13, 2], [13, 14, 2],
                      [14, 15, 2], None, [69, 70, 2]]
    assert scores[0] == 0.75 and scores[12] is None


def test_batch_preserves_human_edits_and_independent_results(tmp_path):
    store = AnnotationSets(tmp_path, lambda split, ident: True)
    record = store.apply(dict(action="create", name="batch"))
    images = [dict(split="val", image_id=n) for n in range(5)]
    ident = record['id']
    def edit(action, **kw):
        return store.apply(dict(id=ident, revision=store.read(ident)['revision'], action=action, **kw))
    edit("add", images=images)
    record = edit("save", people=[], reviewed=True, **images[4])
    proposal = dict(bbox=[1, 2, 3, 4], joints=[[2, 3, 2]] + [None]*13,
                    detection_score=0.8, joint_scores=[0.7]+[None]*13)
    def run(items):
        assert len(items) == 4
        edit("person", **images[1])
        yield 0, [proposal], None
        yield 1, [proposal], None
        yield 2, [], None
        yield 3, [], "unreadable"
    batch = BatchLabels(store, lambda split, n: str(tmp_path / str(n)), run)
    with pytest.raises(ValueError):
        batch.start(ident, record['revision'], [dict(split="test", image_id=0)])
    batch.start(ident, record['revision'], images)
    batch.thread.join(5)
    assert [i['status'] for i in batch.status(ident)['items']] == [
        "saved", "skipped", "no people", "failed", "skipped"]
    rows = store.read(ident)['images']
    person = rows[0]['people'][0]
    assert person['bbox'] == proposal['bbox'] and not rows[0]['reviewed']
    person['joints'][0] = [20, 30, 1]
    edit("save", people=[person], reviewed=False, **images[0])
    assert store.read(ident)['images'][0]['autolabel']['people'][0]['joints'][0] == [2,3,2]
    assert rows[1]['people'][0]['bbox'] is None and rows[2]['people'] == []
    with pytest.raises(Conflict):
        batch.start(ident, record['revision'], images)


def test_interrupted_worker_keeps_completed_images(tmp_path):
    store = AnnotationSets(tmp_path, lambda split, ident: True)
    record = store.apply(dict(action="create", name="interrupted"))
    images = [dict(split="test", image_id=n) for n in (1, 2)]
    record = store.apply(dict(action="add", id=record["id"], revision=record["revision"], images=images))

    def broken_worker(paths):
        yield 0, [], None
        raise RuntimeError("Worker unavailable")

    batch = BatchLabels(store, lambda split, n: str(n), broken_worker)
    batch.start(record["id"], record["revision"], images)
    batch.thread.join(5)
    status = batch.status(record["id"])
    assert not status["running"]
    assert [(i["status"], i["detail"]) for i in status["items"]] == [
        ("no people", "0 detected"), ("failed", "Worker unavailable")]
    first, second = store.read(record["id"])["images"]
    assert first["autolabel"]["people"] == [] and not first["reviewed"]
    assert "autolabel" not in second and not second["reviewed"]


def test_yolo_mapping_keeps_box_pose_and_confidence_separate():
    from scripts.data.yolo26_worker import map_yolo_joints

    keypoints = [[index + 10, index + 20, 0.9] for index in range(17)]
    keypoints[6][2] = 0.1
    keypoints[7][0] = -1
    joints, scores = map_yolo_joints(keypoints, 100, 100, 0.25)
    assert joints[0] == [15, 25, 2]
    assert joints[1:3] == [None, None]
    assert joints[11] == [26, 36, 2]
    assert joints[12:] == [None, None]
    assert scores[1] == 0.1 and scores[2] == 0.9
    with pytest.raises(ValueError, match="COCO-17"):
        map_yolo_joints(keypoints[:14], 100, 100, 0.25)


def test_profiles_validate_settings_and_keep_sapiens_default(tmp_path):
    from scripts.data.crowdpose_autolabel import AutolabelProfiles

    profiles = AutolabelProfiles(tmp_path / "sapiens", tmp_path / "python", tmp_path)
    _, source = profiles.resolve(None, {})
    assert source["profile"] == "sapiens2" and source["detector_threshold"] == 0.3
    _, source = profiles.resolve("yolo26-pose", {"confidence": 0.4, "joint_confidence": 0.5})
    assert source["confidence"] == 0.4 and source["joint_confidence"] == 0.5
    with pytest.raises(ValueError, match="profile"):
        profiles.resolve("/unknown", {})
    for invalid in (True, -0.1, 10, float("nan"), "0.2"):
        with pytest.raises(ValueError, match="confidence"):
            profiles.resolve("yolo26-boxes", {"confidence": invalid})


def test_local_profile_config_resolves_paths_without_exposing_them(tmp_path):
    import json

    from scripts.data.crowdpose_autolabel import AutolabelProfiles

    checkpoint = tmp_path / "custom.pt"
    checkpoint.touch()
    runtime = tmp_path / "python"
    runtime.touch()
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({
        "custom-pose": {"label": "Custom pose", "task": "pose", "checkpoint": "custom.pt", "imgsz": 800,
                        "confidence": 0.4, "joint_confidence": 0.6},
    }))
    profiles = AutolabelProfiles(runtime, runtime, tmp_path, path)
    profile = next(row for row in profiles.catalog() if row["id"] == "custom-pose")
    assert profile["available"] and profile["joint_confidence"] == 0.6
    assert "checkpoint" not in profile
    _, source = profiles.resolve("custom-pose", {"confidence": 0.3})
    assert source["model"] == "custom.pt" and source["imgsz"] == 800 and source["confidence"] == 0.3
