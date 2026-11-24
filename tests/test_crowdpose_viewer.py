import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from scripts.data import crowdpose_viewer


@pytest.fixture
def viewer(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "1.jpg").write_bytes(b"image")
    (images / "outside.jpg").symlink_to(tmp_path / "outside.jpg")
    (tmp_path / "outside.jpg").write_bytes(b"outside the image root")
    annotation = {"id": 7, "image_id": 1, "iscrowd": 1, "num_keypoints": 0,
                  "bbox": [1, 2, 3, 4], "keypoints": [11, 12, 1, 21, 22, 2] + [0] * 36}
    dataset = {
        "images": [
            {"id": 1, "file_name": "1.jpg", "width": 50, "height": 100, "crowdIndex": 0.8},
            {"id": 2, "file_name": "missing.jpg"},
            {"id": 3, "file_name": "../outside.jpg"},
            {"id": 4, "file_name": "outside.jpg"},
        ],
        "annotations": [annotation],
    }
    annotation_file = tmp_path / "train.json"
    annotation_file.write_text(json.dumps(dataset))
    empty_file = tmp_path / "val.json"
    empty_file.write_text(json.dumps({"images": [], "annotations": []}))
    notes_path = tmp_path / "notes" / "review.jsonl"
    server = crowdpose_viewer.make_server(
        0, {"train": annotation_file, "val": empty_file}, images, notes_path
    )
    with server:
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", notes_path, annotation
        finally:
            server.shutdown()
            worker.join()


def read_json(url):
    with urlopen(url, timeout=5) as response:
        return json.load(response)


def post_note(url, image_id, note):
    request = Request(
        url + "/api/notes",
        data=json.dumps({"split": "train", "image_id": image_id, "text": note}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=5) as response:
        assert response.status == 201
        return json.load(response)


def test_raw_annotations_and_image_paths(viewer):
    url, _, annotation = viewer
    assert read_json(url + "/api/live-reload") == {"version": None}
    assert read_json(url + "/api/dataset")["splits"] == {"train": 4, "val": 0}
    record = read_json(url + "/api/sample/train/0")
    assert read_json(url + "/api/sample-id/train/1")["position"] == 0
    assert record["annotations"] == [annotation]
    assert record["image"]["crowdIndex"] == 0.8
    with urlopen(url + record["image_url"], timeout=5) as response:
        assert response.read() == b"image"
    assert read_json(url + "/api/sample/train/1")["annotations"] == []
    for path in ("/api/image/train/2", "/api/image/train/3", "/api/image/train/4",
                 "/api/sample/val/0", "/api/sample/train/-1", "/api/sample/train/4"):
        with pytest.raises(HTTPError) as error:
            urlopen(url + path, timeout=5)
        assert error.value.code == 404


def test_live_reload_reports_changes_to_served_assets(tmp_path, monkeypatch):
    stylesheet = tmp_path / "viewer.css"
    stylesheet.write_text("body {}")
    monkeypatch.setattr(crowdpose_viewer, "ASSETS", tmp_path)
    monkeypatch.setattr(crowdpose_viewer, "ASSET_ROUTES", {"/viewer.css": ("viewer.css", "text/css")})
    with crowdpose_viewer.make_server(0, {}, tmp_path, tmp_path / "notes.jsonl", live_reload=True) as server:
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/api/live-reload"
            initial = read_json(url)
            assert initial["version"]
            assert read_json(url) == initial
            (tmp_path / "unserved.txt").write_text("Ignored")
            assert read_json(url) == initial
            stylesheet.write_text("body { color: white; }")
            assert read_json(url)["version"] != initial["version"]
        finally:
            server.shutdown()
            worker.join()


def test_notes_persist_and_invalid_writes_do_not_append(viewer):
    url, notes_path, _ = viewer
    first = post_note(url, 1, " Occluded wrist\nNeeds a second look. ")
    second = post_note(url, 1, "Checked the source.")
    assert first["text"] == "Occluded wrist\nNeeds a second look."
    assert first["created_at"].endswith("+00:00")
    assert read_json(url + "/api/sample/train/0")["notes"] == [first, second]
    assert read_json(url + "/api/sample/train/1")["notes"] == []
    for image_id, note, status in ((1, " ", 400), (1, "a" * 4001, 400), (9, "text", 404)):
        with pytest.raises(HTTPError) as error:
            post_note(url, image_id, note)
        assert error.value.code == status
    assert [json.loads(line) for line in notes_path.read_text().splitlines()] == [first, second]
    assert crowdpose_viewer.NoteStore(notes_path).for_image("train", 1) == [first, second]


def test_note_write_failure_does_not_create_a_saved_record(viewer):
    url, notes_path, _ = viewer
    notes_path.parent.write_text("A file occupies the notes directory path.")
    with pytest.raises(HTTPError) as error:
        post_note(url, 1, "Try saving a note.")
    assert error.value.code == 500
    assert read_json(url + "/api/sample/train/0")["notes"] == []


def test_audit_routes_attach_reviews_to_the_right_split(viewer):
    url, notes_path, _ = viewer
    request = Request(
        url + "/api/audit",
        data=json.dumps(dict(split="train", image_id=1, action="tag", kind="clean",
                             person=None, joint=None)).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=5) as response:
        review = json.load(response)
    assert review["reviewed"]
    assert read_json(url + "/api/sample/train/0")["audit"] == review
    index = read_json(url + "/api/audit/train")
    assert index["reviews"] == {"1": review}
    assert index["rows"][0]["people"] == 0  # The fixture annotation is a crowd record.
    assert read_json(url + "/api/audit/val") == {"rows": [], "reviews": {}}
    assert not notes_path.exists()


def test_wiki_is_served_and_linked_from_picker(viewer):
    url, *_ = viewer
    with urlopen(url + "/wiki", timeout=5) as response:
        assert response.headers.get_content_type() == "text/html"
        wiki = response.read().decode()
    assert 'id="standards"' in wiki
    assert 'id="image-picker"' in wiki
    with urlopen(url + "/", timeout=5) as response:
        assert 'href="/wiki#image-picker"' in response.read().decode()
