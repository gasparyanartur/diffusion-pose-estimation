"""Serve the local CrowdPose annotations and image review notes."""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from functools import cached_property
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import yaml

from scripts.data.crowdpose_audit import (
    COHORTS, OVERLAP_IOU, SMALL_HEIGHT, TAGS, AuditStore, image_stats, review_update,
)
from scripts.data.crowdpose_autolabel import AutolabelProfiles, BatchLabels, predict_batch
from scripts.data.manual_annotations import AnnotationSets, Conflict

from src.data.skeleton import CROWDPOSE_EDGE_LIST, CROWDPOSE_JOINT_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).with_name("crowdpose_viewer_web")
ASSET_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/wiki": ("wiki.html", "text/html; charset=utf-8"),
    "/viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
    "/autolabel.js": ("autolabel.js", "text/javascript; charset=utf-8"),
    "/copy_annotations.js": ("copy_annotations.js", "text/javascript; charset=utf-8"),
    "/layers.js": ("layers.js", "text/javascript; charset=utf-8"),
    "/audit.js": ("audit.js", "text/javascript; charset=utf-8"),
    "/editor.js": ("editor.js", "text/javascript; charset=utf-8"),
    "/viewer.css": ("viewer.css", "text/css; charset=utf-8"),
}
IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
NOTE_LIMIT = 4000


class CrowdPoseSplit:
    def __init__(self, annotation_file: Path):
        with annotation_file.open(encoding="utf-8") as source:
            dataset = json.load(source)
        self.images = dataset["images"]
        self.positions = {image["id"]: position for position, image in enumerate(self.images)}
        self.annotations = defaultdict(list)
        for annotation in dataset["annotations"]:
            self.annotations[annotation["image_id"]].append(annotation)

    @cached_property
    def audit_stats(self):
        return [image_stats(image, self.annotations.get(image["id"], ())) for image in self.images]


class NoteStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = Lock()
        self.records = defaultdict(list)
        if path.exists():
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        self.records[record["split"], record["image_id"]].append(record)
                    except (ValueError, TypeError, KeyError) as error:
                        raise ValueError(f"Invalid note in {path}, line {line_number}") from error

    def for_image(self, split: str, image_id: int) -> list:
        with self.lock:
            return list(self.records.get((split, image_id), ()))

    def append(self, split: str, image_id: int, text: str) -> dict:
        record = {
            "split": split,
            "image_id": image_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "text": text,
        }
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as destination:
                destination.write(json.dumps(record) + "\n")
            self.records[split, image_id].append(record)
        return record


def make_server(port: int, annotation_paths: dict[str, Path], images: Path,
                notes_path: Path, sets_dir: Path | None = None,
                autolabel_python: Path | None = None, *, yolo_python: Path | None = None,
                yolo_weights: Path | None = None, autolabel_profiles: Path | None = None,
                live_reload: bool = False) -> ThreadingHTTPServer:
    splits = {name: CrowdPoseSplit(path) for name, path in annotation_paths.items()}
    image_root = images.resolve()
    notes = NoteStore(notes_path)
    audit = AuditStore(notes_path.parent / "audit")
    sets = AnnotationSets(sets_dir or notes_path.parent / "annotation_sets",
                          lambda split, ident: split in splits and ident in splits[split].positions,
                          lambda split, ident: splits[split].annotations.get(ident, ()))

    def image_path(image):
        path = (image_root / image.get("file_name", "")).resolve()
        if not path.is_relative_to(image_root) or path.suffix.lower() not in IMAGE_TYPES or not path.is_file():
            raise ValueError("Image file unavailable.")
        return path

    def queued_path(split, ident):
        dataset = splits[split]
        return image_path(dataset.images[dataset.positions[ident]])

    python = autolabel_python or PROJECT_ROOT / "data/crowdpose_viewer/sapiens-env/bin/python"
    weights = yolo_weights or PROJECT_ROOT / "data/crowdpose_viewer/yolo-weights"
    profiles = AutolabelProfiles(python, yolo_python or Path(sys.executable), weights, autolabel_profiles)
    batches = BatchLabels(sets, queued_path, lambda paths: predict_batch(python.absolute(), paths))

    class ViewerHandler(BaseHTTPRequestHandler):
        def log_request(self, code="-", size="-"):
            if self.path != "/api/live-reload" or code != 200:
                super().log_request(code, size)

        def send_content(self, status, content_type, length):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def send_json(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_content(status, "application/json", len(body))
            self.wfile.write(body)

        def send_file(self, path, content_type):
            try:
                source = path.open("rb")
            except OSError:
                self.send_json(404, {"error": "Image or file unavailable."})
                return
            with source:
                self.send_content(200, content_type, source.seek(0, 2))
                source.seek(0)
                shutil.copyfileobj(source, self.wfile)

        def send_sample(self, route, split, position):
            dataset = splits.get(split)
            if dataset is None or not 0 <= position < len(dataset.images):
                self.send_json(404, {"error": "Image is outside this split."})
                return
            image = dataset.images[position]
            if route == "image":
                self.send_image(image)
                return
            self.send_json(200, {
                "image": image,
                "position": position,
                "image_url": f"/api/image/{split}/{position}",
                "annotations": dataset.annotations.get(image["id"], []),
                "notes": notes.for_image(split, image["id"]),
                "audit": audit.read(split, image["id"]),
            })

        def send_image(self, image):
            try:
                path = image_path(image)
            except (ValueError, OSError):
                self.send_json(404, {"error": "Image file unavailable."})
                return
            self.send_file(path, IMAGE_TYPES[path.suffix.lower()])

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ASSET_ROUTES:
                name, content_type = ASSET_ROUTES[path]
                self.send_file(ASSETS / name, content_type)
            elif path == "/api/live-reload":
                try:
                    version = None
                    if live_reload:
                        stats = [(ASSETS / name).stat() for name, _ in ASSET_ROUTES.values()]
                        version = ":".join(f"{stat.st_mtime_ns}-{stat.st_size}" for stat in stats)
                    self.send_json(200, {"version": version})
                except OSError:
                    self.send_json(503, {"error": "Viewer assets are being updated."})
            elif path.startswith("/api/audit/"):
                split = path.removeprefix("/api/audit/")
                if split not in splits:
                    self.send_json(404, {"error": "Unknown split."})
                    return
                self.send_json(200, {"rows": splits[split].audit_stats, "reviews": audit.for_split(split)})
            elif path == "/api/autolabel/profiles":
                self.send_json(200, profiles.catalog())
            elif path.startswith("/api/autolabel/"):
                self.send_json(200, batches.status(path.rsplit("/", 1)[1]))
            elif path == "/api/sets" or path.startswith("/api/sets/"):
                try:
                    data = sets.catalog() if path == "/api/sets" else sets.read(path[10:])
                    self.send_json(200, data)
                except (ValueError, OSError):
                    self.send_json(404, {"error": "Annotation set unavailable."})
            elif path == "/api/dataset":
                self.send_json(200, {
                    "splits": {name: len(dataset.images) for name, dataset in splits.items()},
                    "joints": list(CROWDPOSE_JOINT_NAMES),
                    "edges": CROWDPOSE_EDGE_LIST,
                    "audit": {"tags": TAGS, "cohorts": COHORTS, "overlap_iou": OVERLAP_IOU,
                              "small_height": SMALL_HEIGHT},
                })
            else:
                parts = path.split("/")
                if (len(parts) == 6 and parts[1:3] == ["api", "copy-source"]
                        and parts[5].isascii() and parts[5].isdigit() and len(parts[5]) <= 10):
                    try:
                        self.send_json(200, sets.copy_source(parts[3], parts[4], int(parts[5])))
                    except (OSError, ValueError):
                        self.send_json(404, {"error": "Source annotations unavailable."})
                elif (len(parts) == 5 and parts[1:3] == ["api", "sample-id"]
                        and parts[3] in splits and parts[4].isascii()
                        and parts[4].isdigit() and len(parts[4]) <= 10):
                    split = parts[3]
                    self.send_sample("sample", split, splits[split].positions.get(int(parts[4]), -1))
                elif (len(parts) == 5 and parts[1] == "api" and parts[2] in {"sample", "image"}
                        and len(parts[4]) <= 10 and parts[4].isascii() and parts[4].isdigit()):
                    self.send_sample(parts[2], parts[3], int(parts[4]))
                else:
                    self.send_json(404, {"error": "Page not found."})

        def do_POST(self):
            if self.path not in {"/api/notes", "/api/sets", "/api/autolabel", "/api/audit"}:
                self.send_json(404, {"error": "Page not found."})
                return
            host = self.headers.get("Host")
            origin = self.headers.get("Origin")
            port = self.server.server_port
            if host not in {f"127.0.0.1:{port}", f"localhost:{port}"} or (
                origin is not None and origin != f"http://{host}"
            ):
                self.send_json(403, {"error": "Writes require the local viewer."})
                return
            if self.headers.get_content_type() != "application/json":
                self.send_json(415, {"error": "A JSON request is required."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= (2_000_000 if self.path == "/api/sets" else 64_000):
                    raise ValueError
                payload = json.loads(self.rfile.read(length))
                if self.path in {"/api/sets", "/api/autolabel", "/api/audit"}:
                    try:
                        if self.path == "/api/autolabel":
                            profile = profiles.resolve(payload.get("profile"), payload.get("settings", {}))
                            result = batches.start(payload["id"], payload["revision"], payload["images"], profile)
                        elif self.path == "/api/audit":
                            split, ident = payload["split"], payload["image_id"]
                            if (split not in splits or type(ident) is not int
                                    or ident not in splits[split].positions):
                                raise ValueError("Unknown image.")
                            source = splits[split].annotations.get(ident, ())
                            result = audit.apply(split, ident, lambda record: review_update(record, payload, source))
                        else:
                            result = sets.apply(payload)
                    except Conflict as error:
                        self.send_json(409, {"error": str(error)})
                    except (KeyError, ValueError, TypeError, OverflowError) as error:
                        self.send_json(400, {"error": str(error)})
                    except OSError:
                        self.send_json(500, {"error": "Could not access the local record."})
                    else:
                        self.send_json(200, result)
                    return
                split, image_id, text = payload["split"], payload["image_id"], payload["text"]
                if not isinstance(split, str) or type(image_id) is not int:
                    raise ValueError
                if not isinstance(text, str) or not 1 <= len(text.strip()) <= NOTE_LIMIT:
                    raise ValueError
            except (ValueError, TypeError, KeyError):
                self.send_json(400, {"error": f"Enter a note of 1–{NOTE_LIMIT} characters."})
                return
            if split not in splits or image_id not in splits[split].positions:
                self.send_json(404, {"error": "Unknown image."})
                return
            try:
                record = notes.append(split, image_id, text.strip())
            except OSError as error:
                self.log_error("Could not save note: %s", error)
                self.send_json(500, {"error": "Could not write the notes file."})
                return
            self.send_json(201, record)

    return ThreadingHTTPServer(("127.0.0.1", port), ViewerHandler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "conf/data/crowdpose.yaml")
    parser.add_argument("--notes", type=Path, default=PROJECT_ROOT / "data/crowdpose_viewer/notes.jsonl")
    parser.add_argument("--sets-dir", type=Path, default=PROJECT_ROOT / "data/crowdpose_viewer/annotation_sets")
    parser.add_argument("--autolabel-python", type=Path, help="Python executable for the Sapiens2 worker")
    parser.add_argument("--yolo-python", type=Path, help="Python executable for the YOLO26 worker")
    parser.add_argument("--yolo-weights", type=Path, help="Local directory containing the official YOLO26 weights")
    parser.add_argument("--autolabel-profiles", type=Path, help="Optional JSON file of named local YOLO profiles")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true", help="Reload the browser when viewer web files change")
    args = parser.parse_args()
    try:
        with args.config.open(encoding="utf-8") as source:
            config = yaml.safe_load(source)
        data_root = PROJECT_ROOT / config["root"]
        annotations = {split: data_root / config[f"{split}_ann"] for split in ("train", "val", "test")}
        server = make_server(args.port, annotations, data_root / config["img_dir"], args.notes, args.sets_dir, args.autolabel_python,
                             yolo_python=args.yolo_python, yolo_weights=args.yolo_weights,
                             autolabel_profiles=args.autolabel_profiles, live_reload=args.reload)
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
        parser.exit(1, f"Could not start CrowdPose viewer: {error}\n")
    with server:
        print(f"Local: http://127.0.0.1:{server.server_port}/", flush=True)
        print(f"Notes: {args.notes.resolve()}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
