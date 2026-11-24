const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const {join} = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");

function load(name, globals = {}) {
  const source = join(__dirname, "..", "scripts", "data", "crowdpose_viewer_web", name);
  return vm.createContext({element: () => ({}), ...globals},
    {name: source});
}

function read(name, context) {
  vm.runInContext(readFileSync(join(__dirname, "..", "scripts", "data", "crowdpose_viewer_web", name), "utf8"), context);
}

test("position ranges are one-based, inclusive, ordered, and deduplicated", () => {
  const context = load("audit.js");
  read("audit.js", context);
  const parse = vm.runInContext("parseAuditPositions", context);
  assert.deepEqual(Array.from(parse("3 – 5, 1, 4-6", 6)), [2, 3, 4, 0, 5]);
  for (const invalid of ["", "0", "6-7", "5-3", "2,,3", "3.5", "train/4", "4-"]) {
    assert.throws(() => parse(invalid, 6));
  }
});

test("geometry hit-testing identifies joints and bounding boxes", () => {
  const context = load("layers.js");
  read("layers.js", context);
  const hit = vm.runInContext("hitAnnotationGeometry", context);
  const people = [
    {key: "source:1", bbox: [0, 0, 50, 50], joints: [], showBox: true, showPose: false},
    {key: "source:2", bbox: [5, 5, 20, 20], joints: [], showBox: true, showPose: false},
    {key: "custom:3", id: "person-3", bbox: null, joints: [null, {x: 15, y: 15}], showBox: false, showPose: true},
  ];
  const joint = hit(people, 15, 15, 4);
  assert.equal(joint.id, "person-3");
  assert.equal(joint.joint, 1);
  assert.equal(hit(people, 8, 8, 4).key, "source:2");
  assert.equal(hit(people, 30, 30, 4).key, "source:1");
  assert.equal(hit(people, 8, 8, 4).joint, null);
  people[1].showBox = people[2].showPose = false;
  assert.equal(hit(people, 15, 15, 4).key, "source:1");
  assert.equal(hit(people, 80, 80, 4), null);
});

test("picker commands skip existing selections, follow filters and sort, and handle shortfalls", () => {
  const context = load("audit.js");
  read("audit.js", context);
  const pick = vm.runInContext("selectAuditRows", context);
  const source = ["a", "b", "c", "d", "e", "f"];
  const matching = ["f", "d", "e", "c", "b"];
  const available = row => row !== "d" && row !== "b";
  const select = command => pick(command, source, matching, available, () => 0.99);

  assert.deepEqual(Array.from(select("first 2").rows), ["f", "e"]);
  assert.deepEqual(Array.from(select("LAST   2").rows), ["e", "c"]);
  assert.deepEqual(Array.from(select("all matching").rows), ["f", "e", "c"]);
  assert.deepEqual(Array.from(select("1–4").rows), ["a", "c"]);
  assert.deepEqual(Array.from(select("2 random").rows), ["c", "f"]);
  assert.deepEqual(Array.from(select("Random 2").rows), ["c", "f"]);
  const short = select("64 random");
  assert.equal(short.requested, 64);
  assert.deepEqual(Array.from(short.rows), ["c", "f", "e"]);
  assert.equal(pick("64 random", source, [], available).rows.length, 0);
  assert.deepEqual(matching, ["f", "d", "e", "c", "b"]);
  for (const invalid of ["0 random", "random -4", "first 2.5", "last 9007199254740992", "sixty random"]) {
    assert.throws(() => select(invalid), /command|positive whole number|positions/i);
  }
});

test("placement wraps to unfinished slots and preserves explicitly unlabeled joints", () => {
  const context = load("editor.js");
  read("editor.js", context);
  const next = vm.runInContext("nextUnplacedJoint", context);
  const joints = [null, [0, 0, 0], [4, 5, 2], null];
  assert.equal(next(joints, 0), 3);
  assert.equal(next(joints, 3), 0);
  assert.equal(next(joints.slice(1, 3), 1), -1);
});

test("opposite corners yield the same positive box in either order", () => {
  const context = load("editor.js");
  read("editor.js", context);
  const box = vm.runInContext("boxFromCorners", context);
  assert.deepEqual(Array.from(box([75, 99], [10, 20])), [10, 20, 65, 79]);
  assert.deepEqual(Array.from(box([10, 20], [75, 99])), [10, 20, 65, 79]);
  assert.equal(box([10, 20], [10, 30]), null);
});
