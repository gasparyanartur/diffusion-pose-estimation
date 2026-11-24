"use strict";

function nextUnplacedJoint(joints, current) {
  for (let step = 1; step <= joints.length; step++) {
    const index = (current + step) % joints.length;
    if (joints[index] === null) return index;
  }
  return -1;
}

function boxFromCorners([x1, y1], [x2, y2]) {
  const width = Math.abs(x2 - x1);
  const height = Math.abs(y2 - y1);
  return width && height ? [Math.min(x1, x2), Math.min(y1, y2), width, height] : null;
}

const editor = (() => {
  const ui = element;
  let active = null;
  let pending = false;
  let personId = "";
  let mode = "select";
  let targetType = "joint";
  let jointIndex = 0;
  let draft = null;
  let firstCorner = null;
  let focusOrigin = null;
  const focusNudges = [];
  let nudging = false;
  let inspectorOpen = false;
  let boxAction = "position";
  const fieldIds = ["coord-a", "coord-b", "coord-width", "coord-height"];
  let visibilityChoice = null;
  const jointButtons = [];
  const visibilityNames = ["Unlabeled", "Occluded", "Visible"];
  const jointList = new Intl.ListFormat("en", {type: "disjunction"});
  const toolButtons = {"select-tool": "select", "set-joints": "joint", "set-box": "box",
    "edit-annotations": "continuous", "focus-mode": "focus"};
  const nudgeKeys = new Map([
    ["w", [0, -1]], ["arrowup", [0, -1]], ["a", [-1, 0]], ["arrowleft", [-1, 0]],
    ["s", [0, 1]], ["arrowdown", [0, 1]], ["d", [1, 0]], ["arrowright", [1, 0]],
  ]);
  const inQueue = () => Boolean(active && !ui("browse-pool").checked);
  const entry = () => inQueue() ? active.images[position] : null;
  const person = () => entry()?.people.find(p => p.id === personId);
  const total = () => inQueue() ? active.images.length : dataset?.splits[splitSelect.value] ?? 0;
  const message = (text) => { ui("edit-status").textContent = text; };
  const dirty = () => Boolean(draft || firstCorner || visibilityChoice !== null);
  const placing = () => mode === "joint" || mode === "box" || inspectorOpen;
  const visibility = () => visibilityChoice ?? person()?.joints[jointIndex]?.[2] ?? 2;
  const targetValue = () => targetType === "box" ? person()?.bbox : person()?.joints[jointIndex];
  const focusMode = () => inQueue() && mode === "focus";
  const jointName = index => dataset.joints[index].replaceAll("_", " ").replace(/^./, char => char.toUpperCase());
  const personIndex = () => entry()?.people.findIndex(p => p.id === personId) ?? -1;
  function canNavigate() {
    if (pending || saving) return false;
    return !dirty() || window.confirm("Discard the unconfirmed placement?");
  }

  async function write(action, args) {
    if (pending || saving) return false;
    pending = true;
    document.body.inert = true;
    message("Saving…");
    try {
      active = await fetchJSON("/api/sets", {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({id: active?.id, revision: active?.revision, action, ...args}),
      });
      message("Saved locally");
      return true;
    } catch (error) {
      message(error.message);
      return false;
    } finally {
      pending = false;
      document.body.inert = false;
    }
  }

  function imageKey() {
    return {split: entry().split, image_id: entry().image_id};
  }

  async function save(people, reviewed = false) {
    return write("save", {...imageKey(), people, reviewed});
  }

  async function catalog() {
    const options = (await fetchJSON("/api/sets")).map(s => new Option(s.name, s.id));
    ui("set-select").replaceChildren(new Option("Original CrowdPose", ""), ...options);
    ui("set-select").value = active?.id ?? "";
  }

  function clearDraft() {
    draft = null;
    firstCorner = null;
    focusOrigin = null;
    inspectorOpen = false;
    boxAction = "position";
    visibilityChoice = null;
    frame.style.transform = "";
  }

  function resetImage() {
    clearDraft();
    mode = "select";
    targetType = "joint";
    jointIndex = 0;
    personId = "";
    message("");
    render();
  }

  function render() {
    const custom = Boolean(active);
    const editing = Boolean(entry() && sample && sample.image.id === entry().image_id);
    ui("image-heading").textContent = custom ? "Source image details" : "Image details";
    ui("autolabel-open").disabled = !active?.images.length;
    ui("editor-pane").hidden = !custom;
    ui("queue-tools").hidden = !custom;
    ui("image-editor").hidden = !editing;
    ui("add-current").hidden = !custom || inQueue();
    ui("add-current").disabled = !sample;
    ui("queue-progress").textContent = custom ?
      active.images.filter(i => i.reviewed).length + " / " + active.images.length + " reviewed" : "";
    stage.classList.toggle("editing", editing && mode !== "select");
    if (!editing) return;
    const item = entry();
    if (!person()) personId = item.people[0]?.id ?? "";
    if (!person()) mode = "select";
    const index = personIndex();
    const choices = () => item.people.map((p, i) => new Option("Person " + (i + 1), p.id));
    ui("person").replaceChildren(...choices());
    ui("person").value = personId;
    ui("person").disabled = index < 0;
    ui("delete-person").disabled = index < 0;
    ui("previous-person").disabled = index <= 0;
    ui("next-person").disabled = index < 0 || index + 1 >= item.people.length;
    const owner = ui("owner").value;
    ui("owner").replaceChildren(...choices().filter(opt => opt.value !== personId));
    if ([...ui("owner").options].some(option => option.value === owner)) ui("owner").value = owner;
    const selected = targetValue();
    const box = targetType === "box";
    const select = mode === "select";
    const focusing = focusMode();
    if (focusing && !geometryFields(selected).length) focusOrigin = null;
    const stored = person()?.joints[jointIndex];
    const undoCount = active.undo?.length ?? 0;
    ui("undo-edit").disabled = undoCount === 0;
    ui("undo-edit").textContent = undoCount ? "Undo (" + undoCount + ")" : "Undo";
    ui("geometry").disabled = !person();
    ui("review-image").textContent = item.reviewed ? "Reopen image" : "Mark reviewed";
    ui("review-image").setAttribute("aria-pressed", item.reviewed);
    for (const [id, value] of Object.entries(toolButtons)) ui(id).setAttribute("aria-pressed", mode === value);

    const jointState = visibilityChoice !== null || draft ? visibilityNames[visibility()] :
      stored ? visibilityNames[stored[2]] : "Unplaced";
    ui("active-target").textContent = index < 0 ? "No person selected" :
      focusing && !focusOrigin ? "Choose a joint or box to focus" : "Person " + (index + 1) + " · " +
      (box ? "Bounding box" + (selected ? "" : " · Unplaced") : jointName(jointIndex) + " · " + jointState);
    ui("joint-row").hidden = focusing || (box && !select);
    ui("select-box").hidden = !select;
    ui("select-box").setAttribute("aria-pressed", box && select);
    ui("previous-joint").disabled = !box && adjacentJoint(-1) < 0;
    ui("next-joint").disabled = !box && adjacentJoint(1) < 0;
    for (const [i, button] of jointButtons.entries()) {
      const slot = person()?.joints[i];
      const state = slot ? visibilityNames[slot[2]] : "Unplaced";
      button.setAttribute("aria-pressed", !box && i === jointIndex);
      button.setAttribute("aria-label", jointName(i) + " · " + state);
      button.title = state;
      button.dataset.state = slot?.[2] ? "placed" : "empty";
      button.disabled = mode === "joint" && slot !== null;
    }
    ui("geometry-actions").hidden = !select;
    ui("edit-geometry").disabled = !selected;
    ui("delete-geometry").disabled = !selected || (focusing && !focusOrigin);
    ui("move-row").hidden = !select;
    ui("move-geometry").disabled = !selected || !ui("owner").options.length;
    ui("placement-controls").hidden = !placing();
    ui("visibility-row").hidden = box;
    ui("box-dimensions").hidden = !box;
    ui("box-edit-actions").hidden = !box || !inspectorOpen;
    ui("box-position").setAttribute("aria-pressed", boxAction === "position");
    ui("box-size").setAttribute("aria-pressed", boxAction === "size");
    for (const button of ui("visibility-row").querySelectorAll("button")) {
      button.setAttribute("aria-pressed", Number(button.dataset.visibility) === visibility());
    }

    const original = item.autolabel?.people.find(p => p.id === personId);
    const score = box ? original?.detection_score : original?.joint_scores?.[jointIndex];
    ui("proposal-score").hidden = !Number.isFinite(score);
    ui("proposal-score").textContent = Number.isFinite(score) ?
      (box ? "Original box score " : "Original joint score ") + score.toFixed(3) +
        " · " + (item.autolabel.label ?? "Sapiens2") : "";
    const pair = draft ?? geometryFields(selected);
    ui("label-a").textContent = box ? "Center X" : "X";
    ui("label-b").textContent = box ? "Center Y" : "Y";
    for (const [index, id] of fieldIds.entries()) {
      ui(id).value = Number.isFinite(pair[index]) ? pair[index] : "";
    }
    ui("confirm-edit").textContent = inspectorOpen ? "Apply" : "Confirm";
    ui("confirm-edit").hidden = !box;
    ui("cancel-edit").textContent = mode === "continuous" && !box ? "Done" : "Cancel";
    ui("placement-zoom").closest("label").hidden = focusing;
    ui("edit-hint").textContent = !person() ? "" : editHint();
    const connection = !box && (stored?.[2] || anchor()) ? connectionHint(jointIndex) : "";
    ui("joint-connection-hint").textContent = connection;
    ui("joint-connection-hint").hidden = !connection;
    drawOverlays();
  }

  function geometryFields(shape) {
    if (!shape) return [];
    if (targetType === "joint") return shape[2] ? shape.slice(0, 2) : [];
    const [x, y, width, height] = shape;
    return [x + width / 2, y + height / 2, width, height];
  }

  function editHint() {
    if (mode === "select") return "Select a joint or box on the image or in the sidebar.";
    if (mode === "focus") return focusOrigin ?
      "WASD or arrow keys move one image pixel. Esc exits Focus mode." :
      "Click an existing joint or box to zoom in.";
    if (inspectorOpen) {
      if (targetType === "joint") return "Click to move and save the joint. Use Enter to save typed coordinates.";
      return boxAction === "position" ? "Click the new box center. Apply or press Enter to save." :
        "Click a corner to set the size. Apply or press Enter to save.";
    }
    if (mode === "continuous") return "Click a joint or box to edit it.";
    if (mode === "box") return firstCorner ? "Click the opposite corner." : "Click the first corner.";
    return "Click to save " + jointName(jointIndex).toLowerCase() + ".";
  }

  function connectionHint(index) {
    const selected = person();
    if (!selected) return "";
    const neighbors = dataset.edges.flatMap(([start, end]) =>
      start === index ? [end] : end === index ? [start] : []);
    if (!neighbors.length || neighbors.some(joint => selected.joints[joint]?.[2])) return "";
    return "Place " + jointList.format(neighbors.map(jointName)) + " to connect this joint.";
  }

  function jointPreview(id) {
    if (id !== personId || targetType !== "joint") return null;
    const point = anchor();
    if (!point) return null;
    return {index: jointIndex, joint: {x: point[0], y: point[1], visibility: visibility()}};
  }

  function pixelPoint(event) {
    const {x, y} = pointerToSource(event);
    return [Math.round(x), Math.round(y)];
  }

  function anchor() {
    if (!draft?.slice(0, 2).every(Number.isFinite)) return null;
    if (targetType === "joint" && visibility() === 0) return null;
    return draft.slice(0, 2);
  }

  function drawSelectedJoint([x, y], scale) {
    context.save();
    context.beginPath();
    context.arc(x, y, 7 / scale, 0, Math.PI * 2);
    context.strokeStyle = "#101519";
    context.lineWidth = 5 / scale;
    context.stroke();
    context.strokeStyle = "#f4ff82";
    context.lineWidth = 2 / scale;
    context.stroke();
    context.restore();
  }

  function draw(scale) {
    if (!inQueue() || !entry() || !sample) return;
    frame.style.transform = "";
    if (focusOrigin) {
      zoomAt(focusOrigin, scale);
      const [x, y] = geometryFields(targetValue());
      if (Number.isFinite(x) && layers.isVisible(personId, targetType)) {
        drawReticle(context, x, y, scale * 3);
      }
      return;
    }
    const point = anchor();
    const zoomPoint = mode === "edit" && ui("placement-zoom").checked ?
      point ?? (targetType === "joint" && targetValue()?.[2] ? targetValue() : null) : null;
    if (zoomPoint) zoomAt(zoomPoint, scale);
    const screenScale = zoomPoint ? scale * 3 : scale;
    if (point) {
      if (targetType === "box") {
        const [x, y, width, height] = draft;
        drawBox({bbox:[x - width / 2, y - height / 2, width, height]}, "#f4ff82", screenScale);
        if (firstCorner && !(width > 0 && height > 0)) {
          drawReticle(context, firstCorner[0], firstCorner[1], screenScale);
        }
      }
      drawReticle(context, point[0], point[1], screenScale);
    } else if (firstCorner) {
      drawReticle(context, firstCorner[0], firstCorner[1], scale);
    } else if (targetType === "joint" && person()?.joints[jointIndex]?.[2] &&
               layers.isVisible(personId, "joint")) {
      drawSelectedJoint(person().joints[jointIndex], screenScale);
    }
  }

  function zoomAt([x, y], scale) {
    const dx = (sample.image.width / 2 - x) * scale * 3;
    const dy = (sample.image.height / 2 - y) * scale * 3;
    frame.style.transform = `translate(${dx}px,${dy}px) scale(3)`;
  }

  function setDraft(fields, [x, y] = fields) {
    draft = fields;
    setMagnifierPoint(x, y);
    render();
  }

  function validFields(fields) {
    if (!fields.every(Number.isFinite)) return false;
    if (targetType === "box") return fields.length === 4 && fields[2] > 0 && fields[3] > 0;
    return fields.length === 2;
  }

  function values() {
    return fieldIds.slice(0, targetType === "box" ? 4 : 2).map(id => ui(id).valueAsNumber);
  }

  function withTarget(value) {
    const people = structuredClone(entry().people);
    const target = people.find(p => p.id === personId);
    if (targetType === "joint") target.joints[jointIndex] = value;
    else target.bbox = value;
    return people;
  }

  async function confirmEdit() {
    if (!person() || !placing() || pending || saving) return;
    const flag = visibility();
    const unset = targetType === "joint" && flag === 0;
    const fields = unset ? [0, 0] : values();
    if (!unset && !validFields(fields)) {
      message(targetType === "joint" ? "Enter valid X and Y coordinates." :
        "Enter valid coordinates and positive box dimensions.");
      return;
    }
    const [x, y, width, height] = fields;
    const shape = targetType === "joint" ? [x, y, flag] : [x - width / 2, y - height / 2, width, height];
    const current = targetValue();
    if (!shape.every((value, index) => value === current?.[index]) && !await save(withTarget(shape))) return;
    const keepJoint = mode === "continuous" && targetType === "joint";
    clearDraft();
    inspectorOpen = keepJoint;
    const hint = targetType === "joint" && flag > 0 ? connectionHint(jointIndex) : "";
    if (hint) message(jointName(jointIndex) + " saved. " + hint);
    if (mode === "edit") mode = "select";
    else if (mode === "joint") {
      const next = nextUnplacedJoint(person().joints, jointIndex);
      if (next < 0) {
        mode = "select";
        message(hint ? "All joints placed. " + hint : "All joints placed.");
      } else jointIndex = next;
    }
    render();
  }

  function changeTool(next) {
    if (mode === next || !canNavigate()) return;
    clearDraft();
    mode = next;
    if (next === "box") targetType = "box";
    if (next === "edit") inspectorOpen = true;
    if (next === "joint") {
      targetType = "joint";
      if (person().joints[jointIndex] !== null) {
        const nextJoint = nextUnplacedJoint(person().joints, jointIndex);
        if (nextJoint < 0) {
          mode = "select";
          message("All joints are placed. Use Edit to change one.");
        } else jointIndex = nextJoint;
      }
    }
    render();
  }

  function choosePerson(id) {
    if (personId !== id && entry()?.people.some(p => p.id === id) && canNavigate()) {
      clearDraft();
      personId = id;
      jointIndex = 0;
      if (mode === "edit") mode = "select";
      if (mode === "joint") {
        const next = nextUnplacedJoint(person().joints, -1);
        if (next < 0) mode = "select";
        else jointIndex = next;
      }
    }
    render();
  }

  function adjacentJoint(direction) {
    for (let index = jointIndex + direction; index >= 0 && index < jointButtons.length; index += direction) {
      if (mode !== "joint" || person()?.joints[index] === null) return index;
    }
    return -1;
  }

  function chooseJoint(index) {
    if (index >= 0 && index < jointButtons.length &&
        (mode !== "joint" || person()?.joints[index] === null) && canNavigate()) {
      clearDraft();
      if (mode === "edit") mode = "select";
      targetType = "joint";
      jointIndex = index;
      if (mode === "continuous" && targetValue()) inspectorOpen = true;
      render();
    }
  }

  function cancelPlacement() {
    clearDraft();
    if (mode === "edit" || mode === "continuous" || mode === "focus") mode = "select";
    render();
  }

  async function nudgeFocus(delta) {
    focusNudges.push(delta);
    if (nudging) return;
    nudging = true;
    while (focusNudges.length) {
      const [dx, dy] = focusNudges.shift();
      const [x, y, ...attributes] = targetValue();
      if (!await save(withTarget([x + dx, y + dy, ...attributes]))) {
        focusNudges.length = 0;
        break;
      }
      const [nextX, nextY] = geometryFields(targetValue());
      setMagnifierPoint(nextX, nextY);
      render();
    }
    nudging = false;
  }

  function keydown(event) {
    if (document.querySelector("dialog[open]") || !inQueue() || !person() ||
        event.ctrlKey || event.metaKey || event.altKey) return;
    const editingText = event.target.closest("input,textarea,select,[contenteditable]");
    const delta = focusMode() && !editingText && nudgeKeys.get(event.key.toLowerCase());
    if (delta) {
      event.preventDefault();
      if (focusOrigin && !saving && (!pending || nudging)) nudgeFocus(delta);
      return;
    }
    if (pending || saving) return;
    if (event.key === "Escape" && mode !== "select") {
      event.preventDefault();
      cancelPlacement();
      return;
    }
    if (event.key === "Delete") {
      if (!event.repeat && !editingText && targetValue() && (!focusMode() || focusOrigin)) {
        event.preventDefault();
        deleteGeometry();
      }
      return;
    }
    if (!placing()) return;
    const field = fieldIds.includes(event.target.id);
    if (event.key === "Enter" && !event.repeat && (inspectorOpen || dirty() || field) &&
        (!editingText || field) && !event.target.closest("button")) {
      event.preventDefault();
      confirmEdit();
      return;
    }
  }

  async function onCanvasClick(event) {
    if (!person() || !sample || frame.hidden || pending || saving || !frame.contains(event.target)) return;
    stage.focus({preventScroll:true});
    if (mode === "select" || mode === "focus" || (mode === "continuous" && !inspectorOpen)) {
      const hit = layers.hitEditable(event);
      if (!hit) return;
      personId = hit.id;
      targetType = hit.joint === null ? "box" : "joint";
      if (hit.joint !== null) jointIndex = hit.joint;
      if (mode === "continuous") inspectorOpen = true;
      if (mode === "focus") {
        focusOrigin = geometryFields(targetValue()).slice(0, 2);
        setMagnifierPoint(...focusOrigin);
      }
      render();
      return;
    }
    const [x, y] = pixelPoint(event);
    if (mode === "box") {
      if (!firstCorner) {
        firstCorner = [x, y];
        draft = null;
        setMagnifierPoint(x, y);
        render();
        return;
      }
      const box = boxFromCorners(firstCorner, [x, y]);
      if (!box) {
        message("Choose a corner with positive width and height.");
        return;
      }
      setDraft(geometryFields(box));
      await confirmEdit();
      return;
    }
    if (targetType === "box") {
      const [cx, cy, width, height] = values();
      setDraft(boxAction === "position" ? [x, y, width, height] :
        [cx, cy, 2 * Math.abs(x - cx), 2 * Math.abs(y - cy)], [x, y]);
      return;
    }
    if (visibility() === 0) visibilityChoice = 2;
    setDraft([x, y]);
    await confirmEdit();
  }

  async function deleteGeometry() {
    if (!sample || pending || saving || !targetValue()) return;
    if (dirty() && !canNavigate()) return;
    if (await save(withTarget(null))) {
      clearDraft();
      if (mode === "edit") mode = "select";
      render();
    }
  }

  async function undoEdit() {
    const snapshot = active?.undo?.at(-1);
    if (!snapshot || !canNavigate()) return;
    if (!await write("undo", {})) return;
    const keepJoint = mode === "continuous" && inspectorOpen && targetType === "joint" && !dirty();
    if (!focusMode() && !keepJoint) clearDraft();
    if (mode === "edit") mode = "select";
    const targetPosition = active.images.findIndex(item =>
      item.split === snapshot.split && item.image_id === snapshot.image_id);
    if (targetPosition !== position) await loadSample(targetPosition);
    else {
      render();
      if (focusOrigin) setMagnifierPoint(...geometryFields(targetValue()).slice(0, 2));
    }
    message("Edit undone");
  }

  async function moveGeometry() {
    if (!person() || !canNavigate()) return;
    const people = structuredClone(entry().people);
    const from = people.find(p=>p.id === personId);
    const to = people.find(p=>p.id === ui("owner").value);
    if (!to) return;
    const joint = targetType === "joint";
    const value = joint ? from.joints[jointIndex] : from.bbox;
    if (!value) { message("Set the geometry before moving it."); return; }
    if ((joint ? to.joints[jointIndex] : to.bbox) && !window.confirm("Replace the destination person's existing geometry?")) return;
    if (joint) { to.joints[jointIndex] = value; from.joints[jointIndex] = null; }
    else { to.bbox = value; from.bbox = null; }
    if (await save(people)) { personId = to.id; clearDraft(); render(); }
  }

  async function addImages(images) {
    if (!active || !canNavigate()) return {error: "Queue update canceled."};
    const previousLength = active.images.length;
    if (!await write("add", {images})) return {error: ui("edit-status").textContent};
    ui("queue-ids").value = "";
    clearDraft();
    if (inQueue()) await loadSample(position);
    else render();
    message("Queue saved");
    return {added: active.images.length - previousLength};
  }

  async function init(reloadContext) {
    await catalog();
    if (reloadContext?.setId && [...ui("set-select").options].some(option => option.value === reloadContext.setId)) {
      try {
        active = await fetchJSON("/api/sets/" + reloadContext.setId);
        ui("set-select").value = active.id;
        ui("browse-pool").checked = Boolean(reloadContext.browsePool);
      } catch (error) {
        message(error.message);
      }
    }
    initCopyAnnotations({
      getTarget: () => ({id: active.id, ...imageKey()}),
      prepare() {
        if (!canNavigate()) return false;
        clearDraft();
        if (mode === "edit") mode = "select";
        render();
        return true;
      },
      async copy(target, options) {
        if (target.id !== active?.id || target.split !== entry()?.split || target.image_id !== entry()?.image_id) {
          return "The active image changed. Reopen the copy dialog.";
        }
        const before = entry().people.length;
        if (!await write("copy", {...imageKey(), ...options})) return ui("edit-status").textContent;
        const count = entry().people.length - before;
        personId = entry().people[before].id;
        jointIndex = 0;
        targetType = "joint";
        mode = "select";
        clearDraft(); render();
        message(count + (count === 1 ? " person copied." : " people copied."));
      },
    });
    initAutolabel({
      getSet: () => active,
      async refresh() {
        const id = active.id;
        const record = await fetchJSON("/api/sets/" + id);
        if (active?.id !== id) throw Error("Active set changed.");
        active = record;
        render();
      },
      prepare() {
        if (!canNavigate()) return false;
        clearDraft();
        if (mode === "edit") mode = "select";
        render();
        return true;
      },
    });
    ui("set-select").onchange = async event => {
      const id = event.target.value;
      if (!canNavigate()) { event.target.value = active?.id ?? ""; return; }
      clearDraft();
      try {
        const next = id ? await fetchJSON("/api/sets/"+id) : null;
        if (ui("set-select").value !== id) return;
        active = next;
        ui("browse-pool").checked = false;
        await loadSample(0);
      } catch (error) { window.alert(error.message); ui("set-select").value = active?.id ?? ""; }
    };
    ui("new-set").onclick = async () => {
      if (!canNavigate()) return;
      const name = window.prompt("Name this annotation set");
      if (!name) return;
      if (!await write("create",{name})) { window.alert(ui("edit-status").textContent); return; }
      clearDraft();
      await catalog();
      ui("browse-pool").checked = false;
      ui("queue-import").open = true;
      await loadSample(0);
    };
    ui("browse-pool").onchange = async () => {
      if (!canNavigate()) { ui("browse-pool").checked = !ui("browse-pool").checked; return; }
      clearDraft(); await loadSample(0);
    };
    ui("add-current").onclick = () => sample && addImages([{split:splitSelect.value,image_id:sample.image.id}]);
    ui("add-ids").onclick = () => {
      const images = [];
      for (const token of ui("queue-ids").value.trim().split(/[\s,]+/)) {
        const match = /^(?:(train|val|test)[/:])?(\d+)$/.exec(token);
        if (!match) { message("Use image IDs such as val/114203."); return; }
        images.push({split:match[1] ?? splitSelect.value,image_id:Number(match[2])});
      }
      addImages(images);
    };
    ui("add-person").onclick = async () => {
      if (canNavigate() && await write("person", imageKey())) {
        clearDraft();
        personId = entry().people.at(-1).id;
        jointIndex = 0;
        targetType = "joint";
        mode = "select";
        render();
      }
    };
    ui("person").onchange = event => choosePerson(event.target.value);
    ui("previous-person").onclick = () => choosePerson(entry().people[personIndex() - 1]?.id);
    ui("next-person").onclick = () => choosePerson(entry().people[personIndex() + 1]?.id);
    for (const [index] of dataset.joints.entries()) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = jointName(index);
      button.onclick = () => chooseJoint(index);
      jointButtons.push(button);
      ui("joint-buttons").append(button);
    }
    ui("previous-joint").onclick = () => chooseJoint(targetType === "box" ? jointIndex : adjacentJoint(-1));
    ui("next-joint").onclick = () => chooseJoint(targetType === "box" ? jointIndex : adjacentJoint(1));
    ui("select-box").onclick = () => {
      if (mode !== "select" || !canNavigate()) return;
      targetType = "box";
      render();
    };
    for (const button of ui("visibility-row").querySelectorAll("button")) {
      button.onclick = async () => {
        const next = Number(button.dataset.visibility);
        if (next === visibility()) return;
        visibilityChoice = next;
        render();
        if (next === 0 || validFields(values())) await confirmEdit();
      };
    }
    for (const id of fieldIds) {
      ui(id).oninput = () => {
        draft = values();
        setMagnifierPoint(draft[0], draft[1]);
        drawOverlays();
      };
      ui(id).addEventListener("focusout", event => {
        if (targetType === "joint" && draft && !fieldIds.includes(event.relatedTarget?.id)) {
          confirmEdit();
        }
      });
    }
    for (const action of ["position", "size"]) {
      ui("box-" + action).onclick = () => { boxAction = action; render(); };
    }
    ui("delete-person").onclick = async () => {
      if (!person() || !canNavigate()) return;
      const label = "Person " + (personIndex() + 1);
      if (!window.confirm("Delete " + label + " and all their annotations from this custom image? You can undo this.")) return;
      if (await save(entry().people.filter(p => p.id !== personId))) { clearDraft(); render(); }
    };
    for (const [id, value] of Object.entries(toolButtons)) ui(id).onclick = () => changeTool(value);
    ui("edit-geometry").onclick = () => changeTool("edit");
    ui("delete-geometry").onclick = deleteGeometry;
    ui("undo-edit").onclick = undoEdit;
    ui("placement-zoom").onchange = drawOverlays;
    ui("cancel-edit").onclick = () => {
      if (mode === "continuous" && targetType === "joint" && !dirty()) {
        clearDraft();
        render();
      } else cancelPlacement();
    };
    ui("confirm-edit").onclick = confirmEdit;
    ui("move-geometry").onclick = moveGeometry;
    ui("review-image").onclick = async () => {
      if (!canNavigate()) return;
      const item = entry();
      if (!item.reviewed) {
        const unfinished = item.people.some(p=>!p.bbox || p.joints.some(j=>j === null));
        if (!item.people.length && !window.confirm("Mark this image as reviewed with no people?")) return;
        if (unfinished && !window.confirm("Some boxes or joints are unfinished. Mark this image reviewed anyway?")) return;
      }
      if (await save(item.people,!item.reviewed)) { clearDraft(); render(); }
    };
    ui("reference").onchange = drawOverlays;
    stage.tabIndex = 0;
    stage.addEventListener("click", onCanvasClick);
    document.addEventListener("keydown",keydown);
    window.addEventListener("beforeunload", e => { if (dirty() || pending) e.preventDefault(); });
  }
  return {init, inQueue, total, entry, queue: () => active?.images ?? [], canNavigate, resetImage, render, draw,
    addImages, jointPreview, focusMode, inspectionEnabled: () => !inQueue() || mode === "select",
    setId: () => active?.id ?? "", reloadReady: () => !pending && !dirty() && !ui("queue-ids").value.trim(),
    selectedPersonId: () => personId, selectLayer: choosePerson, boxSelected: () => targetType === "box"};
})();
