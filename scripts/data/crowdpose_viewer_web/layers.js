"use strict";

function validOverlayBox(box) {
  return Array.isArray(box) && box.length === 4 && box.every(Number.isFinite) && box[2] > 0 && box[3] > 0;
}

function hitAnnotationGeometry(people, x, y, radius) {
  let jointHit = null;
  let nearest = radius;
  let boxHit = null;
  let smallest = Infinity;
  for (const person of people) {
    if (person.showPose) {
      for (const [index, joint] of person.joints.entries()) {
        if (!joint) continue;
        const distance = Math.hypot(x - joint.x, y - joint.y);
        if (distance <= nearest && (jointHit === null || distance < nearest)) {
          jointHit = {key: person.key, id: person.id, joint: index};
          nearest = distance;
        }
      }
    }
    if (!person.showBox) continue;
    const [left, top, width, height] = person.bbox;
    const area = width * height;
    if (area < smallest && x >= left && x <= left + width && y >= top && y <= top + height) {
      boxHit = {key: person.key, id: person.id, joint: null};
      smallest = area;
    }
  }
  return jointHit ?? boxHit;
}

const layers = (() => {
  const ui = id => element("layers-" + id);
  const hidden = new Map();
  const controls = new Map();
  let people = [];
  let previousSample;
  let previousEntry;
  let previousReference;
  let pinned = null;
  let hovered = null;

  function reset() {
    hidden.clear();
    controls.clear();
    people = [];
    previousSample = previousEntry = previousReference = undefined;
    pinned = hovered = null;
    ui("rows").replaceChildren();
    ui("clear").hidden = true;
    ui("pane").hidden = true;
  }

  function describePeople(entry, source) {
    const custom = (entry?.people ?? []).map((person, index) => ({
      key: "custom:" + person.id, id: person.id, number: index + 1, label: "Person " + (index + 1),
      bbox: person.bbox, color: colorFor(index),
      joints: person.joints.map(joint => joint?.[2] ? {x:joint[0], y:joint[1], visibility:joint[2]} : null),
    }));
    if (!source) return custom;
    return custom.concat(sample.annotations.map((person, index) => ({
      key: "source:" + (person.id ?? "row-" + index),
      label: (person.iscrowd ? "Crowd " : "Source ") + (person.id == null ? "row " + (index + 1) : "#" + person.id),
      bbox: person.bbox, joints: getJoints(person), iscrowd: person.iscrowd,
      color: entry ? "#b5bbc4" : colorFor(index),
    })));
  }

  function visiblePerson(person) {
    const state = hidden.get(person.key);
    const shown = !isolateSelectedPerson() || person.id === editor.selectedPersonId();
    return {...person, showBox: shown && boxesToggle.checked && person.hasBox && state?.box !== true,
      showPose: shown && posesToggle.checked && person.hasPose && state?.pose !== true};
  }

  function setHover(key) {
    if (!editor.inspectionEnabled()) return;
    if (hovered === key) return;
    hovered = key;
    drawOverlays();
  }

  function rowFor(person) {
    const row = document.createElement("li");
    const identify = document.createElement("button");
    identify.type = "button";
    identify.textContent = person.label;
    identify.setAttribute("aria-pressed", "false");
    identify.style.setProperty("--person-color", person.color);
    if (person.number) {
      const ident = document.createElement("small");
      ident.textContent = person.id.slice(0, 8);
      ident.title = person.id;
      identify.title = "Person ID " + person.id;
      identify.append(ident);
    }
    identify.onclick = () => {
      if (!editor.inspectionEnabled()) return;
      if (person.number) editor.selectLayer(person.id);
      pinned = pinned === person.key ? null : person.key;
      drawOverlays();
    };
    row.onpointerenter = () => setHover(person.key);
    row.onpointerleave = () => setHover(null);
    identify.onfocus = () => setHover(person.key);
    identify.onblur = () => setHover(null);
    row.append(identify);
    const inputs = {};
    for (const [kind, name] of [["box", "Box"], ["pose", "Pose"]]) {
      const label = document.createElement("label");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.setAttribute("aria-label", name + " for " + person.label);
      checkbox.onchange = () => {
        const state = hidden.get(person.key) ?? {};
        hidden.set(person.key, {...state, [kind]: !checkbox.checked});
        drawOverlays();
      };
      inputs[kind] = checkbox;
      label.append(checkbox, name);
      row.append(label);
    }
    controls.set(person.key, {row, identify, inputs});
    return row;
  }

  function sync() {
    const entry = editor.entry();
    const source = !entry || element("reference").checked;
    if (sample === previousSample && entry === previousEntry && source === previousReference) return;
    previousSample = sample;
    previousEntry = entry;
    previousReference = source;
    people = describePeople(entry, source).map(person => ({
      ...person, hasBox: validOverlayBox(person.bbox), hasPose: person.joints.some(Boolean),
    }));
    controls.clear();
    ui("pane").hidden = false;
    ui("empty").hidden = people.length > 0;
    ui("rows").replaceChildren(...people.map(rowFor));
    if (!controls.has(pinned)) pinned = null;
    if (!controls.has(hovered)) hovered = null;
  }

  function updateRow(person, highlighted) {
    const {row, identify, inputs} = controls.get(person.key);
    row.classList.toggle("highlighted", highlighted);
    row.classList.toggle("selected", person.id === editor.selectedPersonId());
    identify.setAttribute("aria-pressed", pinned === person.key);
    inputs.box.checked = person.hasBox && hidden.get(person.key)?.box !== true;
    inputs.pose.checked = person.hasPose && hidden.get(person.key)?.pose !== true;
    inputs.box.disabled = !boxesToggle.checked || !person.hasBox;
    inputs.pose.disabled = !posesToggle.checked || !person.hasPose;
  }

  function drawLabel(person, text, scale) {
    const joint = person.joints.find(Boolean);
    const anchor = person.showBox ? person.bbox : [joint.x, joint.y];
    const padding = 4 / scale;
    context.font = "bold " + 13 / scale + "px sans-serif";
    context.textBaseline = "top";
    const width = context.measureText(text).width + 2 * padding;
    const height = 18 / scale;
    const transform = context.getTransform();
    const x = Math.max(-transform.e / transform.a, Math.min(anchor[0],
      (overlay.width - transform.e) / transform.a - width));
    const y = Math.max(-transform.f / transform.d, Math.min(anchor[1],
      (overlay.height - transform.f) / transform.d - height));
    context.fillStyle = person.id === editor.selectedPersonId() ? "#fff" : person.color;
    context.fillRect(x, y, width, height);
    context.fillStyle = "#111619";
    context.fillText(text, x + padding, y + 2 / scale);
  }

  function draw(scale) {
    sync();
    const visible = people.map(visiblePerson);
    const inspecting = editor.inspectionEnabled();
    if (!inspecting) hovered = null;
    const target = inspecting ? pinned ?? hovered : null;
    const active = visible.find(person => person.key === target && (person.showBox || person.showPose));
    ui("clear").hidden = pinned === null;
    for (const person of visible) {
      updateRow(person, person === active);
      if (person === active) continue;
      context.save();
      if (active) context.globalAlpha = 0.28;
      drawGeometry(person, scale);
      if (person.id === editor.selectedPersonId() && (person.showBox || person.showPose)) {
        drawLabel(person, person.label, scale);
      } else if (person.showBox && person.number) {
        drawLabel(person, String(person.number), scale);
      }
      context.restore();
    }
    if (active) {
      drawGeometry(active, scale);
      drawLabel(active, active.label, scale);
    }
  }

  function drawGeometry(person, scale) {
    const selected = person.id === editor.selectedPersonId();
    if (person.showBox) drawBox(person, selected && editor.boxSelected() ? "#f4ff82" : person.color, scale);
    if (person.showPose) {
      const preview = editor.jointPreview(person.id);
      const joints = preview ? person.joints.map((joint, index) =>
        index === preview.index ? preview.joint : joint) : person.joints;
      drawPose(joints, person.color, scale);
    }
  }

  function hitAt(event, editableOnly) {
    const {x, y, scale} = pointerToSource(event);
    const pool = editableOnly ? people.filter(person => person.number) : people;
    return hitAnnotationGeometry(pool.map(visiblePerson), x, y, 9 / scale);
  }

  function isVisible(id, type) {
    const person = people.find(item => item.id === id);
    if (!person) return false;
    const visible = visiblePerson(person);
    return type === "box" ? visible.showBox : visible.showPose;
  }

  function init() {
    frame.addEventListener("pointermove", event => {
      if (!sample || !editor.inspectionEnabled() || pinned) {
        setHover(null);
        return;
      }
      setHover(hitAt(event, false)?.key ?? null);
    });
    frame.addEventListener("pointerleave", () => setHover(null));
    ui("clear").onclick = () => {
      pinned = null;
      hovered = null;
      drawOverlays();
    };
  }

  return {init, reset, draw, isVisible, hitEditable: event => hitAt(event, true)};
})();
