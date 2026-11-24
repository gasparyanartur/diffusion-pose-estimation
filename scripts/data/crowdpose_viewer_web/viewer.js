"use strict";

const element = (id) => document.getElementById(id);
const splitSelect = element("split");
const positionInput = element("position");
const stage = element("stage");
const frame = element("image-frame");
const imageStatus = element("image-status");
const photo = element("photo");
const overlay = element("overlay");
const context = overlay.getContext("2d");
const lensDock = element("lens-dock");
const lens = element("lens");
const lensContext = lens.getContext("2d");
const brightnessSample = document.createElement("canvas");
brightnessSample.width = brightnessSample.height = 11;
const brightnessContext = brightnessSample.getContext("2d", {willReadFrequently: true});
const stageBackground = "#0b0f12";
const boxesToggle = element("boxes");
const posesToggle = element("poses");
const selectedPersonToggle = element("selected-person-only");
const noteText = element("note-text");
const noteStatus = element("note-status");
const drafts = new Map();
const annotationColors = ["#65f2b5", "#ffcc70", "#7ad6ff", "#ff85b7", "#d3a6ff", "#ff9777"];
const dateFormat = new Intl.DateTimeFormat(undefined, {dateStyle: "medium", timeStyle: "short"});

let dataset;
let position = 0;
let sample;
let request;
let loading = false;
let saving = false;
let lastWheelStep = -Infinity;
let magnifierPoint = null;
let magnifierFrame = null;

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed.");
  }
  return data;
}

function colorFor(index) {
  return annotationColors[index % annotationColors.length];
}

function getJoints(annotation) {
  const keypoints = Array.isArray(annotation.keypoints) ? annotation.keypoints : [];
  return dataset.joints.map((_, index) => {
    const [x, y, visibility] = keypoints.slice(index * 3, index * 3 + 3);
    if ((visibility === 1 || visibility === 2) && Number.isFinite(x) && Number.isFinite(y)) {
      return {x, y, visibility};
    }
    return null;
  });
}

function pointerToSource(event) {
  const rect = photo.getBoundingClientRect();
  const scale = rect.width / sample.image.width;
  return {
    x: (event.clientX - rect.left) / scale,
    y: (event.clientY - rect.top) * sample.image.height / rect.height,
    scale,
  };
}

function luminanceNear(x, y) {
  const size = brightnessSample.width;
  brightnessContext.fillStyle = stageBackground;
  brightnessContext.fillRect(0, 0, size, size);
  brightnessContext.drawImage(photo, Math.round(x) - 5, Math.round(y) - 5, size, size, 0, 0, size, size);
  const pixels = brightnessContext.getImageData(0, 0, size, size).data;
  let brightness = 0;
  for (let index = 0; index < pixels.length; index += 4) {
    brightness += 0.2126 * pixels[index] + 0.7152 * pixels[index + 1] + 0.0722 * pixels[index + 2];
  }
  return brightness / (size * size);
}

function drawReticle(ctx, x, y, scale, sourcePoint = [x, y]) {
  const darkCore = luminanceNear(...sourcePoint) > 140;
  const arm = 18 / scale;
  const gap = 6 / scale;
  ctx.save();
  ctx.shadowBlur = 0;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x - arm, y);
  ctx.lineTo(x - gap, y);
  ctx.moveTo(x + gap, y);
  ctx.lineTo(x + arm, y);
  ctx.moveTo(x, y - arm);
  ctx.lineTo(x, y - gap);
  ctx.moveTo(x, y + gap);
  ctx.lineTo(x, y + arm);
  ctx.strokeStyle = darkCore ? "#fff" : "#101519";
  ctx.lineWidth = 7 / scale;
  ctx.stroke();
  ctx.strokeStyle = darkCore ? "#101519" : "#fff";
  ctx.lineWidth = 3 / scale;
  ctx.stroke();
  ctx.restore();
}

function renderMagnifier() {
  magnifierFrame = null;
  if (lensDock.hidden || !magnifierPoint) return;
  const [x, y] = magnifierPoint;
  const size = lens.width;
  const crop = size / 3;
  lensContext.fillStyle = stageBackground;
  lensContext.fillRect(0, 0, size, size);
  lensContext.imageSmoothingEnabled = false;
  lensContext.drawImage(photo, x + 0.5 - crop / 2, y + 0.5 - crop / 2, crop, crop, 0, 0, size, size);
  drawReticle(lensContext, size / 2, size / 2, 1, magnifierPoint);
  element("lens-coordinates").textContent = `X ${x} · Y ${y}`;
}

function setMagnifierPoint(x, y) {
  if (lensDock.hidden || !Number.isFinite(x) || !Number.isFinite(y) ||
      (magnifierPoint?.[0] === x && magnifierPoint[1] === y)) return;
  magnifierPoint = [x, y];
  if (magnifierFrame === null) magnifierFrame = requestAnimationFrame(renderMagnifier);
}

function isolateSelectedPerson() {
  return editor.inQueue() && selectedPersonToggle.checked;
}

function drawOverlays() {
  if (frame.hidden || !sample) return;
  const {width, height} = sample.image;
  const workArea = editor.inQueue();
  const stageWidth = stage.clientWidth;
  const stageHeight = stage.clientHeight;
  if (!stageWidth || !stageHeight) return;
  const gutter = workArea ? Math.min(64, stageWidth * 0.1, stageHeight * 0.1) : 0;
  const scale = Math.min((stageWidth - 2 * gutter) / width, (stageHeight - 2 * gutter) / height, 1.5);
  const displayWidth = width * scale;
  const displayHeight = height * scale;
  const frameWidth = workArea ? stageWidth : displayWidth;
  const frameHeight = workArea ? stageHeight : displayHeight;
  const left = (frameWidth - displayWidth) / 2;
  const top = (frameHeight - displayHeight) / 2;
  frame.classList.toggle("annotation-work-area", workArea);
  frame.style.width = `${frameWidth}px`;
  frame.style.height = `${frameHeight}px`;
  Object.assign(photo.style, {
    left: `${left}px`, top: `${top}px`, width: `${displayWidth}px`, height: `${displayHeight}px`,
  });
  overlay.width = Math.max(1, Math.round(frameWidth * window.devicePixelRatio));
  overlay.height = Math.max(1, Math.round(frameHeight * window.devicePixelRatio));
  const ratioX = overlay.width / frameWidth;
  const ratioY = overlay.height / frameHeight;
  context.setTransform(scale * ratioX, 0, 0, scale * ratioY, left * ratioX, top * ratioY);
  context.lineJoin = "round";
  context.lineCap = "round";
  context.shadowColor = "#000";
  context.shadowBlur = 3;

  layers.draw(scale);
  editor.draw(scale);
  if (!isolateSelectedPerson()) audit.draw(scale);
}

function drawBox(annotation, color, scale) {
  const box = annotation.bbox;
  if (!validOverlayBox(box)) return;
  const [x, y, width, height] = box;
  context.strokeStyle = color;
  context.lineWidth = 1.5 / scale;
  context.setLineDash(annotation.iscrowd ? [6 / scale, 4 / scale] : []);
  context.strokeRect(x, y, width, height);
  context.setLineDash([]);
}

function drawPose(joints, color, scale) {
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = 2 / scale;
  for (const [start, end] of dataset.edges) {
    const from = joints[start];
    const to = joints[end];
    if (!from || !to) continue;
    context.setLineDash(from.visibility === 2 && to.visibility === 2 ? [] : [4 / scale, 3 / scale]);
    context.beginPath();
    context.moveTo(from.x, from.y);
    context.lineTo(to.x, to.y);
    context.stroke();
  }
  context.setLineDash([]);
  for (const joint of joints) {
    if (!joint) continue;
    context.beginPath();
    context.arc(joint.x, joint.y, 3.1 / scale, 0, Math.PI * 2);
    if (joint.visibility === 2) context.fill();
    else context.stroke();
  }
}

function setDetails(fields) {
  for (const [id, value] of Object.entries(fields)) element(id).textContent = value;
}

function showDetails() {
  const list = element("annotations");
  list.replaceChildren();
  if (!sample) {
    setDetails({
      filename: "", "crowd-index": "—", "people-count": "—", "image-id": "—",
      dimensions: "—", "visible-count": "—", "occluded-count": "—",
      "unlabeled-count": "—", "crowd-count": "—",
    });
    return;
  }
  const {image, annotations} = sample;
  const totals = [0, 0, 0];
  let crowds = 0;
  annotations.forEach((annotation, index) => {
    const counts = [0, 0, 0];
    for (const joint of getJoints(annotation)) counts[joint?.visibility ?? 0]++;
    counts.forEach((count, visibility) => { totals[visibility] += count; });
    if (annotation.iscrowd) crowds++;
    const row = document.createElement("li");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.backgroundColor = colorFor(index);
    const label = annotation.iscrowd ? "Crowd" : "Person";
    const description = counts[1] + counts[2]
      ? `${counts[2]} visible, ${counts[1]} occluded`
      : "no labeled joints";
    row.append(swatch, `${label} #${annotation.id ?? "—"} · ${description}`);
    list.append(row);
  });
  if (!annotations.length) {
    const row = document.createElement("li");
    row.textContent = "No annotations";
    list.append(row);
  }
  setDetails({
    filename: image.file_name ?? "",
    "crowd-index": image.crowdIndex ?? "—",
    "people-count": annotations.length - crowds,
    "image-id": image.id,
    dimensions: image.width > 0 && image.height > 0 ? `${image.width} × ${image.height} px` : "—",
    "visible-count": totals[2],
    "occluded-count": totals[1],
    "unlabeled-count": totals[0],
    "crowd-count": crowds,
  });
}

function showNotes() {
  const list = element("notes");
  const notes = sample?.notes ?? [];
  element("note-count").textContent = notes.length ? String(notes.length) : "";
  list.replaceChildren(...notes.slice().reverse().map((note) => {
    const row = document.createElement("li");
    const time = document.createElement("time");
    time.dateTime = note.created_at;
    const date = new Date(note.created_at);
    time.textContent = Number.isNaN(date.valueOf()) ? note.created_at : dateFormat.format(date);
    const text = document.createElement("p");
    text.textContent = note.text;
    row.append(time, text);
    return row;
  }));
}

function draftKey() {
  return `${splitSelect.value}/${sample.image.id}`;
}

function updateControls() {
  const total = editor.total();
  splitSelect.disabled = saving || !dataset || editor.inQueue();
  element("previous").disabled = saving || position <= 0;
  element("next").disabled = saving || position + 1 >= total;
  positionInput.value = total ? position + 1 : "";
  positionInput.max = total;
  positionInput.disabled = saving || !total;
  element("total").textContent = `/ ${total.toLocaleString()}`;
  updateNoteControls();
}

function updateNoteControls() {
  noteText.disabled = loading || saving || !sample;
  element("save-note").disabled = noteText.disabled || !noteText.value.trim();
}

function showImageStatus(message) {
  frame.hidden = true;
  lensDock.hidden = true;
  magnifierPoint = null;
  element("selection-overlay").hidden = true;
  imageStatus.hidden = false;
  imageStatus.textContent = message;
}

async function showPhoto(record) {
  if (!(record.image.width > 0 && record.image.height > 0)) {
    showImageStatus("Image dimensions unavailable.");
    return;
  }
  photo.alt = `CrowdPose image ${record.image.id}`;
  photo.src = record.image_url;
  try {
    await photo.decode();
  } catch {
    if (sample === record) showImageStatus("Image file unavailable.");
    return;
  }
  if (sample !== record) return;
  imageStatus.hidden = true;
  frame.hidden = false;
  lensDock.hidden = !editor.inQueue();
  element("selection-overlay").hidden = !editor.inQueue();
  setMagnifierPoint(Math.round(record.image.width / 2), Math.round(record.image.height / 2));
  drawOverlays();
}

async function loadSample(nextPosition, flag = null) {
  if (saving || !dataset || !editor.canNavigate()) return;
  const split = splitSelect.value;
  const total = editor.total();
  position = total ? Math.max(0, Math.min(total - 1, nextPosition)) : 0;
  request?.abort();
  const controller = new AbortController();
  request = controller;
  sample = null;
  layers.reset();
  audit.setFocus(flag);
  audit.showReview();
  editor.resetImage();
  loading = Boolean(total);
  noteText.value = "";
  noteStatus.textContent = "";
  showDetails();
  showNotes();
  showImageStatus(total ? "Loading image…" : "This split is empty.");
  updateControls();
  if (!total) {
    if (editor.inQueue()) showImageStatus("Add images by ID or browse source images to build your queue.");
    return;
  }

  try {
    const item = editor.entry();
    const url = item ? `/api/sample-id/${item.split}/${item.image_id}` : `/api/sample/${split}/${position}`;
    if (item) splitSelect.value = item.split;
    const record = await fetchJSON(url, {signal: controller.signal});
    if (controller.signal.aborted) return;
    sample = record;
    audit.showReview();
    editor.render();
    noteText.value = drafts.get(draftKey()) ?? "";
    showDetails();
    showNotes();
    await showPhoto(record);
  } catch (error) {
    if (!controller.signal.aborted) showImageStatus(error.message);
  } finally {
    if (!controller.signal.aborted) {
      loading = false;
      updateNoteControls();
    }
  }
}

function step(direction) {
  const total = editor.total();
  const target = position + direction;
  if (target >= 0 && target < total) loadSample(target);
}

element("previous").addEventListener("click", () => step(-1));
element("next").addEventListener("click", () => step(1));
splitSelect.addEventListener("change", () => loadSample(0));
element("jump").addEventListener("submit", (event) => {
  event.preventDefault();
  if (Number.isInteger(positionInput.valueAsNumber)) loadSample(positionInput.valueAsNumber - 1);
});
for (const toggle of [boxesToggle, posesToggle, selectedPersonToggle]) toggle.addEventListener("change", drawOverlays);
frame.addEventListener("pointermove", event => {
  if (lensDock.hidden) return;
  const {x, y} = pointerToSource(event);
  setMagnifierPoint(Math.round(x), Math.round(y));
});
new ResizeObserver(drawOverlays).observe(stage);

document.addEventListener("keydown", (event) => {
  if (document.querySelector("dialog[open]") || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey ||
      event.target.closest("input, select, textarea, [contenteditable]")) return;
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    if (editor.focusMode()) return;
    event.preventDefault();
    step(event.key === "ArrowLeft" ? -1 : 1);
  }
});
stage.addEventListener("wheel", (event) => {
  if (editor.inQueue() || event.ctrlKey || event.deltaY === 0 || saving) return;
  const direction = Math.sign(event.deltaY);
  const total = editor.total();
  if (position + direction < 0 || position + direction >= total) return;
  event.preventDefault();
  if (event.timeStamp - lastWheelStep >= 200) {
    lastWheelStep = event.timeStamp;
    step(direction);
  }
}, {passive: false});

noteText.addEventListener("input", () => {
  if (sample) {
    if (noteText.value) drafts.set(draftKey(), noteText.value);
    else drafts.delete(draftKey());
  }
  noteStatus.textContent = "";
  updateNoteControls();
});

element("note-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (loading || saving || !sample || !noteText.value.trim()) return;
  saving = true;
  updateControls();
  noteStatus.textContent = "Saving…";
  try {
    const record = await fetchJSON("/api/notes", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({split: splitSelect.value, image_id: sample.image.id, text: noteText.value}),
    });
    sample.notes.push(record);
    drafts.delete(draftKey());
    noteText.value = "";
    showNotes();
    noteStatus.textContent = "Saved";
  } catch (error) {
    noteStatus.textContent = error.message;
  } finally {
    saving = false;
    updateControls();
    noteText.focus();
  }
});

const reloadContextKey = "crowdpose-viewer:reload";

function takeReloadContext() {
  try {
    const saved = sessionStorage.getItem(reloadContextKey);
    sessionStorage.removeItem(reloadContextKey);
    return JSON.parse(saved);
  } catch {
    return null;
  }
}

async function restoredPosition(reloadContext) {
  if (!reloadContext) return 0;
  const fallback = Number.isInteger(reloadContext.position) ? reloadContext.position : 0;
  const {split, imageId} = reloadContext;
  if (!Object.hasOwn(dataset.splits, split) || !Number.isInteger(imageId)) return fallback;
  if (editor.inQueue()) {
    const index = editor.queue().findIndex(item => item.split === split && item.image_id === imageId);
    return index < 0 ? fallback : index;
  }
  try {
    const record = await fetchJSON(`/api/sample-id/${split}/${imageId}`);
    return Number.isInteger(record.position) ? record.position : fallback;
  } catch {
    return fallback;
  }
}

function startLiveReload() {
  let version;
  async function poll() {
    try {
      const response = await fetch("/api/live-reload");
      if (response.status === 404) return;
      if (response.ok) {
        const {version: latest} = await response.json();
        if (latest === null) return;
        version ??= latest;
        if (latest !== version && !loading && !saving && !drafts.size && editor.reloadReady() &&
            !document.querySelector("dialog[open]") && element("set-select").value === editor.setId()) {
          sessionStorage.setItem(reloadContextKey, JSON.stringify({
            setId: editor.setId(), browsePool: element("browse-pool").checked,
            split: splitSelect.value, imageId: sample?.image.id, position,
          }));
          location.reload();
          return;
        }
      }
    } catch {
      // The local server may be restarting.
    }
    setTimeout(poll, 1000);
  }
  poll();
}

document.addEventListener("DOMContentLoaded", async () => {
  const reloadContext = takeReloadContext();
  try {
    dataset = await fetchJSON("/api/dataset");
    if (reloadContext && Object.hasOwn(dataset.splits, reloadContext.split)) splitSelect.value = reloadContext.split;
    await editor.init(reloadContext);
    audit.init();
    layers.init();
    await loadSample(await restoredPosition(reloadContext));
    startLiveReload();
  } catch (error) {
    showImageStatus(error.message);
  }
});
