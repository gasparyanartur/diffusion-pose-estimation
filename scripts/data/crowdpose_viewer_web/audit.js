"use strict";

function parseAuditPositions(text, total) {
  const selected = new Set();
  for (const token of text.split(",")) {
    const match = /^(\d+)(?:\s*[-–]\s*(\d+))?$/.exec(token.trim());
    const start = Number(match?.[1]);
    const end = Number(match?.[2] ?? start);
    if (!match || start < 1 || end < start || end > total) {
      throw new Error("Enter positions from 1 to " + total + " in ascending ranges.");
    }
    for (let index = start - 1; index < end; index++) selected.add(index);
  }
  return [...selected];
}

function selectAuditRows(text, source, matching, available, random = Math.random) {
  const command = text.trim().toLowerCase().replace(/\s+/g, " ")
    .replace(/^(\d+) random$/, "random $1");
  if (!/[a-z]/.test(command)) {
    const rows = parseAuditPositions(command, source.length).map(index => source[index]);
    return {rows: rows.filter(available)};
  }
  if (command === "all matching") return {rows: matching.filter(available)};

  const match = /^(random|first|last) (\d+)$/.exec(command);
  if (!match) throw new Error("Use split positions or a command listed under Info.");
  const [, kind, count] = match;
  const requested = Number(count);
  if (!Number.isSafeInteger(requested) || requested < 1) {
    throw new Error("Enter a positive whole number for this command.");
  }

  const pool = matching.filter(available);
  const size = Math.min(requested, pool.length);
  if (kind === "last") return {rows: pool.slice(pool.length - size), requested};
  if (kind === "random") {
    for (let index = 0; index < size; index++) {
      const next = index + Math.floor(random() * (pool.length - index));
      [pool[index], pool[next]] = [pool[next], pool[index]];
    }
  }
  return {rows: pool.slice(0, size), requested};
}

const audit = (() => {
  const ui = name => element("audit-" + name);
  const value = name => ui(name).value;
  const dialog = ui("window");
  const pageSize = 40;
  const flagLabels = {duplicate: "Shared joint", outside: "Outside box"};
  const emptyReview = {cohort: "", reviewed: false, tags: []};
  let rows = [];
  let page = 0;
  let group = "";
  let focus = null;
  let writing = false;
  let tableRequest;
  let picker = null;

  function imageKey(row) {
    return row.split + "/" + row.image_id;
  }

  function queuedKeys() {
    return new Set(editor.queue().map(imageKey));
  }

  function updatePicker() {
    if (!picker) return;
    const count = picker.selected.size;
    ui("add-selection").textContent = count ? "Add selected (" + count + ")" : "Add selected";
    ui("add-selection").disabled = !count || picker.saving;
    ui("clear-selection").disabled = !count || picker.saving;
    ui("range-select").disabled = picker.loading || picker.saving || !rows.length;
    ui("select-page").disabled = picker.loading || picker.saving || !picker.visible.length;
  }

  function selectRows(selectedRows) {
    if (!picker || picker.saving) return;
    const queued = queuedKeys();
    for (const row of selectedRows) {
      if (!queued.has(imageKey(row))) picker.selected.set(imageKey(row), row);
    }
    ui("picker-status").textContent = "";
    renderTable();
  }

  async function addSelection() {
    if (!picker || picker.saving || !picker.selected.size) return;
    const images = [...picker.selected.values()].map(({split, image_id}) => ({split, image_id}));
    picker.saving = true;
    updatePicker();
    try {
      const {added, error} = await editor.addImages(images);
      if (error) {
        ui("picker-status").textContent = error;
        return;
      }
      picker.selected.clear();
      ui("picker-status").textContent = added + (added === 1 ? " image added." : " images added.");
    } finally {
      picker.saving = false;
      renderTable();
    }
  }

  function jointName(index) {
    return dataset.joints[index].replaceAll("_", " ");
  }

  function sourcePeople() {
    return sample?.annotations.filter(person => !person.iscrowd) ?? [];
  }

  function setFocus(flag) {
    focus = flag;
    ui("person").value = "";
    ui("tag-joint").value = "";
    ui("edit-status").textContent = "";
  }

  function showReview() {
    const review = sample?.audit ?? emptyReview;
    ui("edit").disabled = !sample || writing;
    ui("cohort").value = review.cohort;
    ui("reviewed").checked = review.reviewed;
    const people = sourcePeople();
    const personValue = value("person");
    ui("person").replaceChildren(new Option("Whole image", ""), ...people.map(
      item => new Option("Source #" + item.id, item.id)));
    const highlighted = focus && people.find(item => focus.ids.includes(item.id));
    if (people.some(item => String(item.id) === personValue)) ui("person").value = personValue;
    else if (highlighted && value("kind") !== "clean") {
      ui("person").value = highlighted.id;
      ui("tag-joint").value = focus.joint;
    }
    updateTarget();
    ui("focus").hidden = !highlighted;
    if (highlighted) {
      ui("focus").replaceChildren(flagLabels[focus.kind] + " · " + jointName(focus.joint) + " · source " +
        focus.ids.map(id => "#" + id).join(" and "), button("Clear highlight", () => {
          focus = null;
          showReview();
          drawOverlays();
        }));
    }

    ui("tags").replaceChildren(...review.tags.map(tag => {
      const item = document.createElement("li");
      const person = tag.person === null ? "" : " · #" + tag.person;
      const joint = tag.joint === null ? "" : " · " + jointName(tag.joint);
      item.append(dataset.audit.tags[tag.kind] + person + joint,
                  button("Remove", () => saveReview({...tag, action: "remove"})));
      return item;
    }));
  }

  function updateTarget() {
    const clean = value("kind") === "clean";
    if (clean) ui("person").value = "";
    ui("person").disabled = clean;
    ui("tag-joint").disabled = !value("person");
    if (!value("person")) ui("tag-joint").value = "";
  }

  async function saveReview(change) {
    if (!sample || writing) return;
    const record = sample;
    const split = splitSelect.value;
    writing = true;
    ui("edit").disabled = true;
    ui("edit-status").textContent = "Saving…";
    try {
      record.audit = await fetchJSON("/api/audit", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({...change, split, image_id: record.image.id}),
      });
      if (sample === record) ui("edit-status").textContent = "Saved";
    } catch (error) {
      if (sample === record) ui("edit-status").textContent = error.message;
    } finally {
      writing = false;
      showReview();
    }
  }

  function draw(scale) {
    if (!focus) return;
    context.save();
    context.lineWidth = 2.5 / scale;
    context.strokeStyle = "#fff";
    for (const annotation of sourcePeople()) {
      if (!focus.ids.includes(annotation.id)) continue;
      drawBox(annotation, "#fff", scale);
      const point = getJoints(annotation)[focus.joint];
      if (!point) continue;
      context.beginPath();
      context.arc(point.x, point.y, 10 / scale, 0, 2 * Math.PI);
      context.stroke();
    }
    context.restore();
  }

  function button(label, handler) {
    const control = document.createElement("button");
    control.type = "button";
    control.textContent = label;
    control.onclick = handler;
    return control;
  }

  function peopleGroup(row) {
    return row.people <= 1 ? "0–1 people" : row.people <= 4 ? "2–4 people" : "5+ people";
  }

  function labelGroup(row) {
    if (value("group") === "people") return peopleGroup(row);
    if (value("group") === "cohort") return dataset.audit.cohorts[row.review.cohort];
    const overlap = row.overlapping ? "Overlapping" : "No overlapping pair";
    const size = row.min_height === null ? "no box size" : row.small ? "small person" : "no small person";
    return overlap + " · " + size;
  }

  function jointCount(row, field) {
    const counts = row[field];
    return value("joint") ? counts[Number(value("joint"))] : counts.reduce((sum, count) => sum + count, 0);
  }

  function baseMatches(row) {
    const people = value("people");
    const overlap = value("overlap");
    const crowd = value("crowd-filter");
    const crowdBand = row.crowd_index === null ? "unknown" :
      row.crowd_index < 0.2 ? "low" : row.crowd_index < 0.8 ? "medium" : "high";
    const box = value("box");
    const visibility = value("visibility");
    const flag = value("flag");
    const cohort = value("cohort-filter");
    const reviewed = value("review-filter");
    return (!people || peopleGroup(row) === people) &&
      (!crowd || crowdBand === crowd) && (!overlap || Boolean(row.overlapping) === (overlap === "high")) &&
      (!box || row[box] > 0) && (!visibility || jointCount(row, visibility) > 0) &&
      (!flag || row[flag]) && (!cohort || row.review.cohort === (cohort === "unassigned" ? "" : cohort)) &&
      (!reviewed || row.review.reviewed === (reviewed === "reviewed"));
  }

  function matchesTag(row, kind) {
    return row.review.tags.some(tag => kind ? tag.kind === kind : tag.kind !== "clean");
  }

  function groupRows(filtered, kind) {
    const totals = new Map();
    for (const row of filtered) {
      const label = labelGroup(row);
      if (!totals.has(label)) totals.set(label, {images: 0, reviewed: 0, tagged: 0});
      const stats = totals.get(label);
      stats.images++;
      if (row.review.reviewed) {
        stats.reviewed++;
        if (matchesTag(row, kind)) stats.tagged++;
      }
    }
    ui("groups").replaceChildren(...[...totals].sort().map(([label, stats]) => {
      const imageCount = stats.images + (stats.images === 1 ? " image" : " images");
      const reviewCount = stats.reviewed ? stats.tagged + "/" + stats.reviewed +
        " reviewed " + (kind ? "with tag" : "with issues") : "0 reviewed";
      const result = button(label + " · " + imageCount + " · " + reviewCount, () => {
        group = group === label ? "" : label;
        page = 0;
        renderTable();
      });
      result.setAttribute("aria-pressed", group === label);
      return result;
    }));
  }

  function statRow(row, queued) {
    const line = document.createElement("tr");
    function cell(...content) {
      const item = document.createElement("td");
      item.append(...content);
      line.append(item);
    }
    function open(flag) {
      dialog.close();
      loadSample(row.position, flag);
    }
    const number = (n, digits) => n === null ? "—" : n.toFixed(digits);
    if (picker) {
      const key = imageKey(row);
      const label = document.createElement("label");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = picker.selected.has(key);
      checkbox.disabled = queued.has(key) || picker.saving;
      checkbox.onchange = () => {
        if (checkbox.checked) picker.selected.set(key, row);
        else picker.selected.delete(key);
        ui("picker-status").textContent = "";
        updatePicker();
      };
      label.append(checkbox, (row.position + 1) + " · " + key + (queued.has(key) ? " · In set" : ""));
      cell(label);
    } else {
      cell(button(imageKey(row), () => open(null)));
    }
    cell(number(row.crowd_index, 2));
    cell(String(row.people));
    cell(number(row.max_iou, 2) + " / " + row.overlapping);
    cell(number(row.min_height, 0) + " / " + row.small);
    cell((row.edge ?? "—") + " / " + row.box_only);
    cell(jointCount(row, "missing") + " / " + jointCount(row, "occluded"));
    const flags = Object.entries(flagLabels).filter(([key]) => row[key]).map(([key, label]) =>
      picker ? label + " · " + jointName(row[key].joint) :
        button(label + " · " + jointName(row[key].joint), () => open({...row[key], kind: key})));
    cell(...(picker ? [flags.join(", ")] : flags));
    cell((row.review.reviewed ? "Reviewed" : "Unreviewed") +
      (row.review.cohort ? " · " + dataset.audit.cohorts[row.review.cohort] : ""));
    return line;
  }

  function renderTable() {
    const kind = value("tag-filter");
    const eligible = rows.filter(baseMatches);
    if (value("group")) groupRows(eligible, kind);
    else ui("groups").replaceChildren();
    const matches = eligible.filter(row => (!group || labelGroup(row) === group) &&
      (!kind || matchesTag(row, kind)));
    const sortKey = {
      people: row => -row.people,
      iou: row => -(row.max_iou ?? -1),
      height: row => row.min_height ?? Infinity,
      visibility: row => -jointCount(row, value("visibility") || "missing"),
      crowd: row => -(row.crowd_index ?? -1),
    }[value("sort")];
    if (sortKey) matches.sort((a, b) => sortKey(a) - sortKey(b));
    const start = page * pageSize;
    const end = Math.min(start + pageSize, matches.length);
    const visible = matches.slice(start, end);
    const queued = picker ? queuedKeys() : null;
    if (picker) {
      picker.visible = visible;
      picker.matching = matches;
    }
    ui("rows").replaceChildren(...visible.map(row => statRow(row, queued)));
    updatePicker();
    ui("prev").disabled = page === 0;
    ui("next").disabled = end >= matches.length;
    const range = end === start + 1 ? String(end) : (start + 1) + "–" + end;
    ui("status").textContent = matches.length ?
      range + " of " + matches.length + (matches.length === 1 ? " matching image" : " matching images") :
      "No matching images";
    ui("table-scroll").scrollTop = 0;
  }

  async function loadRows(split, queue) {
    const splits = queue ? [...new Set(queue.map(item => item.split))] : [split];
    ui("scope").textContent = queue ? "Custom queue · source labels" : split;
    rows = [];
    page = 0;
    group = "";
    if (picker) picker.loading = true;
    renderTable();
    ui("status").textContent = "Loading statistics…";
    tableRequest?.abort();
    const controller = new AbortController();
    tableRequest = controller;
    try {
      const datasets = await Promise.all(splits.map(name =>
        fetchJSON("/api/audit/" + name, {signal: controller.signal})));
      if (controller.signal.aborted) return;
      const sourceRows = datasets.flatMap(({rows: source, reviews}, index) => source.map((row, position) => ({
        ...row, position, split: splits[index], review: reviews[row.image_id] ?? emptyReview,
      })));
      if (queue) {
        const byId = new Map(sourceRows.map(row => [imageKey(row), row]));
        rows = queue.map((item, position) => ({...byId.get(imageKey(item)), position}));
      } else {
        rows = sourceRows;
      }
      if (picker) picker.loading = false;
      renderTable();
    } catch (error) {
      if (controller.signal.aborted) return;
      if (picker) picker.loading = false;
      updatePicker();
      ui("status").textContent = error.message;
    }
  }

  function chooseSplit(split) {
    for (const branch of ui("splits").querySelectorAll("button")) {
      branch.setAttribute("aria-current", branch.dataset.split === split ? "true" : "false");
    }
    ui("range").value = "";
    loadRows(split, null);
  }

  function openTable(selecting = false) {
    picker = selecting ? {selected: new Map(), visible: [], loading: false, saving: false} : null;
    ui("picker").hidden = !picker;
    ui("dialog-label").textContent = picker ? "Add images to set" : "Image table";
    ui("picker-status").textContent = "";
    dialog.showModal();
    if (picker) chooseSplit(splitSelect.value);
    else loadRows(splitSelect.value, editor.inQueue() ? editor.queue() : null);
  }

  function init() {
    const joints = label => [new Option(label, ""),
      ...dataset.joints.map((_, index) => new Option(jointName(index), index))];
    const options = map => Object.entries(map).map(([key, label]) => new Option(label, key));
    ui("joint").replaceChildren(...joints("All joints"));
    ui("tag-joint").replaceChildren(...joints("Whole person"));
    ui("cohort").replaceChildren(...options(dataset.audit.cohorts));
    ui("cohort-filter").replaceChildren(new Option("Any", ""), new Option("Unassigned", "unassigned"),
      ...options(dataset.audit.cohorts).filter(option => option.value));
    ui("kind").replaceChildren(...options(dataset.audit.tags));
    ui("tag-filter").replaceChildren(new Option("Any", ""), ...options(dataset.audit.tags));
    ui("person").onchange = updateTarget;
    ui("kind").onchange = updateTarget;
    ui("cohort").onchange = () => saveReview({action: "cohort", value: value("cohort")});
    ui("reviewed").onchange = () => saveReview({action: "reviewed", value: ui("reviewed").checked});
    ui("add").onclick = () => saveReview({
      action: "tag", kind: value("kind"), person: value("person") ? Number(value("person")) : null,
      joint: value("tag-joint") ? Number(value("tag-joint")) : null,
    });
    ui("open").disabled = false;
    ui("open").onclick = () => openTable();
    element("queue-picker").onclick = () => openTable(true);
    for (const option of splitSelect.options) {
      const branch = document.createElement("li");
      const toggle = button(option.textContent, () => chooseSplit(option.value));
      toggle.dataset.split = option.value;
      branch.append(toggle);
      ui("splits").append(branch);
    }
    ui("range-form").onsubmit = event => {
      event.preventDefault();
      if (!picker || picker.loading || picker.saving || !rows.length) return;
      try {
        const queued = queuedKeys();
        const available = row => !queued.has(imageKey(row)) && !picker.selected.has(imageKey(row));
        const result = selectAuditRows(value("range"), rows, picker.matching, available);
        selectRows(result.rows);
        const count = result.rows.length;
        if (result.requested !== undefined && count < result.requested) {
          ui("picker-status").textContent = count + " of " + result.requested +
            " selected. " + (count ? "That’s all that matched." : "No eligible images match.");
        } else {
          ui("picker-status").textContent = count ? count + " selected." : "No new images to select.";
        }
      } catch (error) {
        ui("picker-status").textContent = error.message;
      }
    };
    ui("select-page").onclick = () => selectRows(picker.visible);
    ui("clear-selection").onclick = () => { picker.selected.clear(); renderTable(); };
    ui("add-selection").onclick = addSelection;
    dialog.oncancel = event => { if (picker?.saving) event.preventDefault(); };
    ui("close").onclick = () => { if (!picker?.saving) dialog.close(); };
    dialog.onclose = () => { if (!dialog.open) tableRequest?.abort(); };
    ui("filters").onchange = () => { page = 0; group = ""; renderTable(); };
    ui("clear").onclick = () => {
      ui("filters").querySelectorAll("select").forEach(select => { select.selectedIndex = 0; });
      ui("filters").onchange();
    };
    ui("prev").onclick = () => { page--; renderTable(); };
    ui("next").onclick = () => { page++; renderTable(); };
  }

  return {init, showReview, draw, setFocus};
})();
