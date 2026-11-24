"use strict";

function initCopyAnnotations({getTarget, prepare, copy}) {
  const ui = id => element("copy-" + id);
  const dialog = ui("window");
  const people = ui("people");
  const status = ui("status");
  let target;
  let source;
  let request;
  let saving = false;

  function decisions() {
    const boxes = ui("boxes").checked;
    const joints = ui("joints").checked;
    const selected = [...people.querySelectorAll("input:checked:not(:disabled)")];
    ui("save").disabled = saving || !selected.length || !(boxes || joints);
    ui("save").textContent = selected.length ? "Copy selected (" + selected.length + ")" : "Copy selected";
    return {people: selected.map(input => source.people[Number(input.value)].id), boxes, joints};
  }

  function refreshEligibility() {
    for (const input of people.querySelectorAll("input")) {
      const person = source.people[Number(input.value)];
      input.disabled = !(ui("boxes").checked && person.bbox) &&
        !(ui("joints").checked && person.joints.some(slot => slot !== null));
    }
    decisions();
  }

  async function loadSource() {
    request?.abort();
    const controller = new AbortController();
    request = controller;
    const sourceId = ui("source").value;
    people.replaceChildren();
    ui("all").disabled = true;
    decisions();
    status.textContent = "Loading source…";
    try {
      const result = await fetchJSON("/api/copy-source/" + sourceId + "/" + target.split + "/" + target.image_id,
        {signal: controller.signal});
      if (controller.signal.aborted) return;
      source = result;
      for (const [index, person] of source.people.entries()) {
        const label = document.createElement("label");
        const input = document.createElement("input");
        input.type = "checkbox";
        input.value = index;
        input.onchange = decisions;
        const name = sourceId === "original" ? "Source #" + person.id : "Person " + (index + 1);
        const placed = person.joints.filter(joint => joint?.[2]).length;
        const details = placed + (placed === 1 ? " joint" : " joints") + (person.bbox ? " · Box" : "");
        label.append(input, name, Object.assign(document.createElement("span"), {textContent: details}));
        people.append(label);
      }
      ui("all").disabled = !source.people.length;
      refreshEligibility();
      status.textContent = source.people.length ? "" : "This source has no people for the current image.";
    } catch (error) {
      if (!controller.signal.aborted) status.textContent = error.message;
    }
  }

  ui("open").onclick = async () => {
    if (!prepare()) return;
    target = getTarget();
    dialog.showModal();
    ui("source").replaceChildren(new Option("CrowdPose Ground Truth", "original"));
    const controller = new AbortController();
    request = controller;
    people.replaceChildren();
    ui("all").disabled = true;
    ui("source").disabled = true;
    decisions();
    status.textContent = "Loading sets…";
    try {
      const sets = await fetchJSON("/api/sets", {signal: controller.signal});
      if (controller.signal.aborted) return;
      ui("source").append(...sets.filter(set => set.id !== target.id).map(set => new Option(set.name, set.id)));
      ui("source").disabled = false;
      await loadSource();
    } catch (error) {
      if (!controller.signal.aborted) status.textContent = error.message;
    }
  };
  ui("source").onchange = loadSource;
  ui("boxes").onchange = ui("joints").onchange = refreshEligibility;
  ui("all").onclick = () => {
    for (const input of people.querySelectorAll("input:not(:disabled)")) input.checked = true;
    decisions();
  };
  ui("save").onclick = async () => {
    if (saving || ui("save").disabled) return;
    const options = decisions();
    saving = true;
    ui("save").disabled = true;
    status.textContent = "Copying…";
    try {
      const error = await copy(target, {...options, source: ui("source").value, source_revision: source.revision});
      if (error) status.textContent = error;
      else dialog.close();
    } finally {
      saving = false;
      decisions();
    }
  };
  ui("close").onclick = () => { if (!saving) dialog.close(); };
  dialog.oncancel = event => { if (saving) event.preventDefault(); };
  dialog.onclose = () => request?.abort();
}
