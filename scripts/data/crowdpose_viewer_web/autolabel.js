"use strict";

function initAutolabel({getSet, refresh, prepare}) {
  const dialog = element("autolabel-window");
  const run = element("autolabel-run");
  const summary = element("autolabel-summary");
  const rows = new Map();
  const profiles = new Map();
  const model = element("autolabel-model");
  const confidence = element("autolabel-confidence");
  const jointConfidence = element("autolabel-joint-confidence");
  const current = () => profiles.get(model.value);
  const key = item => item.split + "/" + item.image_id;
  let busy = false;
  let timer;

  function updateSelection() {
    const selected = [...rows.values()].filter(row => row.input.checked).length;
    const profile = current();
    const settingsValid = confidence.checkValidity() &&
      (profile?.joint_confidence == null || jointConfidence.checkValidity());
    run.disabled = busy || !selected || !profile?.available || !settingsValid;
    run.textContent = selected ? "Run " + (profile?.label ?? "autolabel") + " (" + selected + ")" :
      "Run " + (profile?.label ?? "autolabel");
  }

  function changeModel() {
    const profile = current();
    confidence.value = profile?.confidence ?? "";
    jointConfidence.value = profile?.joint_confidence ?? "";
    element("autolabel-joint-setting").hidden = profile?.joint_confidence == null;
    element("autolabel-description").textContent = profile?.available ? profile.geometry + " · editable proposals" :
      "Runtime or checkpoint unavailable.";
    updateSelection();
  }

  function show(job) {
    busy = job.running;
    const results = new Map(job.items.map(item => [key(item), item]));
    for (const image of getSet().images) {
      const row = rows.get(key(image));
      if (!row) continue;
      const item = results.get(key(image));
      const blocked = image.reviewed || image.people.length > 0;
      row.input.disabled = busy || blocked;
      if (blocked) row.input.checked = false;
      row.status.textContent = item ? [item.status, item.detail].filter(Boolean).join(" · ") :
        image.reviewed ? "Reviewed" : blocked ? "Already labeled" : "Eligible";
    }
    const complete = job.items.filter(i => !["queued", "processing"].includes(i.status)).length;
    summary.textContent = job.items.length ?
      (job.label ? job.label + " · " : "") + complete + " / " + job.items.length +
      (busy ? " finished · running" : " finished") : "";
    element("autolabel-all").disabled = element("autolabel-none").disabled = busy;
    element("autolabel-close").disabled = busy;
    model.disabled = confidence.disabled = jointConfidence.disabled = busy;
    updateSelection();
  }

  function listImages() {
    const list = element("autolabel-images");
    list.replaceChildren();
    rows.clear();
    for (const image of getSet().images) {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.onchange = updateSelection;
      const status = document.createElement("span");
      label.append(input, document.createTextNode(key(image)), status);
      list.append(label);
      rows.set(key(image), {input, status, image});
    }
  }

  function failed(error) {
    busy = false;
    element("autolabel-close").disabled = false;
    run.disabled = true;
    summary.textContent = error.message + " Reopen this window to refresh.";
  }

  async function poll() {
    try {
      const job = await fetchJSON("/api/autolabel/" + getSet().id);
      if (!dialog.open) return;
      if (!job.running) await refresh();
      show(job);
      if (job.running) timer = setTimeout(poll, 900);
    } catch (error) { failed(error); }
  }

  element("autolabel-open").onclick = async () => {
    if (!prepare()) return;
    dialog.showModal();
    busy = true;
    element("autolabel-close").disabled = true;
    run.disabled = true;
    summary.textContent = "Loading images…";
    try {
      const previous = model.value;
      const catalog = await fetchJSON("/api/autolabel/profiles");
      profiles.clear();
      for (const profile of catalog) profiles.set(profile.id, profile);
      model.replaceChildren(...catalog.map(profile => {
        const option = new Option(profile.label + (profile.available ? "" : " · unavailable"), profile.id);
        option.disabled = !profile.available;
        return option;
      }));
      model.value = profiles.get(previous)?.available ? previous :
        profiles.get("yolo26-pose")?.available ? "yolo26-pose" : catalog.find(profile => profile.available)?.id ?? "";
      changeModel();
      await refresh();
      listImages();
      await poll();
    } catch (error) { failed(error); }
  };
  dialog.oncancel = event => { if (busy) event.preventDefault(); };
  dialog.onclose = () => clearTimeout(timer);
  element("autolabel-close").onclick = () => dialog.close();
  for (const [id, checked] of [["autolabel-all", true], ["autolabel-none", false]]) {
    element(id).onclick = () => {
      for (const {input} of rows.values()) if (!input.disabled) input.checked = checked;
      updateSelection();
    };
  }
  model.onchange = changeModel;
  confidence.oninput = jointConfidence.oninput = updateSelection;
  run.onclick = async () => {
    if (busy || run.disabled) return;
    const profile = current();
    const settings = {confidence: confidence.valueAsNumber};
    if (profile.joint_confidence != null) settings.joint_confidence = jointConfidence.valueAsNumber;
    const images = [...rows.values()].filter(row => row.input.checked)
      .map(({image}) => ({split:image.split, image_id:image.image_id}));
    const set = getSet();
    busy = true;
    element("autolabel-close").disabled = run.disabled = true;
    try {
      const job = await fetchJSON("/api/autolabel", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({id:set.id, revision:set.revision, images, profile:profile.id, settings}),
      });
      show(job);
      await poll();
    } catch (error) { failed(error); }
  };
}
