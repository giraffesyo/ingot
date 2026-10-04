"use strict";

const $ = (id) => document.getElementById(id);
const form = $("generate-form");
let references = [];
let current = null;
let connected = false;
let submitting = false;
let lastImage = "";
let previewVersion = 0;
let previewError = "";
let pollTimer;
let pollVersion = 0;
let queue = { jobs: [], paused: true, batch_size: 4 };
let selectedID = null;
let jobSignature = "";
let mutating = false;
let outputInitialized = false;
let folderVersion = 0;
let folderListing = null;

function message(id, text) {
  $(id).textContent = text;
  $(id).hidden = !text;
}

function isRunning() {
  return current && ["running", "cancelling"].includes(current.status);
}

function updateControls() {
  $("settings").disabled = submitting;
  $("generate").disabled = submitting || !connected;
  $("generate").textContent = submitting ? "Adding…" : connected ? "Add to queue" : "Reconnecting…";
  $("cancel").hidden = !current || terminal(current.status);
  $("cancel").disabled = !connected || mutating || current?.status === "cancelling";
  $("cancel").textContent = current?.status === "cancelling" ? "Cancellation requested…" : "Cancel selected job";
  $("reuse").hidden = !current;
  $("toggle-queue").disabled = !connected || mutating;
  $("toggle-queue").textContent = queue.paused ? "Start queue" : "Pause queue";
  $("batch-size").disabled = !connected || mutating;
  $("clear-finished").disabled = !connected || mutating || !queue.jobs.some((job) => terminal(job.status));
}

function terminal(status) { return ["completed", "cancelled", "failed"].includes(status); }

function renderQueue(state) {
  queue = state;
  if (!outputInitialized) {
    $("output").value = state.output;
    outputInitialized = true;
  }
  $("batch-size").value = state.batch_size;
  $("queue-count").textContent = state.jobs.filter((job) => !terminal(job.status)).length;
  const active = state.jobs.filter((job) => ["running", "cancelling"].includes(job.status)).length;
  $("queue-hint").textContent = state.paused
    ? active ? "Pausing after the current batch. Admitted jobs finish together." : "Queue is paused. Add a few jobs to share model loads."
    : active ? `Processing ${active} jobs together. New arrivals join the next batch.` : "Queue is running. New jobs will start automatically.";
  $("queue-empty").hidden = state.jobs.length > 0;
  if (!state.jobs.some((job) => job.id === selectedID)) selectedID = state.jobs.at(-1)?.id || null;
  const signature = JSON.stringify(state.jobs.map((job) => [job.id, job.status, job.settings.prompt, job.settings.seed])) + selectedID;
  if (signature !== jobSignature) {
    const focused = document.activeElement?.dataset.job;
    const scroll = $("jobs").scrollTop;
    jobSignature = signature;
    $("jobs").replaceChildren(...state.jobs.map((job, index) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "job-select";
      button.dataset.job = job.id;
      button.setAttribute("aria-pressed", String(job.id === selectedID));
      const number = document.createElement("span"); number.className = "job-number"; number.textContent = index + 1;
      const body = document.createElement("span");
      const prompt = document.createElement("span"); prompt.className = "job-prompt"; prompt.textContent = job.settings.prompt;
      const meta = document.createElement("span"); meta.className = "job-meta"; meta.textContent = `Seed ${job.settings.seed} · ${job.settings.steps} steps`;
      body.append(prompt, meta);
      const status = document.createElement("span"); status.className = "job-status"; status.dataset.status = job.status;
      status.textContent = { queued: "Queued", running: "In batch", cancelling: "Cancelling", cancelled: "Cancelled", completed: "Ready", failed: "Failed" }[job.status];
      button.append(number, body, status);
      button.addEventListener("click", () => { selectedID = job.id; renderQueue(queue); });
      item.append(button);
      return item;
    }));
    $("jobs").scrollTop = scroll;
    if (focused) [...$("jobs").querySelectorAll("button")].find((button) => button.dataset.job === focused)?.focus({ preventScroll: true });
  }
  renderJob(state.jobs.find((job) => job.id === selectedID) || null);
}

async function request(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { "X-Ingot-UI": "1", ...options.headers } });
  const text = await response.text();
  let data;
  try { data = JSON.parse(text); } catch { throw new Error(text || `Request failed (${response.status}).`); }
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}

function restoreSettings(settings) {
  for (const [key, value] of Object.entries(settings)) {
    if (!$(key)) continue;
    if (key === "fast") $(key).checked = value;
    else $(key).value = value;
  }
}

function renderJob(job) {
  current = job;
  message("job-output", job ? `Output folder: ${job.settings.output}` : "");

  updateControls();
  $("progress").hidden = !isRunning();
  if (!job) {
    $("status").textContent = "Select a queued job to see its progress.";
    $("log").textContent = "Stage timings and model messages will appear here.";
    message("run-error", "");
    $("result").hidden = true;
    $("empty").hidden = false;
    $("download").hidden = true;
    $("result-meta").textContent = "";
    clearPreview();
    $("elapsed").textContent = "";
    return;
  }
  $("log").textContent = job.log || "Waiting for this job’s turn.";
  message("run-error", job.error || previewError);
  const steps = [...job.log.matchAll(/step\s+(\d+)\/(\d+)/g)];
  const latest = steps.at(-1);
  if (latest) {
    $("progress").max = Number(latest[2]);
    $("progress").value = Number(latest[1]);
  } else {
    $("progress").removeAttribute("value");
  }
  const stages = [...job.log.matchAll(/stage: (.+)/g)];
  const stage = stages.at(-1)?.[1];
  const labels = {
    queued: queue.paused ? "Queued. Start or resume the queue to run this image." : "Queued · waiting for the next available batch.",
    running: stage === "decode" ? "Decoding image…" : latest ? Number(latest[1]) === Number(latest[2]) ? "Denoised · waiting for batch decode…" : `Generating · step ${latest[1]} of ${latest[2]}` : stage ? `Processing ${stage}…` : "Admitted to batch. Waiting for shared model stage…",
    cancelling: "Cancelling after the current step or stage finishes…",
    cancelled: "Generation cancelled. Adjust your prompt and try again.",
    completed: "Image ready.",
    failed: "Generation couldn’t finish.",
  };
  $("status").textContent = labels[job.status] || job.status;
  if (job.status === "failed") $("log-details").open = true;
  if (job.status === "completed") {
    if (job.image !== lastImage) loadPreview(job);
  } else if (lastImage) {
    clearPreview();
    message("run-error", job.error || "");
  }
  updateElapsed();
}

function clearPreview() {
  previewVersion += 1;
  previewError = "";
  lastImage = "";
  $("result").hidden = true;
  $("result").removeAttribute("src");
  $("empty").hidden = false;
  $("download").hidden = true;
  $("retry-preview").hidden = true;
  $("result-meta").textContent = "";
}

function loadPreview(job) {
  clearPreview();
  lastImage = job.image;
  message("run-error", job.error || "");
  const version = previewVersion;
  const image = new Image();
  const stillSelected = () => version === previewVersion && selectedID === job.id && current?.image === job.image;
  image.onload = () => {
    if (!stillSelected()) return;
    $("result").src = image.src;
    $("result").alt = `Generated image: ${job.settings.prompt}`;
    $("result").hidden = false;
    $("empty").hidden = true;
    $("download").href = job.image;
    $("download").download = `qwenimage-${job.id.slice(0, 8)}.png`;
    $("download").hidden = false;
    $("result-meta").textContent = `${image.naturalWidth} × ${image.naturalHeight} px · ${job.settings.steps} steps · Seed ${job.settings.seed}`;
  };
  image.onerror = () => {
    if (!stillSelected()) return;
    previewError = "The preview could not load. Retry it, or find the saved PNG in the output directory.";
    message("run-error", previewError);
    $("retry-preview").hidden = false;
  };
  image.src = job.image;
}

$("retry-preview").addEventListener("click", () => {
  if (current?.image) loadPreview(current);
});

$("output").addEventListener("input", () => { outputInitialized = true; });

async function browseFolder(path) {
  const version = ++folderVersion;
  folderListing = null;
  $("folder-use").disabled = true;
  $("folder-up").disabled = true;
  $("folder-list").replaceChildren();
  $("folder-path").value = path;
  message("folder-error", "");
  message("folder-status", "Loading folders…");
  try {
    const listing = await request(`/api/folders?path=${encodeURIComponent(path)}`, { signal: AbortSignal.timeout(10000) });
    if (version !== folderVersion || !$("folder-dialog").open) return;
    folderListing = listing;
    $("folder-path").value = listing.path;
    $("folder-up").disabled = listing.parent === listing.path;
    $("folder-use").disabled = false;
    message("folder-status", listing.folders.length ? "Open a subfolder, or use this folder." : "No subfolders. You can use this folder.");
    $("folder-list").replaceChildren(...listing.folders.map((name) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = name;
      button.addEventListener("click", () => browseFolder(`${listing.path}/${name}`));
      item.append(button);
      return item;
    }));
    $("folder-path").focus({ preventScroll: true });
  } catch (error) {
    if (version !== folderVersion || !$("folder-dialog").open) return;
    message("folder-status", "");
    message("folder-error", error.message);
  }
}

$("browse-output").addEventListener("click", () => {
  $("folder-dialog").showModal();
  browseFolder($("output").value);
});
$("folder-form").addEventListener("submit", (event) => {
  event.preventDefault();
  browseFolder($("folder-path").value);
});
$("folder-path").addEventListener("input", () => {
  folderVersion += 1; // Ignore a response for a path the user has already changed.
  $("folder-use").disabled = $("folder-path").value !== folderListing?.path;
  message("folder-status", $("folder-use").disabled ? "Press Go to open this path." : "You can use this folder.");
  message("folder-error", "");
});
$("folder-up").addEventListener("click", () => { if (folderListing) browseFolder(folderListing.parent); });
$("folder-home").addEventListener("click", () => browseFolder("~"));
$("folder-cancel").addEventListener("click", () => $("folder-dialog").close());
$("folder-dialog").addEventListener("close", () => { folderVersion += 1; });
$("folder-use").addEventListener("click", () => {
  if (!folderListing || $("folder-path").value !== folderListing.path) return;
  $("output").value = folderListing.path;
  outputInitialized = true;
  $("folder-dialog").close();
});

function updateElapsed() {
  if (!current?.started || current.status === "queued") { $("elapsed").textContent = ""; return; }
  const end = current.finished ? Date.parse(current.finished) : Date.now();
  const seconds = Math.max(0, Math.floor((end - Date.parse(current.started)) / 1000));
  $("elapsed").textContent = seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}
setInterval(updateElapsed, 1000);

async function poll() {
  const version = ++pollVersion;
  try {
    const state = await request("/api/queue", { signal: AbortSignal.timeout(5000) });
    if (version !== pollVersion) return;
    connected = true;
    renderQueue(state);
  } catch {
    if (version !== pollVersion) return;
    connected = false;
    $("status").textContent = "Can’t reach the local server. Reconnecting…";
    updateControls();
  } finally {
    if (version === pollVersion) pollTimer = setTimeout(poll, queue.jobs.some((job) => !terminal(job.status)) ? 1000 : 2500);
  }
}

function pausePolling() {
  clearTimeout(pollTimer);
  pollVersion += 1; // Ignore stale status responses while submitting a mutation.
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity() || submitting) return;
  message("form-error", "");
  const data = new FormData(form);
  data.set("fast", String($("fast").checked));
  for (const { file } of references) data.append("images", file);
  submitting = true;
  pausePolling();
  updateControls();
  try {
    const state = await request("/api/generate", { method: "POST", body: data });
    selectedID = state.jobs.at(-Number(data.get("count")))?.id;
    renderQueue(state);
    $("added").textContent = `${data.get("count")} added. ${state.paused ? "Start the queue when you’re ready." : "The queue will pick them up."}`;
  } catch (error) {
    message("form-error", error.message);
  } finally {
    submitting = false;
    updateControls();
    poll();
  }
});

async function mutate(path, body) {
  pausePolling();
  mutating = true;
  updateControls();
  message("form-error", "");
  try {
    const options = { method: "POST" };
    if (body) { options.headers = { "Content-Type": "application/json" }; options.body = JSON.stringify(body); }
    renderQueue(await request(path, options));
  } catch (error) {
    message("form-error", error.message);
  } finally {
    mutating = false;
    updateControls();
    poll();
  }
}

$("cancel").addEventListener("click", () => {
  if (current && !terminal(current.status)) mutate(`/api/cancel/${current.id}`);
});
$("toggle-queue").addEventListener("click", () => mutate("/api/queue", { paused: !queue.paused }));
$("batch-size").addEventListener("change", () => mutate("/api/queue", { batch_size: Number($("batch-size").value) }));
$("clear-finished").addEventListener("click", () => mutate("/api/queue", { clear_finished: true }));
$("reuse").addEventListener("click", () => {
  if (!current) return;
  restoreSettings(current.settings);
  for (const ref of references) URL.revokeObjectURL(ref.url);
  references = [];
  renderReferences();
  $("prompt").focus();
  $("added").textContent = "Settings copied. Reattach reference images if this was an edit.";
});

$("random-seed").addEventListener("click", () => {
  const words = crypto.getRandomValues(new Uint32Array(2));
  $("seed").value = ((BigInt(words[0]) << 32n) | BigInt(words[1])).toString();
});

$("images").addEventListener("change", () => {
  const files = [...$("images").files];
  $("images").value = "";
  if (references.length + files.length > 10) {
    message("form-error", "You can add up to 10 reference images. Remove one before adding more.");
    return;
  }
  if (files.some((file) => !["image/png", "image/jpeg"].includes(file.type))) {
    message("form-error", "Reference images must be PNG or JPEG files.");
    return;
  }
  if ([...references.map((r) => r.file), ...files].reduce((total, file) => total + file.size, 0) >= 64 * 1024 * 1024) {
    message("form-error", "Keep reference images below 64 MB in total.");
    return;
  }
  message("form-error", "");
  references.push(...files.map((file) => ({ file, url: URL.createObjectURL(file) })));
  renderReferences();
});

function renderReferences() {
  $("references").replaceChildren();
  references.forEach((ref, index) => {
    const item = document.createElement("li");
    const image = document.createElement("img");
    image.src = ref.url;
    image.alt = "";
    const name = document.createElement("span");
    name.textContent = `${index + 1}. ${ref.file.name}`;
    name.title = ref.file.name;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "secondary";
    remove.textContent = "Remove";
    remove.setAttribute("aria-label", `Remove reference ${index + 1}: ${ref.file.name}`);
    remove.addEventListener("click", () => {
      URL.revokeObjectURL(ref.url);
      references.splice(index, 1);
      renderReferences();
      $("images").focus();
    });
    item.append(image, name, remove);
    $("references").append(item);
  });
}

form.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    if (!$("generate").disabled) form.requestSubmit();
  }
});
poll();
