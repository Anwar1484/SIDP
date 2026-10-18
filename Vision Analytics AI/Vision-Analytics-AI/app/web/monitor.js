"use strict";
const $ = id => document.getElementById(id);
const canvas = $("feed"), ctx = canvas.getContext("2d");
const controls = ["boxes", "skeleton", "labels", "confidence", "trackIds"];
const bones = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
const percent = value => Number.isFinite(value) ? `${Math.round(value * 100)}%` : "—";
const escape = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let socket, reconnectTimer, observation, bitmap, pendingPacket, decoding = false;
let cameraState = "connecting", connected = false, busy = false, lastFrameAt = 0;
let generation = 0, cardsKey = "", lastSession = null, frameRates = [];
let statusInitialized = false;

function notice(message = "") {
  $("notice").hidden = !message;
  $("notice").textContent = message;
}

function updateButtons() {
  const active = cameraState === "running" || cameraState === "starting";
  $("cameraButton").disabled = !connected || busy || cameraState === "stopping";
  $("cameraButton").dataset.running = String(active);
  $("cameraButton").querySelector("use").setAttribute("href", active ? "#i-stop" : "#i-play");
  $("cameraButton").querySelector("span").textContent = busy ? "Please wait…" : active ? "Stop monitoring" : "Start monitoring";
  $("cameraIndex").disabled = active || busy;
  $("quality").disabled = active || busy;
  $("resetBaseline").disabled = !connected || busy || cameraState !== "running";
}

function clearPreview(title, message) {
  generation += 1;
  pendingPacket = null;
  observation = null;
  if (bitmap) { bitmap.close(); bitmap = null; }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("stageEmpty").hidden = false;
  $("emptyTitle").textContent = title;
  $("emptyMessage").textContent = message;
  $("feedBadge").hidden = true;
  $("frameGuidance").hidden = true;
  $("fps").textContent = $("latency").textContent = $("frameAge").textContent = "—";
  $("people").textContent = $("trackCount").textContent = "0";
  renderPeople([]);
  frameRates = [];
  $("sparkline").setAttribute("points", "");
}

function applyStatus(status) {
  connected = true;
  const next = status.camera;
  if (next !== cameraState) {
    cameraState = next;
    if (["stopped", "disabled", "failed", "stopping"].includes(next)) {
      clearPreview(next === "failed" ? "Camera unavailable." : next === "stopping" ? "Releasing the camera…" : "Your monitoring space.",
        next === "failed" ? status.camera_error || "Check the camera source and close other apps using it." : "Start the camera to see people and their movement.");
    }
    if (next === "starting") {
      $("stageEmpty").hidden = false;
      $("emptyTitle").textContent = "Opening your camera…";
      $("emptyMessage").textContent = "The first frame will appear here shortly.";
    }
  }
  $("sessionState").dataset.state = next;
  $("sessionLabel").textContent = {running:"Monitoring", starting:"Starting", stopped:"Camera stopped", disabled:"Ready to start", stopping:"Stopping", failed:"Camera unavailable"}[next] || "Ready";
  $("cameraTitle").textContent = `Camera ${status.camera_index}`;
  $("sourceDetail").textContent = next === "running" ? "Live camera · independent person tracks" : "Live camera · local processing";
  $("device").textContent = status.device || "CPU";
  if (!busy && (!statusInitialized || ["running", "starting"].includes(next))) {
    $("cameraIndex").value = String(status.camera_index);
    if ([416,512,640].includes(status.image_size)) $("quality").value = String(status.image_size);
  }
  statusInitialized = true;
  const models = status.models || [];
  $("modelGrid").innerHTML = models.map(model => `<div class="model-cell"><div class="model-role">${escape(model.role)}</div><div class="model-name"><span class="small-dot"></span>${escape(model.name)}</div><div class="model-status">${model.enabled === false ? "Disabled" : model.error ? escape(model.error) : model.ready ? "Loaded · local inference" : "Unavailable"}</div></div>`).join("");
  const enabled = models.filter(m => m.enabled !== false);
  const ready = enabled.filter(m => m.ready && !m.error).length;
  $("pipelineSummary").textContent = `${ready} / ${enabled.length} models ready`;
  const missing = enabled.filter(m => !m.ready || m.error);
  if (next === "failed") notice(status.camera_error || "The camera could not start.");
  else if (missing.length) notice(`${missing.map(m => m.name).join(", ")} unavailable. Run setup_windows.ps1 and restart the monitor. Details are in Model pipeline.`);
  else if (!busy) notice();
  updateButtons();
}

function riskDisplay(person) {
  const risk = person.risk || {};
  if (person.identity_uncertain) return {label:"Tracking overlap", hint:"Waiting for the people to separate.", severity:"", value:null};
  if (risk.status === "calibrating") return {label:"Preparing baseline", hint:"Stand briefly with shoulders and hips visible.", severity:"", value:risk.calibration_progress};
  if (risk.status === "collecting") return {label:"Reading movement", hint:"Collecting a short movement sequence.", severity:"", value:Math.min(1, (risk.window_frames || 0)/15)};
  if (risk.status === "pose_incomplete") return {label:"Full view needed", hint:"Step back so shoulders, hips and legs are visible.", severity:"", value:null};
  if (risk.status !== "ready") return {label:"Temporal model offline", hint:"Check Model pipeline for setup details.", severity:"", value:null};
  const state = risk.advisory || risk.state;
  if (state === "FALL") return {label:"Fall detected", hint:"Temporal model evidence · review the live view.", severity:"high", value:risk.fall_confidence};
  if (state === "PRE_FALL") return {label:"Pre-fall advisory", hint:"The model sees a pre-fall movement pattern.", severity:"watch", value:risk.prefall_confidence};
  if (state === "UNASSESSED") return {label:"Reading movement", hint:"Waiting for enough usable pose evidence.", severity:"", value:null};
  return {label:"Monitoring", hint:"", severity:"", value:risk.prefall_confidence};
}

function renderPeople(people) {
  const key = JSON.stringify(people.map(p => [p.track_id, p.activity_label, percent(p.activity_confidence), percent(p.detection_confidence),
    p.posture, p.movement_state, p.risk?.status, p.risk?.state, p.risk?.advisory,
    percent(p.risk?.prefall_confidence), percent(p.risk?.fall_confidence), percent(p.risk?.calibration_progress),
    p.risk?.window_frames, p.identity_uncertain, percent(p.pose_coverage), p.scene?.status]));
  if (key === cardsKey) return;
  cardsKey = key;
  if (!people.length) {
    $("personCards").innerHTML = '<div class="people-empty"><svg class="icon"><use href="#i-person"/></svg><p>No people in view</p><span>Tracks appear here as people enter the frame.</span></div>';
    return;
  }
  $("personCards").innerHTML = people.map(person => {
    const r = riskDisplay(person), risk = person.risk || {};
    const source = person.activity_confidence == null ? "Person detection · YOLO26" : person.activity_source === "RelateAnything" ? "Scene confidence · RelateAnything" : "Posture confidence · RGB model";
    const confidence = person.activity_confidence ?? person.detection_confidence;
    const scoreRow = risk.status === "ready" ? `<div class="risk-scores"><span>Pre-fall ${percent(risk.prefall_confidence)}</span><span>Fall ${percent(risk.fall_confidence)}</span></div>` : "";
    const posture = person.posture === "UNKNOWN" ? "Pose in view" : person.posture.charAt(0) + person.posture.slice(1).toLowerCase();
    const motion = {STATIONARY:"Still",MOVING:"Moving",UNKNOWN:"Acquiring"}[person.movement_state] || "Acquiring";
    return `<article class="person-card" data-severity="${r.severity}"><div class="person-top"><div class="avatar">P${String(person.track_id).padStart(2,"0")}</div><div><div class="person-name">Person ${person.track_id}</div><div class="person-activity">${escape(person.activity_label)}</div></div><div class="person-confidence">${percent(confidence)}</div></div><div class="source-caption">${source}</div><div class="risk-line" style="margin-top:13px"><span class="risk-tag ${r.severity}">${r.label}</span><span class="risk-value">${risk.status === "calibrating" ? percent(risk.calibration_progress) : ""}</span></div><div class="risk-meter"><span style="width:${Math.round((r.value || 0)*100)}%"></span></div>${scoreRow}${r.hint ? `<div class="card-hint">${r.hint}</div>` : ""}<div class="person-foot"><span>${escape(posture)} · ${motion}</span><span>Pose coverage ${percent(person.pose_coverage)}</span></div></article>`;
  }).join("");
}

function drawOverlay() {
  if (!bitmap || !observation) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  const scale = canvas.width/640;
  for (const person of observation.persons) {
    const [x1,y1,x2,y2] = person.bbox;
    const severe = riskDisplay(person).severity;
    ctx.save();
    ctx.lineJoin = "round";
    if ($("skeleton").checked) {
      for (const [a,b] of bones) {
        const ka = person.keypoints[a], kb = person.keypoints[b];
        if (!ka.valid || !kb.valid) continue;
        ctx.beginPath(); ctx.moveTo(ka.x, ka.y); ctx.lineTo(kb.x, kb.y);
        ctx.strokeStyle = "rgba(0,0,0,.75)"; ctx.lineWidth = 3.5*scale; ctx.stroke();
        ctx.strokeStyle = "rgba(255,255,255,.88)"; ctx.lineWidth = 1.6*scale; ctx.stroke();
      }
      for (const keypoint of person.keypoints) if (keypoint.valid) {
        ctx.beginPath(); ctx.arc(keypoint.x, keypoint.y, 2.2*scale, 0, Math.PI*2);
        ctx.fillStyle = "#f5f5f5"; ctx.fill(); ctx.strokeStyle = "rgba(0,0,0,.8)"; ctx.lineWidth = scale; ctx.stroke();
      }
    }
    if ($("boxes").checked) {
      ctx.strokeStyle = "rgba(0,0,0,.6)"; ctx.lineWidth = (severe ? 4 : 3)*scale; ctx.strokeRect(x1,y1,x2-x1,y2-y1);
      ctx.strokeStyle = severe ? "#fff" : "rgba(255,255,255,.8)"; ctx.lineWidth = (severe ? 2 : 1)*scale; ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    }
    const parts = [];
    if ($("trackIds").checked) parts.push(`P${String(person.track_id).padStart(2,"0")}`);
    if ($("labels").checked) parts.push(person.activity_label);
    if ($("confidence").checked) parts.push(`${percent(person.activity_confidence ?? person.detection_confidence)}${person.activity_confidence == null ? " detection" : ""}`);
    if (severe) parts.push(severe === "high" ? "FALL DETECTED" : "PRE-FALL ADVISORY");
    if (parts.length) {
      ctx.font = `500 ${11*scale}px "Segoe UI", Arial, sans-serif`;
      let text = parts.join("  ·  ");
      const limit = canvas.width-20*scale;
      while (ctx.measureText(text).width > limit && text.length > 12) text = text.slice(0,-2);
      const tw = ctx.measureText(text).width + 16*scale, th = 25*scale;
      const x = Math.max(4*scale, Math.min(x1, canvas.width-tw-4*scale));
      const y = Math.max(4*scale, y1-th-4*scale);
      ctx.fillStyle = severe === "high" ? "rgba(245,245,245,.96)" : "rgba(12,12,12,.82)";
      ctx.fillRect(x,y,tw,th);
      ctx.fillStyle = severe === "high" ? "#111" : "#f2f2f2";
      ctx.textBaseline = "middle"; ctx.fillText(text, x+8*scale, y+th/2);
    }
    ctx.restore();
  }
}

function showObservation(data, image) {
  if (lastSession !== data.session_id) {
    lastSession = data.session_id; frameRates = []; cardsKey = "";
  }
  if (bitmap) bitmap.close();
  bitmap = image; observation = data; lastFrameAt = performance.now();
  if (canvas.width !== data.frame_width || canvas.height !== data.frame_height) {
    canvas.width = data.frame_width; canvas.height = data.frame_height;
    $("stage").style.aspectRatio = `${data.frame_width} / ${data.frame_height}`;
  }
  $("stageEmpty").hidden = true; $("feedBadge").hidden = false;
  cameraState = "running";
  $("sessionState").dataset.state = "running"; $("sessionLabel").textContent = "Monitoring";
  $("fps").textContent = data.fps > 0 ? data.fps.toFixed(1) : "—";
  $("latency").textContent = Math.round(data.processing_latency_ms);
  const transfer = Math.max(0, Date.now() - Date.parse(data.timestamp));
  $("frameAge").textContent = Math.round((data.capture_to_publish_ms || data.processing_latency_ms) + transfer);
  $("people").textContent = $("trackCount").textContent = data.persons.length;
  $("feedPeople").textContent = `${data.persons.length} ${data.persons.length === 1 ? "person" : "people"}`;
  $("resolution").textContent = `${data.frame_width} × ${data.frame_height}`;
  frameRates.push(data.fps || 0); if (frameRates.length > 50) frameRates.shift();
  const max = Math.max(24, ...frameRates);
  $("sparkline").setAttribute("points", frameRates.map((v,i) => `${i*280/Math.max(1,frameRates.length-1)},${32-v/max*29}`).join(" "));
  const guidance = data.persons.map(riskDisplay).find(r => r.hint && !r.severity);
  $("frameGuidance").hidden = !guidance;
  $("frameGuidance").textContent = guidance?.hint || "";
  renderPeople(data.persons); drawOverlay(); updateButtons();
}

async function decodeLatest() {
  if (decoding) return;
  decoding = true;
  try {
    while (pendingPacket) {
      const packet = pendingPacket, token = generation;
      pendingPacket = null;
      if (packet.byteLength < 4) continue;
      const length = new DataView(packet).getUint32(0, true);
      if (length > packet.byteLength-4 || length > 2_000_000) continue;
      const data = JSON.parse(new TextDecoder().decode(new Uint8Array(packet, 4, length)));
      const image = await createImageBitmap(new Blob([new Uint8Array(packet,4+length)], {type:"image/jpeg"}));
      if (token !== generation || !connected) { image.close(); continue; }
      showObservation(data, image);
    }
  } catch (error) { console.error("Preview decode failed", error); }
  finally { decoding = false; }
}

function connect() {
  clearTimeout(reconnectTimer);
  socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/monitor`);
  socket.binaryType = "arraybuffer";
  socket.onmessage = event => {
    if (typeof event.data === "string") {
      try { const status = JSON.parse(event.data); if (status.type === "status") applyStatus(status); }
      catch (error) { console.error(error); }
    } else { pendingPacket = event.data; void decodeLatest(); }
  };
  socket.onclose = () => {
    connected = false; cameraState = "offline";
    $("sessionLabel").textContent = "Reconnecting";
    $("sessionState").dataset.state = "offline";
    clearPreview("Monitor disconnected.", "Reconnecting to the local application…");
    updateButtons(); reconnectTimer = setTimeout(connect, 1500);
  };
  socket.onerror = () => socket.close();
}

async function command(path, payload) {
  const response = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload || {})});
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || "The camera operation could not complete.");
  return result;
}

$("cameraButton").addEventListener("click", async () => {
  busy = true; updateButtons(); notice();
  try {
    const active = cameraState === "running" || cameraState === "starting";
    const status = await command(active ? "/api/camera/stop" : "/api/camera/start", active ? {} : {index:Number($("cameraIndex").value), image_size:Number($("quality").value)});
    applyStatus(status);
  } catch (error) { notice(error.message); }
  finally { busy = false; updateButtons(); }
});
$("resetBaseline").addEventListener("click", async () => {
  try { await command("/api/session/reset"); notice("Baseline reset. Stand briefly with shoulders and hips visible."); }
  catch (error) { notice(error.message); }
});
$("cameraIndex").addEventListener("change", () => { $("cameraTitle").textContent = `Camera ${$("cameraIndex").value}`; });
$("fullscreen").addEventListener("click", () => {
  if (document.fullscreenElement) void document.exitFullscreen();
  else void $("stage").requestFullscreen().catch(() => notice("Fullscreen is unavailable in this browser."));
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && document.fullscreenElement) {
    void document.exitFullscreen().catch(() => notice("Fullscreen could not close. Use the browser controls."));
  }
});
for (const name of controls) {
  const saved = localStorage.getItem(`sidp.${name}`); if (saved !== null) $(name).checked = saved === "true";
  $(name).addEventListener("change", () => { localStorage.setItem(`sidp.${name}`, String($(name).checked)); drawOverlay(); });
}
setInterval(() => {
  $("clock").textContent = new Date().toLocaleTimeString("en-GB");
  if (cameraState === "running" && lastFrameAt && performance.now()-lastFrameAt > 3000) {
    $("sessionLabel").textContent = "Waiting for frames";
    clearPreview("Waiting for a fresh frame.", "Check the camera connection if the preview does not resume.");
  }
}, 1000);
window.addEventListener("beforeunload", () => { clearTimeout(reconnectTimer); if (socket) socket.close(); });
connect();
