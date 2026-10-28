import { getAuth, onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-auth.js";
import { equalTo, getDatabase, limitToLast, onValue, orderByChild, push, query, ref, set, update } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-database.js";
import { firebaseApp, firebaseConfig } from "./firebase-config.js";
import { deleteObject, getDownloadURL, getStorage, ref as storageRef, uploadBytes } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-storage.js";
import { demoData } from "./demo-data.js";

const auth = getAuth(firebaseApp);
const searchParams = new URLSearchParams(window.location.search);
const isDemoMode = searchParams.get("demo") === "1";
let isLiveBridgeMode = searchParams.get("live") === "1" || window.location.port === "8502";

const dashboardMessage = document.getElementById("dashboardMessage");
const residentRows = document.getElementById("residentRows");
const residentForm = document.getElementById("residentForm");
const formError = document.getElementById("residentFormError");
const saveResidentButton = document.getElementById("saveResidentButton");
const historyFilterForm = document.getElementById("historyFilterForm");
const historyFilterError = document.getElementById("historyFilterError");

let database;
let residents = {};
let postureByResident = {};
let activityLogs = {};
let selectedResidentId = "";
let activityUnsubscribe;
let appliedDateRange = initializeDateRange();
let storage;
const photoUrlCache = new Map();
let selectedPhotoFile;
let photoPreviewUrl;
let cameraStream;
let liveStreamActive = false;
let livePollTimer;
const demoSnapshotUrls = new Map();

async function probeLocalIcd03Bridge() {
    try {
        const resp = await fetch("/api/icd03/state", { cache: "no-store" });
        if (resp.ok) {
            return await resp.json();
        }
    } catch (_err) {
        // Local ICD-03 bridge not running on this origin
    }
    return null;
}

async function initDashboardMode() {
    if (isDemoMode) {
        residents = demoData.RESIDENTS;
        postureByResident = demoData.CURRENT_POSTURE_STATUS;
        activityLogs = demoData.ACTIVITY_LOGS;
        selectedResidentId = Object.keys(residents)[0] || "";
        document.getElementById("demoModeNotice").classList.remove("d-none");
        document.getElementById("addResidentButton").classList.add("d-none");
        document.getElementById("signOutButton").classList.add("d-none");
        document.getElementById("signedInUser").textContent = "Sample data";
        setConnection("Demo data", "connected");
        syncResidentFilters();
        renderResidents();
        renderSummary();
        renderSelectedResident();
        return;
    }

    const initialBridgeState = await probeLocalIcd03Bridge();
    if (isLiveBridgeMode || initialBridgeState) {
        isLiveBridgeMode = true;
        activateLiveBridgeMode(initialBridgeState);
        return;
    }

    onAuthStateChanged(auth, (user) => {
        if (!user) {
            window.location.replace("login.html");
            return;
        }

        document.getElementById("signedInUser").textContent = user.email || "Signed in";
        connectDatabase();
    }, (error) => {
        showMessage(`Authentication status could not be checked: ${error.message}`, "danger");
    });
}

function activateLiveBridgeMode(initialPayload) {
    const notice = document.getElementById("demoModeNotice");
    if (notice) {
        notice.className = "alert alert-success";
        notice.innerHTML = '<i class="fas fa-satellite-dish mr-1"></i> <strong>ICD-03 Live AI Pipeline Active</strong> — receiving real-time telemetry, SQLite incident logs, and dual video feeds from Subsystems 1, 2 &amp; 3.';
    }
    document.getElementById("signOutButton").classList.add("d-none");
    document.getElementById("signedInUser").textContent = "ICD-03 Live Bridge (Subsystems 1–4)";
    setConnection("ICD-03 Live Connected", "connected");

    if (initialPayload) {
        applyLiveBridgeState(initialPayload);
    }
    startLiveCameraStreams();

    if (livePollTimer) clearInterval(livePollTimer);
    livePollTimer = setInterval(async () => {
        const state = await probeLocalIcd03Bridge();
        if (state) {
            setConnection("ICD-03 Live Connected", "connected");
            applyLiveBridgeState(state);
        } else {
            setConnection("Bridge reconnecting…", "error");
        }
    }, 750);
}

function applyLiveBridgeState(payload) {
    residents = (payload && payload.RESIDENTS && typeof payload.RESIDENTS === "object") ? payload.RESIDENTS : {};
    postureByResident = normalizePostureData(payload ? payload.CURRENT_POSTURE_STATUS : {});
    activityLogs = (payload && payload.ACTIVITY_LOGS && typeof payload.ACTIVITY_LOGS === "object") ? payload.ACTIVITY_LOGS : {};

    const prevSelected = selectedResidentId;
    const residentIds = Object.keys(residents);
    if (!residentIds.includes(selectedResidentId)) {
        selectedResidentId = residentIds[0] || "";
    }

    const selectedEl = document.getElementById("selectedResident");
    const filterEl = document.getElementById("activityResidentFilter");
    if (selectedEl.options.length !== residentIds.length || prevSelected !== selectedResidentId) {
        syncResidentFilters();
    } else {
        renderHistory();
    }
    renderSummary();
    renderResidents();
    renderSelectedResident();
}

function startLiveCameraStreams() {
    const rawImg = document.getElementById("liveRawFeed");
    const skelImg = document.getElementById("liveSkeletonFeed");
    const skelFrame = document.getElementById("skeletonPlaceholder");
    const skelSvg = document.getElementById("skeletonIllustration");
    const skelCopy = document.getElementById("skeletonPlaceholderCopy");

    if (rawImg && skelImg) {
        liveStreamActive = true;
        const ts = Date.now();
        rawImg.src = `/api/icd03/raw.mjpg?t=${ts}`;
        skelImg.src = `/api/icd03/annotated.mjpg?t=${ts}`;
        rawImg.classList.remove("d-none");
        skelImg.classList.remove("d-none");
        document.getElementById("cameraPlaceholder").classList.add("d-none");
        if (skelSvg) skelSvg.classList.add("d-none");
        if (skelCopy) skelCopy.classList.add("d-none");
        if (skelFrame) skelFrame.classList.add("is-streaming");

        document.getElementById("startCameraButton").classList.add("d-none");
        document.getElementById("stopCameraButton").classList.remove("d-none");
        const status = document.getElementById("cameraStatus");
        status.textContent = "Live · Tapo C210 + YOLO26-Pose";
        status.className = "badge badge-success mr-2";
        document.getElementById("cameraMessage").textContent =
            "Streaming live unmasked RTSP video (Subsystem 1) alongside 17-keypoint YOLO26-Pose & Subsystem 3 Safety HUD.";
        renderHistory();
    }
}

initDashboardMode();

document.getElementById("signOutButton").addEventListener("click", async () => {
    try {
        await signOut(auth);
    } catch (error) {
        showMessage(`Could not sign out: ${error.message}`, "danger");
    }
});

async function startCamera() {
    if (isLiveBridgeMode) {
        startLiveCameraStreams();
        return;
    }
    const message = document.getElementById("cameraMessage");
    if (!navigator.mediaDevices?.getUserMedia) {
        message.textContent = "Camera access requires a supported browser and a secure page (HTTPS or localhost).";
        return;
    }

    const startButton = document.getElementById("startCameraButton");
    startButton.disabled = true;
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: { facingMode: { ideal: "environment" } }
        });
        const video = document.getElementById("cameraVideo");
        video.srcObject = cameraStream;
        await video.play();
        video.classList.remove("d-none");
        document.getElementById("cameraPlaceholder").classList.add("d-none");
        document.getElementById("startCameraButton").classList.add("d-none");
        document.getElementById("stopCameraButton").classList.remove("d-none");
        document.getElementById("cameraStatus").textContent = "Live · unmasked";
        document.getElementById("cameraStatus").className = "badge badge-success mr-2";
        message.textContent = "The original feed is displayed without masking and stays in this browser. Snapshots are saved only when attached to a warning.";
        renderHistory();
    } catch (error) {
        stopCamera();
        message.textContent = `Could not start the camera: ${error.message}. Check camera permission and use HTTPS or localhost.`;
    } finally {
        startButton.disabled = false;
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
        cameraStream = undefined;
    }
    liveStreamActive = false;

    const video = document.getElementById("cameraVideo");
    if (video) {
        video.pause();
        video.srcObject = null;
        video.classList.add("d-none");
    }
    const rawImg = document.getElementById("liveRawFeed");
    if (rawImg) {
        rawImg.removeAttribute("src");
        rawImg.classList.add("d-none");
    }
    const skelImg = document.getElementById("liveSkeletonFeed");
    if (skelImg) {
        skelImg.removeAttribute("src");
        skelImg.classList.add("d-none");
    }
    const skelFrame = document.getElementById("skeletonPlaceholder");
    if (skelFrame) skelFrame.classList.remove("is-streaming");
    const skelSvg = document.getElementById("skeletonIllustration");
    if (skelSvg) skelSvg.classList.remove("d-none");
    const skelCopy = document.getElementById("skeletonPlaceholderCopy");
    if (skelCopy) skelCopy.classList.remove("d-none");

    const placeholder = document.getElementById("cameraPlaceholder");
    if (placeholder) placeholder.classList.remove("d-none");
    const startButton = document.getElementById("startCameraButton");
    if (startButton) startButton.classList.remove("d-none");
    const stopButton = document.getElementById("stopCameraButton");
    if (stopButton) stopButton.classList.add("d-none");
    const status = document.getElementById("cameraStatus");
    if (status) {
        status.textContent = "Camera off";
        status.className = "badge badge-secondary mr-2";
    }
    if (residentRows) renderHistory();
}

async function captureWarningSnapshot(record, button) {
    if (isLiveBridgeMode) {
        button.disabled = true;
        try {
            const resp = await fetch("/api/icd03/snapshot", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ log_id: record.id, resident_id: selectedResidentId })
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            if (data.snapshot_url) {
                record.snapshot_url = data.snapshot_url;
                record.snapshot_captured_at = data.snapshot_captured_at || new Date().toISOString();
            }
            historyFilterError.classList.add("d-none");
            renderHistory();
        } catch (err) {
            showHistoryError(`Could not capture snapshot from ICD-03 stream: ${err.message}`);
        } finally {
            button.disabled = false;
        }
        return;
    }

    const video = document.getElementById("cameraVideo");
    if (!cameraStream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        showHistoryError("Start the camera and wait for a live frame before capturing a snapshot.");
        return;
    }
    if (!video.videoWidth || !video.videoHeight) {
        showHistoryError("The camera has not provided a usable frame yet. Try again in a moment.");
        return;
    }
    if (!isDemoMode && (!database || !storage)) {
        showHistoryError("Firebase is not ready. The warning snapshot was not saved.");
        return;
    }

    button.disabled = true;
    try {
        const canvas = document.createElement("canvas");
        const scale = Math.min(1, 1280 / video.videoWidth, 720 / video.videoHeight);
        canvas.width = Math.round(video.videoWidth * scale);
        canvas.height = Math.round(video.videoHeight * scale);
        canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await encodeSnapshot(canvas);
        const capturedAt = new Date().toISOString();

        if (isDemoMode) {
            const previousUrl = demoSnapshotUrls.get(record.id);
            if (previousUrl) URL.revokeObjectURL(previousUrl);
            demoSnapshotUrls.set(record.id, URL.createObjectURL(blob));
            record.demo_snapshot_captured_at = capturedAt;
        } else {
            const path = `POSTURE_ALERT_SNAPSHOTS/${selectedResidentId}/${record.id}.jpg`;
            const imageReference = storageRef(storage, path);
            const previousPath = record.snapshot_storage_path;
            await uploadBytes(imageReference, blob, { contentType: "image/jpeg" });
            try {
                await update(ref(database, `ACTIVITY_LOGS/${record.id}`), {
                    snapshot_storage_path: path,
                    snapshot_captured_at: capturedAt
                });
            } catch (error) {
                await deleteObject(imageReference).catch((cleanupError) => {
                    console.error("Could not remove the snapshot after its history update failed:", cleanupError);
                });
                throw error;
            }
            record.snapshot_storage_path = path;
            record.snapshot_captured_at = capturedAt;
            if (previousPath && previousPath !== path) {
                await deleteObject(storageRef(storage, previousPath)).catch((error) => {
                    console.error("Could not remove the replaced warning snapshot:", error);
                });
            }
        }

        historyFilterError.classList.add("d-none");
        renderHistory();
    } catch (error) {
        showHistoryError(`Could not save the warning snapshot: ${error.message}`);
    } finally {
        button.disabled = false;
    }
}

async function encodeSnapshot(canvas) {
    for (const quality of [0.72, 0.55, 0.4]) {
        const blob = await new Promise((resolve, reject) => {
            canvas.toBlob((imageBlob) => imageBlob
                ? resolve(imageBlob)
                : reject(new Error("The browser could not encode the camera frame.")), "image/jpeg", quality);
        });
        if (blob.size <= 1024 * 1024) return blob;
    }
    throw new Error("The camera snapshot is too large to upload (maximum 1 MB).");
}

async function viewWarningSnapshot(record) {
    const image = document.getElementById("snapshotImage");
    try {
        if (isDemoMode && demoSnapshotUrls.has(record.id)) {
            const url = demoSnapshotUrls.get(record.id);
            image.dataset.demoUrl = url;
            image.src = url;
        } else if (record.snapshot_url) {
            image.src = record.snapshot_url;
        } else if (record.snapshot_storage_path && storage) {
            image.removeAttribute("src");
            image.src = await getDownloadURL(storageRef(storage, record.snapshot_storage_path));
        } else {
            throw new Error("No snapshot is saved for this warning.");
        }
        document.getElementById("snapshotTimestamp").textContent = record.snapshot_captured_at || record.timestamp
            ? `Captured ${formatTimestamp(record.snapshot_captured_at || record.timestamp)}`
            : "Warning snapshot";
        window.jQuery("#snapshotModal").modal("show");
    } catch (error) {
        showHistoryError(`Could not open the warning snapshot: ${error.message}`);
    }
}

document.getElementById("selectedResident").addEventListener("change", (event) => {
    selectedResidentId = event.target.value;
    document.getElementById("activityResidentFilter").value = selectedResidentId;
    renderSelectedResident();
    listenForSelectedActivity();
});

document.getElementById("activityResidentFilter").addEventListener("change", (event) => {
    selectedResidentId = event.target.value;
    document.getElementById("selectedResident").value = selectedResidentId;
    renderSelectedResident();
    listenForSelectedActivity();
});

document.getElementById("resPhoto").addEventListener("change", (event) => {
    const file = event.target.files[0];
    formError.classList.add("d-none");
    if (!file) {
        clearPhotoPreview();
        return;
    }
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type) || file.size > 5 * 1024 * 1024) {
        event.target.value = "";
        clearPhotoPreview();
        formError.textContent = "Choose a JPG, PNG, or WebP image smaller than 5 MB.";
        formError.classList.remove("d-none");
        return;
    }

    selectedPhotoFile = file;
    if (photoPreviewUrl) URL.revokeObjectURL(photoPreviewUrl);
    photoPreviewUrl = URL.createObjectURL(file);
    document.getElementById("photoPreview").src = photoPreviewUrl;
    document.getElementById("photoPreviewWrap").classList.remove("d-none");
});

document.getElementById("clearPhotoButton").addEventListener("click", clearPhotoPreview);
window.jQuery("#addResidentModal").on("hidden.bs.modal", clearPhotoPreview);

historyFilterForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const from = document.getElementById("historyFromDate").value;
    const to = document.getElementById("historyToDate").value;
    if (!from || !to) {
        showHistoryError("Choose both a start date and an end date.");
        return;
    }
    if (from > to) {
        showHistoryError("The start date must be on or before the end date.");
        return;
    }

    appliedDateRange = { from, to };
    historyFilterError.classList.add("d-none");
    renderHistory();
});

document.getElementById("exportHistoryButton").addEventListener("click", exportHistory);
document.getElementById("residentSearch").addEventListener("input", renderResidents);
document.getElementById("startCameraButton").addEventListener("click", startCamera);
document.getElementById("stopCameraButton").addEventListener("click", stopCamera);
window.addEventListener("pagehide", () => {
    stopCamera();
    if (livePollTimer) clearInterval(livePollTimer);
    for (const url of demoSnapshotUrls.values()) URL.revokeObjectURL(url);
});
window.jQuery("#snapshotModal").on("hidden.bs.modal", () => {
    const image = document.getElementById("snapshotImage");
    if (image.dataset.demoUrl) {
        URL.revokeObjectURL(image.dataset.demoUrl);
        image.removeAttribute("src");
        delete image.dataset.demoUrl;
    }
});

residentForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    formError.classList.add("d-none");
    saveResidentButton.disabled = true;

    try {
        if (isLiveBridgeMode) {
            const payload = {
                full_name: document.getElementById("resName").value.trim(),
                age: Number(document.getElementById("resAge").value),
                room_number: document.getElementById("resRoom").value.trim(),
                address: document.getElementById("resAddress").value.trim(),
                risk_category: document.getElementById("resRisk").value,
                created_at: new Date().toISOString()
            };
            const resp = await fetch("/api/icd03/residents", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const updatedState = await probeLocalIcd03Bridge();
            if (updatedState) applyLiveBridgeState(updatedState);
            residentForm.reset();
            clearPhotoPreview();
            window.jQuery("#addResidentModal").modal("hide");
            return;
        }

        const residentRef = push(ref(database, "RESIDENTS"));
        let uploadedPhotoReference;
        let photoStoragePath = "";
        if (selectedPhotoFile) {
            const extension = { "image/jpeg": "jpg", "image/png": "png", "image/webp": "webp" }[selectedPhotoFile.type];
            photoStoragePath = `RESIDENT_PHOTOS/${residentRef.key}/profile.${extension}`;
            uploadedPhotoReference = storageRef(storage, photoStoragePath);
            await uploadBytes(uploadedPhotoReference, selectedPhotoFile, {
                contentType: selectedPhotoFile.type
            });
        }

        try {
            await set(residentRef, {
                full_name: document.getElementById("resName").value.trim(),
                age: Number(document.getElementById("resAge").value),
                room_number: document.getElementById("resRoom").value.trim(),
                address: document.getElementById("resAddress").value.trim(),
                risk_category: document.getElementById("resRisk").value,
                ...(photoStoragePath ? { photo_storage_path: photoStoragePath } : {}),
                created_at: new Date().toISOString()
            });
        } catch (error) {
            if (uploadedPhotoReference) {
                try {
                    await deleteObject(uploadedPhotoReference);
                } catch (cleanupError) {
                    console.error("Could not remove the uploaded profile photo after the resident save failed:", cleanupError);
                }
            }
            throw error;
        }
        residentForm.reset();
        clearPhotoPreview();
        window.jQuery("#addResidentModal").modal("hide");
    } catch (error) {
        formError.textContent = `Could not save the resident profile: ${error.message}`;
        formError.classList.remove("d-none");
    } finally {
        saveResidentButton.disabled = false;
    }
});

function connectDatabase() {
    if (!firebaseConfig.databaseURL) {
        setConnection("Database URL required", "error");
        showMessage("Add your Realtime Database URL to js/firebase-config.js (Firebase Console → Realtime Database) before connecting.", "warning");
        return;
    }

    try {
        database = getDatabase(firebaseApp, firebaseConfig.databaseURL);
        storage = getStorage(firebaseApp);
        hideMessage();
        setConnection("Connecting", "warning");

        listenTo("RESIDENTS", (value) => {
            residents = value && typeof value === "object" ? value : {};
            syncResidentFilters();
            renderResidents();
            renderSummary();
            renderSelectedResident();
            listenForSelectedActivity();
        });

        listenTo("CURRENT_POSTURE_STATUS", (value) => {
            postureByResident = normalizePostureData(value);
            renderSummary();
            renderResidents();
            renderSelectedResident();
        });
    } catch (error) {
        setConnection("Connection error", "error");
        showMessage(`Could not connect to Firebase Realtime Database: ${error.message}`, "danger");
    }
}

function listenTo(path, onData) {
    onValue(ref(database, path), (snapshot) => {
        onData(snapshot.val());
        setConnection("Connected", "connected");
        hideMessage();
    }, (error) => {
        setConnection("Access denied", "error");
        showMessage(`Cannot read ${path}: ${error.message}. Check that you are signed in and your Realtime Database rules allow authorized staff.`, "danger");
    });
}

function normalizePostureData(value) {
    if (!value || typeof value !== "object") {
        return {};
    }

    const residentId = value.resident_id || value.residentId;
    if (residentId || value.posture || value.posture_status || value.pose_status || value.status ||
        value.activity_recognition || value.fall_detected !== undefined || value.inactivity_detected !== undefined) {
        const residentIds = Object.keys(residents);
        const key = residentId || (residentIds.length === 1 ? residentIds[0] : "unknown");
        return { [key]: value };
    }

    return Object.fromEntries(Object.entries(value).map(([key, record]) => [
        key,
        record && typeof record === "object" ? record : { posture: record }
    ]));
}

function syncResidentFilters() {
    const residentList = Object.entries(residents);
    if (!residentList.some(([id]) => id === selectedResidentId)) {
        selectedResidentId = residentList.length ? residentList[0][0] : "";
    }

    const selected = document.getElementById("selectedResident");
    const activityFilter = document.getElementById("activityResidentFilter");
    populateResidentSelect(selected, residentList, "No residents added");
    populateResidentSelect(activityFilter, residentList, "Select a resident");
    selected.value = selectedResidentId;
    activityFilter.value = selectedResidentId;
    renderHistory();
}

function populateResidentSelect(select, residentList, emptyLabel) {
    select.replaceChildren();
    if (!residentList.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = emptyLabel;
        select.append(option);
        return;
    }

    for (const [id, resident] of residentList) {
        const option = document.createElement("option");
        option.value = id;
        option.textContent = `${resident.full_name || "Unnamed resident"}${resident.room_number ? ` · ${resident.room_number}` : ""}`;
        select.append(option);
    }
}

function renderSummary() {
    let safe = 0;
    let unsafe = 0;
    let alerts = 0;

    for (const id of Object.keys(residents)) {
        const status = postureByResident[id];
        const category = postureCategory(status);
        if (category === "safe") safe += 1;
        if (category === "unsafe") unsafe += 1;
        if (status && (isAlertRecord(status) || category === "unsafe")) {
            alerts += 1;
        }
    }

    document.getElementById("totalCount").textContent = String(Object.keys(residents).length);
    document.getElementById("safeCount").textContent = String(safe);
    document.getElementById("unsafeCount").textContent = String(unsafe);
    document.getElementById("alertCount").textContent = String(alerts);
}

function renderResidents() {
    const search = document.getElementById("residentSearch").value.trim().toLowerCase();
    const residentList = Object.entries(residents).filter(([, resident]) => {
        const searchable = `${resident.full_name || ""} ${resident.room_number || ""} ${resident.age || ""} ${resident.address || ""}`.toLowerCase();
        return searchable.includes(search);
    });

    residentRows.replaceChildren();
    if (!residentList.length) {
        addEmptyRow(residentRows, 8, search ? "No residents match your search." : "No resident profiles yet. Add a profile to get started.");
        return;
    }

    for (const [id, resident] of residentList) {
        const row = document.createElement("tr");
        addResidentNameCell(row, resident);
        addCell(row, resident.age || "—");
        addCell(row, resident.address || "—");
        addCell(row, resident.room_number || "—");
        addCell(row, resident.risk_category || "—");
        addPostureCell(row, postureByResident[id]);
        addCell(row, angleText(postureByResident[id]));

        const actionCell = document.createElement("td");
        const viewButton = document.createElement("button");
        viewButton.className = "btn btn-sm btn-outline-primary";
        viewButton.type = "button";
        viewButton.textContent = "View history";
        viewButton.addEventListener("click", () => {
            selectedResidentId = id;
            document.getElementById("selectedResident").value = id;
            document.getElementById("activityResidentFilter").value = id;
            renderSelectedResident();
            listenForSelectedActivity();
            document.getElementById("activityLogs").scrollIntoView({ behavior: "smooth" });
        });
        actionCell.append(viewButton);
        row.append(actionCell);
        residentRows.append(row);
    }
}

function renderSelectedResident() {
    const record = postureByResident[selectedResidentId];
    const profile = residents[selectedResidentId];
    const posture = postureLabel(record);
    const badge = document.getElementById("selectedPosture");
    badge.textContent = posture;
    badge.className = `badge ${postureBadgeClass(record)}`;
    document.getElementById("selectedAngle").textContent = angleText(record);
    document.getElementById("selectedUpdated").textContent = record
        ? formatTimestamp(record.updated_at || record.timestamp || record.created_at)
        : "—";
    document.getElementById("selectedResidentName").textContent = profile?.full_name || "—";
    setResidentPhoto(document.getElementById("selectedResidentPhoto"), profile);
    document.getElementById("selectedAge").textContent = profile?.age ? `${profile.age} years` : "—";
    document.getElementById("selectedAddress").textContent = profile?.address || "—";
    document.getElementById("selectedActivity").textContent = activityLabel(record) || "—";
    setDetectionBadge("selectedFallStatus", record, isFallDetected(record), ["fall_detected", "fallDetected", "fall"]);
    setDetectionBadge("selectedInactivityStatus", record, isInactivityDetected(record), ["inactivity_detected", "inactivityDetected", "inactive"]);
}

function listenForSelectedActivity() {
    if (activityUnsubscribe) {
        activityUnsubscribe();
        activityUnsubscribe = undefined;
    }

    if (isLiveBridgeMode) {
        renderHistory();
        return;
    }

    activityLogs = {};
    renderHistory();

    if (isDemoMode) {
        activityLogs = demoData.ACTIVITY_LOGS;
        renderHistory();
        return;
    }

    if (!database || !selectedResidentId) {
        return;
    }

    const activityQuery = query(
        ref(database, "ACTIVITY_LOGS"),
        orderByChild("resident_id"),
        equalTo(selectedResidentId),
        limitToLast(100)
    );
    activityUnsubscribe = onValue(activityQuery, (snapshot) => {
        const value = snapshot.val();
        activityLogs = value && typeof value === "object" ? value : {};
        renderHistory();
    }, (error) => {
        showMessage(`Cannot read activity logs: ${error.message}. Check the Realtime Database rules.`, "danger");
    });
}

function renderHistory() {
    const fromTime = localDateStart(appliedDateRange.from);
    const toTime = localDateStart(appliedDateRange.to) + 24 * 60 * 60 * 1000;
    const records = Object.entries(activityLogs)
        .filter(([, record]) => record && (!record.resident_id || record.resident_id === selectedResidentId))
        .map(([id, record]) => ({ id, ...record }))
        .filter((record) => {
            const time = recordTime(record);
            return time >= fromTime && time < toTime && isAlertRecord(record);
        })
        .sort((left, right) => recordTime(right) - recordTime(left));
    records.splice(100);

    document.getElementById("historyRangeLabel").textContent = selectedResidentId
        ? `Showing up to ${records.length} warning / unsafe records for ${residentName(selectedResidentId)} from ${appliedDateRange.from} to ${appliedDateRange.to}.`
        : "Select a resident to view warning history.";
    renderDetectionSummary(records);
    renderAlertHistory(records);
}

function renderDetectionSummary(records) {
    const counts = {
        periodAlertCount: records.filter(isAlertRecord).length,
        periodFallCount: records.filter(isFallDetected).length,
        periodInactivityCount: records.filter(isInactivityDetected).length
    };
    for (const [id, count] of Object.entries(counts)) {
        document.getElementById(id).textContent = String(count);
    }
}

function renderAlertHistory(records) {
    const alertRows = document.getElementById("alertRows");
    alertRows.replaceChildren();
    if (!selectedResidentId) {
        addEmptyRow(alertRows, 7, "Select a resident to view warnings.");
        return;
    }
    if (!records.length) {
        addEmptyRow(alertRows, 7, "No warnings or unsafe-posture events were recorded for this resident in the selected date range.");
        return;
    }

    for (const record of records) {
        const row = document.createElement("tr");
        addCell(row, formatTimestamp(record.timestamp || record.created_at || record.detected_at));
        addCell(row, alertLabel(record));
        addCell(row, record.severity || record.alert_severity || (isFallDetected(record) ? "High" : "—"));
        addCell(row, activityLabel(record) || "—");
        addCell(row, poseLabel(record) || "—");
        addCell(row, record.notes || record.alert_message || "—");
        const snapshotCell = document.createElement("td");
        const savedDemoSnapshot = isDemoMode && demoSnapshotUrls.has(record.id);
        const hasSnapshot = Boolean(record.snapshot_storage_path || record.snapshot_url || savedDemoSnapshot);
        const canCapture = Boolean(cameraStream || liveStreamActive);
        if (hasSnapshot) {
            const viewButton = document.createElement("button");
            viewButton.className = "btn btn-sm btn-outline-primary";
            viewButton.type = "button";
            viewButton.textContent = "View snapshot";
            viewButton.addEventListener("click", () => viewWarningSnapshot(record));
            snapshotCell.append(viewButton);
        } else {
            const captureButton = document.createElement("button");
            captureButton.className = "btn btn-sm btn-outline-danger";
            captureButton.type = "button";
            captureButton.disabled = !canCapture;
            captureButton.textContent = canCapture ? "Capture now" : "Camera off";
            captureButton.addEventListener("click", () => captureWarningSnapshot(record, captureButton));
            snapshotCell.append(captureButton);
        }
        row.append(snapshotCell);
        alertRows.append(row);
    }
}

function initializeDateRange() {
    const today = new Date();
    const from = new Date(today.getFullYear(), today.getMonth(), today.getDate() - 6);
    const range = { from: toDateInputValue(from), to: toDateInputValue(today) };
    document.getElementById("historyFromDate").value = range.from;
    document.getElementById("historyToDate").value = range.to;
    return range;
}

function toDateInputValue(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
}

function localDateStart(dateValue) {
    const [year, month, day] = dateValue.split("-").map(Number);
    return new Date(year, month - 1, day).getTime();
}

function recordTime(record) {
    const value = record.timestamp ?? record.created_at ?? record.detected_at ?? record.updated_at;
    if (typeof value === "number") {
        return value < 1000000000000 ? value * 1000 : value;
    }
    const parsed = Date.parse(value || "");
    return Number.isNaN(parsed) ? 0 : parsed;
}

function activityCategory(record) {
    const value = String(
        record.activity_recognition || record.recognized_activity || record.activity_type || record.activity || ""
    ).trim().toLowerCase().replace(/[\s-]+/g, "_");
    if (/^(sit|sitting|seated)$/.test(value)) return "sit";
    if (/^(stand|standing)$/.test(value)) return "stand";
    if (/^(walk|walking)$/.test(value)) return "walk";
    if (/^(bend|bending|bent|stooping)$/.test(value)) return "bend";
    if (/^(lie|lying|lay|laying|lying_down|fall|falling)$/.test(value)) return "lying";
    return "";
}

function activityLabel(record) {
    const labels = {
        sit: "Sitting",
        stand: "Standing",
        walk: "Walking",
        bend: "Bending",
        lying: "Lying"
    };
    const category = activityCategory(record);
    if (category) return labels[category];
    const value = record.activity_recognition || record.recognized_activity || record.activity_type || record.activity;
    return typeof value === "string" ? value : "";
}

function poseLabel(record) {
    const value = record.pose_status || record.posture || record.pose || record.pose_label;
    return typeof value === "string" || typeof value === "number" ? String(value) : "";
}

function isFallDetected(record) {
    return hasTrueFlag(record, ["fall_detected", "fallDetected", "fall"]) ||
        /fall/.test(eventText(record));
}

function isInactivityDetected(record) {
    return hasTrueFlag(record, ["inactivity_detected", "inactivityDetected", "inactive"]) ||
        /inactiv/.test(eventText(record));
}

function hasTrueFlag(record, keys) {
    return keys.some((key) => {
        const value = record[key];
        return value === true || value === 1 || (typeof value === "string" && /^(true|yes|1|detected)$/i.test(value));
    });
}

function eventText(record) {
    return [
        record.event_type, record.event, record.type, record.detection_type,
        record.alert_type, record.alert_message, record.activity_recognition,
        record.activity_type, record.activity, record.posture, record.pose_status
    ].filter((value) => typeof value === "string").join(" ").toLowerCase();
}

function isAlertRecord(record) {
    const explicitAlert = hasTrueFlag(record, ["alert_active", "alertActive", "is_alert", "alert"]);
    const severity = String(record.severity || record.alert_severity || "").toLowerCase();
    const severityAlert = severity && !/^(info|informational|normal|low)$/.test(severity);
    const unsafePosture = postureCategory(record) === "unsafe";
    const alertType = Boolean(record.alert_type || record.alert_message || record.alert_severity);
    return explicitAlert || severityAlert || alertType || isFallDetected(record) || isInactivityDetected(record) ||
        unsafePosture || /\b(alert|fall|inactiv|unsafe|slouch)\b/.test(eventText(record));
}

function alertLabel(record) {
    if (record.alert_type || record.alert_message) return record.alert_type || record.alert_message;
    if (isFallDetected(record)) return "Fall detected";
    if (isInactivityDetected(record)) return "Inactivity detected";
    if (postureCategory(record) === "unsafe") {
        return "Unsafe posture";
    }
    return record.event_type || record.event || record.type || "AI alert";
}

function setDetectionBadge(elementId, record, detected, fields) {
    const badge = document.getElementById(elementId);
    if (!record || (!fields.some((field) => field in record) && !detected)) {
        badge.textContent = "No data";
        badge.className = "badge badge-secondary";
        return;
    }
    badge.textContent = detected ? "Detected" : "Not detected";
    badge.className = `badge ${detected ? "badge-danger" : "badge-success"}`;
}

function residentName(id) {
    const resident = residents[id];
    return resident ? `${resident.full_name || "Unnamed resident"}${resident.room_number ? ` · ${resident.room_number}` : ""}` : id;
}

function exportHistory() {
    if (!selectedResidentId) {
        showHistoryError("Select a resident before exporting the history.");
        return;
    }
    if (!window.XLSX) {
        showHistoryError("The Excel export library could not load. Check your internet connection and try again.");
        return;
    }

    const fromTime = localDateStart(appliedDateRange.from);
    const toTime = localDateStart(appliedDateRange.to) + 24 * 60 * 60 * 1000;
    const records = Object.entries(activityLogs)
        .filter(([, record]) => record && (!record.resident_id || record.resident_id === selectedResidentId))
        .map(([id, record]) => ({ id, ...record }))
        .filter((record) => recordTime(record) >= fromTime && recordTime(record) < toTime && isAlertRecord(record))
        .sort((left, right) => recordTime(right) - recordTime(left));
    records.splice(100);

    const workbook = window.XLSX.utils.book_new();
    const summary = [
        ["Elderly Posture Monitoring - Warning History"],
        ["Resident", residentName(selectedResidentId)],
        ["From date", appliedDateRange.from],
        ["To date", appliedDateRange.to],
        ["Warnings / unsafe events", records.length],
        ["Falls", records.filter(isFallDetected).length],
        ["Inactivity warnings", records.filter(isInactivityDetected).length]
    ];
    const warningData = [[
        "Time", "Warning", "Severity", "Activity", "Pose / posture", "Notes", "Snapshot"
    ], ...records.map((record) => [
        formatTimestamp(record.timestamp || record.created_at || record.detected_at),
        alertLabel(record),
        record.severity || record.alert_severity || (isFallDetected(record) ? "High" : ""),
        activityLabel(record),
        poseLabel(record),
        record.notes || record.alert_message || "",
        record.snapshot_storage_path || record.snapshot_url || (demoSnapshotUrls.has(record.id) ? "Saved in this demo session" : "")
    ])];

    window.XLSX.utils.book_append_sheet(workbook, window.XLSX.utils.aoa_to_sheet(summary), "Summary");
    window.XLSX.utils.book_append_sheet(workbook, window.XLSX.utils.aoa_to_sheet(warningData), "Warnings");
    const safeResidentName = residentName(selectedResidentId).replace(/[^\w-]+/g, "_").replace(/^_+|_+$/g, "");
    window.XLSX.writeFile(workbook, `posture_history_${safeResidentName}_${appliedDateRange.from}_to_${appliedDateRange.to}.xlsx`);
    historyFilterError.classList.add("d-none");
}

function showHistoryError(message) {
    historyFilterError.textContent = message;
    historyFilterError.classList.remove("d-none");
}

function clearPhotoPreview() {
    selectedPhotoFile = undefined;
    if (photoPreviewUrl) {
        URL.revokeObjectURL(photoPreviewUrl);
        photoPreviewUrl = undefined;
    }
    const photoInput = document.getElementById("resPhoto");
    if (photoInput) photoInput.value = "";
    document.getElementById("photoPreviewWrap").classList.add("d-none");
    document.getElementById("photoPreview").removeAttribute("src");
}

function postureCategory(record) {
    if (!record) return "unknown";
    if (record.alert_active === true || record.alertActive === true) return "unsafe";

    const value = String(record.posture || record.posture_status || record.pose_status || record.status || "").toLowerCase();
    if (/unsafe|poor|slouch|bad|alert|abnormal|warning/.test(value)) return "unsafe";
    if (/safe|good|normal|upright|correct/.test(value)) return "safe";
    return "unknown";
}

function postureLabel(record) {
    if (!record) return "No data";
    const value = record.posture || record.posture_status || record.pose_status || record.pose || record.status;
    return value ? String(value) : "Unknown";
}

function postureBadgeClass(record) {
    const category = postureCategory(record);
    if (category === "safe") return "badge-success";
    if (category === "unsafe") return "badge-danger";
    return "badge-secondary";
}

function angleText(record) {
    if (!record) return "—";
    const angle = record.angle_degrees ?? record.angle ?? record.posture_angle;
    return angle === undefined || angle === null || angle === "" ? "—" : `${angle}°`;
}

function timestampValue(value) {
    if (typeof value === "number") return value < 1000000000000 ? value * 1000 : value;
    const parsed = Date.parse(value || "");
    return Number.isNaN(parsed) ? 0 : parsed;
}

function formatTimestamp(value) {
    const time = timestampValue(value);
    return time ? new Date(time).toLocaleString() : "—";
}

function addResidentNameCell(row, resident) {
    const cell = document.createElement("td");
    const wrapper = document.createElement("div");
    wrapper.className = "resident-name-cell";
    const photo = document.createElement("img");
    photo.className = "resident-table-photo";
    setResidentPhoto(photo, resident);
    photo.alt = "";
    const name = document.createElement("span");
    name.className = "font-weight-bold";
    name.textContent = resident.full_name || "Unnamed resident";
    wrapper.append(photo, name);
    cell.append(wrapper);
    row.append(cell);
}

function setResidentPhoto(image, resident) {
    const placeholder = "img/resident-placeholder.svg";
    image.onerror = (event) => {
        event.currentTarget.onerror = null;
        event.currentTarget.src = placeholder;
    };
    if (resident?.photo_url) {
        image.src = resident.photo_url;
        return;
    }
    if (isDemoMode || isLiveBridgeMode || !resident?.photo_storage_path || !storage) {
        image.src = placeholder;
        return;
    }

    const path = resident.photo_storage_path;
    image.dataset.photoPath = path;
    if (!photoUrlCache.has(path)) {
        photoUrlCache.set(path, getDownloadURL(storageRef(storage, path)));
    }
    photoUrlCache.get(path).then((url) => {
        if (image.dataset.photoPath === path) image.src = url;
    }).catch((error) => {
        photoUrlCache.delete(path);
        image.src = placeholder;
        showMessage(`Could not load a resident profile photo: ${error.message}`, "danger");
    });
}

function addCell(row, value) {
    const cell = document.createElement("td");
    cell.textContent = String(value);
    row.append(cell);
}

function addPostureCell(row, record) {
    const cell = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `badge ${postureBadgeClass(record)}`;
    badge.textContent = postureLabel(record);
    cell.append(badge);
    row.append(cell);
}

function addEmptyRow(tableBody, columnCount, message) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = columnCount;
    cell.className = "text-center text-muted py-4";
    cell.textContent = message;
    row.append(cell);
    tableBody.append(row);
}

function setConnection(message, state) {
    document.getElementById("connectionText").textContent = message;
    const dot = document.getElementById("connectionDot");
    dot.classList.toggle("is-connected", state === "connected");
    dot.classList.toggle("is-error", state === "error");
}

function showMessage(message, type) {
    dashboardMessage.textContent = message;
    dashboardMessage.className = `alert alert-${type}`;
}

function hideMessage() {
    dashboardMessage.classList.add("d-none");
}
