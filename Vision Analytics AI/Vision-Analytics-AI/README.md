# SIDP Vision Analytics AI

This is the Vision Posture Detection proof of concept for SIDP Subsystem 2. It
observes people, posture, movement, scene relationships, and temporal fall
patterns from a local webcam or video. Subsystem 3 owns the final
SAFE/WARNING/UNSAFE/ALERT decision; Subsystem 4 owns transmission, storage, and
caregiver notifications. This repository does not send notifications or make a
validated safety decision.

This README is the authoritative setup, implementation, integration, and handoff
guide. Model URLs, exact revisions, and SHA-256 values are machine-readable in
`configs/poc_models.json`. Downloaded third-party notices and license texts remain
separate because they belong to their respective authors.

## Quick start on Windows PowerShell

Unzip the teammate archive, open PowerShell in its `Vision-Analytics-AI` folder,
and install a user-managed **64-bit Python 3.11**. Python >=3.11 is declared by
the package, but 3.11.16 is the version verified here; newer minors and other
platforms have not been verified in this cleanup.

```powershell
Set-Location "C:\path\Vision-Analytics-AI"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1
.\run_demo.cmd --open-browser
```

If Python discovery selects the wrong interpreter:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1 -PythonPath "C:\path\Python311\python.exe"
```

The helper creates or repairs `.venv`, installs `requirements.txt`, runs
`pip check`, and fetches/verifies the pinned model assets. It reuses the ignored
`.runtime/python311` interpreter when that interpreter exists in this checkout;
the archive does not contain that runtime. Changing Python minor version preserves
the previous environment in a `.venv-backup-*` directory. App-managed Python paths
that may be invisible in ordinary PowerShell are rejected with an explanation.

The helper detects NVIDIA hardware with `nvidia-smi` and installs PyTorch
2.11.0/torchvision 0.26.0 CUDA 12.8 wheels when CUDA is not already usable. A
working existing CUDA environment is retained. `-Cpu` skips this GPU installation;
it does not replace an existing CUDA build. `--device cpu` selects CPU inference.
No separate CUDA toolkit, TensorFlow, Node build, or RelateAnything Python package
is needed for the default POC. A compatible NVIDIA driver is needed for CUDA.

For a reproducible CPU environment using the tested PyTorch pair:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\download_models.py
.\run_demo.cmd --device cpu --open-browser
```

Use the selected Python executable in place of `py -3.11` if the Python launcher
is unavailable. These source-checkout commands do not require activation or an
editable installation. `requirements-lock.txt` preserves the earlier tested
Windows CUDA environment, not a universal platform lock; do not use its CUDA
extra index for the CPU recipe. Ruff is a development tool installed separately:
`python -m pip install ruff`. The direct Pillow and threadpoolctl dependencies are
declared explicitly rather than relying on transitive installation.

The dashboard opens at **http://127.0.0.1:8000**. Keep the full body in view and
stand briefly to establish each person's baseline. Stop the camera in the page
when finished, then press **Ctrl+C** in the terminal to stop the server. The
model-loading phase can take several seconds before the page becomes available.

## Current scope and flow

```text
webcam latest frame / offline video media time
  -> OpenCV BGR image -> YOLO26-Pose -> configured tracker (default ByteTrack)
  -> independent context for each person
       + geometry -> smoothed posture / motion diagnostics
       + RGB Standing / Sitting / Fallen detector -> matched posture evidence
       + pretrained tree + ST-GCN/TCN -> Normal / Pre-fall / Fall pattern evidence
       + background RelateAnything ONNX + object regions -> scene/activity label
  -> VisionObservation schema 2.0
       + OpenCV overlay / JSONL / video evaluation
       + FastAPI image inference / observation WebSocket / browser preview
  -> downstream safety subsystem
```

The primary model supplies original-frame boxes and 17 COCO keypoints. Model
weights load once per launcher. The camera monitor has a capacity-one latest-frame
slot: inference skips superseded acquisitions instead of building a queue. The
background scene worker also keeps one pending job. Its results expire and cannot
cross a person's history reset. Image uploads share the loaded pose/RGB models
under a lock and receive a fresh session; they never join camera histories.

ByteTrack is the default. `tracker.backend: centroid` selects the retained gated
Hungarian IoU/centroid tracker with constant-velocity prediction. Track IDs are
local to a session and never reused within it. Empty frames expire stale tracks.
Long gaps, changed frame dimensions, and overlapping ambiguous people reset
temporal contexts. Crossing/occlusion can still switch IDs; there is no guaranteed
re-identification. The same configured tracker is used for sequence extraction.

Each person has independent posture smoothing, a bounded pose buffer, learned
predictor/calibration state, and cached evidence. RGB detections are assigned
one-to-one by IoU; ambiguous matches remain unavailable. Scene labels take priority
for the activity label; RGB posture follows, then geometry. Geometry itself has
no fabricated confidence. Scene predicates beginning with sitting/standing can
also supply the corresponding posture, with their own provenance and reason.

| Capability | Status |
|---|---|
| Webcam preview, Start/Stop/restart, quality selection, overlays, fullscreen | Implemented and exercised in a real browser and camera session |
| YOLO26, ByteTrack, RGB model, tree/fusion, RelateAnything | Available assets loaded and real inference exercised |
| HTTP image inference and both WebSockets | Verified against the real local server; contract regressions covered by tests |
| Video evaluation and sequence/window extraction | Exercised on an RGB-only URFD fixture; no accuracy claim |
| Optional event-classifier calibration workflow | Retained research workflow; no default fitted artifact is supplied |
| Optional Keras fall backend | Explicit manifest selection is connected and mock-tested; no TensorFlow training or real Keras artifact run in this cleanup |
| Posture Keras artifact, reference H5 reuse, caregiver integration | Not active runtime features |

Pre-fall is an upstream learned movement-pattern advisory, with **no guaranteed
future prediction horizon**. Static lying alone is not proof of a fall. Staged
adult footage does not establish performance in older adults or on this webcam.
There is no newly measured sensitivity, specificity, probability calibration, or
clinical validation in this handoff.

## Directory guide

```text
app/
  main.py, monitor.py       OpenCV and browser launchers
  config.py, schemas.py    validated settings and public schema
  capture/                 webcam/video ownership and live worker lifecycle
  inference/               pose, RGB, scene, pretrained and optional adapters
  tracking/                tracker selection, ByteTrack and centroid association
  posture/                 confidence-aware geometry and normalization
  temporal/                buffers, features, evidence states and dataset windows
  fusion/pipeline.py       per-person processing and evidence composition
  api/server.py            HTTP, WebSockets, preview and local controls
  visualization/, web/     OpenCV overlay and browser HTML/CSS/JavaScript
  evaluation/              experimental classification/calibration metrics
configs/                   YAML defaults, model manifest, frozen evaluation split
models/                    local assets and third-party notices; see asset table
scripts/                   source entry points, evaluation, downloads and research
tests/                     hardware-independent regression suite
.github/workflows/tests.yml minimal Linux unit/lint job
pyproject.toml              packaging, console entry points, pytest and Ruff config
requirements*.txt           runtime, optional training and historical snapshot
setup_windows.ps1           Windows environment/model setup
run_demo.cmd               browser launcher (the filename is a compatibility alias)
README.md                   this guide
```

Local `.venv`, `.runtime`, `data`, `outputs`, caches, exported OpenVINO models,
and VS Code settings can remain in the development checkout. They are not part
of the teammate export. No checkpoints, datasets, or personal work were deleted
as part of documentation consolidation.

## Models, provenance, and export contents

`scripts/download_models.py` processes all 28 pinned assets, verifies SHA-256
before replacing a file, and reuses matching files. Retries restart an incomplete
artifact; this is not byte-range resume. A changed upstream file fails its hash
check rather than being silently accepted. Do not substitute arbitrary weights.

| Expected path | Purpose / source | Teammate ZIP |
|---|---|---|
| `yolo26n-pose.pt` | Official Ultralytics YOLO26 nano pose; assets release `v8.4.0` | Included, hash checked |
| `models/human-fall-detection.pt` | YOLO11n RGB Standing/Sitting/Fallen; HF `best.pt`, revision `97261b03632a9715a27481b2cfdac6a78e26d2d9` | Included, hash checked |
| `models/relateanything/relateanything_vits16plus_w16.onnx` | Author's unchanged relation graph | Download during setup |
| `models/relateanything/relateanything_vits16plus_w16.data0` and `.data1` | External graph tensors; both required alongside ONNX | Download during setup |
| `models/relateanything/yolo26n.onnx` and `yolo26n.classes.json` | Author's automatic object detector export and class list | Download during setup |
| `models/relateanything/predicate_bank.json` | Author's predicate embeddings/weights | Download during setup |
| `models/relateanything/THIRD_PARTY.md` | Author's unmodified deployment notices | Included |
| `models/fall-risk/yolo_tail60_prefall_accel_robust_classifier.joblib` | Pretrained tree classifier | Download during setup |
| `models/fall-risk/skeleton_feature_fusion_tuned.pt` | Pretrained ST-GCN/TCN fusion | Download during setup |
| `models/fall-risk/source/fall_prediction/*.py` | 17 pinned upstream inference modules listed below | Download during setup |
| `models/temporal/fall.keras`, `fall_stats.npz`, `fall_manifest.json` | Optional locally trained fall model; never a default dependency | Not supplied; generate only with reviewed labeled data |
| `models/event_classifier.json` | Optional research JSON event classifier | Not supplied; `evidence.model` defaults to null |

The exact source modules under `models/fall-risk/source/fall_prediction/` are:

```text
__init__.py  config.py  deep_dataset.py  deep_model.py  ensemble_predictor.py
features.py  fusion_model.py  landmarks.py  lying_adl_filter.py  ml_features.py
ml_predictor.py  pose.py  predictor.py  risk.py  robustness.py
skeleton_dataset.py  window_dataset.py
```

All excluded required files have URLs and hashes in `configs/poc_models.json`;
the reproducible acquisition command is:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py
```

The source ZIP includes the compact pose/RGB weights and their AGPL-3.0 text in
`models/ULTRALYTICS-LICENSE.txt`. Corresponding implementation source is the
[Ultralytics repository at v8.4.171](https://github.com/ultralytics/ultralytics/tree/v8.4.171);
the [RGB model card](https://huggingface.co/melihuzunoglu/human-fall-detection)
declares AGPL-3.0 and its three detector classes. The card does not provide a
reproducible training split or webcam calibration.

[RelateAnything](https://github.com/Maelic/RelateAnything) code declares AGPL-3.0;
its DINOv3-derived weights have separate notices in the included third-party
file. The pinned temporal manifest records no declared upstream license for
[the tree/fusion repository](https://github.com/JiajunTang2006/pose-based-fall-risk-prediction).
Those downloaded source files/weights are retained locally and excluded from
redistribution. RelateAnything's larger cache is also reacquired from the author.
This project does not invent permission to redistribute either asset set.
The first-party repository has no root license declaration; dependency/model
licenses do not supply a license for first-party code.

The optional benchmark also uses `models/yolo26s-pose.pt` and `outputs/bus.jpg`.
Neither is needed to launch the POC or included in the ZIP. Obtain the former
from official Ultralytics assets if benchmarking it, and use the public
Ultralytics sample image at `https://ultralytics.com/images/bus.jpg` for the latter.
The benchmark's OpenVINO option additionally requires a user-installed OpenVINO
runtime and performs model export. Existing local exports are preserved, not
selected automatically.

## Launch and diagnostic commands

Browser monitor; starts camera 0 unless configured otherwise:

```powershell
.\run_demo.cmd --open-browser
.\run_demo.cmd --camera 0 --device cpu --image-size 416 --open-browser
.\run_demo.cmd --no-camera --port 8000
.\run_demo.cmd --config .\configs\default.yaml --no-camera
```

`scripts/run_monitor.py` and `python -m app.monitor` run the same browser launcher.
The monitor honors `--config`, then `SIDP_CONFIG`, then bundled defaults. Explicit
camera/device/image-size switches override YAML; omitted switches retain YAML.

OpenCV display or bounded headless webcam run:

```powershell
.\.venv\Scripts\python.exe -m app.main --camera 0
.\.venv\Scripts\python.exe scripts\run_demo.py --device cpu --headless --max-frames 15 --jsonl outputs\webcam.jsonl
.\.venv\Scripts\python.exe scripts\test_pose_webcam.py --max-frames 15
```

The two source scripts are preserved compatibility aliases for `app.main`.
`--config`, `--posture-config`, `--width`, `--height`, `--pose-model`,
`--confidence`, `--keypoint-confidence`, and `--debug` are available there.
`--no-temporal-model` disables only an explicit retrained Keras manifest; the
pretrained POC branches remain enabled. Use YAML enabled flags to disable those.

Pose-only local-image smoke test and full configured video evaluator:

```powershell
.\.venv\Scripts\python.exe scripts\test_pose_image.py "C:\path\person.jpg" --output outputs\pose-image.jpg
.\.venv\Scripts\python.exe scripts\evaluate_video.py "C:\path\rgb-video.mp4" --output-dir outputs\evaluation
```

Image smoke tests exercise primary pose/geometry, not temporal evidence. Video
evaluation loads configured optional models and writes `annotated.mp4`,
`results.csv`, `observations.jsonl`, and `summary.json`. It reports counts and
runtime; `accuracy_metrics` is null without ground truth. `--fps 30` supplies a
finite positive fallback only if video FPS metadata is invalid. Video timestamps
use presentation time, with strictly increasing FPS fallback when needed. A
decode failure ends the stream; OpenCV cannot reliably distinguish every corrupt
tail from ordinary end of stream. No decoded frames is an error.

Direct local API without automatic camera capture:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api.server:app --host 127.0.0.1 --port 8000
curl.exe -X POST "http://127.0.0.1:8000/infer/image" -H "Content-Type: image/jpeg" --data-binary "@C:\path\person.jpg"
```

Set `$env:SIDP_CONFIG = "C:\path\settings.yaml"` for the direct API factory.
Set `$env:SIDP_CAMERA_ENABLED = "1"` before direct Uvicorn launch to enable
camera capture; remove it with `Remove-Item Env:SIDP_CAMERA_ENABLED` afterward.
Explicit factory configuration and the browser launcher's `--no-camera` take
precedence over that environment camera override. Use one Uvicorn worker and
avoid reload while a camera is active. Console names `sidp-vision` and
`sidp-monitor` are declared in `pyproject.toml`; the source commands above are the
verified handoff paths and do not depend on installing console entry points.

## Configuration and controls

`configs/default.yaml` and `configs/posture.yaml` are validated with Pydantic;
unknown keys, nonfinite values, and invalid ranges raise errors. An alternate
main YAML still uses bundled posture settings unless the OpenCV launcher receives
`--posture-config`. `temporal.model_manifest` is resolved relative to its YAML;
pose/RGB/model-directory paths are relative to the project root unless absolute.

| Setting | Default / meaning |
|---|---|
| `camera.index`, `width`, `height` | 0, requested 640x480; actual camera dimensions can differ |
| `pose.model`, `device`, `image_size` | `yolo26n-pose.pt`, auto CUDA/CPU, 512 |
| `pose.confidence`, `keypoint_confidence` | 0.1 detection floor, 0.3 valid-joint threshold |
| `tracker.backend`, `stale_seconds`, `ambiguity_iou` | ByteTrack, 1 second, overlap reset at 0.55 IoU |
| `secondary.interval_seconds`, `max_age_seconds`, `min_iou` | RGB every 0.4 s; expire after 0.65 s; association IoU >=0.2 |
| `scene.interval_seconds`, `max_age_seconds`, `confidence` | Scene every 1 s; expire after 2.5 s; relation score >=0.56 |
| `risk.sample_hz`, `max_gap_seconds`, `min_coverage` | 30 Hz causal resampling; reset after 0.5 s gap; coverage >=0.45 and all torso anchors |
| `temporal.capacity`, `fall_window`, `sample_hz` | 120 observations; optional Keras 60x51 input at 30 Hz |
| `temporal.min_valid_ratio`, `inference_interval_seconds` | Optional Keras window coverage >=0.7; infer at most every 0.15 s per person |
| `api.max_image_bytes`, `max_image_pixels` | 10,000,000 bytes and 16,000,000 pixels |
| `api.websocket_hz`, `preview_hz`, `inference_hz` | Maximum requested live publication/preview/inference rates: 20/20/24 Hz; hardware may be slower |

Standing geometry uses torso angle <=30 degrees and knee angle >=155 degrees.
Sitting uses torso <=45, knee <=135, hip <=140. Lying requires torso >=65 and
bbox width/height >=1.15. Geometry coverage must reach 0.4. Smoothing waits 0.25 s
for a changed label and returns UNKNOWN immediately if usable geometry disappears.
These are image-space heuristics, not calibrated class probabilities.

The pretrained tree/fusion branch retains upstream 15-frame, 30 Hz features,
standing calibration, decision policy, and static-lying filter. Its thresholds
are in the pinned upstream adapter/configuration, not `state.*`. The retained
`state.*` settings and `FallStateMachine` belong to the historical experimental
geometry estimator; it is tested independently and not activated by default.
`temporal.posture_window` is used by dataset preparation, not a live learned
posture classifier. Optional event persistence uses its configured artifact
threshold, 0.4 s persistence and 1 s recovery defaults.

| UI/control | Behavior |
|---|---|
| Camera source | UI offers indices 0/1/2; CLI/API can select other valid indices |
| Start / Stop monitoring | Start fresh histories; stop releases capture and clears preview |
| Faster / Balanced / More detail | 416 / 512 / 640 input; stop before changing source or quality |
| Reset standing baseline | Reset each active person's pretrained baseline/window |
| Five overlay switches | Boxes, skeleton, activity, confidence, IDs; saved in browser local storage |
| Fullscreen | Expand the preview; Escape exits fullscreen |
| OpenCV Q or Shift+Q / window close | Exit the display loop and release resources |
| Ctrl+C | Stop either terminal launcher; headless mode has no Q window |

Stop is not pause: restarting deliberately creates a new session/history. Closing
or disconnecting a browser does not stop the local camera/server; use Stop or
Ctrl+C. Multiple preview clients share the same camera, with skipped frames under
backpressure. No conflicting global Q/space bindings are added to the browser.

## Observation and transport contract

Schema version stays **2.0**. Identity is the tuple
`(camera_id, session_id, track_id)`. Original field names and types are retained;
`/status` model entries add an `enabled` flag so deliberately disabled branches
are not mistaken for failed setup.

| Output | Meaning |
|---|---|
| `timestamp`, `elapsed_seconds`, `frame_id` | UTC observation wall time; monotonic/media session seconds; source frame index |
| `fps`, `processing_latency_ms`, `capture_to_publish_ms`, `timings_ms` | Measured rate and milliseconds; live monitor FPS measures publication, offline FPS reflects media cadence |
| `bbox`, `keypoints` | Original-image pixels, `[x1,y1,x2,y2]`, exactly 17 ordered index/x/y/confidence/valid objects |
| `posture`, `posture_confidence`, `posture_source`, `posture_reason` | STANDING/SITTING/LYING/UNKNOWN; geometric confidence null; model scores retain source meaning |
| `fall_probability`, `fall_inference.probability` | Null by default; only the explicitly selected calibrated event classifier supplies a numeric probability |
| `fall_inference.raw_score`, `calibrated`, `status`, `target` | Raw evidence, calibration flag, availability/meaning, and what was assessed |
| `appearance` | Independent RGB Fallen/posture confidence, match status, age and latency; uncalibrated |
| `risk` | UNASSESSED/NORMAL/PRE_FALL/FALL, separate tree/fusion class scores, baseline progress, window count; no calibrated future risk |
| `scene`, `activity_*` | Relationship label/object/score and selected activity provenance; stale/unavailable evidence has null confidence |
| `state` | NORMAL/POSSIBLE_FALL/FALLING/FALLEN; risk PRE_FALL maps to POSSIBLE_FALL, risk FALL to FALLEN; unavailable evidence maps to legacy NORMAL |
| `motion`, `geometry`, `movement_state`, `identity_uncertain` | Nullable diagnostics, STATIONARY/MOVING/UNKNOWN, overlap uncertainty |

**NORMAL with missing evidence does not mean safe.** Read availability/status
and `risk.state` as well as the legacy state. No detection, no matched RGB box,
missing joints, model errors, warmup, and stale results are explicit unavailable
evidence. They do not become zero fall probability. Raw RGB and retrained Keras
scores no longer populate the misleading `fall_probability` field; they remain
available in `raw_score`/their evidence branch. This is an intentional semantic
bug fix with unchanged field types. Consumers that treated those raw values as a
validated probability must use the source-specific score instead.

Positive image y velocity means downward. Linear velocities are fractions of
frame dimension per second; joint speed uses separately normalized x/y axes;
angular velocity is degrees/second; aspect-ratio velocity is per second; motion
energy is mean squared normalized joint speed. Torso angles are 0-90 degrees to
vertical, joint angles are degrees. Hip/shoulder heights are fractions of image
height above the image bottom, not physical floor distance. `processing_latency_ms`
excludes model loading, capture, drawing, JPEG encoding and delivery;
`capture_to_publish_ms` is measured before store JPEG encoding/network transmission.

The following is a **schema-valid documentation example**, not captured runtime
output. An empty person list establishes no safety assessment:

```json
{
  "schema_version": "2.0",
  "timestamp": "2026-10-16T00:00:00Z",
  "elapsed_seconds": 0.0,
  "frame_id": 0,
  "camera_id": "webcam_0",
  "session_id": "example-session",
  "fps": 0.0,
  "processing_latency_ms": 12.0,
  "frame_width": 640,
  "frame_height": 480,
  "timings_ms": {"pose": 10.0, "tracking": 0.5, "features_and_state": 1.5, "temporal": 0.0, "appearance": 0.0, "total": 12.0},
  "persons": [],
  "detected_people": 0,
  "capture_to_publish_ms": 0.0
}
```

| Endpoint | Contract |
|---|---|
| `GET /health`, `/status` | Liveness/schema, camera state/error and enabled/ready model branches; liveness alone does not validate model quality |
| `GET /`, `/assets/*` | Local dashboard and packaged static resources |
| `POST /infer/image` | JPEG/PNG bytes, not multipart; accepted content types image/jpeg, image/png, application/octet-stream |
| `POST /api/camera/start` | JSON `{"index":0,"image_size":512}`; stop first to change a running source/quality; conflict is HTTP 409 |
| `POST /api/camera/stop`, `/api/session/reset` | Clear/release camera; reset standing baseline respectively |
| `WS /ws/vision` | New observation JSON; no raw image; clients may skip frame IDs |
| `WS /ws/monitor` | Binary `[4-byte little-endian JSON length][UTF-8 JSON][JPEG]`; also text status messages with `type: status` |
| `GET /stream.mjpg`, `/docs` | Compatibility MJPEG stream; generated HTTP API documentation |

An uploaded image gets independent tracks, `camera_id: image_request`, null
temporal probability, and no live-stream publication. RGB image evidence can
still be present; a single image cannot establish a temporal sequence. The API
enforces byte/pixel limits and rejects malformed/unsupported images. Cross-origin
browser POST controls and monitor WebSocket origins are rejected; the service is
a localhost POC without authentication or TLS. Remote integration, access control,
retention and missing-evidence policy require agreement with downstream owners.

## Extraction and optional research workflows

COCO order is nose, left/right eye, left/right ear, left/right shoulder,
left/right elbow, left/right wrist, left/right hip, left/right knee, left/right
ankle (indices 0-16). `coco17_midhip_torso_mask_v1` requires both shoulders and
hips, subtracts mid-hip, divides x and y by the same torso length, zeroes invalid
joints, then emits 34 interleaved x/y coordinates plus 17 visibility masks.
Missing anchors or a degenerate torso invalidate normalization. The retained
reference audit adapter uses raw mid-hip offsets and is not the 51-value model
input.

```powershell
.\.venv\Scripts\python.exe scripts\extract_yolo26_keypoints.py "C:\path\videos" --output-dir data\sequences
.\.venv\Scripts\python.exe scripts\prepare_temporal_dataset.py data\sequences --labels data\labels.json --task fall --output data\fall_windows.npz
.\.venv\Scripts\python.exe scripts\prepare_temporal_dataset.py data\sequences --labels data\labels.json --task posture --output data\posture_windows.npz
```

Extraction creates paths such as
`data/sequences/subject-a/session.mp4/track_1.jsonl` to prevent same-stem collisions.
Rows preserve video/track/frame IDs, timestamps, frame dimensions, bbox, raw
keypoints, threshold, normalized coordinates/masks and scale. Existing extracted
tracks require a different destination or explicit `--overwrite`; source media
is not deleted. Review identities around crossings before annotation.

Example labels (format example only):

```json
[
  {"video_id":"subject-a/session.mp4","track_id":1,"group_id":"subject-a","start":0.0,"end":8.0,"posture":"STANDING","fall":0},
  {"video_id":"subject-a/session.mp4","track_id":1,"group_id":"subject-a","start":8.1,"end":11.0,"fall":1}
]
```

Only windows wholly within a matching annotation are labeled; overlaps are
rejected and unlabeled windows retain -1. Define the positive interval semantics
before labeling. Keep the same `group_id` for a subject across videos; otherwise
splits use video ID. No geometric prediction becomes ground truth.

Default fall/posture NPZ shapes are `(N,60,51)` / `(N,30,51)`, plus labels, groups,
track IDs and time bounds; companion JSON records the contract. Stride defaults
to 15 source observations. Resampling uses the configured 30 Hz grid, linear xy
interpolation and the intersection of endpoint masks, with no extrapolation or
large-gap interpolation. Sixty samples span 59/30 seconds. At high acquisition
rates increase buffer capacity to retain that duration.

Optional training is an explicit separate action; setup and demo never start it:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe scripts\train_temporal.py data\fall_windows.npz --output-dir models\temporal --epochs 30
```

At least five labeled groups and all training classes are required. Group splits
precede fitting mean/std; saved NumPy statistics are never refitted at runtime.
Training emits `fall.keras`, `fall_stats.npz`, `fall_manifest.json`, and
`fall_report.json`; posture training emits corresponding `posture_*` files, but
its runtime integration remains future work. Training reports are not calibration.

To explicitly select the Keras fall backend, set `temporal.model_manifest` in a
copy of the YAML to `../models/temporal/fall_manifest.json` (for a config beneath
`configs/`). Its manifest must match YOLO26, normalization version, task fall,
51 features, window size, and cadence. An invalid/missing explicit artifact raises
an error. Explicit Keras selection takes precedence over automatic pretrained
tree/event loading; raw Keras scores are uncalibrated, and do not drive the
calibrated event persistence machine. Missing current torso evidence discards a
cached score. The browser's pretrained risk cards remain unassessed for this
experimental backend; inspect JSON for its score/status.

The unfinished event-classifier experiment is retained because its data tools,
configuration and portable inference adapter form a deliberate optional workflow.
It is not the default POC and no artifact is fabricated to activate it:

```powershell
.\.venv\Scripts\python.exe scripts\download_urfd.py
.\.venv\Scripts\python.exe scripts\extract_event_dataset.py --resume
.\.venv\Scripts\python.exe scripts\calibrate_events.py --fit
.\.venv\Scripts\python.exe scripts\calibrate_events.py --evaluate
```

Use all 70 sequences with the frozen split in `configs/evaluation_split.json` for
this workflow; `download_urfd.py --sequences fall-01 adl-01` is only a download
smoke check. Official URFD previews combine depth and RGB: the extractor requires
240x640 previews and uses only the right 320x240 RGB half. Supplied depth columns
are never model inputs. Consult the [dataset owner's page](https://fenix.ur.edu.pl/~mkepski/ds/uf.html)
for its academic/noncommercial terms; the prior audit records CC BY-NC-SA 4.0.
Datasets and media are excluded from the ZIP.

The experiment fits static pose, temporal pose, RGB and learned fusion variants
on training videos, fits Platt mapping on calibration videos, selects thresholds
on validation videos, and evaluates a frozen artifact on test videos. Brier/ECE
and coverage are reported; missing predictions count as missed positives. Splits
are video-wise, not subject-held-out, and unknown RGB pretraining overlap remains
a limitation. To use a reviewed artifact, set `evidence.model` and disable
`risk.enabled`; otherwise the pretrained risk branch remains primary. No claim
of a calibrated future fall risk follows from this experiment.

`scripts/benchmark_models.py --device 0 --runs 15` measures the local bus fixture
with nano/small pose and RGB models. Its `--openvino` branch is optional, requires
OpenVINO, and is not validated in this cleanup. Existing benchmark outputs remain
local. No training/calibration or full benchmark sweep was initiated here.

## Troubleshooting

| Symptom | Action |
|---|---|
| Broken/missing `.venv` Python | Rerun setup with a user-installed Python 3.11 path; do not copy someone else's virtual environment |
| Missing/corrupt optional model | Read `/status` and terminal error; rerun `scripts/download_models.py`, then restart; matching files are retained |
| Hash mismatch / offline download | Check connectivity and retry the pinned artifact; do not bypass its hash or silently substitute weights |
| Camera unavailable / no frames | Verify index, Windows camera permission, and other camera apps; Stop, then retry; the failure remains explicit |
| Camera still stopping | Let the native operation finish before restart; the preview stays cleared; a scene worker timeout is surfaced as failure |
| No temporal confidence | Stand with torso and legs visible; allow baseline/window warmup; review pose coverage and enabled model status |
| No scene label | A confident object region and matching relationship are required; missing/stale relations are expected |
| CUDA failure or low FPS | Try `--device cpu` or Faster/416 input; frame IDs can skip because old acquisitions are dropped; inspect measured latency |
| UNKNOWN posture | Step back for full-body geometry; low-confidence joints remain unknown rather than safe |
| Single image has null probability | Expected; examine independent RGB evidence, not a fabricated temporal result |
| Browser retains an old script | Reload after updating; this cleanup versions the changed static script URL |
| Slow/offline WebSocket client | Reconnect; track session/frame IDs; latest snapshots are not a durable event log |
| Optional TensorFlow missing | Install `requirements-training.txt` only for explicit Keras use; default inference does not need it |

## Verification and limits

Cleanup verification date: **16 October 2026, Asia/Kuala_Lumpur**. The pre-edit
baseline was a clean Git working tree at `b648f01`, 42 passing tests, passing Ruff,
and `pip check` without broken requirements. A Git bundle and source ZIP checkpoint
were created outside the project before editing. Existing ignored models, datasets,
generated results and environments were preserved.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip install ruff
.\.venv\Scripts\python.exe -m ruff check app scripts tests configs
.\.venv\Scripts\python.exe -m compileall -q app scripts tests configs
.\.venv\Scripts\python.exe -m pip check
```

The final suite has **68 passing tests**. New regressions reproduced failures
before repair for raw-confidence semantics, disconnected temporal inference,
scene continuity, missing branch scores, camera state/late publication, and
invalid fallback FPS. Added checks also cover packet image/JSON alignment,
single-image isolation, origin/option validation, explicit configuration,
tracker choice, temporal cadence and shutdown timeout reporting.

Real checks on Windows/Python 3.11.16, PyTorch 2.11.0+cu128, MX550, Ultralytics
8.4.171, OpenCV 4.14.0, NumPy 2.4.6 and ONNX Runtime 1.30.0 included:

- All 28 pinned model artifacts matched their hashes. Pose, RGB, pretrained
  tree/fusion and scene models loaded; real scene graph inference ran. The public
  bus fixture retained the same four people, boxes, keypoints, posture and evidence
  fields before/after repair, excluding timing and corrected reason text.
- A 90-frame RGB-only crop of local URFD `fall-01-cam0.mp4` was processed with
  valid JSON; temporal states reached calibrating, collecting and ready. The real
  CLI evaluator produced a decodable 90-frame annotated MP4 and matching CSV/JSONL.
  This bounded fixture is not an accuracy evaluation of the full fall episode.
- The real OpenCV launcher ran 15 headless webcam frames with all configured
  models, wrote observation JSONL, and exited/released capture. Q/window close is
  additionally covered by deterministic tests; no controlled participant trial
  was performed.
- Real HTTP liveness/status/image inference, increasing `/ws/vision` frames and
  matching `/ws/monitor` JPEG/metadata dimensions passed. Browser Start/Stop,
  fresh-session restart, 416 quality, baseline reset, all overlay toggles,
  persistence across reload, fullscreen/Escape and cleared stopped preview were
  exercised. Browser console reported no errors. Unavailable camera 20 reported
  an explicit failure and stopped cleanly.
- Real sequence extraction and unlabeled window preparation ran without training
  or inventing labels. An independent CPU virtual environment installed the
  documented PyTorch pair and runtime requirements with passing dependency checks.
  The final archive is extracted and checked using that isolated environment.

The development test client emits one upstream Starlette/httpx deprecation
warning. It is not a test failure. Remaining unverified areas are real Keras
training/inference, optional event fitting/calibration, OpenVINO, other platforms
and GPUs, long-duration/multiple-camera operation, controlled full-body posture
trials, subject-held-out fall accuracy, and downstream subsystem integration.
Synthetic multi-person/history tests are not evidence of perfect real-world
identity tracking. No guarantee beyond these checks is made.

## Verified history and cleanup decisions

Git records these milestones; earlier documentation observations are historical
records, not rerun accuracy claims:

- `2db7995` (1 October 2026): initial repository.
- `27000dc` (2 October): YOLO26 posture/temporal POC, geometry, normalized COCO17
  histories, offline tools and API. `df0a7d1` repaired Windows interpreter discovery
  and added the webcam launcher. The earlier verification guide recorded 42 tests,
  CPU image/webcam/API/evaluation smokes and untrained extraction windows.
- `cfe8b75` (2 October): browser monitoring and model-backed evidence.
- `b648f01` (4 October): RelateAnything deployment graph, pretrained tree/fusion,
  standing baseline, latest-frame camera architecture and atomic preview transport.
  These supersede earlier statements that ByteTrack/RGB/context were future work.

The prior H5 compatibility audit is preserved here because it explains why those
weights remain inactive. `Rchit13/posture-aware-fall-detection` at
`68b177a31300eecc76fdf5a51b7eac2c4f117263` shipped a `(30,20)` posture model and
`(60,24)` fall models/scalers. Its inference used a shared 30-frame setting and
raw mid-hip pixel offsets; fall training instead described `(30,20)` ordinary
LSTMs, incompatible with the shipped BiLSTM artifacts. The posture confidence
handling also differed between training and inference; scalers recorded
scikit-learn 1.6.1. Matching dimensions alone does not resolve preprocessing,
cadence, missing joints, or the YOLOv8-to-YOLO26 distribution change. The audit
adapter and its tests remain, but those H5 weights/scalers are not enabled or
refitted to appear compatible.

Other recorded research decisions: Y1D1R's contour/geometry application
(`82a02761c8a35819d24d5ec4be701cb0f7c480c9`) supplied conceptual persistence
ideas, not a causal pretrained predictor or copied email integration.
Supungamlath's feature approach (`a7dadc5d0d5993d376bfc6032b03e5a3cfe122e3`)
informed independent motion diagnostics. TSSTG was deferred for weight/license,
AlphaPose transfer and action-vocabulary uncertainty; PoseC3D/PYSKL for domain,
preprocessing and dependency cost; DeepFall's depth/thermal anomaly task was not
an RGB drop-in. Earlier notes about withdrawn UP-Fall RGB access or uncertain
Le2i access describe the prior audit, not newly verified dataset availability.
No missing `MODEL_EVIDENCE.md` or unpublished accuracy report is claimed.

Material cleanup changes (16 October): repaired model loading in the OpenCV/video
paths, connected explicit Keras manifests, corrected raw-score/probability and
posture-reason semantics, prevented stale scene results after context reset,
reset mutated risk state after inference errors, made missing-score overlays safe,
honored configuration/tracker choices, repaired stop/publication state and
fullscreen Escape, and declared direct runtime dependencies. First-party setup,
architecture, integration, research, training and validation guides were merged
into this README; the nine superseded Markdown guides were removed. One unused
RGB protocol and an unused detector class attribute were removed. Existing launch
aliases, experimental training/calibration tools, historical geometry/state
modules and audit adapters were retained for compatibility and explicit research
use. No speculative experiments were enabled by default.

## Teammate handoff

1. Extract the ZIP into a writable directory and open its `Vision-Analytics-AI`
   folder. Install/select 64-bit Python 3.11; create a new environment using the
   quick-start helper or CPU recipe. Never copy the developer's `.venv`.
2. Run the pinned downloader to obtain excluded scene/temporal assets; it verifies
   included weights too. Review upstream terms for any downloaded research assets.
3. Launch `.\run_demo.cmd --open-browser`, or add `--device cpu` for CPU. Confirm
   the enabled model branches are loaded and the camera source is correct.
4. Keep the body visible, stand for baseline preparation, then check track IDs,
   pose, posture/activity and measured FPS. Test Stop/Start, a changed quality,
   baseline reset and Escape from fullscreen. Missing evidence must remain
   unassessed. Do not stage an unsafe fall for a smoke check.
5. Run the tests, dependency check, and a bounded 15-frame headless run from the
   commands above. For downstream integration, inspect schema 2.0 JSON and connect
   to `/ws/vision`; agree safety decisions with the downstream owner.

The export deliberately includes first-party source, web resources, configs,
requirements, tests, CI metadata, README, pose/RGB weights and license/notices.
It excludes Git history, environments, caches, editor state, logs, generated
results, credentials, real `.env` files, datasets/media, backups, prior archives,
OpenVINO exports, and the excluded model cache described above. Local copies are
preserved. The final ZIP's exact path, size and SHA-256 are supplied with the
handoff message, outside the archive so it cannot include itself.
