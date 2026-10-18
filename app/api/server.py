"""Local monitoring API. One model owner, fresh tracks per camera session."""
import asyncio
import io
import json
import logging
import os
import struct
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from app.capture.monitor import CameraMonitor
from app.config import AppConfig, load_config
from app.fusion.pipeline import PoseBackend, VisionPipeline
from app.schemas import VisionObservation

logger = logging.getLogger(__name__)
WEB_ROOT = Path(__file__).resolve().parent.parent / "web"


class ObservationStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None
        self.camera_status = "disabled"
        self.error = None
        self.latest_jpeg = None
        self.latest_packet = None
        self.revision = 0

    def publish(self, observation: VisionObservation, frame: np.ndarray | None = None):
        payload = observation.model_dump(mode="json")
        jpeg, packet = None, None
        if frame is not None:
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                jpeg = encoded.tobytes()
                header = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
                # Keep image and observation atomic under client backpressure.
                packet = struct.pack("<I", len(header)) + header + jpeg
        with self.lock:
            self.latest, self.latest_jpeg, self.latest_packet = payload, jpeg, packet
            self.camera_status, self.error = "running", None
            self.revision += 1

    def snapshot(self):
        with self.lock:
            return self.latest

    def snapshot_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def monitor_snapshot(self):
        with self.lock:
            return self.revision, self.latest_packet, self.camera_status

    def clear(self, status, error=None):
        with self.lock:
            self.latest = self.latest_jpeg = self.latest_packet = None
            self.camera_status, self.error = status, error
            self.revision += 1

    def set_status(self, status, error=None):
        with self.lock:
            self.camera_status, self.error = status, error


class CameraStart(BaseModel):
    source: str | None = Field(default=None, pattern="^(rtsp|webcam)$")
    index: int | None = Field(default=None, ge=0, le=20)
    image_size: int | None = Field(default=None, ge=320, le=960, multiple_of=32)


class PTZMove(BaseModel):
    direction: str = Field(pattern="^(up|down|left|right)$")


class DatasetClassSelect(BaseModel):
    class_name: str


def create_app(config: AppConfig | None = None, pose_backend: PoseBackend | None = None,
               store: ObservationStore | None = None) -> FastAPI:
    cfg = config or load_config(os.getenv("SIDP_CONFIG"))
    if config is None and os.getenv("SIDP_CAMERA_ENABLED") is not None:
        cfg.api.camera_enabled = os.environ["SIDP_CAMERA_ENABLED"] == "1"
    observations = store or ObservationStore()
    model_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        backend = pose_backend
        if backend is None:
            from app.inference.yolo26_pose import Yolo26PoseBackend
            backend = await asyncio.to_thread(Yolo26PoseBackend, cfg.pose)
        api.state.backend = backend
        if hasattr(backend, "warmup"):
            await asyncio.to_thread(backend.warmup, cfg.camera.width, cfg.camera.height)
        models = VisionPipeline(cfg, backend)
        await asyncio.to_thread(models.load_optional_models)
        api.state.pipeline = models
        api.state.fall_backend = models.fall_backend
        monitor = CameraMonitor(cfg, models, observations, model_lock)
        api.state.monitor = monitor
        if cfg.api.camera_enabled:
            monitor.start()
        try:
            yield
        finally:
            await asyncio.to_thread(monitor.stop)
            models.close()

    api = FastAPI(title="SIDP Vision Monitor", version="0.2.0", lifespan=lifespan)
    api.state.store = observations
    api.mount("/assets", StaticFiles(directory=WEB_ROOT), name="assets")

    @api.middleware("http")
    async def local_controls(request, call_next):
        origin = request.headers.get("origin")
        if request.method == "POST" and origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Use the controls on this monitor page"}, status_code=403)
        return await call_next(request)

    @api.get("/health")
    def health():
        return {"status": "ok", "schema_version": "2.0"}

    @api.get("/status")
    def status():
        latest = observations.snapshot()
        with observations.lock:
            camera_status, error = observations.camera_status, observations.error
        models = api.state.pipeline
        monitor = api.state.monitor
        live = monitor.pipeline or models
        ptz = monitor.ptz
        return {
            "camera": camera_status, "camera_error": error,
            "camera_source": cfg.camera.source,
            "camera_address": cfg.camera.display_address,
            "camera_index": cfg.camera.index, "image_size": cfg.pose.image_size,
            "last_frame_id": latest["frame_id"] if latest else None,
            "device": getattr(api.state.backend, "device_name", getattr(api.state.backend, "device", cfg.pose.device)),
            "pose_model": Path(cfg.pose.model).stem, "tracking": cfg.tracker.backend,
            "ptz": {
                "available": bool(ptz and ptz.available),
                "patrolling": bool(ptz and ptz.patrolling),
                "mode": ptz.mode_label if ptz else "OFFLINE",
                "error": ptz.error if ptz else None,
            },
            "dataset": monitor.collector.status(),
            "models": [
                {"name": Path(cfg.pose.model).stem, "role": "People & skeleton", "ready": True, "error": None},
                {"name": "RelateAnything", "role": "Scene labels", "ready": models.scene_backend is not None,
                 "enabled": cfg.scene.enabled,
                 "error": live.scene_worker.error if live.scene_worker else models.scene_error},
                {"name": ("Retrained Keras" if models.fall_backend else "Event classifier"
                          if models.event_classifier and not models.risk_backend else "Tree + ST-GCN / TCN"),
                 "role": "Temporal fall evidence",
                 "enabled": bool(models.fall_backend or cfg.risk.enabled or cfg.evidence.model),
                 "ready": any(model is not None for model in (
                     models.fall_backend, models.risk_backend, models.event_classifier)),
                 "error": live.risk_error or models.risk_error or live.event_error or models.event_error},
                {"name": "RGB posture", "role": "Posture confidence", "ready": models.appearance_backend is not None,
                 "enabled": cfg.secondary.enabled,
                 "error": live.appearance_error or models.appearance_error},
            ],
            "temporal": ("retrained Keras (uncalibrated)" if models.fall_backend
                         else "pretrained Normal / Pre-fall / Fall" if models.risk_backend
                         else "event classifier" if models.event_classifier else "unavailable"),
            "schema_version": "2.0",
        }

    @api.post("/api/camera/start")
    def start_camera_endpoint(options: CameraStart):
        try:
            source = options.source
            if source is None and options.index is not None:
                source = "webcam"
            api.state.monitor.start(options.index, options.image_size, source=source)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None
        return status()

    @api.post("/api/camera/stop")
    def stop_camera_endpoint():
        api.state.monitor.stop()
        return status()

    @api.post("/api/session/reset")
    def reset_baseline_endpoint():
        api.state.monitor.reset_risk()
        return {"status": "baseline_reset"}

    @api.post("/api/ptz/move")
    def ptz_move_endpoint(cmd: PTZMove):
        ptz = api.state.monitor.ptz
        if ptz is None or not ptz.available:
            raise HTTPException(409, "ONVIF PTZ is offline or camera is not running in RTSP mode")
        actions = {"up": ptz.tilt_up, "down": ptz.tilt_down, "left": ptz.pan_left, "right": ptz.pan_right}
        actions[cmd.direction]()
        return status()

    @api.post("/api/ptz/patrol")
    def ptz_patrol_endpoint():
        ptz = api.state.monitor.ptz
        if ptz is None or not ptz.available:
            raise HTTPException(409, "ONVIF PTZ is offline or camera is not running in RTSP mode")
        ptz.toggle_patrol()
        return status()

    @api.post("/api/dataset/class")
    def dataset_class_endpoint(cmd: DatasetClassSelect):
        api.state.monitor.collector.select_class(cmd.class_name)
        return status()

    @api.post("/api/dataset/snap")
    def dataset_snap_endpoint():
        try:
            saved = api.state.monitor.snap_clean_frame()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None
        res = status()
        res["saved_path"] = str(saved)
        return res

    @api.post("/api/dataset/record")
    def dataset_record_endpoint():
        api.state.monitor.collector.toggle_recording()
        return status()

    @api.get("/", response_class=HTMLResponse)
    def dashboard():
        return (WEB_ROOT / "index.html").read_text(encoding="utf-8")

    @api.get("/stream.mjpg")
    async def stream_mjpg():
        async def generator():
            previous = None
            while True:
                revision, _, camera = observations.monitor_snapshot()
                jpeg = observations.snapshot_jpeg()
                if jpeg is not None and revision != previous and camera == "running":
                    previous = revision
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode()
                           + b"\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(1/cfg.api.preview_hz)
        return StreamingResponse(generator(), media_type="multipart/x-mixed-replace; boundary=frame")

    @api.post("/infer/image", response_model=VisionObservation)
    async def infer_image(request: Request):
        if request.headers.get("content-type", "").split(";")[0] not in ("image/jpeg", "image/png", "application/octet-stream"):
            raise HTTPException(415, "send JPEG or PNG bytes as the request body")
        payload = bytearray()
        async for chunk in request.stream():
            payload.extend(chunk)
            if len(payload) > cfg.api.max_image_bytes:
                raise HTTPException(413, "image exceeds configured byte limit")
        try:
            with Image.open(io.BytesIO(payload)) as image:
                if image.width*image.height > cfg.api.max_image_pixels:
                    raise HTTPException(413, "image exceeds configured pixel limit")
                if image.format not in ("JPEG", "PNG"):
                    raise HTTPException(415, "only JPEG and PNG encodings are accepted")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(400, "invalid image") from None
        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(400, "invalid image")

        def run():
            with model_lock:
                models = api.state.pipeline
                isolated = VisionPipeline(cfg, api.state.backend, camera_id="image_request",
                                          fall_backend=models.fall_backend,
                                          appearance_backend=models.appearance_backend,
                                          event_classifier=models.event_classifier)
                return isolated.process(frame, perf_counter(), 0)
        return await asyncio.to_thread(run)

    @api.websocket("/ws/vision")
    async def vision_stream(websocket: WebSocket):
        await websocket.accept()
        previous = None
        try:
            while True:
                current = observations.snapshot()
                if current and (current["session_id"], current["frame_id"]) != previous:
                    await websocket.send_json(current)
                    previous = current["session_id"], current["frame_id"]
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=1/cfg.api.websocket_hz)
                    if message["type"] == "websocket.disconnect":
                        break
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            pass

    @api.websocket("/ws/monitor")
    async def monitor_stream(websocket: WebSocket):
        origin = websocket.headers.get("origin")
        expected = str(websocket.url).replace("ws://", "http://").replace("wss://", "https://").split("/ws/")[0]
        if origin and origin.rstrip("/") != expected:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        previous = -1
        last_status = -float("inf")
        try:
            while True:
                revision, packet, camera = observations.monitor_snapshot()
                if revision != previous and packet is not None and camera == "running":
                    await websocket.send_bytes(packet)
                previous = revision
                if perf_counter()-last_status >= .75:
                    await websocket.send_json({"type": "status", **status()})
                    last_status = perf_counter()
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=1/cfg.api.preview_hz)
                    if message["type"] == "websocket.disconnect":
                        break
                except asyncio.TimeoutError:
                    pass
        except (WebSocketDisconnect, RuntimeError):
            pass

    return api


app = create_app()
