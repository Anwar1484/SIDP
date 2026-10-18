"""Latest-frame camera producer (RTSP or webcam), PTZ & dataset hooks, and inference consumer."""
import logging
import threading
from time import perf_counter

from app.capture.dataset import DatasetCollector
from app.capture.ptz import PTZController
from app.capture.webcam import WebcamCapture
from app.fusion.pipeline import VisionPipeline

logger = logging.getLogger(__name__)


class CameraMonitor:
    def __init__(self, config, models, store, model_lock):
        self.config, self.models, self.store, self.model_lock = config, models, store, model_lock
        self.lock = threading.RLock()
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.capture_thread = self.inference_thread = None
        self.latest = None
        self.latest_clean_frame = None
        self.pipeline = None
        self.ptz: PTZController | None = None
        self.collector = DatasetCollector(config.dataset)

    def alive(self):
        scene = getattr(self.pipeline, "scene_worker", None)
        return any(t is not None and t.is_alive() for t in (
            self.capture_thread, self.inference_thread, scene.thread if scene else None,
        ))

    def start(self, index=None, image_size=None, source=None):
        with self.lock:
            if self.alive():
                if self.stop_event.is_set():
                    raise RuntimeError("Camera is still stopping; retry in a moment")
                if ((index is not None and index != self.config.camera.index)
                        or (source is not None and source != self.config.camera.source)
                        or (image_size is not None and image_size != self.config.pose.image_size)):
                    raise RuntimeError("Stop monitoring before changing the camera or image size")
                return
            if source is not None:
                self.config.camera.source = source
            if index is not None:
                self.config.camera.index = index
                if source is None and index >= 0 and self.config.camera.source == "webcam":
                    self.config.camera.id = f"webcam_{index}"
            if self.config.camera.source == "rtsp":
                self.config.camera.id = "tapo_c210"
            else:
                self.config.camera.id = f"webcam_{self.config.camera.index}"
            if image_size is not None:
                self.config.pose.image_size = image_size
                self.models.pose_backend.config.image_size = image_size
            self.pipeline = VisionPipeline(
                self.config, self.models.pose_backend, fall_backend=self.models.fall_backend,
                appearance_backend=self.models.appearance_backend,
                event_classifier=self.models.event_classifier, risk_backend=self.models.risk_backend,
                scene_backend=self.models.scene_backend,
            )
            if self.ptz is not None:
                self.ptz.close()
            self.ptz = PTZController(self.config.camera, self.config.ptz)
            self.latest = None
            self.latest_clean_frame = None
            self.stop_event.clear()
            self.store.clear("starting")
            self.capture_thread = threading.Thread(target=self._capture, name="sidp-capture", daemon=True)
            self.inference_thread = threading.Thread(target=self._infer, name="sidp-inference", daemon=True)
            self.capture_thread.start()
            self.inference_thread.start()

    def stop(self):
        with self.lock:
            with self.condition:
                self.stop_event.set()
                self.latest = None
                self.store.clear("stopping" if self.alive() else "stopped")
                self.condition.notify_all()
            if self.ptz is not None:
                self.ptz.close()
                self.ptz = None
            for thread in (self.capture_thread, self.inference_thread):
                if thread and thread is not threading.current_thread():
                    thread.join(timeout=3)
            if not self.alive():
                if self.pipeline:
                    self.pipeline.close()
                self.store.clear("stopped")

    def snap_clean_frame(self):
        with self.condition:
            frame = None if self.latest_clean_frame is None else self.latest_clean_frame.copy()
        if frame is None:
            raise RuntimeError("No active camera frame available to capture")
        return self.collector.snap(frame)

    def _finished(self):
        with self.condition:
            others = (t for t in (self.capture_thread, self.inference_thread)
                      if t is not None and t is not threading.current_thread())
            if not any(t.is_alive() for t in others):
                self.latest = None
                if self.store.monitor_snapshot()[2] == "stopping":
                    self.store.clear("stopped")

    def _failed(self, exc):
        with self.condition:
            if not self.stop_event.is_set():
                self.store.clear("failed", str(exc))
            self.latest = None
            self.stop_event.set()
            self.condition.notify_all()

    def reset_risk(self):
        with self.model_lock:
            if self.pipeline:
                self.pipeline.reset_risk()

    def _capture(self):
        try:
            with WebcamCapture(self.config.camera) as camera:
                frame_id = 0
                while not self.stop_event.is_set():
                    frame, timestamp = camera.read()
                    self.collector.maybe_record(frame)
                    with self.condition:
                        if self.stop_event.is_set():
                            break
                        # Overwrite the last acquisition. No inference backlog.
                        self.latest_clean_frame = frame
                        self.latest = frame_id, frame, timestamp
                        self.condition.notify()
                    frame_id += 1
        except Exception as exc:
            logger.exception("camera_capture_failed")
            self._failed(exc)
        finally:
            self._finished()

    def _infer(self):
        previous_id = -1
        published_at = None
        fps = 0.0
        try:
            while not self.stop_event.is_set():
                iteration_start = perf_counter()
                with self.condition:
                    self.condition.wait_for(lambda: self.stop_event.is_set() or
                                            (self.latest is not None and self.latest[0] != previous_id), timeout=1)
                    if self.stop_event.is_set():
                        break
                    if self.latest is None or self.latest[0] == previous_id:
                        continue
                    frame_id, frame, timestamp = self.latest
                with self.model_lock:
                    observation = self.pipeline.process(frame, timestamp, frame_id)
                now = perf_counter()
                if published_at is not None:
                    instantaneous = 1 / max(now-published_at, 1e-6)
                    fps = instantaneous if not fps else fps*.8+instantaneous*.2
                observation.fps = fps
                observation.capture_to_publish_ms = max(0, (now-timestamp)*1000)
                # Serialize publication with stop/failure so a late frame cannot revive the preview.
                with self.condition:
                    if self.stop_event.is_set():
                        break
                    self.store.publish(observation, frame)
                published_at, previous_id = now, frame_id
                self.stop_event.wait(max(0, 1/self.config.api.inference_hz-(perf_counter()-iteration_start)))
        except Exception as exc:
            logger.exception("camera_inference_failed")
            self._failed(exc)
        finally:
            try:
                if self.pipeline:
                    self.pipeline.close()
            except Exception as exc:
                logger.exception("camera_shutdown_failed")
                self.store.clear("failed", str(exc))
            finally:
                self._finished()
