"""Subsystem 1 ONVIF PTZ motor controller for the Tapo C210 camera."""
import logging
import threading
import time
from time import perf_counter

from app.config import CameraConfig, PTZConfig

logger = logging.getLogger(__name__)


class PTZController:
    """Non-blocking ONVIF pan/tilt stepping and continuous 35-degree patrol sweep."""

    def __init__(self, camera: CameraConfig, config: PTZConfig):
        self.camera = camera
        self.config = config
        self.available = False
        self.patrolling = False
        self.stopped = False
        self.error: str | None = None
        self.last_motion_time: float = -float("inf")
        self._lock = threading.Lock()
        self._ptz = None
        self._token = None
        self._patrol_thread: threading.Thread | None = None

        if not config.enabled or camera.source != "rtsp":
            self.error = "PTZ disabled for local webcam source"
            return

        try:
            from onvif import ONVIFCamera
            cam = ONVIFCamera(camera.host, config.port, camera.username, camera.password)
            media = cam.create_media_service()
            self._ptz = cam.create_ptz_service()
            profiles = media.GetProfiles()
            self._token = profiles[0].token
            self.available = True
            logger.info("onvif_ptz_ready host=%s port=%d", camera.host, config.port)
        except Exception as exc:
            self.error = str(exc)
            logger.warning("onvif_ptz_unavailable host=%s port=%d error=%s", camera.host, config.port, exc)

    @property
    def mode_label(self) -> str:
        if not self.available:
            return "OFFLINE"
        return "PATROL 35°" if self.patrolling else "MANUAL"

    def is_moving(self, now: float | None = None) -> bool:
        current = perf_counter() if now is None else now
        return self.patrolling or (current - self.last_motion_time) <= self.config.settle_seconds

    def step(self, pan: float = 0.0, tilt: float = 0.0, duration: float | None = None) -> bool:
        if not self.available or self.patrolling or self.stopped:
            return False
        step_duration = duration if duration is not None else self.config.step_seconds

        def _actuate():
            with self._lock:
                if not self.available or self.patrolling or self.stopped:
                    return
                try:
                    self.last_motion_time = perf_counter()
                    request = self._ptz.create_type("ContinuousMove")
                    request.ProfileToken = self._token
                    request.Velocity = {"PanTilt": {"x": float(pan), "y": float(tilt)}}
                    self._ptz.ContinuousMove(request)
                    time.sleep(step_duration)
                    self._ptz.Stop({"ProfileToken": self._token})
                    self.last_motion_time = perf_counter()
                except Exception as exc:
                    self.error = str(exc)
                    logger.warning("ptz_step_failed error=%s", exc)

        threading.Thread(target=_actuate, name="sidp-ptz-step", daemon=True).start()
        return True

    def pan_left(self) -> bool:
        return self.step(pan=-self.config.pan_speed, tilt=0.0)

    def pan_right(self) -> bool:
        return self.step(pan=self.config.pan_speed, tilt=0.0)

    def tilt_up(self) -> bool:
        return self.step(pan=0.0, tilt=self.config.tilt_speed)

    def tilt_down(self) -> bool:
        return self.step(pan=0.0, tilt=-self.config.tilt_speed)

    def toggle_patrol(self) -> bool:
        if not self.available:
            return False
        if self.patrolling:
            self.stop_patrol()
        else:
            self.start_patrol()
        return self.patrolling

    def start_patrol(self) -> bool:
        if not self.available or self.stopped:
            return False
        if self.patrolling and self._patrol_thread and self._patrol_thread.is_alive():
            return True
        self.patrolling = True
        self._patrol_thread = threading.Thread(target=self._patrol_worker, name="sidp-ptz-patrol", daemon=True)
        self._patrol_thread.start()
        return True

    def stop_patrol(self):
        self.patrolling = False
        self._brake()

    def _brake(self):
        if not self.available or self._ptz is None:
            return
        try:
            self._ptz.Stop({"ProfileToken": self._token})
            self.last_motion_time = perf_counter()
        except Exception:
            pass

    def _patrol_worker(self):
        direction = -1
        sweep_steps = max(1, int(round(self.config.patrol_sweep_seconds / 0.1)))
        dwell_steps = max(1, int(round(self.config.patrol_dwell_seconds / 0.1)))
        while self.patrolling and not self.stopped:
            try:
                self.last_motion_time = perf_counter()
                request = self._ptz.create_type("ContinuousMove")
                request.ProfileToken = self._token
                request.Velocity = {"PanTilt": {"x": direction * self.config.pan_speed, "y": 0.0}}
                self._ptz.ContinuousMove(request)

                for _ in range(sweep_steps):
                    if not self.patrolling or self.stopped:
                        break
                    self.last_motion_time = perf_counter()
                    time.sleep(0.1)

                self._brake()

                for _ in range(dwell_steps):
                    if not self.patrolling or self.stopped:
                        break
                    time.sleep(0.1)

                direction *= -1
            except Exception as exc:
                self.error = str(exc)
                logger.warning("ptz_patrol_failed error=%s", exc)
                break
        self._brake()

    def close(self):
        self.patrolling = False
        self.stopped = True
        if self._patrol_thread and self._patrol_thread.is_alive():
            self._patrol_thread.join(timeout=1.5)
        self._brake()
