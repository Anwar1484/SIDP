"""Subsystem 1 Tri-Mode ONVIF PTZ motor controller for the Tapo C210 camera."""
import logging
import threading
import time
from dataclasses import dataclass
from time import perf_counter

from app.config import CameraConfig, PTZConfig

logger = logging.getLogger(__name__)

MODES = ("STATIC", "MANUAL", "AUTOMATIC")


@dataclass
class AutoTrackingState:
    active: bool = False
    occupant_count: int = 0
    group_bbox: tuple[int, int, int, int] | None = None
    collective_center: tuple[int, int] | None = None
    error_x: float = 0.0
    error_y: float = 0.0
    in_deadband: bool = True
    cmd_pan: float = 0.0
    cmd_tilt: float = 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


class PTZController:
    """
    Stateful Tri-Mode ONVIF PTZ controller:
      - STATIC: Disables all motor movement to minimize mechanical wear.
      - MANUAL: Non-blocking WASD velocity vectors via ONVIF ContinuousMove.
      - AUTOMATIC: Multi-person collective centroid proportional controller
        with 15% center deadband and 300 ms SOAP command rate-limiting.
    Also preserves 35-degree patrol sweep support (`Z`).
    """

    DEADBAND: float = 0.15
    KP: float = 0.65
    THROTTLE_SECONDS: float = 0.30

    def __init__(self, camera: CameraConfig, config: PTZConfig, initial_mode: str = "MANUAL"):
        self.camera = camera
        self.config = config
        self.available = False
        self.patrolling = False
        self.stopped = False
        self.mode = initial_mode if initial_mode in MODES else "MANUAL"
        self.error: str | None = None
        self.last_motion_time: float = -float("inf")
        self.last_auto_cmd_time: float = 0.0
        self.auto_state = AutoTrackingState(active=(self.mode == "AUTOMATIC"))
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
            logger.info("onvif_ptz_ready host=%s port=%d mode=%s", camera.host, config.port, self.mode)
        except Exception as exc:
            self.error = str(exc)
            logger.warning("onvif_ptz_unavailable host=%s port=%d error=%s", camera.host, config.port, exc)

    @property
    def mode_label(self) -> str:
        if self.patrolling:
            return "PATROL 35°"
        if self.mode == "AUTOMATIC":
            return "AUTO-TRACKING"
        return self.mode

    def set_mode(self, new_mode: str) -> str:
        upper = new_mode.upper()
        if upper not in MODES:
            return self.mode
        if self.patrolling:
            self.stop_patrol()
        self.mode = upper
        self.auto_state = AutoTrackingState(active=(self.mode == "AUTOMATIC"))
        self._brake_async()
        logger.info("ptz_mode_changed mode=%s", self.mode)
        return self.mode

    def cycle_mode(self) -> str:
        """Toggle sequentially: STATIC -> MANUAL -> AUTOMATIC -> STATIC."""
        idx = MODES.index(self.mode) if self.mode in MODES else 0
        next_mode = MODES[(idx + 1) % len(MODES)]
        return self.set_mode(next_mode)

    def is_moving(self, now: float | None = None) -> bool:
        current = perf_counter() if now is None else now
        return self.patrolling or (current - self.last_motion_time) <= self.config.settle_seconds

    def step(self, pan: float = 0.0, tilt: float = 0.0, duration: float | None = None,
             allow_in_auto: bool = False) -> bool:
        if self.stopped or self.patrolling:
            return False
        if self.mode == "STATIC":
            return False
        if self.mode == "AUTOMATIC" and not allow_in_auto:
            return False
        if not self.available:
            return False
        step_duration = duration if duration is not None else self.config.step_seconds

        def _actuate():
            with self._lock:
                if not self.available or self.patrolling or self.stopped or self.mode == "STATIC":
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

    def update_auto_tracking(
        self,
        bboxes: list[list[float] | tuple[float, float, float, float]],
        frame_width: int,
        frame_height: int,
        now: float | None = None,
    ) -> AutoTrackingState:
        """
        Mode 3 (AUTOMATIC): Calculates collective bounding box & centroid (Cx, Cy)
        across all detected occupants, normalized error (ex, ey) in [-1, 1],
        15% deadband, and rate-limited (300 ms) proportional velocity actuation.
        """
        if self.mode != "AUTOMATIC":
            self.auto_state = AutoTrackingState(active=False, occupant_count=len(bboxes))
            return self.auto_state

        now_ts = perf_counter() if now is None else now
        if not bboxes or frame_width <= 0 or frame_height <= 0:
            self.auto_state = AutoTrackingState(active=True, occupant_count=0, in_deadband=True)
            return self.auto_state

        x_min = min(float(b[0]) for b in bboxes)
        y_min = min(float(b[1]) for b in bboxes)
        x_max = max(float(b[2]) for b in bboxes)
        y_max = max(float(b[3]) for b in bboxes)

        c_x = 0.5 * (x_min + x_max)
        c_y = 0.5 * (y_min + y_max)

        half_w = 0.5 * frame_width
        half_h = 0.5 * frame_height
        e_x = _clamp((c_x - half_w) / max(half_w, 1e-6), -1.0, 1.0)
        e_y = _clamp((c_y - half_h) / max(half_h, 1e-6), -1.0, 1.0)

        v_pan = 0.0
        v_tilt = 0.0
        if abs(e_x) > self.DEADBAND:
            v_pan = _clamp(self.KP * e_x, -self.config.pan_speed, self.config.pan_speed)
        if abs(e_y) > self.DEADBAND:
            v_tilt = _clamp(-self.KP * e_y, -self.config.tilt_speed, self.config.tilt_speed)

        in_deadband = (v_pan == 0.0 and v_tilt == 0.0)
        if not in_deadband and (now_ts - self.last_auto_cmd_time) >= self.THROTTLE_SECONDS:
            self.last_auto_cmd_time = now_ts
            self.step(pan=v_pan, tilt=v_tilt, duration=0.22, allow_in_auto=True)

        self.auto_state = AutoTrackingState(
            active=True,
            occupant_count=len(bboxes),
            group_bbox=(int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max))),
            collective_center=(int(round(c_x)), int(round(c_y))),
            error_x=e_x,
            error_y=e_y,
            in_deadband=in_deadband,
            cmd_pan=v_pan,
            cmd_tilt=v_tilt,
        )
        return self.auto_state

    def toggle_patrol(self) -> bool:
        if not self.available or self.mode == "STATIC":
            return False
        if self.patrolling:
            self.stop_patrol()
        else:
            self.start_patrol()
        return self.patrolling

    def start_patrol(self) -> bool:
        if not self.available or self.stopped or self.mode == "STATIC":
            return False
        if self.patrolling and self._patrol_thread and self._patrol_thread.is_alive():
            return True
        self.patrolling = True
        self._patrol_thread = threading.Thread(target=self._patrol_worker, name="sidp-ptz-patrol", daemon=True)
        self._patrol_thread.start()
        return True

    def stop_patrol(self):
        self.patrolling = False
        self._brake_async()

    def _brake(self):
        if not self.available or self._ptz is None:
            return
        try:
            self._ptz.Stop({"ProfileToken": self._token})
            self.last_motion_time = perf_counter()
        except Exception:
            pass

    def _brake_async(self):
        if not self.available or self._ptz is None:
            return
        threading.Thread(target=self._brake, name="sidp-ptz-stop", daemon=True).start()

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
