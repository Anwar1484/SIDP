#!/usr/bin/env python3
"""
subsystem1_2_integrated_v2.py
================================================================================
RBB4013 System Integrated Design Project (SIDP) - Universiti Teknologi PETRONAS
Project: Vision-Based Elderly Care Safety Monitoring System (Subsystems 1 & 2)
Target Architecture: Python 3.11+, OpenCV, Ultralytics YOLO-Pose (17 COCO keypoints),
                     Auxiliary RGB Fall Detector, ONVIF PTZ (Tapo C210)

Technical Upgrades in v2:
1. Expanded 6-Class Posture Detection Pipeline (Subsystem 2):
   - STANDING, SITTING, WALKING, BENDING (with >=60 deg hazard warning),
     FALLING, and LYING DOWN using deterministic 17-keypoint COCO geometry,
     bounding-box kinematics, rolling temporal trajectories, and secondary
     RGB fall detection (`models/human-fall-detection.pt`).
2. Tri-Mode PTZ Camera Controller (Subsystem 1):
   - STATIC: Blocks all motor actuation to eliminate mechanical wear.
   - MANUAL: Non-blocking WASD continuous velocity steering via ONVIF port 2020.
   - AUTOMATIC: Multi-person collective bounding box & centroid (Cx, Cy)
     proportional controller with a 15% center deadband and 300 ms SOAP command
     rate-limiting.
3. Thread-Safe RTSP Ingestion & Asynchronous ONVIF Execution:
   - Lock-protected zero-lag RTSP frame reader and background ONVIF worker
     thread so the OpenCV GUI never stutters.
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import queue
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Ensure workspace root is importable
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SIDP.IntegratedV2")


# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

DEFAULT_CAMERA_IP = os.getenv("SIDP_CAMERA_HOST", "172.20.10.4")
DEFAULT_CAMERA_USER = os.getenv("SIDP_CAMERA_USER", "SIDPProject")
DEFAULT_CAMERA_PASS = os.getenv("SIDP_CAMERA_PASSWORD", "SIDPUTP2026")
DEFAULT_CAMERA_STREAM = os.getenv("SIDP_CAMERA_STREAM", "stream2")
DEFAULT_RTSP_PORT = 554
DEFAULT_ONVIF_PORT = 2020

# COCO 17 Keypoint Indices
KP_NOSE = 0
KP_L_EYE = 1
KP_R_EYE = 2
KP_L_EAR = 3
KP_R_EAR = 4
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_ELBOW = 7
KP_R_ELBOW = 8
KP_L_WRIST = 9
KP_R_WRIST = 10
KP_L_HIP = 11
KP_R_HIP = 12
KP_L_KNEE = 13
KP_R_KNEE = 14
KP_L_ANKLE = 15
KP_R_ANKLE = 16

COCO_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


class PostureClass(str, Enum):
    """Expanded 6-posture taxonomy for Subsystem 2."""
    STANDING = "STANDING"
    SITTING = "SITTING"
    WALKING = "WALKING"
    BENDING = "BENDING"
    FALLING = "FALLING"
    LYING_DOWN = "LYING DOWN"


class CameraMode(str, Enum):
    """Tri-Mode Camera Control Architecture for Subsystem 1."""
    STATIC = "STATIC"
    MANUAL = "MANUAL"
    AUTOMATIC = "AUTOMATIC"

    @property
    def hud_tag(self) -> str:
        if self == CameraMode.AUTOMATIC:
            return "[AUTO-TRACKING]"
        return f"[{self.value}]"


# Color-coded BGR banners for OSD rendering
POSTURE_COLORS_BGR: dict[PostureClass, tuple[int, int, int]] = {
    PostureClass.STANDING: (50, 205, 50),      # Green
    PostureClass.SITTING: (50, 205, 50),       # Green
    PostureClass.WALKING: (50, 205, 50),       # Green
    PostureClass.BENDING: (0, 175, 255),       # Orange / Yellow
    PostureClass.FALLING: (30, 30, 240),       # Red
    PostureClass.LYING_DOWN: (30, 30, 240),    # Red
}


def clamp(val: float, low: float, high: float) -> float:
    """Clamp a scalar value to [low, high]."""
    return float(max(low, min(high, val)))


# ==============================================================================
# SUBSYSTEM 1: THREAD-SAFE RTSP STREAM INGESTION
# ==============================================================================

class ThreadSafeStreamReader:
    """
    Background buffer-flushing RTSP/Webcam stream reader.
    Protects frame read/write operations using `threading.Lock()` so the
    inference and UI loop always receives the most recent frame (<200 ms latency).
    """

    def __init__(
        self,
        rtsp_url: str,
        use_webcam: bool = False,
        webcam_index: int = 0,
        width: int = 640,
        height: int = 480,
        reconnect_delay: float = 2.0,
    ) -> None:
        self.rtsp_url = rtsp_url
        self.use_webcam = use_webcam
        self.webcam_index = webcam_index
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0.0
        self._frame_id: int = 0
        self._connected: bool = False
        self._stopped = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "ThreadSafeStreamReader":
        self._open_capture()
        self._thread = threading.Thread(
            target=self._ingest_loop,
            name="SIDP-RTSP-Ingestion",
            daemon=True,
        )
        self._thread.start()
        return self

    def _open_capture(self) -> bool:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        if self.use_webcam:
            cap = cv2.VideoCapture(self.webcam_index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.webcam_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        else:
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp|fflags;nobuffer|flags;low_delay")
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self._connected = False
            return False

        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            cap.release()
            self._connected = False
            return False

        with self._lock:
            self._cap = cap
            self._frame = first_frame
            self._timestamp = time.perf_counter()
            self._frame_id += 1
            self._connected = True
        return True

    def _ingest_loop(self) -> None:
        while not self._stopped.is_set():
            if self._cap is None or not self._connected:
                if not self._open_capture():
                    time.sleep(self.reconnect_delay)
                    continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                logger.warning("Stream read interrupted; attempting reconnect...")
                self._connected = False
                time.sleep(0.25)
                continue

            now = time.perf_counter()
            with self._lock:
                self._frame = frame
                self._timestamp = now
                self._frame_id += 1
                self._connected = True

    def read(self) -> tuple[bool, Optional[np.ndarray], float, int]:
        """Thread-safe retrieval of a defensive copy of the latest frame."""
        with self._lock:
            if self._frame is None:
                return False, None, 0.0, self._frame_id
            return self._connected, self._frame.copy(), self._timestamp, self._frame_id

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            self._connected = False


# ==============================================================================
# SUBSYSTEM 1: TRI-MODE ONVIF PTZ CAMERA CONTROLLER
# ==============================================================================

@dataclass
class AutoTrackTelemetry:
    """Telemetry computed by the AUTOMATIC group-centering controller."""
    active: bool = False
    occupant_count: int = 0
    group_bbox: Optional[tuple[int, int, int, int]] = None
    collective_center: Optional[tuple[int, int]] = None
    error_x: float = 0.0
    error_y: float = 0.0
    in_deadband: bool = True
    cmd_pan: float = 0.0
    cmd_tilt: float = 0.0


class TriModePTZController:
    """
    Stateful Tri-Mode ONVIF Motor Controller (`STATIC`, `MANUAL`, `AUTOMATIC`).
    Executes all ONVIF SOAP calls asynchronously on a dedicated background worker
    thread so the OpenCV rendering loop never blocks or stutters.
    """

    MODE_SEQUENCE: tuple[CameraMode, ...] = (
        CameraMode.STATIC,
        CameraMode.MANUAL,
        CameraMode.AUTOMATIC,
    )

    # Control parameters per specification
    MANUAL_PAN_SPEED: float = 0.4
    MANUAL_TILT_SPEED: float = 0.3
    MANUAL_PULSE_DURATION: float = 0.28

    AUTO_DEADBAND: float = 0.15
    AUTO_KP: float = 0.65
    AUTO_MAX_PAN: float = 0.4
    AUTO_MAX_TILT: float = 0.3
    AUTO_THROTTLE_SECONDS: float = 0.30  # 300 ms rate-limiting (within 250-350 ms spec)
    AUTO_PULSE_DURATION: float = 0.22

    def __init__(
        self,
        ip: str = DEFAULT_CAMERA_IP,
        port: int = DEFAULT_ONVIF_PORT,
        username: str = DEFAULT_CAMERA_USER,
        password: str = DEFAULT_CAMERA_PASS,
        enabled: bool = True,
        initial_mode: CameraMode = CameraMode.STATIC,
    ) -> None:
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.enabled = enabled

        self._mode = initial_mode
        self._lock = threading.Lock()
        self.available = False
        self.status_error: Optional[str] = None

        self._ptz_service = None
        self._media_profile_token: Optional[str] = None
        self._last_auto_dispatch_ts: float = 0.0
        self._last_motion_ts: float = -float("inf")
        self._is_moving_flag: bool = False
        self.telemetry = AutoTrackTelemetry()

        self._cmd_queue: queue.Queue[tuple[str, float, float, float]] = queue.Queue(maxsize=4)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        if self.enabled:
            self._connect_onvif()
        else:
            self.status_error = "PTZ disabled via CLI"

        self._worker_thread = threading.Thread(
            target=self._command_worker,
            name="SIDP-ONVIF-Worker",
            daemon=True,
        )
        self._worker_thread.start()

    def _connect_onvif(self) -> None:
        try:
            from onvif import ONVIFCamera

            cam = ONVIFCamera(self.ip, self.port, self.username, self.password)
            media_service = cam.create_media_service()
            ptz_service = cam.create_ptz_service()
            profiles = media_service.GetProfiles()
            if not profiles:
                raise RuntimeError("No ONVIF media profiles returned by camera")
            self._ptz_service = ptz_service
            self._media_profile_token = profiles[0].token
            self.available = True
            self.status_error = None
            logger.info("ONVIF PTZ initialized at %s:%d (token=%s)", self.ip, self.port, self._media_profile_token)
        except Exception as exc:
            self.available = False
            self.status_error = str(exc)
            logger.warning(
                "ONVIF PTZ unavailable at %s:%d (%s). Running in fallback simulation mode.",
                self.ip,
                self.port,
                exc,
            )

    @property
    def mode(self) -> CameraMode:
        with self._lock:
            return self._mode

    def set_mode(self, new_mode: CameraMode) -> CameraMode:
        with self._lock:
            self._mode = new_mode
            self.telemetry = AutoTrackTelemetry(active=(new_mode == CameraMode.AUTOMATIC))
        # Always brake motors on mode transition to prevent runaway movement
        self._enqueue_command("STOP", 0.0, 0.0, 0.0)
        logger.info("Camera control mode switched -> %s", new_mode.value)
        return new_mode

    def cycle_mode(self) -> CameraMode:
        """Cycle sequentially: STATIC -> MANUAL -> AUTOMATIC -> STATIC."""
        with self._lock:
            idx = self.MODE_SEQUENCE.index(self._mode)
            next_mode = self.MODE_SEQUENCE[(idx + 1) % len(self.MODE_SEQUENCE)]
        return self.set_mode(next_mode)

    def manual_steer(self, key_char: str) -> bool:
        """
        Handle W/A/S/D steering in MANUAL mode:
          W: Tilt Up   (tilt_speed = +0.3)
          S: Tilt Down (tilt_speed = -0.3)
          A: Pan Left  (pan_speed  = -0.4)
          D: Pan Right (pan_speed  = +0.4)
        Blocked in STATIC and AUTOMATIC modes.
        """
        if self.mode != CameraMode.MANUAL:
            return False

        k = key_char.upper()
        pan_v, tilt_v = 0.0, 0.0
        if k == "W":
            tilt_v = +self.MANUAL_TILT_SPEED
        elif k == "S":
            tilt_v = -self.MANUAL_TILT_SPEED
        elif k == "A":
            pan_v = -self.MANUAL_PAN_SPEED
        elif k == "D":
            pan_v = +self.MANUAL_PAN_SPEED
        else:
            return False

        self._enqueue_command("PULSE", pan_v, tilt_v, self.MANUAL_PULSE_DURATION)
        return True

    def update_automatic_tracking(
        self,
        bboxes: list[tuple[float, float, float, float]],
        frame_width: int,
        frame_height: int,
        now: Optional[float] = None,
    ) -> AutoTrackTelemetry:
        """
        Mode 3: AUTOMATIC GROUP-CENTERING.
        Calculates collective bounding box (Xmin, Ymin, Xmax, Ymax) and center (Cx, Cy)
        for all N >= 1 occupants, computes normalized error (ex, ey) in [-1, 1],
        applies deadband |ex|, |ey| <= 0.15, and dispatches rate-limited (300 ms)
        proportional velocity commands.
        """
        current_mode = self.mode
        if current_mode != CameraMode.AUTOMATIC:
            self.telemetry = AutoTrackTelemetry(active=False, occupant_count=len(bboxes))
            return self.telemetry

        now_ts = time.perf_counter() if now is None else now
        n_persons = len(bboxes)

        if n_persons == 0 or frame_width <= 0 or frame_height <= 0:
            # Hold last position when no persons are detected (do not drift)
            if self._is_moving_flag:
                self._enqueue_command("STOP", 0.0, 0.0, 0.0)
            self.telemetry = AutoTrackTelemetry(
                active=True,
                occupant_count=0,
                in_deadband=True,
            )
            return self.telemetry

        # 1. Multi-Person Collective Centroid
        x_min = min(b[0] for b in bboxes)
        y_min = min(b[1] for b in bboxes)
        x_max = max(b[2] for b in bboxes)
        y_max = max(b[3] for b in bboxes)

        c_x = 0.5 * (x_min + x_max)
        c_y = 0.5 * (y_min + y_max)

        # 2. Normalized Error Calculation in [-1.0, 1.0]
        half_w = 0.5 * frame_width
        half_h = 0.5 * frame_height
        e_x = clamp((c_x - half_w) / max(half_w, 1e-6), -1.0, 1.0)
        e_y = clamp((c_y - half_h) / max(half_h, 1e-6), -1.0, 1.0)

        # 3. Deadband Zone & Proportional Control
        v_pan = 0.0
        v_tilt = 0.0
        if abs(e_x) > self.AUTO_DEADBAND:
            v_pan = clamp(self.AUTO_KP * e_x, -self.AUTO_MAX_PAN, self.AUTO_MAX_PAN)
        if abs(e_y) > self.AUTO_DEADBAND:
            v_tilt = clamp(-self.AUTO_KP * e_y, -self.AUTO_MAX_TILT, self.AUTO_MAX_TILT)

        in_deadband = (v_pan == 0.0 and v_tilt == 0.0)

        # 4. Command Throttling / Rate-Limiting (once every 250-350 ms)
        if in_deadband:
            if self._is_moving_flag:
                self._enqueue_command("STOP", 0.0, 0.0, 0.0)
        elif (now_ts - self._last_auto_dispatch_ts) >= self.AUTO_THROTTLE_SECONDS:
            self._last_auto_dispatch_ts = now_ts
            self._enqueue_command("PULSE", v_pan, v_tilt, self.AUTO_PULSE_DURATION)

        self.telemetry = AutoTrackTelemetry(
            active=True,
            occupant_count=n_persons,
            group_bbox=(int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max))),
            collective_center=(int(round(c_x)), int(round(c_y))),
            error_x=e_x,
            error_y=e_y,
            in_deadband=in_deadband,
            cmd_pan=v_pan,
            cmd_tilt=v_tilt,
        )
        return self.telemetry

    def _enqueue_command(self, action: str, pan: float, tilt: float, duration: float) -> None:
        if self._stop_event.is_set():
            return
        # Drain stale queued commands so latest intention executes immediately
        while not self._cmd_queue.empty():
            try:
                self._cmd_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._cmd_queue.put_nowait((action, float(pan), float(tilt), float(duration)))
        except queue.Full:
            pass

    def _command_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                action, pan, tilt, duration = self._cmd_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if action == "STOP" or self.mode == CameraMode.STATIC:
                self._execute_stop()
                continue

            if action == "PULSE":
                self._execute_continuous_pulse(pan, tilt, duration)

    def _execute_continuous_pulse(self, pan: float, tilt: float, duration: float) -> None:
        self._is_moving_flag = True
        self._last_motion_ts = time.perf_counter()
        if not self.available or self._ptz_service is None or self._media_profile_token is None:
            time.sleep(min(duration, 0.1))
            self._is_moving_flag = False
            return

        try:
            request = self._ptz_service.create_type("ContinuousMove")
            request.ProfileToken = self._media_profile_token
            request.Velocity = {"PanTilt": {"x": float(pan), "y": float(tilt)}}
            self._ptz_service.ContinuousMove(request)

            end_time = time.perf_counter() + max(0.05, duration)
            while time.perf_counter() < end_time:
                if self._stop_event.is_set() or self.mode == CameraMode.STATIC or not self._cmd_queue.empty():
                    break
                time.sleep(0.03)

            self._ptz_service.Stop({"ProfileToken": self._media_profile_token})
            self._last_motion_ts = time.perf_counter()
        except Exception as exc:
            self.status_error = str(exc)
            logger.debug("ONVIF ContinuousMove error: %s", exc)
        finally:
            self._is_moving_flag = False

    def _execute_stop(self) -> None:
        self._is_moving_flag = False
        if not self.available or self._ptz_service is None or self._media_profile_token is None:
            return
        try:
            self._ptz_service.Stop({"ProfileToken": self._media_profile_token})
            self._last_motion_ts = time.perf_counter()
        except Exception as exc:
            logger.debug("ONVIF Stop error: %s", exc)

    def is_camera_moving(self, settle_window: float = 0.45) -> bool:
        return self._is_moving_flag or (time.perf_counter() - self._last_motion_ts) <= settle_window

    def shutdown(self) -> None:
        """Clean shutdown: halt motors via ptz.Stop() and terminate worker thread."""
        self._stop_event.set()
        self._execute_stop()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)


# ==============================================================================
# SUBSYSTEM 1: DATASET COLLECTOR (SUPERVISOR TRAINING SUPPORT)
# ==============================================================================

class DatasetCollector:
    """Saves clean unannotated frames for posture classifier training."""

    CLASSES: tuple[str, ...] = (
        "standing",
        "sitting",
        "walking",
        "bending",
        "falling",
        "lying_down",
    )

    def __init__(self, root_dir: Path = WORKSPACE_ROOT / "dataset", record_hz: float = 2.0) -> None:
        self.root_dir = root_dir
        self.record_hz = record_hz
        self.current_index = 0
        self.auto_recording = False
        self._last_save_ts = 0.0
        for cls_name in self.CLASSES:
            (self.root_dir / cls_name).mkdir(parents=True, exist_ok=True)

    @property
    def current_class(self) -> str:
        return self.CLASSES[self.current_index]

    def select_class_index(self, idx: int) -> str:
        if 0 <= idx < len(self.CLASSES):
            self.current_index = idx
        return self.current_class

    def toggle_auto_record(self) -> bool:
        self.auto_recording = not self.auto_recording
        return self.auto_recording

    def save_frame(self, clean_frame: np.ndarray) -> Path:
        target_dir = self.root_dir / self.current_class
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        out_path = target_dir / f"{self.current_class}_{stamp}.jpg"
        cv2.imwrite(str(out_path), clean_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        self._last_save_ts = time.perf_counter()
        return out_path

    def maybe_auto_record(self, clean_frame: np.ndarray) -> Optional[Path]:
        if not self.auto_recording:
            return None
        if (time.perf_counter() - self._last_save_ts) >= (1.0 / max(self.record_hz, 0.1)):
            return self.save_frame(clean_frame)
        return None


# ==============================================================================
# SUBSYSTEM 2: 6-CLASS GEOMETRIC & TEMPORAL POSTURE ENGINE
# ==============================================================================

@dataclass
class FrameKinematicsSample:
    """Single-frame kinematic snapshot stored in a track's rolling history."""
    timestamp: float
    centroid_x: float
    centroid_y: float
    bbox_w: float
    bbox_h: float
    aspect_ratio: float
    torso_angle: Optional[float]
    torso_length: Optional[float]
    hip_y: Optional[float]
    shoulder_y: Optional[float]
    ankle_sep_x: Optional[float]
    left_ankle_x: Optional[float]
    right_ankle_x: Optional[float]
    is_upright: bool
    raw_posture: PostureClass


@dataclass
class TrackHistoryState:
    """Rolling temporal state for an individual tracked occupant."""
    track_id: int
    samples: deque[FrameKinematicsSample] = field(default_factory=lambda: deque(maxlen=30))
    posture_votes: deque[PostureClass] = field(default_factory=lambda: deque(maxlen=5))
    fall_latched_until: float = 0.0
    reference_height_px: float = 280.0
    last_seen_ts: float = 0.0


@dataclass
class PersonAnalysisResult:
    """Complete per-person output from the 6-class posture classification pipeline."""
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    keypoints_xyc: np.ndarray  # shape (17, 3)
    posture: PostureClass
    display_label: str
    torso_angle_deg: Optional[float]
    vertical_velocity_mps: float
    bending_warning: bool
    rgb_fall_conf: Optional[float]


def compute_joint_angle_deg(
    a: Optional[tuple[float, float]],
    b: Optional[tuple[float, float]],
    c: Optional[tuple[float, float]],
) -> Optional[float]:
    """Compute interior angle ABC in degrees at vertex B."""
    if a is None or b is None or c is None:
        return None
    ux, uy = a[0] - b[0], a[1] - b[1]
    vx, vy = c[0] - b[0], c[1] - b[1]
    norm_u = math.hypot(ux, uy)
    norm_v = math.hypot(vx, vy)
    if norm_u < 1e-5 or norm_v < 1e-5:
        return None
    cos_theta = clamp((ux * vx + uy * vy) / (norm_u * norm_v), -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


class SixClassPostureEngine:
    """
    Deterministic geometric and temporal posture classification engine for the
    6 required postures:
      1. Standing
      2. Sitting
      3. Walking
      4. Bending (with >=60 deg hazard warning)
      5. Falling
      6. Lying Down
    """

    # Geometric & temporal thresholds per specification
    STANDING_MAX_TORSO_DEG: float = 25.0
    SITTING_MAX_TORSO_DEG: float = 35.0
    WALKING_MAX_TORSO_DEG: float = 30.0
    BENDING_MIN_TORSO_DEG: float = 45.0
    BENDING_WARNING_TORSO_DEG: float = 60.0

    SITTING_THIGH_TORSO_RATIO: float = 0.65
    LYING_MIN_ASPECT_RATIO: float = 1.15
    FALL_MIN_VELOCITY_MPS: float = 1.5
    FALL_TRANSITION_WINDOW_SEC: float = 1.0
    FALL_LATCH_DURATION_SEC: float = 0.9

    ASSUMED_HUMAN_HEIGHT_M: float = 1.65

    def __init__(self, kp_conf_threshold: float = 0.30) -> None:
        self.kp_conf_threshold = kp_conf_threshold
        self.tracks: dict[int, TrackHistoryState] = {}

    def prune_stale_tracks(self, current_ts: float, max_stale_sec: float = 2.0) -> None:
        stale = [
            tid for tid, st in self.tracks.items()
            if (current_ts - st.last_seen_ts) > max_stale_sec
        ]
        for tid in stale:
            self.tracks.pop(tid, None)

    def _valid_point(self, kps: np.ndarray, idx: int) -> Optional[tuple[float, float]]:
        x, y, c = float(kps[idx, 0]), float(kps[idx, 1]), float(kps[idx, 2])
        if c >= self.kp_conf_threshold and x > 0.0 and y > 0.0 and math.isfinite(x) and math.isfinite(y):
            return (x, y)
        return None

    def _pair_midpoint(
        self, kps: np.ndarray, idx_a: int, idx_b: int
    ) -> Optional[tuple[float, float]]:
        pa = self._valid_point(kps, idx_a)
        pb = self._valid_point(kps, idx_b)
        if pa is not None and pb is not None:
            return (0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]))
        return pa if pa is not None else pb

    def evaluate_person(
        self,
        track_id: int,
        bbox: tuple[float, float, float, float],
        det_conf: float,
        keypoints_xyc: np.ndarray,
        timestamp: float,
        frame_height: int,
        rgb_fall_confidence: Optional[float] = None,
        camera_moving: bool = False,
    ) -> PersonAnalysisResult:
        state = self.tracks.get(track_id)
        if state is None:
            state = TrackHistoryState(track_id=track_id, last_seen_ts=timestamp)
            self.tracks[track_id] = state
        state.last_seen_ts = timestamp

        x1, y1, x2, y2 = bbox
        bbox_w = max(1.0, x2 - x1)
        bbox_h = max(1.0, y2 - y1)
        aspect_ratio = bbox_w / bbox_h
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        # Extract anatomical landmarks
        mid_shoulder = self._pair_midpoint(keypoints_xyc, KP_L_SHOULDER, KP_R_SHOULDER)
        mid_hip = self._pair_midpoint(keypoints_xyc, KP_L_HIP, KP_R_HIP)
        mid_knee = self._pair_midpoint(keypoints_xyc, KP_L_KNEE, KP_R_KNEE)
        mid_ankle = self._pair_midpoint(keypoints_xyc, KP_L_ANKLE, KP_R_ANKLE)

        l_hip = self._valid_point(keypoints_xyc, KP_L_HIP)
        r_hip = self._valid_point(keypoints_xyc, KP_R_HIP)
        l_knee = self._valid_point(keypoints_xyc, KP_L_KNEE)
        r_knee = self._valid_point(keypoints_xyc, KP_R_KNEE)
        l_ankle = self._valid_point(keypoints_xyc, KP_L_ANKLE)
        r_ankle = self._valid_point(keypoints_xyc, KP_R_ANKLE)

        # Torso vector & inclination angle relative to vertical axis (0 deg = upright, 90 deg = horizontal)
        torso_angle: Optional[float] = None
        torso_length: Optional[float] = None
        if mid_shoulder is not None and mid_hip is not None:
            dx = mid_shoulder[0] - mid_hip[0]
            dy = mid_hip[1] - mid_shoulder[1]  # positive when shoulders are above hips
            torso_length = math.hypot(dx, dy)
            if torso_length > 4.0:
                torso_angle = math.degrees(math.atan2(abs(dx), max(1e-4, abs(dy))))
                if dy < -0.15 * torso_length:
                    # Shoulders inverted below hips
                    torso_angle = max(torso_angle, 75.0)

        # Knee extension angles (180 deg = straight leg)
        knee_angles = [
            ang for ang in (
                compute_joint_angle_deg(l_hip, l_knee, l_ankle),
                compute_joint_angle_deg(r_hip, r_knee, r_ankle),
            )
            if ang is not None
        ]
        min_knee_angle = min(knee_angles) if knee_angles else None
        max_knee_angle = max(knee_angles) if knee_angles else None

        # Vertical thigh distance |y_hip - y_knee|
        thigh_vertical_dist: Optional[float] = None
        if mid_hip is not None and mid_knee is not None:
            thigh_vertical_dist = abs(mid_knee[1] - mid_hip[1])

        # Lower-limb vertical extension |y_ankle - y_hip|
        leg_vertical_dist: Optional[float] = None
        if mid_hip is not None and mid_ankle is not None:
            leg_vertical_dist = abs(mid_ankle[1] - mid_hip[1])

        # Ankle horizontal separation |x_ankle_left - x_ankle_right|
        ankle_sep_x: Optional[float] = None
        if l_ankle is not None and r_ankle is not None:
            ankle_sep_x = abs(l_ankle[0] - r_ankle[0])

        # Maintain rolling reference standing pixel height for m/s velocity conversion
        is_upright_now = (
            (torso_angle is not None and torso_angle < 35.0)
            and aspect_ratio < 0.85
        )
        if is_upright_now and bbox_h > 0.25 * frame_height:
            state.reference_height_px = 0.85 * state.reference_height_px + 0.15 * bbox_h

        px_per_meter = max(80.0, state.reference_height_px / self.ASSUMED_HUMAN_HEIGHT_M)

        # Compute temporal kinematics across rolling window (5-10 frames & <= 1.0 s window)
        v_y_mps = 0.0
        v_y_px_per_sec = 0.0
        lateral_disp_norm = 0.0
        cyclic_ankle_swing = False
        had_recent_vertical = False

        recent_samples = [s for s in state.samples if (timestamp - s.timestamp) <= self.FALL_TRANSITION_WINDOW_SEC]
        window_5_10 = list(state.samples)[-9:]  # last up to 9 previous frames + current = 10 frames

        if recent_samples and not camera_moving:
            # Peak downward velocity over recent short interval (0.12s - 0.60s)
            for prev in recent_samples:
                dt = timestamp - prev.timestamp
                if 0.06 <= dt <= 0.75:
                    dy_px = cy - prev.centroid_y  # positive = downward motion
                    if mid_hip is not None and prev.hip_y is not None:
                        dy_px = max(dy_px, mid_hip[1] - prev.hip_y)
                    vy_px = dy_px / dt
                    vy_m = vy_px / px_per_meter
                    if vy_m > v_y_mps:
                        v_y_mps = vy_m
                        v_y_px_per_sec = vy_px
                if prev.is_upright or (prev.torso_angle is not None and prev.torso_angle < 38.0) or prev.aspect_ratio < 0.85:
                    had_recent_vertical = True

        if len(window_5_10) >= 3 and not camera_moving:
            oldest = window_5_10[0]
            ref_scale = max(40.0, torso_length or (0.35 * bbox_h))
            lateral_disp_norm = abs(cx - oldest.centroid_x) / ref_scale

            # Check alternating cyclic ankle displacement across rolling 5-10 frame history
            ankle_seps = [s.ankle_sep_x for s in window_5_10 if s.ankle_sep_x is not None]
            if ankle_sep_x is not None:
                ankle_seps.append(ankle_sep_x)
            if len(ankle_seps) >= 3:
                max_sep = max(ankle_seps)
                min_sep = min(ankle_seps)
                sep_threshold = 0.24 * ref_scale
                # Cyclic stride: ankle separation exceeds threshold and exhibits dynamic variation
                if max_sep > sep_threshold and (max_sep - min_sep) > 0.08 * ref_scale:
                    cyclic_ankle_swing = True

        # Check horizontal alignment of mid-shoulder and mid-hip for Lying Down / Fall impact
        shoulder_hip_equal_y = False
        if mid_shoulder is not None and mid_hip is not None and torso_length is not None:
            shoulder_hip_equal_y = abs(mid_shoulder[1] - mid_hip[1]) <= 0.42 * torso_length
        elif aspect_ratio > self.LYING_MIN_ASPECT_RATIO:
            shoulder_hip_equal_y = True

        is_horizontal_now = (
            aspect_ratio > self.LYING_MIN_ASPECT_RATIO
            or (torso_angle is not None and torso_angle >= 62.0 and shoulder_hip_equal_y)
        )
        centroid_low_to_floor = (cy >= 0.48 * frame_height) or (y2 >= 0.72 * frame_height)

        # ------------------------------------------------------------------
        # DETERMINISTIC 6-CLASS DECISION HIERARCHY
        # ------------------------------------------------------------------
        raw_posture = PostureClass.STANDING
        bending_warning = False

        # 1. FALLING:
        #    - High downward vertical centroid velocity (v_y > 1.5 m/s or rapid downward pixel trajectory > 210 px/s)
        #    - Sudden transition from vertical to horizontal orientation within <= 1.0 s
        #    - Secondary RGB fall model confidence trigger if present
        high_downward_velocity = (v_y_mps > self.FALL_MIN_VELOCITY_MPS) or (v_y_px_per_sec > 210.0)
        rapid_vertical_to_horizontal = had_recent_vertical and is_horizontal_now and (v_y_mps > 0.75 or v_y_px_per_sec > 110.0)
        rgb_fall_triggered = (
            rgb_fall_confidence is not None
            and rgb_fall_confidence >= 0.55
            and (high_downward_velocity or had_recent_vertical)
        )

        if high_downward_velocity and (is_horizontal_now or (torso_angle is not None and torso_angle >= 42.0) or aspect_ratio >= 0.90):
            raw_posture = PostureClass.FALLING
            state.fall_latched_until = timestamp + self.FALL_LATCH_DURATION_SEC
        elif rapid_vertical_to_horizontal or rgb_fall_triggered:
            raw_posture = PostureClass.FALLING
            state.fall_latched_until = timestamp + self.FALL_LATCH_DURATION_SEC
        elif timestamp < state.fall_latched_until and (is_horizontal_now or aspect_ratio >= 0.95):
            raw_posture = PostureClass.FALLING

        # 2. LYING DOWN:
        #    - Horizontal aspect ratio (width / height > 1.15) or horizontal torso
        #    - Mid-shoulder and mid-hip share approximately equal vertical y-coordinates
        #    - Centroid low to floor plane with low kinetic velocity
        elif (
            (aspect_ratio > self.LYING_MIN_ASPECT_RATIO or (torso_angle is not None and torso_angle >= 65.0))
            and shoulder_hip_equal_y
            and centroid_low_to_floor
            and v_y_mps <= self.FALL_MIN_VELOCITY_MPS
        ):
            raw_posture = PostureClass.LYING_DOWN

        # 3. BENDING:
        #    - Torso inclination angle >= 45 deg (hazard warning at >= 60 deg)
        #    - Knees remain mostly extended (distinguishing bending from sitting/crouching)
        elif (
            torso_angle is not None
            and torso_angle >= self.BENDING_MIN_TORSO_DEG
            and (
                (max_knee_angle is not None and max_knee_angle >= 125.0)
                or (thigh_vertical_dist is not None and torso_length is not None and thigh_vertical_dist >= 0.55 * torso_length)
                or (min_knee_angle is None and aspect_ratio <= 1.15)
            )
        ):
            raw_posture = PostureClass.BENDING
            bending_warning = torso_angle >= self.BENDING_WARNING_TORSO_DEG

        # 4. SITTING:
        #    - Torso remains upright (theta_torso < 35 deg)
        #    - Vertical thigh distance |y_hip - y_knee| < 0.65 * torso_length (or flexed knees <= 135 deg)
        #    - Stable center of mass
        elif (
            (torso_angle is not None and torso_angle < self.SITTING_MAX_TORSO_DEG)
            and (
                (
                    thigh_vertical_dist is not None
                    and torso_length is not None
                    and thigh_vertical_dist < (self.SITTING_THIGH_TORSO_RATIO * torso_length)
                )
                or (min_knee_angle is not None and min_knee_angle <= 135.0)
            )
            and lateral_disp_norm <= 0.45
        ):
            raw_posture = PostureClass.SITTING

        # 5. WALKING:
        #    - Torso remains substantially upright (theta_torso < 30 deg)
        #    - Alternating cyclic ankle displacement (|x_ankle_L - x_ankle_R| > threshold)
        #      and lateral centroid movement across rolling 5-10 frame history
        elif (
            (torso_angle is None or torso_angle < self.WALKING_MAX_TORSO_DEG)
            and not camera_moving
            and (
                (cyclic_ankle_swing and lateral_disp_norm >= 0.14)
                or (
                    ankle_sep_x is not None
                    and torso_length is not None
                    and ankle_sep_x > 0.28 * torso_length
                    and lateral_disp_norm >= 0.22
                )
                or lateral_disp_norm >= 0.42
            )
        ):
            raw_posture = PostureClass.WALKING

        # 6. STANDING:
        #    - Torso angle relative to vertical baseline < 25 deg (or upright fallback)
        #    - Hip-knee-ankle joint extension near vertical
        #    - Low horizontal displacement across consecutive frames
        else:
            raw_posture = PostureClass.STANDING

        # Append current sample to rolling history
        sample = FrameKinematicsSample(
            timestamp=timestamp,
            centroid_x=cx,
            centroid_y=cy,
            bbox_w=bbox_w,
            bbox_h=bbox_h,
            aspect_ratio=aspect_ratio,
            torso_angle=torso_angle,
            torso_length=torso_length,
            hip_y=mid_hip[1] if mid_hip is not None else None,
            shoulder_y=mid_shoulder[1] if mid_shoulder is not None else None,
            ankle_sep_x=ankle_sep_x,
            left_ankle_x=l_ankle[0] if l_ankle is not None else None,
            right_ankle_x=r_ankle[0] if r_ankle is not None else None,
            is_upright=is_upright_now,
            raw_posture=raw_posture,
        )
        state.samples.append(sample)

        # Temporal smoothing with immediate priority override for FALLING
        state.posture_votes.append(raw_posture)
        if raw_posture == PostureClass.FALLING:
            final_posture = PostureClass.FALLING
        else:
            counts = Counter(state.posture_votes)
            final_posture = counts.most_common(1)[0][0]

        # Build human-readable display banner
        if final_posture == PostureClass.BENDING and torso_angle is not None:
            warn_suffix = " [WARN >=60°]" if torso_angle >= self.BENDING_WARNING_TORSO_DEG else ""
            display_label = f"BENDING ({torso_angle:.0f} deg){warn_suffix}"
        elif final_posture == PostureClass.FALLING:
            display_label = f"FALLING (vy={v_y_mps:.1f}m/s)"
        else:
            display_label = final_posture.value

        return PersonAnalysisResult(
            track_id=track_id,
            bbox=bbox,
            confidence=det_conf,
            keypoints_xyc=keypoints_xyc,
            posture=final_posture,
            display_label=display_label,
            torso_angle_deg=torso_angle,
            vertical_velocity_mps=v_y_mps,
            bending_warning=bending_warning,
            rgb_fall_conf=rgb_fall_confidence,
        )


# ==============================================================================
# SUBSYSTEM 2: YOLO-POSE + SECONDARY RGB FALL MODEL RUNNER
# ==============================================================================

def _bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-5, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1e-5, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return inter / (area_a + area_b - inter)


class FallbackCentroidTracker:
    """Lightweight IoU + Centroid tracker if YOLO `.track()` omits IDs on a frame."""

    def __init__(self, max_stale_sec: float = 1.2) -> None:
        self.max_stale_sec = max_stale_sec
        self._next_id = 1
        self._tracks: dict[int, tuple[tuple[float, float, float, float], float]] = {}

    def assign(
        self,
        bboxes: list[tuple[float, float, float, float]],
        now: float,
    ) -> list[int]:
        stale = [tid for tid, (_, ts) in self._tracks.items() if (now - ts) > self.max_stale_sec]
        for tid in stale:
            self._tracks.pop(tid, None)

        assigned_ids: list[int] = []
        used_existing: set[int] = set()

        for box in bboxes:
            best_id: Optional[int] = None
            best_score = 0.15
            for tid, (prev_box, _) in self._tracks.items():
                if tid in used_existing:
                    continue
                score = _bbox_iou(box, prev_box)
                if score > best_score:
                    best_score = score
                    best_id = tid
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            used_existing.add(best_id)
            self._tracks[best_id] = (box, now)
            assigned_ids.append(best_id)

        return assigned_ids


class VisionAnalyticsRunner:
    """
    Wraps Ultralytics YOLO-Pose (17 COCO keypoints) + optional secondary RGB
    fall detector (`models/human-fall-detection.pt`) + `SixClassPostureEngine`.
    """

    def __init__(
        self,
        pose_weights: Optional[str] = None,
        fall_weights: str = "models/human-fall-detection.pt",
        device: str = "auto",
        conf_threshold: float = 0.30,
        kp_conf_threshold: float = 0.30,
        imgsz: int = 640,
    ) -> None:
        import torch
        from ultralytics import YOLO

        self.conf_threshold = conf_threshold
        self.kp_conf_threshold = kp_conf_threshold
        self.imgsz = imgsz

        if device == "auto":
            self.device = "0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.use_half = self.device != "cpu" and torch.cuda.is_available()

        # Resolve best available YOLO-Pose checkpoint
        resolved_pose = self._resolve_pose_weights(pose_weights)
        logger.info("Loading YOLO-Pose model: %s (device=%s, half=%s)", resolved_pose, self.device, self.use_half)
        self.pose_model = YOLO(str(resolved_pose))

        # Load secondary RGB fall detector if available
        self.fall_model = None
        self.fallen_class_ids: set[int] = set()
        fall_path = WORKSPACE_ROOT / fall_weights
        if fall_path.is_file():
            try:
                self.fall_model = YOLO(str(fall_path))
                for cid, cname in self.fall_model.names.items():
                    if "fall" in str(cname).lower():
                        self.fallen_class_ids.add(int(cid))
                logger.info("Secondary RGB fall detector loaded: %s", fall_path)
            except Exception as exc:
                logger.warning("Secondary RGB fall model unavailable (%s); continuing without it.", exc)
                self.fall_model = None
        else:
            logger.info("Secondary RGB fall model not found at %s; continuing in pose-only mode.", fall_path)

        self.posture_engine = SixClassPostureEngine(kp_conf_threshold=kp_conf_threshold)
        self.fallback_tracker = FallbackCentroidTracker()
        self._last_fall_infer_ts: float = 0.0
        self._cached_fall_boxes: list[tuple[tuple[float, float, float, float], float]] = []

    @staticmethod
    def _resolve_pose_weights(requested: Optional[str]) -> Path:
        candidates = []
        if requested:
            candidates.append(Path(requested))
            candidates.append(WORKSPACE_ROOT / requested)
        candidates.extend([
            WORKSPACE_ROOT / "models" / "yolo26m-pose.pt",
            WORKSPACE_ROOT / "models" / "yolo26n-pose.pt",
            WORKSPACE_ROOT / "yolo26n-pose.pt",
        ])
        for path in candidates:
            if path.is_file():
                return path
        return Path(requested or "yolo26n-pose.pt")

    def warmup(self, width: int = 640, height: int = 480) -> None:
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        self.pose_model.predict(dummy, imgsz=self.imgsz, device=self.device, verbose=False)
        if self.fall_model is not None:
            self.fall_model.predict(dummy, imgsz=416, device=self.device, verbose=False)

    def _infer_secondary_fall(self, frame: np.ndarray, now: float) -> list[tuple[tuple[float, float, float, float], float]]:
        if self.fall_model is None:
            return []
        if (now - self._last_fall_infer_ts) < 0.35:
            return self._cached_fall_boxes
        self._last_fall_infer_ts = now
        try:
            res = self.fall_model.predict(
                frame,
                imgsz=416,
                conf=0.25,
                device=self.device,
                verbose=False,
            )[0]
            boxes_out: list[tuple[tuple[float, float, float, float], float]] = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()
                for b, c, k in zip(xyxy, confs, clss):
                    if int(k) in self.fallen_class_ids:
                        boxes_out.append(((float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(c)))
            self._cached_fall_boxes = boxes_out
        except Exception as exc:
            logger.debug("Secondary RGB fall inference error: %s", exc)
        return self._cached_fall_boxes

    def analyze_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        camera_moving: bool = False,
    ) -> list[PersonAnalysisResult]:
        h, w = frame.shape[:2]
        self.posture_engine.prune_stale_tracks(timestamp)

        fp16_opts = {"quantize": 16} if self.use_half else {}
        results = self.pose_model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            **fp16_opts,
        )
        if not results:
            return []

        res = results[0]
        if res.boxes is None or res.keypoints is None or len(res.boxes) == 0:
            return []

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        kps_data = res.keypoints.data.cpu().numpy()  # (N, 17, 3)

        raw_bboxes: list[tuple[float, float, float, float]] = [
            (
                clamp(float(b[0]), 0.0, float(w)),
                clamp(float(b[1]), 0.0, float(h)),
                clamp(float(b[2]), 0.0, float(w)),
                clamp(float(b[3]), 0.0, float(h)),
            )
            for b in boxes_xyxy
        ]

        if res.boxes.id is not None:
            track_ids = [int(tid) for tid in res.boxes.id.int().cpu().tolist()]
        else:
            track_ids = self.fallback_tracker.assign(raw_bboxes, timestamp)

        fall_boxes = self._infer_secondary_fall(frame, timestamp)

        persons: list[PersonAnalysisResult] = []
        for idx, (tid, bbox, det_conf, kps) in enumerate(zip(track_ids, raw_bboxes, confs, kps_data)):
            if kps.shape != (17, 3):
                continue

            # Match auxiliary RGB fall detector box via IoU
            matched_fall_conf: Optional[float] = None
            for fbox, fconf in fall_boxes:
                if _bbox_iou(bbox, fbox) >= 0.25:
                    if matched_fall_conf is None or fconf > matched_fall_conf:
                        matched_fall_conf = fconf

            person_res = self.posture_engine.evaluate_person(
                track_id=tid,
                bbox=bbox,
                det_conf=float(det_conf),
                keypoints_xyc=kps,
                timestamp=timestamp,
                frame_height=h,
                rgb_fall_confidence=matched_fall_conf,
                camera_moving=camera_moving,
            )
            persons.append(person_res)

        return persons


# ==============================================================================
# PART C: ON-SCREEN DISPLAY (OSD) HUD & TARGET RETICLE RENDERER
# ==============================================================================

def render_osd_hud(
    frame: np.ndarray,
    persons: list[PersonAnalysisResult],
    camera_mode: CameraMode,
    auto_telemetry: AutoTrackTelemetry,
    fps: float,
    ai_overlay_enabled: bool,
    ptz_online: bool,
    dataset_collector: DatasetCollector,
    kp_conf_threshold: float = 0.30,
) -> np.ndarray:
    """
    Renders:
      1. Individual COCO-17 skeletons, bounding boxes, and color-coded posture banners:
         - STANDING, SITTING, WALKING -> Green
         - BENDING -> Orange / Yellow
         - FALLING, LYING DOWN -> Red
      2. AUTOMATIC mode Target Reticle:
         - Center deadband zone box (|e_x|, |e_y| <= 0.15)
         - Collective group bounding box (Xmin, Ymin, Xmax, Ymax)
         - Crosshair reticle at collective centroid (Cx, Cy) + error vector
      3. Top-Left OSD Telemetry HUD:
         - Active Mode indicator ([STATIC], [MANUAL], or [AUTO-TRACKING]),
           Stream FPS, Occupant Count, and interactive keybind bar.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 1. Draw AI Skeletal & Posture Overlay (when enabled via 'P')
    if ai_overlay_enabled:
        for p in persons:
            x1, y1, x2, y2 = map(int, p.bbox)
            color = POSTURE_COLORS_BGR.get(p.posture, (50, 205, 50))

            # Skeleton bones
            for i_a, i_b in COCO_SKELETON_EDGES:
                xa, ya, ca = p.keypoints_xyc[i_a]
                xb, yb, cb = p.keypoints_xyc[i_b]
                if ca >= kp_conf_threshold and cb >= kp_conf_threshold and xa > 0 and ya > 0 and xb > 0 and yb > 0:
                    pt1 = (int(round(xa)), int(round(ya)))
                    pt2 = (int(round(xb)), int(round(yb)))
                    cv2.line(canvas, pt1, pt2, (15, 15, 15), 4, cv2.LINE_AA)
                    cv2.line(canvas, pt1, pt2, (240, 240, 240), 2, cv2.LINE_AA)

            # Skeleton joints
            for j_idx in range(17):
                jx, jy, jc = p.keypoints_xyc[j_idx]
                if jc >= kp_conf_threshold and jx > 0 and jy > 0:
                    jpt = (int(round(jx)), int(round(jy)))
                    cv2.circle(canvas, jpt, 4, (15, 15, 15), -1, cv2.LINE_AA)
                    cv2.circle(canvas, jpt, 2, color, -1, cv2.LINE_AA)

            # Person Bounding Box
            thickness = 3 if p.posture in (PostureClass.FALLING, PostureClass.LYING_DOWN) else 2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (10, 10, 10), thickness + 2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

            # Color-coded Posture Status Banner above bounding box
            banner_text = f"ID:{p.track_id} | {p.display_label} ({p.confidence:.0%})"
            (tw, th), _ = cv2.getTextSize(banner_text, font, 0.50, 1)
            bx1 = max(4, min(x1, w - tw - 16))
            by2 = y1 - 4 if y1 - th - 16 >= 68 else min(h - 6, y2 + th + 14)
            by1 = by2 - th - 10
            bx2 = bx1 + tw + 12

            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (15, 15, 15), 1)
            text_color = (15, 15, 15) if p.posture != PostureClass.FALLING and p.posture != PostureClass.LYING_DOWN else (255, 255, 255)
            cv2.putText(canvas, banner_text, (bx1 + 6, by2 - 6), font, 0.50, text_color, 1, cv2.LINE_AA)

    # 2. Target Reticle & Deadband Box when in AUTOMATIC mode
    if camera_mode == CameraMode.AUTOMATIC:
        cx_frame, cy_frame = w // 2, h // 2
        db_w = int(round(TriModePTZController.AUTO_DEADBAND * (w / 2.0)))
        db_h = int(round(TriModePTZController.AUTO_DEADBAND * (h / 2.0)))

        # Center Deadband Box (|e_x|, |e_y| <= 0.15)
        db_color = (180, 220, 180) if auto_telemetry.in_deadband else (0, 215, 255)
        cv2.rectangle(
            canvas,
            (cx_frame - db_w, cy_frame - db_h),
            (cx_frame + db_w, cy_frame + db_h),
            db_color,
            1,
            cv2.LINE_AA,
        )
        # Frame center small plus
        cv2.line(canvas, (cx_frame - 8, cy_frame), (cx_frame + 8, cy_frame), db_color, 1, cv2.LINE_AA)
        cv2.line(canvas, (cx_frame, cy_frame - 8), (cx_frame, cy_frame + 8), db_color, 1, cv2.LINE_AA)

        # Group Composite Bounding Box & Collective Centroid Crosshair (Cx, Cy)
        if auto_telemetry.group_bbox is not None and auto_telemetry.collective_center is not None:
            gx1, gy1, gx2, gy2 = auto_telemetry.group_bbox
            gcx, gcy = auto_telemetry.collective_center
            reticle_color = (50, 230, 50) if auto_telemetry.in_deadband else (0, 200, 255)

            # Group bounding box with corner brackets
            cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), reticle_color, 1, cv2.LINE_AA)
            c_len = 14
            for px, py, dx, dy in ((gx1, gy1, 1, 1), (gx2, gy1, -1, 1), (gx1, gy2, 1, -1), (gx2, gy2, -1, -1)):
                cv2.line(canvas, (px, py), (px + dx * c_len, py), reticle_color, 2, cv2.LINE_AA)
                cv2.line(canvas, (px, py), (px, py + dy * c_len), reticle_color, 2, cv2.LINE_AA)

            # Crosshair at collective centroid (Cx, Cy)
            cv2.circle(canvas, (gcx, gcy), 10, reticle_color, 2, cv2.LINE_AA)
            cv2.line(canvas, (gcx - 16, gcy), (gcx + 16, gcy), reticle_color, 2, cv2.LINE_AA)
            cv2.line(canvas, (gcx, gcy - 16), (gcx, gcy + 16), reticle_color, 2, cv2.LINE_AA)

            # Steering error vector line from frame center to collective centroid
            cv2.arrowedLine(canvas, (cx_frame, cy_frame), (gcx, gcy), reticle_color, 1, cv2.LINE_AA, tipLength=0.12)

            coord_tag = f"Group C({gcx},{gcy}) e=({auto_telemetry.error_x:+.2f},{auto_telemetry.error_y:+.2f})"
            cv2.putText(canvas, coord_tag, (gx1, max(78, gy1 - 6)), font, 0.44, reticle_color, 1, cv2.LINE_AA)

    # 3. Top-Left OSD Telemetry Header
    cv2.rectangle(canvas, (0, 0), (w, 60), (14, 14, 14), -1)
    cv2.line(canvas, (0, 60), (w, 60), (180, 180, 180), 1)

    mode_color = {
        CameraMode.STATIC: (200, 200, 200),
        CameraMode.MANUAL: (255, 200, 60),
        CameraMode.AUTOMATIC: (60, 235, 100),
    }[camera_mode]

    # Mode badge
    mode_str = camera_mode.hud_tag
    cv2.putText(canvas, f"MODE: {mode_str}", (12, 23), font, 0.56, mode_color, 2, cv2.LINE_AA)

    ptz_link_str = "ONVIF:ONLINE" if ptz_online else "ONVIF:SIM/OFFLINE"
    hud_line_1 = f"|  FPS: {fps:.1f}  |  Occupants: {len(persons)}  |  AI(P): {'ON' if ai_overlay_enabled else 'OFF'}  |  {ptz_link_str}"
    cv2.putText(canvas, hud_line_1, (235, 23), font, 0.48, (240, 240, 240), 1, cv2.LINE_AA)

    rec_state = "REC 2FPS" if dataset_collector.auto_recording else "READY"
    hud_line_2 = (
        f"[M/Tab] Mode  |  [WASD] Manual PTZ  |  [P] AI Overlay  |  "
        f"[1-6/C/R] Dataset: {dataset_collector.current_class.upper()} ({rec_state})  |  [Q] Quit"
    )
    cv2.putText(canvas, hud_line_2, (12, 47), font, 0.41, (185, 185, 185), 1, cv2.LINE_AA)

    return canvas


# ==============================================================================
# MAIN INTEGRATED APPLICATION LOOP
# ==============================================================================

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SIDP Subsystem 1 & 2 Integrated Pipeline v2 (6-Posture + Tri-Mode PTZ)"
    )
    parser.add_argument("--ip", default=DEFAULT_CAMERA_IP, help="Tapo C210 IP address")
    parser.add_argument("--user", default=DEFAULT_CAMERA_USER, help="Camera RTSP/ONVIF username")
    parser.add_argument("--password", default=DEFAULT_CAMERA_PASS, help="Camera RTSP/ONVIF password")
    parser.add_argument("--stream", default=DEFAULT_CAMERA_STREAM, help="RTSP stream path (stream1 or stream2)")
    parser.add_argument("--webcam", action="store_true", help="Use local USB/laptop webcam instead of RTSP")
    parser.add_argument("--camera-index", type=int, default=0, help="Local webcam index when --webcam is set")
    parser.add_argument("--pose-model", default=None, help="Path to YOLO-Pose weights (default: models/yolo26m-pose.pt)")
    parser.add_argument("--fall-model", default="models/human-fall-detection.pt", help="Secondary RGB fall model")
    parser.add_argument("--device", default="auto", help="Inference device: auto, 0, cpu")
    parser.add_argument("--mode", choices=["STATIC", "MANUAL", "AUTOMATIC"], default="STATIC", help="Initial PTZ mode")
    parser.add_argument("--no-ptz", action="store_true", help="Disable ONVIF PTZ motor initialization")
    parser.add_argument("--headless", action="store_true", help="Run without OpenCV GUI window (for smoke tests)")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop automatically after N frames")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    rtsp_url = f"rtsp://{args.user}:{args.password}@{args.ip}:{DEFAULT_RTSP_PORT}/{args.stream}"
    initial_mode = CameraMode(args.mode)

    # 1. Initialize Vision Analytics (Subsystem 2)
    vision = VisionAnalyticsRunner(
        pose_weights=args.pose_model,
        fall_weights=args.fall_model,
        device=args.device,
    )
    vision.warmup()

    # 2. Initialize Tri-Mode ONVIF PTZ Controller (Subsystem 1)
    ptz = TriModePTZController(
        ip=args.ip,
        port=DEFAULT_ONVIF_PORT,
        username=args.user,
        password=args.password,
        enabled=(not args.no_ptz and not args.webcam),
        initial_mode=initial_mode,
    )

    # 3. Initialize Dataset Collector & Thread-Safe Stream Reader
    collector = DatasetCollector()
    stream = ThreadSafeStreamReader(
        rtsp_url=rtsp_url,
        use_webcam=args.webcam,
        webcam_index=args.camera_index,
    ).start()

    window_title = "SIDP Integrated v2 - 6-Class Posture & Tri-Mode PTZ"
    ai_overlay_enabled = True
    fps_ema = 0.0
    prev_loop_ts = time.perf_counter()
    last_processed_frame_id = -1
    processed_count = 0

    logger.info(
        "System started | Mode=%s | Source=%s",
        ptz.mode.value,
        f"webcam:{args.camera_index}" if args.webcam else f"rtsp://{args.ip}:{DEFAULT_RTSP_PORT}/{args.stream}",
    )

    try:
        while args.max_frames is None or processed_count < args.max_frames:
            connected, frame, frame_ts, frame_id = stream.read()
            if not connected or frame is None or frame_id == last_processed_frame_id:
                time.sleep(0.005)
                continue

            last_processed_frame_id = frame_id
            clean_frame = frame.copy()
            collector.maybe_auto_record(clean_frame)

            # Subsystem 2: 6-Posture Detection & Tracking
            persons = vision.analyze_frame(
                frame=frame,
                timestamp=frame_ts,
                camera_moving=ptz.is_camera_moving(),
            )

            # Subsystem 1: Update Automatic Group-Centering PTZ Controller
            h, w = frame.shape[:2]
            person_bboxes = [p.bbox for p in persons]
            auto_telemetry = ptz.update_automatic_tracking(
                bboxes=person_bboxes,
                frame_width=w,
                frame_height=h,
                now=frame_ts,
            )

            # Compute smoothed FPS
            now_loop = time.perf_counter()
            inst_fps = 1.0 / max(1e-4, now_loop - prev_loop_ts)
            prev_loop_ts = now_loop
            fps_ema = inst_fps if fps_ema == 0.0 else (0.85 * fps_ema + 0.15 * inst_fps)
            processed_count += 1

            if not args.headless:
                hud_frame = render_osd_hud(
                    frame=frame,
                    persons=persons,
                    camera_mode=ptz.mode,
                    auto_telemetry=auto_telemetry,
                    fps=fps_ema,
                    ai_overlay_enabled=ai_overlay_enabled,
                    ptz_online=ptz.available,
                    dataset_collector=collector,
                )
                cv2.imshow(window_title, hud_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                    logger.info("Shutdown requested by user.")
                    break
                elif key in (ord("m"), ord("M"), 9):  # M or Tab
                    ptz.cycle_mode()
                elif key in (ord("p"), ord("P")):
                    ai_overlay_enabled = not ai_overlay_enabled
                    logger.info("AI Skeletal Overlay -> %s", "ON" if ai_overlay_enabled else "OFF")
                elif key in (ord("w"), ord("W"), ord("a"), ord("A"), ord("s"), ord("S"), ord("d"), ord("D")):
                    ptz.manual_steer(chr(key))
                elif ord("1") <= key <= ord("6"):
                    cls_name = collector.select_class_index(key - ord("1"))
                    logger.info("Dataset target class -> %s", cls_name)
                elif key in (ord("c"), ord("C")):
                    saved_path = collector.save_frame(clean_frame)
                    logger.info("Captured clean dataset frame -> %s", saved_path)
                elif key in (ord("r"), ord("R")):
                    rec_on = collector.toggle_auto_record()
                    logger.info("Dataset 2 FPS auto-recording -> %s", "ON" if rec_on else "OFF")

                if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted via Ctrl+C.")
    finally:
        logger.info("Executing clean teardown of RTSP stream, ONVIF motors, and UI...")
        stream.stop()
        ptz.shutdown()
        if not args.headless:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

