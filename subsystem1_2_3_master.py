#!/usr/bin/env python3
"""
subsystem1_2_3_master.py
================================================================================
RBB4013 System Integrated Design Project (SIDP) - Universiti Teknologi PETRONAS
Unified Master Pipeline:
  - Subsystem 1: Tapo C210 RTSP Stream Ingestion, Tri-Mode ONVIF PTZ Controller
                 (STATIC / MANUAL WASD / AUTOMATIC Group-Centering + Z Patrol),
                 and Clean Dataset Collector (1-4/1-6, C, R).
  - Subsystem 2: Ultralytics YOLO26-Pose (17 COCO keypoints, RTX 4050 FP16) +
                 Auxiliary RGB Fall Detector (`models/human-fall-detection.pt`) +
                 6-Class Geometric & Temporal Posture Classifier.
  - Subsystem 3: Safety Decision Engine (`./Subsystem3-Safety-Engine`), 4-Tier
                 State Machine (`Safe`, `Warning`, `Unsafe`, `Alert`), Temporal
                 Persistence Timers (3.0s bending >60 deg escalation), SQLite
                 Event Sink, Non-Blocking `pyttsx3` Voice Alerts, and Telegram
                 Webhook Notifications with 15-second per-resident cooldown.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import queue
import sys
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

# ==============================================================================
# PATH RESOLUTION FOR WORKSPACE & SUBSYSTEM 3
# ==============================================================================

WORKSPACE_ROOT = Path(__file__).resolve().parent
SUBSYSTEM3_ROOT = WORKSPACE_ROOT / "Subsystem3-Safety-Engine"
SUBSYSTEM3_SRC = SUBSYSTEM3_ROOT / "src"

for _p in (WORKSPACE_ROOT, SUBSYSTEM3_SRC):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Import Subsystem 3 core modules from ./Subsystem3-Safety-Engine/src/alert_engine
from alert_engine.config import Config as S3Config, load_config as load_s3_config  # noqa: E402
from alert_engine.debounce import DebounceTracker  # noqa: E402
from alert_engine.ml_scoring import HybridScorer  # noqa: E402
from alert_engine.schemas import ClassifiedEvent, PostureEventIn  # noqa: E402
from alert_engine.scoring import ScoreResult, score_posture  # noqa: E402
from alert_engine.sinks import LocalSqliteSink  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SIDP.Master123")


# ==============================================================================
# GLOBAL CONSTANTS & TAXONOMY
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
    """Subsystem 2 6-Class Posture Taxonomy."""
    STANDING = "STANDING"
    SITTING = "SITTING"
    WALKING = "WALKING"
    BENDING = "BENDING"
    FALLING = "FALLING"
    LYING_DOWN = "LYING DOWN"

    @property
    def s3_name(self) -> str:
        """Map Subsystem 2 posture class to Subsystem 3 canonical event name."""
        return {
            PostureClass.STANDING: "standing",
            PostureClass.SITTING: "sitting",
            PostureClass.WALKING: "walking",
            PostureClass.BENDING: "bending",
            PostureClass.FALLING: "fall",
            PostureClass.LYING_DOWN: "lying",
        }[self]


class SafetyTier(str, Enum):
    """Subsystem 3 4-Tier Safety Classification."""
    SAFE = "safe"
    WARNING = "warning"
    UNSAFE = "unsafe"
    ALERT = "alert"

    @property
    def badge_text(self) -> str:
        return self.value.upper()


class CameraMode(str, Enum):
    """Subsystem 1 Tri-Mode PTZ Control Architecture."""
    STATIC = "STATIC"
    MANUAL = "MANUAL"
    AUTOMATIC = "AUTOMATIC"

    @property
    def hud_tag(self) -> str:
        if self == CameraMode.AUTOMATIC:
            return "[AUTO-TRACKING]"
        return f"[{self.value}]"


def clamp(val: float, low: float, high: float) -> float:
    return float(max(low, min(high, val)))


# ==============================================================================
# SUBSYSTEM 1: THREAD-SAFE RTSP STREAM INGESTION
# ==============================================================================

class ThreadSafeStreamReader:
    """
    Multi-threaded buffer-flushing RTSP/Webcam stream reader.
    Uses `threading.Lock()` so the main vision/HUD loop always reads the newest
    frame with <200 ms latency.
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
            name="SIDP-RTSP-Worker",
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
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                "rtsp_transport;udp|fflags;nobuffer|flags;low_delay",
            )
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
                logger.warning("RTSP stream interrupted; reconnecting...")
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
# SUBSYSTEM 1: TRI-MODE ONVIF PTZ CONTROLLER + 35° PATROL SWEEP
# ==============================================================================

@dataclass
class AutoTrackTelemetry:
    active: bool = False
    occupant_count: int = 0
    group_bbox: Optional[tuple[int, int, int, int]] = None
    collective_center: Optional[tuple[int, int]] = None
    error_x: float = 0.0
    error_y: float = 0.0
    in_deadband: bool = True
    cmd_pan: float = 0.0
    cmd_tilt: float = 0.0


class UnifiedPTZController:
    """
    Non-blocking ONVIF PTZ controller supporting:
      - Tri-Mode Switching (`M` / `Tab`): `STATIC` -> `MANUAL` -> `AUTOMATIC`
      - Manual Steering (`W/A/S/D` when in `MANUAL` mode)
      - Continuous 35° Patrol Sweep (`Z` toggle)
      - Automatic Multi-Person Group-Centering (`AUTOMATIC` mode) with 15%
        deadband and 300 ms SOAP command rate-limiting.
    """

    MODE_SEQUENCE: tuple[CameraMode, ...] = (
        CameraMode.STATIC,
        CameraMode.MANUAL,
        CameraMode.AUTOMATIC,
    )

    MANUAL_PAN_SPEED: float = 0.4
    MANUAL_TILT_SPEED: float = 0.3
    MANUAL_PULSE_DURATION: float = 0.28

    AUTO_DEADBAND: float = 0.15
    AUTO_KP: float = 0.65
    AUTO_MAX_PAN: float = 0.4
    AUTO_MAX_TILT: float = 0.3
    AUTO_THROTTLE_SECONDS: float = 0.30
    AUTO_PULSE_DURATION: float = 0.22

    PATROL_SWEEP_SECONDS: float = 1.0
    PATROL_DWELL_SECONDS: float = 0.4

    def __init__(
        self,
        ip: str = DEFAULT_CAMERA_IP,
        port: int = DEFAULT_ONVIF_PORT,
        username: str = DEFAULT_CAMERA_USER,
        password: str = DEFAULT_CAMERA_PASS,
        enabled: bool = True,
        initial_mode: CameraMode = CameraMode.MANUAL,
    ) -> None:
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.enabled = enabled

        self._mode = initial_mode
        self._lock = threading.Lock()
        self.available = False
        self.patrolling = False
        self.status_error: Optional[str] = None

        self._ptz_service = None
        self._media_profile_token: Optional[str] = None
        self._last_auto_dispatch_ts: float = 0.0
        self._last_motion_ts: float = -float("inf")
        self._is_moving_flag: bool = False
        self.telemetry = AutoTrackTelemetry(active=(initial_mode == CameraMode.AUTOMATIC))

        self._cmd_queue: queue.Queue[tuple[str, float, float, float]] = queue.Queue(maxsize=4)
        self._stop_event = threading.Event()
        self._patrol_thread: Optional[threading.Thread] = None

        if self.enabled:
            self._connect_onvif()
        else:
            self.status_error = "PTZ disabled"

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
                raise RuntimeError("No ONVIF media profiles returned")
            self._ptz_service = ptz_service
            self._media_profile_token = profiles[0].token
            self.available = True
            self.status_error = None
            logger.info("ONVIF PTZ online at %s:%d (token=%s)", self.ip, self.port, self._media_profile_token)
        except Exception as exc:
            self.available = False
            self.status_error = str(exc)
            logger.warning("ONVIF PTZ unavailable (%s); continuing in fallback mode.", exc)

    @property
    def mode(self) -> CameraMode:
        with self._lock:
            return self._mode

    @property
    def hud_mode_label(self) -> str:
        if self.patrolling:
            return "[PATROL 35°]"
        return self.mode.hud_tag

    def set_mode(self, new_mode: CameraMode) -> CameraMode:
        if self.patrolling:
            self.stop_patrol()
        with self._lock:
            self._mode = new_mode
            self.telemetry = AutoTrackTelemetry(active=(new_mode == CameraMode.AUTOMATIC))
        self._enqueue_command("STOP", 0.0, 0.0, 0.0)
        logger.info("PTZ Mode -> %s", new_mode.value)
        return new_mode

    def cycle_mode(self) -> CameraMode:
        with self._lock:
            idx = self.MODE_SEQUENCE.index(self._mode)
            next_mode = self.MODE_SEQUENCE[(idx + 1) % len(self.MODE_SEQUENCE)]
        return self.set_mode(next_mode)

    def manual_steer(self, key_char: str) -> bool:
        if self.patrolling:
            self.stop_patrol()
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

    def toggle_patrol(self) -> bool:
        if self.mode == CameraMode.STATIC:
            logger.info("Cannot start patrol while in STATIC mode (press M to switch mode).")
            return False
        if self.patrolling:
            self.stop_patrol()
        else:
            self.start_patrol()
        return self.patrolling

    def start_patrol(self) -> bool:
        if self._stop_event.is_set() or self.mode == CameraMode.STATIC:
            return False
        if self.patrolling and self._patrol_thread and self._patrol_thread.is_alive():
            return True
        self.patrolling = True
        self._patrol_thread = threading.Thread(
            target=self._patrol_worker,
            name="SIDP-PTZ-Patrol",
            daemon=True,
        )
        self._patrol_thread.start()
        logger.info("35° Continuous Patrol Sweep -> ON")
        return True

    def stop_patrol(self) -> None:
        if self.patrolling:
            self.patrolling = False
            self._enqueue_command("STOP", 0.0, 0.0, 0.0)
            logger.info("35° Continuous Patrol Sweep -> OFF")

    def _patrol_worker(self) -> None:
        direction = -1
        sweep_ticks = max(1, int(round(self.PATROL_SWEEP_SECONDS / 0.1)))
        dwell_ticks = max(1, int(round(self.PATROL_DWELL_SECONDS / 0.1)))
        while self.patrolling and not self._stop_event.is_set():
            self._enqueue_command("PULSE", direction * self.MANUAL_PAN_SPEED, 0.0, self.PATROL_SWEEP_SECONDS)
            for _ in range(sweep_ticks):
                if not self.patrolling or self._stop_event.is_set():
                    break
                time.sleep(0.1)
            self._enqueue_command("STOP", 0.0, 0.0, 0.0)
            for _ in range(dwell_ticks):
                if not self.patrolling or self._stop_event.is_set():
                    break
                time.sleep(0.1)
            direction *= -1
        self._enqueue_command("STOP", 0.0, 0.0, 0.0)

    def update_automatic_tracking(
        self,
        bboxes: list[tuple[float, float, float, float]],
        frame_width: int,
        frame_height: int,
        now: Optional[float] = None,
    ) -> AutoTrackTelemetry:
        if self.mode != CameraMode.AUTOMATIC or self.patrolling:
            self.telemetry = AutoTrackTelemetry(active=False, occupant_count=len(bboxes))
            return self.telemetry

        now_ts = time.perf_counter() if now is None else now
        n_persons = len(bboxes)

        if n_persons == 0 or frame_width <= 0 or frame_height <= 0:
            if self._is_moving_flag:
                self._enqueue_command("STOP", 0.0, 0.0, 0.0)
            self.telemetry = AutoTrackTelemetry(active=True, occupant_count=0, in_deadband=True)
            return self.telemetry

        x_min = min(b[0] for b in bboxes)
        y_min = min(b[1] for b in bboxes)
        x_max = max(b[2] for b in bboxes)
        y_max = max(b[3] for b in bboxes)

        c_x = 0.5 * (x_min + x_max)
        c_y = 0.5 * (y_min + y_max)

        half_w = 0.5 * frame_width
        half_h = 0.5 * frame_height
        e_x = clamp((c_x - half_w) / max(half_w, 1e-6), -1.0, 1.0)
        e_y = clamp((c_y - half_h) / max(half_h, 1e-6), -1.0, 1.0)

        v_pan = 0.0
        v_tilt = 0.0
        if abs(e_x) > self.AUTO_DEADBAND:
            v_pan = clamp(self.AUTO_KP * e_x, -self.AUTO_MAX_PAN, self.AUTO_MAX_PAN)
        if abs(e_y) > self.AUTO_DEADBAND:
            v_tilt = clamp(-self.AUTO_KP * e_y, -self.AUTO_MAX_TILT, self.AUTO_MAX_TILT)

        in_deadband = (v_pan == 0.0 and v_tilt == 0.0)

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
        except Exception:
            pass

    def is_camera_moving(self, settle_window: float = 0.45) -> bool:
        return self.patrolling or self._is_moving_flag or (time.perf_counter() - self._last_motion_ts) <= settle_window

    def shutdown(self) -> None:
        self.patrolling = False
        self._stop_event.set()
        self._execute_stop()
        if self._patrol_thread is not None and self._patrol_thread.is_alive():
            self._patrol_thread.join(timeout=1.5)
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)


# ==============================================================================
# SUBSYSTEM 1: DATASET COLLECTOR (`1-4`, `C`, `R`)
# ==============================================================================

class DatasetCollector:
    """Saves clean unannotated frames into `dataset/<class>/` for posture training."""

    CLASSES: tuple[str, ...] = (
        "standing",
        "sitting",
        "bending",
        "falling_lying",
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
# SUBSYSTEM 2: 6-CLASS GEOMETRIC & TEMPORAL POSTURE CLASSIFIER
# ==============================================================================

@dataclass
class FrameKinematicsSample:
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
    is_upright: bool
    raw_posture: PostureClass


@dataclass
class TrackHistoryState:
    track_id: int
    samples: deque[FrameKinematicsSample] = field(default_factory=lambda: deque(maxlen=30))
    posture_votes: deque[PostureClass] = field(default_factory=lambda: deque(maxlen=5))
    fall_latched_until: float = 0.0
    reference_height_px: float = 280.0
    last_seen_ts: float = 0.0


@dataclass
class PersonTelemetry:
    """Structured per-person telemetry from Subsystem 2 + Subsystem 3 safety evaluation."""
    track_id: int
    person_id: str
    bbox: tuple[float, float, float, float]
    confidence: float
    keypoints_xyc: np.ndarray  # (17, 3)
    posture: PostureClass
    torso_angle_deg: Optional[float]
    knee_angle_deg: Optional[float]
    vertical_velocity_mps: float
    rgb_fall_conf: Optional[float]
    # Subsystem 3 fields populated by SafetyDecisionEngine
    safety_tier: SafetyTier = SafetyTier.SAFE
    risk_score: int = 0
    posture_duration_s: float = 0.0
    severe_bend_duration_s: float = 0.0
    safety_reasons: list[str] = field(default_factory=list)
    audio_dispatched: bool = False
    telegram_dispatched: bool = False


def compute_joint_angle_deg(
    a: Optional[tuple[float, float]],
    b: Optional[tuple[float, float]],
    c: Optional[tuple[float, float]],
) -> Optional[float]:
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
    Deterministic geometric and temporal posture analysis using 17 COCO keypoints:
      - STANDING, SITTING, WALKING, BENDING, FALLING, LYING DOWN.
    """

    STANDING_MAX_TORSO_DEG: float = 25.0
    SITTING_MAX_TORSO_DEG: float = 35.0
    WALKING_MAX_TORSO_DEG: float = 30.0
    BENDING_MIN_TORSO_DEG: float = 45.0
    BENDING_SEVERE_TORSO_DEG: float = 60.0

    SITTING_THIGH_TORSO_RATIO: float = 0.65
    LYING_MIN_ASPECT_RATIO: float = 1.15
    FALL_MIN_VELOCITY_MPS: float = 1.5
    FALL_TRANSITION_WINDOW_SEC: float = 1.0
    FALL_LATCH_DURATION_SEC: float = 0.9
    ASSUMED_HUMAN_HEIGHT_M: float = 1.65

    def __init__(self, kp_conf_threshold: float = 0.30) -> None:
        self.kp_conf_threshold = kp_conf_threshold
        self.tracks: dict[int, TrackHistoryState] = {}

    def prune_stale_tracks(self, current_ts: float, max_stale_sec: float = 2.0) -> list[int]:
        stale = [
            tid for tid, st in self.tracks.items()
            if (current_ts - st.last_seen_ts) > max_stale_sec
        ]
        for tid in stale:
            self.tracks.pop(tid, None)
        return stale

    def _valid_point(self, kps: np.ndarray, idx: int) -> Optional[tuple[float, float]]:
        x, y, c = float(kps[idx, 0]), float(kps[idx, 1]), float(kps[idx, 2])
        if c >= self.kp_conf_threshold and x > 0.0 and y > 0.0 and math.isfinite(x) and math.isfinite(y):
            return (x, y)
        return None

    def _pair_midpoint(self, kps: np.ndarray, idx_a: int, idx_b: int) -> Optional[tuple[float, float]]:
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
    ) -> PersonTelemetry:
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

        mid_shoulder = self._pair_midpoint(keypoints_xyc, KP_L_SHOULDER, KP_R_SHOULDER)
        mid_hip = self._pair_midpoint(keypoints_xyc, KP_L_HIP, KP_R_HIP)
        mid_knee = self._pair_midpoint(keypoints_xyc, KP_L_KNEE, KP_R_KNEE)

        l_hip = self._valid_point(keypoints_xyc, KP_L_HIP)
        r_hip = self._valid_point(keypoints_xyc, KP_R_HIP)
        l_knee = self._valid_point(keypoints_xyc, KP_L_KNEE)
        r_knee = self._valid_point(keypoints_xyc, KP_R_KNEE)
        l_ankle = self._valid_point(keypoints_xyc, KP_L_ANKLE)
        r_ankle = self._valid_point(keypoints_xyc, KP_R_ANKLE)

        # Torso inclination angle relative to vertical baseline (0 deg = vertical, 90 deg = horizontal)
        torso_angle: Optional[float] = None
        torso_length: Optional[float] = None
        if mid_shoulder is not None and mid_hip is not None:
            dx = mid_shoulder[0] - mid_hip[0]
            dy = mid_hip[1] - mid_shoulder[1]
            torso_length = math.hypot(dx, dy)
            if torso_length > 4.0:
                torso_angle = math.degrees(math.atan2(abs(dx), max(1e-4, abs(dy))))
                if dy < -0.15 * torso_length:
                    torso_angle = max(torso_angle, 75.0)

        knee_angles = [
            ang for ang in (
                compute_joint_angle_deg(l_hip, l_knee, l_ankle),
                compute_joint_angle_deg(r_hip, r_knee, r_ankle),
            )
            if ang is not None
        ]
        min_knee_angle = min(knee_angles) if knee_angles else None
        max_knee_angle = max(knee_angles) if knee_angles else None

        thigh_vertical_dist: Optional[float] = None
        if mid_hip is not None and mid_knee is not None:
            thigh_vertical_dist = abs(mid_knee[1] - mid_hip[1])

        ankle_sep_x: Optional[float] = None
        if l_ankle is not None and r_ankle is not None:
            ankle_sep_x = abs(l_ankle[0] - r_ankle[0])

        is_upright_now = (torso_angle is not None and torso_angle < 35.0) and aspect_ratio < 0.85
        if is_upright_now and bbox_h > 0.25 * frame_height:
            state.reference_height_px = 0.85 * state.reference_height_px + 0.15 * bbox_h

        px_per_meter = max(80.0, state.reference_height_px / self.ASSUMED_HUMAN_HEIGHT_M)

        v_y_mps = 0.0
        v_y_px_per_sec = 0.0
        lateral_disp_norm = 0.0
        cyclic_ankle_swing = False
        had_recent_vertical = False

        recent_samples = [s for s in state.samples if (timestamp - s.timestamp) <= self.FALL_TRANSITION_WINDOW_SEC]
        window_5_10 = list(state.samples)[-9:]

        if recent_samples and not camera_moving:
            for prev in recent_samples:
                dt = timestamp - prev.timestamp
                if 0.06 <= dt <= 0.75:
                    dy_px = cy - prev.centroid_y
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

            ankle_seps = [s.ankle_sep_x for s in window_5_10 if s.ankle_sep_x is not None]
            if ankle_sep_x is not None:
                ankle_seps.append(ankle_sep_x)
            if len(ankle_seps) >= 3:
                max_sep = max(ankle_seps)
                min_sep = min(ankle_seps)
                if max_sep > 0.24 * ref_scale and (max_sep - min_sep) > 0.08 * ref_scale:
                    cyclic_ankle_swing = True

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

        # 6-Class Classification Rules
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
        elif (
            (aspect_ratio > self.LYING_MIN_ASPECT_RATIO or (torso_angle is not None and torso_angle >= 65.0))
            and shoulder_hip_equal_y
            and centroid_low_to_floor
            and v_y_mps <= self.FALL_MIN_VELOCITY_MPS
        ):
            raw_posture = PostureClass.LYING_DOWN
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
        else:
            raw_posture = PostureClass.STANDING

        state.samples.append(
            FrameKinematicsSample(
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
                is_upright=is_upright_now,
                raw_posture=raw_posture,
            )
        )

        state.posture_votes.append(raw_posture)
        if raw_posture == PostureClass.FALLING:
            final_posture = PostureClass.FALLING
        else:
            final_posture = Counter(state.posture_votes).most_common(1)[0][0]

        return PersonTelemetry(
            track_id=track_id,
            person_id=f"p-{track_id:03d}",
            bbox=bbox,
            confidence=det_conf,
            keypoints_xyc=keypoints_xyc,
            posture=final_posture,
            torso_angle_deg=torso_angle,
            knee_angle_deg=min_knee_angle,
            vertical_velocity_mps=v_y_mps,
            rgb_fall_conf=rgb_fall_confidence,
        )


# ==============================================================================
# SUBSYSTEM 3: SAFETY DECISION ENGINE, TEMPORAL TIMERS & NON-BLOCKING ALERTS
# ==============================================================================

@dataclass
class ResidentPersistenceTimer:
    """
    Tracks temporal persistence per resident for Subsystem 3 safety escalation:
      - Continuous bending > 60 deg (or sustained bending) >= 3.0s -> Escalates Warning to Unsafe.
      - Continuous lying down >= 6.0s -> Escalates Unsafe to Alert.
      - Per-resident 15.0s audio & Telegram dispatch cooldown.
    """
    person_id: str
    active_posture: Optional[PostureClass] = None
    posture_since_ts: float = 0.0
    severe_bend_since_ts: Optional[float] = None
    last_audio_alert_ts: float = -float("inf")
    last_audio_tier: Optional[SafetyTier] = None
    last_telegram_ts: float = -float("inf")
    last_status_db_ts: float = -float("inf")


class NonBlockingAlertDispatcher:
    """
    Dedicated background daemon threads for:
      1. Local TTS (`pyttsx3`) + acoustic attention tone alerts (`sounddevice` / `pyttsx3`).
      2. Telegram Bot webhook notifications (`requests.post`).
    Enforces a strict 15-second per-resident cooldown so speech synthesis or
    network requests NEVER block or lag the OpenCV video loop or PTZ controls.
    """

    def __init__(
        self,
        cooldown_seconds: float = 15.0,
        audio_enabled: bool = True,
        telegram_enabled: bool = True,
        zone_name: str = "monitoring area",
    ) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.audio_enabled = audio_enabled
        self.zone_name = zone_name

        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.telegram_active = bool(telegram_enabled and self.telegram_token and self.telegram_chat_id)

        self._speech_queue: queue.Queue[Optional[tuple[SafetyTier, str]]] = queue.Queue(maxsize=8)
        self._telegram_queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=16)
        self._stop_event = threading.Event()

        self._speech_thread = threading.Thread(
            target=self._speech_worker,
            name="SIDP-S3-TTS-Worker",
            daemon=True,
        )
        self._telegram_thread = threading.Thread(
            target=self._telegram_worker,
            name="SIDP-S3-Telegram-Worker",
            daemon=True,
        )
        self._speech_thread.start()
        self._telegram_thread.start()

    def _play_attention_beep(self, tier: SafetyTier) -> None:
        """Synthesize and play Subsystem 3's attention tone before spoken TTS."""
        try:
            from alert_engine.audio import TONE_SAMPLERATE, _make_beep_tone
            import sounddevice as sd

            tone = _make_beep_tone(tier.value, TONE_SAMPLERATE)
            sd.play(tone, TONE_SAMPLERATE)
            sd.wait()
        except Exception:
            pass

    def _speech_worker(self) -> None:
        """Background daemon thread running pyttsx3 speech synthesis."""
        # On Windows, COM must be initialized per background thread for SAPI5 pyttsx3
        if os.name == "nt":
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass

        while not self._stop_event.is_set():
            try:
                item = self._speech_queue.get(timeout=0.15)
            except queue.Empty:
                continue

            if item is None:
                self._speech_queue.task_done()
                break

            tier, speech_text = item
            if self.audio_enabled and not self._stop_event.is_set():
                try:
                    self._play_attention_beep(tier)
                    import pyttsx3

                    engine = pyttsx3.init()
                    engine.setProperty("rate", 175)
                    engine.setProperty("volume", 1.0)
                    engine.say(speech_text)
                    engine.runAndWait()
                    engine.stop()
                except Exception as exc:
                    logger.warning("Background TTS playback error: %s", exc)
            self._speech_queue.task_done()

    def _telegram_worker(self) -> None:
        """Background daemon thread sending Telegram Bot HTTP POST notifications."""
        while not self._stop_event.is_set():
            try:
                msg = self._telegram_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if msg is None:
                self._telegram_queue.task_done()
                break

            if self.telegram_active and not self._stop_event.is_set():
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                try:
                    requests.post(
                        url,
                        json={"chat_id": self.telegram_chat_id, "text": msg},
                        timeout=4.0,
                    )
                except Exception as exc:
                    logger.warning("Telegram webhook notification failed: %s", exc)
            self._telegram_queue.task_done()

    def enqueue_speech(self, tier: SafetyTier, message: str) -> bool:
        if not self.audio_enabled or self._stop_event.is_set():
            return False
        # Prevent speech backlog buildup
        if self._speech_queue.full():
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
            except queue.Empty:
                pass
        try:
            self._speech_queue.put_nowait((tier, message))
            return True
        except queue.Full:
            return False

    def enqueue_telegram(self, message: str) -> bool:
        if not self.telegram_active or self._stop_event.is_set():
            return False
        try:
            self._telegram_queue.put_nowait(message)
            return True
        except queue.Full:
            return False

    def shutdown(self) -> None:
        self._stop_event.set()
        try:
            self._speech_queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            self._telegram_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._speech_thread.is_alive():
            self._speech_thread.join(timeout=1.5)
        if self._telegram_thread.is_alive():
            self._telegram_thread.join(timeout=1.5)


class Subsystem3SafetyEngine:
    """
    Unifies Subsystem 3 (`./Subsystem3-Safety-Engine`) with Subsystem 2's live
    pose telemetry:
      - Loads `Subsystem3-Safety-Engine/config/config.yaml` and `models/risk_model.npz`.
      - Runs `DebounceTracker` and `HybridScorer` (`ml_scoring.py` + `scoring.py`).
      - Enforces the 4-tier safety state machine (`Safe`, `Warning`, `Unsafe`, `Alert`)
        and the 3.0-second temporal persistence timer for bending (>60 deg or sustained).
      - Logs statuses and alerts to SQLite via `LocalSqliteSink` (`data/events.db`).
      - Dispatches non-blocking `pyttsx3` voice alerts and Telegram notifications
        with a 15.0-second per-resident cooldown.
    """

    BENDING_ESCALATION_SECONDS: float = 3.0
    SEVERE_BENDING_ANGLE_DEG: float = 60.0
    LYING_ALERT_ESCALATION_SECONDS: float = 6.0
    RESIDENT_ALERT_COOLDOWN_SECONDS: float = 15.0

    def __init__(
        self,
        camera_id: str = "tapo_c210",
        zone_name: str = "living room",
        audio_enabled: bool = True,
        telegram_enabled: bool = True,
        cooldown_seconds: float = 15.0,
    ) -> None:
        self.camera_id = camera_id
        self.zone_name = zone_name
        self.cooldown_seconds = cooldown_seconds

        # Load Subsystem 3 YAML configuration & resolve model/SQLite paths
        s3_cfg_path = SUBSYSTEM3_ROOT / "config" / "config.yaml"
        self.s3_config: S3Config = load_s3_config(s3_cfg_path)
        model_rel = Path(self.s3_config.ml.model_path)
        if not model_rel.is_absolute():
            self.s3_config.ml.model_path = str(SUBSYSTEM3_ROOT / model_rel)

        sqlite_rel = Path(self.s3_config.sink.sqlite_path)
        if not sqlite_rel.is_absolute():
            self.s3_config.sink.sqlite_path = str(WORKSPACE_ROOT / "outputs" / sqlite_rel.name)

        self.debounce = DebounceTracker(self.s3_config.debounce)
        self.scorer = HybridScorer.from_config(self.s3_config)
        self.sink = LocalSqliteSink(sqlite_path=self.s3_config.sink.sqlite_path)
        self.sink.start()

        self.alert_dispatcher = NonBlockingAlertDispatcher(
            cooldown_seconds=cooldown_seconds,
            audio_enabled=audio_enabled,
            telegram_enabled=telegram_enabled,
            zone_name=zone_name,
        )
        self.timers: dict[str, ResidentPersistenceTimer] = {}
        logger.info(
            "Subsystem 3 Safety Engine ready (scorer=%s, cooldown=%.1fs, db=%s)",
            self.scorer.mode,
            self.cooldown_seconds,
            self.s3_config.sink.sqlite_path,
        )

    def reset_resident(self, person_id: str) -> None:
        self.debounce.reset_person(person_id)
        self.timers.pop(person_id, None)

    def evaluate(self, telemetry: PersonTelemetry, now_perf: float) -> PersonTelemetry:
        now_utc = datetime.now(timezone.utc)
        person_id = telemetry.person_id
        s3_posture = telemetry.posture.s3_name

        # 1. Update per-resident temporal persistence timers
        timer = self.timers.get(person_id)
        if timer is None:
            timer = ResidentPersistenceTimer(
                person_id=person_id,
                active_posture=telemetry.posture,
                posture_since_ts=now_perf,
            )
            self.timers[person_id] = timer

        if timer.active_posture != telemetry.posture:
            timer.active_posture = telemetry.posture
            timer.posture_since_ts = now_perf

        posture_duration_s = max(0.0, now_perf - timer.posture_since_ts)

        # Track continuous severe bending (>60 deg) or sustained bending persistence timer
        if telemetry.posture == PostureClass.BENDING:
            is_severe_angle = (
                telemetry.torso_angle_deg is not None
                and telemetry.torso_angle_deg >= self.SEVERE_BENDING_ANGLE_DEG
            )
            if is_severe_angle:
                if timer.severe_bend_since_ts is None:
                    timer.severe_bend_since_ts = now_perf
                severe_bend_duration_s = max(0.0, now_perf - timer.severe_bend_since_ts)
            else:
                timer.severe_bend_since_ts = None
                severe_bend_duration_s = 0.0
        else:
            timer.severe_bend_since_ts = None
            severe_bend_duration_s = 0.0

        # 2. Feed structured event into Subsystem 3 DebounceTracker & HybridScorer
        event_in = PostureEventIn(
            event_id=str(uuid.uuid4()),
            timestamp=now_utc,
            camera_id=self.camera_id,
            person_id=person_id,
            posture=s3_posture,
            confidence=clamp(telemetry.confidence, 0.0, 1.0),
            bend_angle_deg=telemetry.torso_angle_deg,
            zone=self.zone_name,
        )
        debounced = self.debounce.process(event_in)
        effective_duration_s = max(posture_duration_s, debounced.duration_s)

        s3_score: ScoreResult = self.scorer.score(
            posture=s3_posture,
            duration_s=effective_duration_s,
            person_id=person_id,
            confidence=event_in.confidence,
            bend_angle_deg=telemetry.torso_angle_deg,
        )
        risk_score = int(s3_score.risk_score)
        reasons = list(s3_score.reasons)

        # 3. Deterministic 4-Tier Safety State Machine (`Safe`, `Warning`, `Unsafe`, `Alert`)
        if telemetry.posture == PostureClass.FALLING:
            tier = SafetyTier.ALERT
            risk_score = max(risk_score, 95)
            reasons.insert(0, f"FALLING detected (vy={telemetry.vertical_velocity_mps:.2f} m/s)")

        elif telemetry.posture == PostureClass.LYING_DOWN:
            if effective_duration_s >= self.LYING_ALERT_ESCALATION_SECONDS or risk_score >= 80:
                tier = SafetyTier.ALERT
                risk_score = max(risk_score, 85)
                reasons.insert(0, f"Prolonged lying on floor ({effective_duration_s:.1f}s)")
            else:
                tier = SafetyTier.UNSAFE
                risk_score = max(risk_score, 65)
                reasons.insert(0, f"Lying down detected ({effective_duration_s:.1f}s)")

        elif telemetry.posture == PostureClass.BENDING:
            # 3.0s persistence timer before escalating Bending from Warning to Unsafe
            angle_val = telemetry.torso_angle_deg or 45.0
            if severe_bend_duration_s >= self.BENDING_ESCALATION_SECONDS or (
                angle_val >= self.SEVERE_BENDING_ANGLE_DEG and effective_duration_s >= self.BENDING_ESCALATION_SECONDS
            ):
                tier = SafetyTier.UNSAFE
                risk_score = max(risk_score, 68)
                reasons.insert(
                    0,
                    f"Sustained bending >60° ({angle_val:.0f}°) for {max(severe_bend_duration_s, effective_duration_s):.1f}s (>=3.0s threshold)",
                )
            elif effective_duration_s >= self.BENDING_ESCALATION_SECONDS and risk_score >= 60:
                tier = SafetyTier.UNSAFE
                risk_score = max(risk_score, 62)
                reasons.insert(0, f"Sustained bending ({angle_val:.0f}°) for {effective_duration_s:.1f}s")
            else:
                tier = SafetyTier.WARNING
                risk_score = clamp(max(risk_score, 38), 30, 59)
                reasons.insert(0, f"Bending inclination {angle_val:.0f}° ({effective_duration_s:.1f}s / 3.0s timer)")

        else:
            # STANDING, SITTING, WALKING -> Safe
            tier = SafetyTier.SAFE
            risk_score = min(risk_score, 25)

        telemetry.safety_tier = tier
        telemetry.risk_score = int(risk_score)
        telemetry.posture_duration_s = effective_duration_s
        telemetry.severe_bend_duration_s = severe_bend_duration_s
        telemetry.safety_reasons = reasons

        # 4. Non-blocking SQLite Status Logging & 15s Cooldown Alert Dispatch
        if (now_perf - timer.last_status_db_ts) >= 2.0 or debounced.changed:
            timer.last_status_db_ts = now_perf
            self.sink.write_status(person_id, s3_posture, tier.value, telemetry.risk_score, now_utc)

        if tier in (SafetyTier.UNSAFE, SafetyTier.ALERT):
            cooldown_elapsed = (now_perf - timer.last_audio_alert_ts) >= self.cooldown_seconds
            escalated_to_alert = (tier == SafetyTier.ALERT and timer.last_audio_tier != SafetyTier.ALERT)

            if cooldown_elapsed or escalated_to_alert:
                timer.last_audio_alert_ts = now_perf
                timer.last_audio_tier = tier

                classified_evt = ClassifiedEvent(
                    alert_id=str(uuid.uuid4()),
                    event_id=event_in.event_id,
                    person_id=person_id,
                    camera_id=self.camera_id,
                    zone=self.zone_name,
                    posture=s3_posture,
                    tier=tier.value,
                    risk_score=telemetry.risk_score,
                    duration_s=effective_duration_s,
                    reasons=reasons,
                    detected_at=now_utc,
                    dispatched_at=now_utc,
                )
                self.sink.write_alert(classified_evt)

                if tier == SafetyTier.ALERT:
                    speech_msg = (
                        f"Emergency Alert! Resident {telemetry.track_id} {telemetry.posture.value.lower()} "
                        f"detected in the {self.zone_name}. Please check immediately."
                    )
                else:
                    angle_info = f" at {telemetry.torso_angle_deg:.0f} degrees" if telemetry.torso_angle_deg else ""
                    speech_msg = (
                        f"Safety Warning. Resident {telemetry.track_id} unsafe {telemetry.posture.value.lower()}"
                        f"{angle_info} sustained for {effective_duration_s:.0f} seconds."
                    )

                telemetry.audio_dispatched = self.alert_dispatcher.enqueue_speech(tier, speech_msg)

                tg_text = (
                    f"[{tier.badge_text}] SIDP Elderly Care Alert\n"
                    f"Resident: {person_id} (Track #{telemetry.track_id})\n"
                    f"Posture: {telemetry.posture.value}\n"
                    f"Risk Score: {telemetry.risk_score}/100 (Duration: {effective_duration_s:.1f}s)\n"
                    f"Reason: {reasons[0] if reasons else 'Safety threshold exceeded'}"
                )
                telemetry.telegram_dispatched = self.alert_dispatcher.enqueue_telegram(tg_text)
                logger.warning(
                    "S3 ALERT DISPATCHED | resident=%s tier=%s posture=%s score=%d duration=%.1fs",
                    person_id,
                    tier.badge_text,
                    telemetry.posture.value,
                    telemetry.risk_score,
                    effective_duration_s,
                )

        return telemetry

    def shutdown(self) -> None:
        self.alert_dispatcher.shutdown()
        self.sink.stop()


# ==============================================================================
# SUBSYSTEM 2 + 3 VISION & SAFETY PIPELINE RUNNER
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
    def __init__(self, max_stale_sec: float = 1.2) -> None:
        self.max_stale_sec = max_stale_sec
        self._next_id = 1
        self._tracks: dict[int, tuple[tuple[float, float, float, float], float]] = {}

    def assign(self, bboxes: list[tuple[float, float, float, float]], now: float) -> list[int]:
        stale = [tid for tid, (_, ts) in self._tracks.items() if (now - ts) > self.max_stale_sec]
        for tid in stale:
            self._tracks.pop(tid, None)

        assigned_ids: list[int] = []
        used: set[int] = set()
        for box in bboxes:
            best_id: Optional[int] = None
            best_score = 0.15
            for tid, (prev_box, _) in self._tracks.items():
                if tid in used:
                    continue
                score = _bbox_iou(box, prev_box)
                if score > best_score:
                    best_score = score
                    best_id = tid
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            used.add(best_id)
            self._tracks[best_id] = (box, now)
            assigned_ids.append(best_id)
        return assigned_ids


class UnifiedVisionSafetyPipeline:
    """
    Runs Subsystem 2 (YOLO26-Pose + Auxiliary RGB Fall Detector + 6-Class Posture)
    and pipes each tracked person directly through Subsystem 3 (`Subsystem3SafetyEngine`).
    """

    def __init__(
        self,
        safety_engine: Subsystem3SafetyEngine,
        pose_weights: Optional[str] = None,
        fall_weights: str = "models/human-fall-detection.pt",
        device: str = "auto",
        conf_threshold: float = 0.30,
        kp_conf_threshold: float = 0.30,
        imgsz: int = 640,
    ) -> None:
        import torch
        from ultralytics import YOLO

        self.safety_engine = safety_engine
        self.conf_threshold = conf_threshold
        self.kp_conf_threshold = kp_conf_threshold
        self.imgsz = imgsz

        if device == "auto":
            self.device = "0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.use_half = self.device != "cpu" and torch.cuda.is_available()

        resolved_pose = self._resolve_pose_weights(pose_weights)
        logger.info("Loading YOLO-Pose model: %s (device=%s, fp16=%s)", resolved_pose, self.device, self.use_half)
        self.pose_model = YOLO(str(resolved_pose))

        self.fall_model = None
        self.fallen_class_ids: set[int] = set()
        fall_path = WORKSPACE_ROOT / fall_weights
        if fall_path.is_file():
            try:
                self.fall_model = YOLO(str(fall_path))
                for cid, cname in self.fall_model.names.items():
                    if "fall" in str(cname).lower():
                        self.fallen_class_ids.add(int(cid))
                logger.info("Loaded auxiliary RGB fall detector: %s", fall_path)
            except Exception as exc:
                logger.warning("Auxiliary RGB fall detector unavailable (%s)", exc)
                self.fall_model = None

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
            res = self.fall_model.predict(frame, imgsz=416, conf=0.25, device=self.device, verbose=False)[0]
            boxes_out: list[tuple[tuple[float, float, float, float], float]] = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()
                for b, c, k in zip(xyxy, confs, clss):
                    if int(k) in self.fallen_class_ids:
                        boxes_out.append(((float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(c)))
            self._cached_fall_boxes = boxes_out
        except Exception:
            pass
        return self._cached_fall_boxes

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        camera_moving: bool = False,
    ) -> list[PersonTelemetry]:
        h, w = frame.shape[:2]
        stale_ids = self.posture_engine.prune_stale_tracks(timestamp)
        for tid in stale_ids:
            self.safety_engine.reset_resident(f"p-{tid:03d}")

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
        kps_data = res.keypoints.data.cpu().numpy()

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

        outputs: list[PersonTelemetry] = []
        for tid, bbox, det_conf, kps in zip(track_ids, raw_bboxes, confs, kps_data):
            if kps.shape != (17, 3):
                continue

            matched_fall_conf: Optional[float] = None
            for fbox, fconf in fall_boxes:
                if _bbox_iou(bbox, fbox) >= 0.25:
                    if matched_fall_conf is None or fconf > matched_fall_conf:
                        matched_fall_conf = fconf

            # Subsystem 2 evaluation
            telemetry = self.posture_engine.evaluate_person(
                track_id=tid,
                bbox=bbox,
                det_conf=float(det_conf),
                keypoints_xyc=kps,
                timestamp=timestamp,
                frame_height=h,
                rgb_fall_confidence=matched_fall_conf,
                camera_moving=camera_moving,
            )
            # Subsystem 3 safety state machine & non-blocking alert evaluation
            telemetry = self.safety_engine.evaluate(telemetry, timestamp)
            outputs.append(telemetry)

        return outputs


# ==============================================================================
# UNIFIED SUBSYSTEM 1 + 2 + 3 HUD RENDERER
# ==============================================================================

def _tier_visual_style(tier: SafetyTier, now_ts: float) -> tuple[tuple[int, int, int], tuple[int, int, int], bool]:
    """
    Returns (badge_bg_bgr, text_fg_bgr, is_flashing_phase) for Subsystem 3 tiers:
      - SAFE    -> GREEN
      - WARNING -> YELLOW
      - UNSAFE  -> RED
      - ALERT   -> FLASHING RED (4 Hz strobe between vivid red and high-contrast white/red)
    """
    if tier == SafetyTier.SAFE:
        return (45, 200, 55), (10, 10, 10), False
    if tier == SafetyTier.WARNING:
        return (0, 215, 255), (10, 10, 10), False
    if tier == SafetyTier.UNSAFE:
        return (25, 25, 230), (250, 250, 250), False

    # SafetyTier.ALERT -> 4 Hz flashing red strobe
    flash_on = (int(now_ts * 8.0) % 2) == 0
    if flash_on:
        return (15, 15, 255), (255, 255, 255), True
    return (245, 245, 255), (0, 0, 235), False


def render_master_hud(
    frame: np.ndarray,
    persons: list[PersonTelemetry],
    ptz: UnifiedPTZController,
    auto_telemetry: AutoTrackTelemetry,
    fps: float,
    ai_overlay_enabled: bool,
    collector: DatasetCollector,
    kp_conf_threshold: float = 0.30,
) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    now_ts = time.perf_counter()

    any_alert = any(p.safety_tier == SafetyTier.ALERT for p in persons)

    if ai_overlay_enabled:
        # Flashing red screen perimeter border when any resident is in ALERT tier
        if any_alert and (int(now_ts * 8.0) % 2 == 0):
            cv2.rectangle(canvas, (0, 62), (w - 1, h - 1), (15, 15, 255), 6)

        for p in persons:
            x1, y1, x2, y2 = map(int, p.bbox)
            badge_bg, badge_fg, _ = _tier_visual_style(p.safety_tier, now_ts)

            # 1. Skeleton wireframe (17 COCO keypoints)
            for i_a, i_b in COCO_SKELETON_EDGES:
                xa, ya, ca = p.keypoints_xyc[i_a]
                xb, yb, cb = p.keypoints_xyc[i_b]
                if ca >= kp_conf_threshold and cb >= kp_conf_threshold and xa > 0 and ya > 0 and xb > 0 and yb > 0:
                    pt1 = (int(round(xa)), int(round(ya)))
                    pt2 = (int(round(xb)), int(round(yb)))
                    cv2.line(canvas, pt1, pt2, (10, 10, 10), 4, cv2.LINE_AA)
                    cv2.line(canvas, pt1, pt2, (245, 245, 245), 2, cv2.LINE_AA)

            for j_idx in range(17):
                jx, jy, jc = p.keypoints_xyc[j_idx]
                if jc >= kp_conf_threshold and jx > 0 and jy > 0:
                    jpt = (int(round(jx)), int(round(jy)))
                    cv2.circle(canvas, jpt, 4, (10, 10, 10), -1, cv2.LINE_AA)
                    cv2.circle(canvas, jpt, 2, badge_bg, -1, cv2.LINE_AA)

            # 2. Bounding box with tier color
            box_thick = 4 if p.safety_tier == SafetyTier.ALERT else (3 if p.safety_tier == SafetyTier.UNSAFE else 2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (10, 10, 10), box_thick + 2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), badge_bg, box_thick)

            # 3. Subsystem 2 + Subsystem 3 Multi-line Telemetry Badge
            angle_str = f"{p.torso_angle_deg:.0f}deg" if p.torso_angle_deg is not None else "N/A"
            line_1 = f"ID:{p.track_id} [{p.safety_tier.badge_text}] {p.posture.value} | Risk:{p.risk_score}/100"
            line_2 = f"Torso:{angle_str} | Hold:{p.posture_duration_s:.1f}s | Conf:{p.confidence:.0%}"

            lines = [line_1, line_2]
            if p.posture == PostureClass.BENDING and p.safety_tier in (SafetyTier.WARNING, SafetyTier.UNSAFE):
                timer_val = max(p.severe_bend_duration_s, p.posture_duration_s)
                lines.append(f"S3 Bend Timer: {timer_val:.1f}s / 3.0s -> {p.safety_tier.badge_text}")
            elif p.safety_tier in (SafetyTier.UNSAFE, SafetyTier.ALERT) and p.safety_reasons:
                lines.append(f"S3: {p.safety_reasons[0][:44]}")

            pad_x, pad_y, line_h = 7, 5, 17
            widths = [cv2.getTextSize(t, font, 0.44, 1)[0][0] for t in lines]
            box_w = (max(widths) if widths else 120) + pad_x * 2
            box_h = len(lines) * line_h + pad_y * 2

            bx1 = max(4, min(x1, w - box_w - 4))
            by1 = max(66, y1 - box_h - 4) if (y1 - box_h - 4) >= 66 else min(h - box_h - 6, y2 + 4)
            bx2, by2 = bx1 + box_w, by1 + box_h

            # Dark body panel + color-coded Subsystem 3 header ribbon
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (16, 16, 16), -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by1 + line_h + 6), badge_bg, -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), badge_bg, 1)

            for idx, text in enumerate(lines):
                ty = by1 + pad_y + 12 + idx * line_h
                fg = badge_fg if idx == 0 else (240, 240, 240)
                cv2.putText(canvas, text, (bx1 + pad_x, ty), font, 0.44, fg, 1, cv2.LINE_AA)

    # 4. AUTOMATIC Mode Target Reticle & Deadband Box
    if ptz.mode == CameraMode.AUTOMATIC and not ptz.patrolling:
        cx_f, cy_f = w // 2, h // 2
        db_w = int(round(UnifiedPTZController.AUTO_DEADBAND * (w / 2.0)))
        db_h = int(round(UnifiedPTZController.AUTO_DEADBAND * (h / 2.0)))
        db_col = (180, 220, 180) if auto_telemetry.in_deadband else (0, 215, 255)
        cv2.rectangle(canvas, (cx_f - db_w, cy_f - db_h), (cx_f + db_w, cy_f + db_h), db_col, 1, cv2.LINE_AA)
        cv2.line(canvas, (cx_f - 8, cy_f), (cx_f + 8, cy_f), db_col, 1, cv2.LINE_AA)
        cv2.line(canvas, (cx_f, cy_f - 8), (cx_f, cy_f + 8), db_col, 1, cv2.LINE_AA)

        if auto_telemetry.group_bbox is not None and auto_telemetry.collective_center is not None:
            gx1, gy1, gx2, gy2 = auto_telemetry.group_bbox
            gcx, gcy = auto_telemetry.collective_center
            r_col = (50, 230, 50) if auto_telemetry.in_deadband else (0, 200, 255)
            cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), r_col, 1, cv2.LINE_AA)
            cv2.circle(canvas, (gcx, gcy), 10, r_col, 2, cv2.LINE_AA)
            cv2.line(canvas, (gcx - 16, gcy), (gcx + 16, gcy), r_col, 2, cv2.LINE_AA)
            cv2.line(canvas, (gcx, gcy - 16), (gcx, gcy + 16), r_col, 2, cv2.LINE_AA)
            cv2.arrowedLine(canvas, (cx_f, cy_f), (gcx, gcy), r_col, 1, cv2.LINE_AA, tipLength=0.12)

    # 5. Top OSD Telemetry Bar
    cv2.rectangle(canvas, (0, 0), (w, 60), (12, 12, 12), -1)
    cv2.line(canvas, (0, 60), (w, 60), (190, 190, 190), 1)

    highest_tier = SafetyTier.SAFE
    tier_order = [SafetyTier.SAFE, SafetyTier.WARNING, SafetyTier.UNSAFE, SafetyTier.ALERT]
    for p in persons:
        if tier_order.index(p.safety_tier) > tier_order.index(highest_tier):
            highest_tier = p.safety_tier

    sys_bg, sys_fg, _ = _tier_visual_style(highest_tier, now_ts)
    cv2.rectangle(canvas, (10, 6), (148, 30), sys_bg, -1)
    cv2.putText(canvas, f"S3: {highest_tier.badge_text}", (18, 23), font, 0.50, sys_fg, 2, cv2.LINE_AA)

    hud_top = (
        f"PTZ: {ptz.hud_mode_label}  |  FPS: {fps:.1f}  |  Occupants: {len(persons)}  |  "
        f"AI(P): {'ON' if ai_overlay_enabled else 'OFF'}"
    )
    cv2.putText(canvas, hud_top, (160, 23), font, 0.46, (240, 240, 240), 1, cv2.LINE_AA)

    rec_tag = "REC 2FPS" if collector.auto_recording else "STANDBY"
    hud_bottom = (
        f"[M/Tab] Mode  [WASD] Move  [Z] Patrol  [P] HUD  "
        f"[1-4/C/R] Data: {collector.current_class.upper()} ({rec_tag})  [Q] Quit"
    )
    cv2.putText(canvas, hud_bottom, (10, 48), font, 0.40, (185, 185, 185), 1, cv2.LINE_AA)

    return canvas


# ==============================================================================
# CLI & MAIN EXECUTION LOOP
# ==============================================================================

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SIDP Unified Master Pipeline: Subsystems 1 (RTSP/PTZ) + 2 (YOLO26-Pose) + 3 (Safety & Alerts)"
    )
    parser.add_argument("--ip", default=DEFAULT_CAMERA_IP, help="Tapo C210 IP address")
    parser.add_argument("--user", default=DEFAULT_CAMERA_USER, help="Camera RTSP/ONVIF username")
    parser.add_argument("--password", default=DEFAULT_CAMERA_PASS, help="Camera RTSP/ONVIF password")
    parser.add_argument("--stream", default=DEFAULT_CAMERA_STREAM, help="RTSP stream path (stream1 or stream2)")
    parser.add_argument("--webcam", action="store_true", help="Use local webcam instead of RTSP stream")
    parser.add_argument("--camera-index", type=int, default=0, help="Local webcam index when --webcam is set")
    parser.add_argument("--pose-model", default=None, help="YOLO-Pose checkpoint (default: models/yolo26m-pose.pt)")
    parser.add_argument("--fall-model", default="models/human-fall-detection.pt", help="Auxiliary RGB fall model")
    parser.add_argument("--device", default="auto", help="Inference device: auto, 0, cpu")
    parser.add_argument("--mode", choices=["STATIC", "MANUAL", "AUTOMATIC"], default="MANUAL", help="Initial PTZ mode")
    parser.add_argument("--zone", default="elderly care room", help="Monitored zone name for voice/Telegram alerts")
    parser.add_argument("--cooldown", type=float, default=15.0, help="Per-resident audio/Telegram cooldown in seconds")
    parser.add_argument("--no-ptz", action="store_true", help="Disable ONVIF PTZ motor initialization")
    parser.add_argument("--no-audio", action="store_true", help="Disable pyttsx3 voice alert synthesis")
    parser.add_argument("--headless", action="store_true", help="Run without OpenCV GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Bounded frame count for smoke tests")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv(WORKSPACE_ROOT / ".env")
    load_dotenv(SUBSYSTEM3_ROOT / ".env")
    args = parse_args(argv)

    rtsp_url = f"rtsp://{args.user}:{args.password}@{args.ip}:{DEFAULT_RTSP_PORT}/{args.stream}"
    initial_mode = CameraMode(args.mode)

    # 1. Initialize Subsystem 3 Safety Decision Engine & Non-Blocking Alert Threads
    safety_engine = Subsystem3SafetyEngine(
        camera_id="tapo_c210" if not args.webcam else f"webcam_{args.camera_index}",
        zone_name=args.zone,
        audio_enabled=(not args.no_audio),
        telegram_enabled=True,
        cooldown_seconds=args.cooldown,
    )

    # 2. Initialize Subsystem 2 Vision Analytics Pipeline
    pipeline = UnifiedVisionSafetyPipeline(
        safety_engine=safety_engine,
        pose_weights=args.pose_model,
        fall_weights=args.fall_model,
        device=args.device,
    )
    pipeline.warmup()

    # 3. Initialize Subsystem 1 ONVIF PTZ Controller, Dataset Collector, and RTSP Stream
    ptz = UnifiedPTZController(
        ip=args.ip,
        port=DEFAULT_ONVIF_PORT,
        username=args.user,
        password=args.password,
        enabled=(not args.no_ptz and not args.webcam),
        initial_mode=initial_mode,
    )
    collector = DatasetCollector()
    stream = ThreadSafeStreamReader(
        rtsp_url=rtsp_url,
        use_webcam=args.webcam,
        webcam_index=args.camera_index,
    ).start()

    window_title = "SIDP Master (Subsystems 1 + 2 + 3) - Vision, PTZ & Safety Engine"
    ai_overlay_enabled = True
    fps_ema = 0.0
    prev_loop_ts = time.perf_counter()
    last_frame_id = -1
    processed_frames = 0

    logger.info(
        "SIDP Subsystems 1+2+3 Master running | PTZ=%s | Source=%s",
        ptz.mode.value,
        f"webcam:{args.camera_index}" if args.webcam else f"rtsp://{args.ip}:{DEFAULT_RTSP_PORT}/{args.stream}",
    )

    try:
        while args.max_frames is None or processed_frames < args.max_frames:
            connected, frame, frame_ts, frame_id = stream.read()
            if not connected or frame is None or frame_id == last_frame_id:
                time.sleep(0.005)
                continue

            last_frame_id = frame_id
            clean_frame = frame.copy()
            collector.maybe_auto_record(clean_frame)

            # Execute Subsystem 2 (Vision) + Subsystem 3 (Safety State Machine & Alerts)
            persons = pipeline.process_frame(
                frame=frame,
                timestamp=frame_ts,
                camera_moving=ptz.is_camera_moving(),
            )

            # Execute Subsystem 1 Automatic Group-Centering PTZ update
            h, w = frame.shape[:2]
            auto_telemetry = ptz.update_automatic_tracking(
                bboxes=[p.bbox for p in persons],
                frame_width=w,
                frame_height=h,
                now=frame_ts,
            )

            now_loop = time.perf_counter()
            inst_fps = 1.0 / max(1e-4, now_loop - prev_loop_ts)
            prev_loop_ts = now_loop
            fps_ema = inst_fps if fps_ema == 0.0 else (0.85 * fps_ema + 0.15 * inst_fps)
            processed_frames += 1

            if not args.headless:
                hud_frame = render_master_hud(
                    frame=frame,
                    persons=persons,
                    ptz=ptz,
                    auto_telemetry=auto_telemetry,
                    fps=fps_ema,
                    ai_overlay_enabled=ai_overlay_enabled,
                    collector=collector,
                )
                cv2.imshow(window_title, hud_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    logger.info("Clean shutdown initiated by user (Q).")
                    break
                elif key in (ord("p"), ord("P")):
                    ai_overlay_enabled = not ai_overlay_enabled
                    logger.info("AI & Safety HUD Overlay -> %s", "ON" if ai_overlay_enabled else "OFF")
                elif key in (ord("m"), ord("M"), 9):
                    ptz.cycle_mode()
                elif key in (ord("z"), ord("Z")):
                    ptz.toggle_patrol()
                elif key in (ord("w"), ord("W"), ord("a"), ord("A"), ord("s"), ord("S"), ord("d"), ord("D")):
                    ptz.manual_steer(chr(key))
                elif ord("1") <= key <= ord("4"):
                    cls_name = collector.select_class_index(key - ord("1"))
                    logger.info("Dataset target class -> %s", cls_name)
                elif key in (ord("c"), ord("C")):
                    saved = collector.save_frame(clean_frame)
                    logger.info("Saved clean dataset snapshot -> %s", saved)
                elif key in (ord("r"), ord("R")):
                    rec_state = collector.toggle_auto_record()
                    logger.info("Dataset 2 FPS auto-capture -> %s", "ON" if rec_state else "OFF")

                if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted via Ctrl+C.")
    finally:
        logger.info("Shutting down PTZ motors, RTSP stream, SQLite sink, and audio threads...")
        stream.stop()
        ptz.shutdown()
        safety_engine.shutdown()
        if not args.headless:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

