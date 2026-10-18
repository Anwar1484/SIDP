"""Context-managed RTSP (Tapo C210) or local webcam capture with zero-lag frame draining."""
import logging
import os
import threading
import time
from time import perf_counter

import cv2
import numpy as np

from app.config import CameraConfig

logger = logging.getLogger(__name__)


class WebcamCapture:
    """
    Subsystem 1 unified video capture:
    - When config.source == 'rtsp', connects to the Tapo C210 RTSP stream via FFmpeg
      and drains stale frames on a background thread so read() always returns the newest frame.
    - When config.source == 'webcam', opens the local camera index.
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: cv2.VideoCapture | None = None
        self._cond = threading.Condition()
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._latest_frame: np.ndarray | None = None
        self._frame_seq = 0
        self._read_seq = 0

    def __enter__(self):
        self._stopped = False
        if self.config.source == "rtsp":
            if self.config.rtsp_transport:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{self.config.rtsp_transport}"
            logger.info("opening_rtsp_stream url=%s", self.config.display_address)
            self.cap = cv2.VideoCapture(self.config.rtsp_url, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                self.cap.release()
                raise RuntimeError(
                    f"cannot open RTSP stream at {self.config.display_address}; "
                    f"verify Wi-Fi/hotspot connection to {self.config.host} or pass --camera 0 for local webcam"
                )
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Only start background RTSP buffer-draining thread on real OpenCV VideoCapture objects.
            if hasattr(self.cap, "grab"):
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    self.cap.release()
                    raise RuntimeError(f"RTSP stream opened at {self.config.display_address} but returned no frame")
                self._latest_frame = frame
                self._frame_seq = 1
                self._read_seq = 0
                self._thread = threading.Thread(target=self._drain_rtsp, name="sidp-rtsp-drain", daemon=True)
                self._thread.start()
            return self

        self.cap = cv2.VideoCapture(self.config.index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if not self.cap.isOpened() and os.name == "nt":
            self.cap.release()
            self.cap = cv2.VideoCapture(self.config.index)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"cannot open camera {self.config.index}; check camera index, permissions and other apps")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self

    def _drain_rtsp(self):
        """Continuously pulls RTSP frames so the OpenCV FFmpeg buffer never accumulates lag."""
        last_ok = perf_counter()
        while not self._stopped and self.cap is not None:
            if not self.cap.isOpened():
                break
            ok, frame = self.cap.read()
            now = perf_counter()
            if not ok or frame is None:
                if (now - last_ok) > self.config.frame_timeout_seconds:
                    with self._cond:
                        self._stopped = True
                        self._cond.notify_all()
                    break
                time.sleep(0.01)
                continue
            last_ok = now
            with self._cond:
                self._latest_frame = frame
                self._frame_seq += 1
                self._cond.notify_all()

    @property
    def resolution(self) -> tuple[int, int]:
        if self.cap is None:
            raise RuntimeError("camera is not open")
        if self._latest_frame is not None:
            h, w = self._latest_frame.shape[:2]
            return int(w), int(h)
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        if self.cap is None:
            raise RuntimeError("camera is not open")
        if self._thread is not None:
            with self._cond:
                if self._frame_seq == self._read_seq and not self._stopped:
                    self._cond.wait(timeout=0.05)
                if self._latest_frame is None or (self._stopped and self._frame_seq == 0):
                    raise RuntimeError("camera stopped returning frames")
                frame = self._latest_frame.copy()
                self._read_seq = self._frame_seq
            return frame, perf_counter()

        ok, frame = self.cap.read()
        timestamp = perf_counter()
        if not ok:
            raise RuntimeError("camera stopped returning frames")
        return frame, timestamp

    def __exit__(self, *_):
        self._stopped = True
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
