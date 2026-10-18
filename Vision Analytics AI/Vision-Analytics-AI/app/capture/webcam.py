"""Context-managed OpenCV webcam capture with monotonic acquisition timestamps."""
import os
from time import perf_counter

import cv2

from app.config import CameraConfig


class WebcamCapture:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: cv2.VideoCapture | None = None

    def __enter__(self):
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

    @property
    def resolution(self) -> tuple[int, int]:
        if self.cap is None:
            raise RuntimeError("camera is not open")
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        if self.cap is None:
            raise RuntimeError("camera is not open")
        ok, frame = self.cap.read()
        timestamp = perf_counter()
        if not ok:
            raise RuntimeError("camera stopped returning frames")
        return frame, timestamp

    def __exit__(self, *_):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
