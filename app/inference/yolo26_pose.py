"""Official YOLO26-Pose API. Load once, preserve original-frame pixel coordinates."""
import logging
import re
from pathlib import Path

import numpy as np

from app.config import PoseConfig
from app.posture.geometry import clip_bbox
from app.runtime import ROOT, prepare_runtime
from app.schemas import Keypoint, PersonPose

logger = logging.getLogger(__name__)


def select_device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if requested in ("cuda", "cuda:0"):
        return "0"
    return requested


class Yolo26PoseBackend:
    def __init__(self, config: PoseConfig):
        prepare_runtime()
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(config.cpu_threads)
        self.config = config
        if not re.fullmatch(r"yolo26[nsmlx]-pose\.pt", Path(config.model).name):
            raise ValueError("pose backbone must be an official yolo26[n/s/m/l/x]-pose.pt checkpoint")
        self.device = select_device(config.device)
        self.device_name = torch.cuda.get_device_name(int(self.device)) if self.device.isdigit() else "CPU"
        checkpoint = Path(config.model)
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        self.model = YOLO(str(checkpoint))
        if self.model.task != "pose":
            raise ValueError("checkpoint is not a pose model")
        # Family validation prevents an older checkpoint simply renamed to yolo26.
        architecture = self.model.model.yaml.get("yaml_file", "")
        if "yolo26" not in str(architecture):
            raise ValueError(f"checkpoint architecture is not YOLO26: {architecture}")

    def warmup(self, width: int = 640, height: int = 480):
        """Perform one blank inference before live capture, excluding cold-start delay from tracks."""
        self.infer(np.zeros((height, width, 3), dtype=np.uint8))

    def infer(self, frame_bgr: np.ndarray) -> list[PersonPose]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
            raise ValueError("expected uint8 HxWx3 BGR image")
        height, width = frame_bgr.shape[:2]
        options = {"quantize": 16} if self.config.half and self.device != "cpu" else {}
        result = self.model.predict(source=frame_bgr, conf=self.config.confidence,
                                    imgsz=self.config.image_size, device=self.device,
                                    verbose=False, **options)[0]
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy()
        if keypoints.shape != (len(boxes), 17, 3):
            raise RuntimeError(f"expected (N,17,3) COCO keypoints; got {keypoints.shape}")
        persons = []
        for bbox, score, joints in zip(boxes, scores, keypoints):
            box = clip_bbox(bbox, width, height)
            if box is None or not np.isfinite(score):
                logger.warning("discarded_invalid_bbox")
                continue
            typed = []
            for index, (x, y, confidence) in enumerate(joints):
                finite = bool(np.isfinite([x, y, confidence]).all())
                visible = finite and 0 <= x <= width and 0 <= y <= height
                typed.append(Keypoint(index=index, x=float(x) if finite else 0,
                                      y=float(y) if finite else 0,
                                      confidence=float(np.clip(confidence, 0, 1)) if finite else 0,
                                      valid=visible and confidence >= self.config.keypoint_confidence))
            persons.append(PersonPose(bbox=box, detection_confidence=float(score), keypoints=typed))
        return persons
