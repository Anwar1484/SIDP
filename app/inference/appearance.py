"""RGB posture evidence, associated one-to-one with YOLO26 pose tracks."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.config import SecondaryConfig
from app.posture.geometry import clip_bbox
from app.runtime import ROOT, prepare_runtime
from app.schemas import AppearanceEvidence, BoundingBox, Posture
from app.tracking.person_tracker import Track, iou


@dataclass(frozen=True)
class FallenBox:
    bbox: BoundingBox
    confidence: float
    posture: Posture = Posture.LYING


class AppearanceBackend:
    def __init__(self, config: SecondaryConfig, device: str = "cpu"):
        prepare_runtime()
        from ultralytics import YOLO
        self.config, self.device = config, device
        path = ROOT / config.model
        if not path.is_file():
            raise FileNotFoundError(f"Fall checkpoint missing: {path}. Run scripts/download_models.py.")
        self.model = YOLO(str(path))
        names = {int(k): str(v).lower() for k, v in self.model.names.items()}
        if self.model.task != "detect" or sorted(names.values()) != ["fallen", "sitting", "standing"]:
            raise ValueError(f"Unexpected fall detector classes: {names}")
        self.classes = {k: {"fallen": Posture.LYING, "sitting": Posture.SITTING,
                            "standing": Posture.STANDING}[v] for k, v in names.items()}
        self.latency_ms = 0.0

    def warmup(self, width=640, height=480):
        self.infer(np.zeros((height, width, 3), np.uint8))

    def infer(self, frame: np.ndarray) -> list[FallenBox]:
        start = perf_counter()
        result = self.model.predict(frame, imgsz=self.config.image_size, device=self.device,
                                    conf=self.config.confidence_floor,
                                    verbose=False)[0]
        found = []
        if result.boxes is not None:
            for box, score, label in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                         result.boxes.cls.cpu().numpy()):
                bbox = clip_bbox(box, frame.shape[1], frame.shape[0])
                if bbox is not None and np.isfinite(score):
                    found.append(FallenBox(bbox, float(score), self.classes[int(label)]))
        self.latency_ms = (perf_counter()-start)*1000
        return found


def match_appearance(tracks: list[Track], detections: list[FallenBox], timestamp: float,
                     min_iou: float, latency_ms: float = 0) -> dict[int, AppearanceEvidence]:
    """One-to-one assignment; the same box cannot supply evidence for two people."""
    output = {track.id: AppearanceEvidence(status="no_matched_fallen_box", sampled_at=timestamp,
                                          age_seconds=0, latency_ms=latency_ms) for track in tracks}
    if not tracks or not detections:
        return output
    overlaps = np.array([[iou(t.pose.bbox, d.bbox) for d in detections] for t in tracks])
    rows, cols = linear_sum_assignment(1-overlaps)
    for row, col in zip(rows, cols):
        if overlaps[row, col] < min_iou:
            continue
        # An RGB detection spanning overlapping people has no safe track assignment.
        competitors = np.delete(overlaps[:, col], row)
        if len(competitors) and competitors.max() >= overlaps[row, col] - .1:
            output[tracks[row].id] = output[tracks[row].id].model_copy(update={"status": "ambiguous_match"})
            continue
        output[tracks[row].id] = AppearanceEvidence(
            status="matched", fallen_confidence=(detections[col].confidence
                                                   if detections[col].posture == Posture.LYING else None),
            posture=detections[col].posture, posture_confidence=detections[col].confidence,
            sampled_at=timestamp, age_seconds=0, matched_iou=float(overlaps[row, col]), latency_ms=latency_ms,
        )
    return output
