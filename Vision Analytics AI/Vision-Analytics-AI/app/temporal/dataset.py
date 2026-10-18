"""Training windows use the same temporal resampling and quality gates as inference."""
import json
from pathlib import Path

import numpy as np

from app.config import TemporalConfig
from app.posture.geometry import extract_geometry
from app.posture.normalization import NORMALIZATION_VERSION, normalize_pose
from app.schemas import BoundingBox, Keypoint, PersonPose, Posture, PostureResult
from app.temporal.buffer import PoseSample, TemporalBuffer


def prepare_windows(sequences: Path, labels: Path | None, config: TemporalConfig,
                    frames: int, stride: int, task: str) -> dict[str, np.ndarray]:
    annotations = json.loads(labels.read_text(encoding="utf-8")) if labels else []
    windows, targets, groups, identities, bounds = [], [], [], [], []
    if stride <= 0:
        raise ValueError("stride must be positive")
    posture_labels = {"STANDING": 0, "SITTING": 1, "LYING": 2}
    for file in sorted(sequences.rglob("track_*.jsonl")):
        buffer = TemporalBuffer(config)
        for index, line in enumerate(file.read_text(encoding="utf-8").splitlines()):
            row = json.loads(line)
            if row.get("normalization") != NORMALIZATION_VERSION or row.get("pose_family") != "YOLO26":
                raise ValueError(f"incompatible feature contract: {file}")
            pose = PersonPose(bbox=BoundingBox(**dict(zip(("x1", "y1", "x2", "y2"), row["bbox"]))),
                              detection_confidence=1, keypoints=[Keypoint(**k) for k in row["keypoints"]])
            w, h = row["frame_size"]
            geometry = extract_geometry(pose, h)
            # Recompute from raw poses so stale or edited normalized fields cannot alter the contract.
            sample = PoseSample(row["timestamp"], pose, normalize_pose(pose),
                                PostureResult(label=Posture.UNKNOWN, confidence=None,
                                              reason="unlabeled training sample", features=geometry), geometry, (w, h))
            buffer.append(sample)
            if index % stride:
                continue
            window, _ = buffer.window(frames)
            if window is None:
                continue
            end = row["timestamp"]
            start = end - (frames-1) / config.sample_hz
            matches = [a for a in annotations if a["video_id"] == row["video_id"]
                       and int(a["track_id"]) == row["track_id"] and a["start"] <= start
                       and a["end"] >= end and task in a]
            if len(matches) > 1:
                raise ValueError("overlapping labels for the same track/window")
            target = -1
            if matches:
                label = matches[0][task]
                if task == "fall":
                    if label not in (0, 1) or isinstance(label, str):
                        raise ValueError("fall labels must be 0 or 1")
                    target = int(label)
                else:
                    if label not in posture_labels:
                        raise ValueError("posture labels must be STANDING, SITTING or LYING")
                    target = posture_labels[label]
            windows.append(window)
            targets.append(target)
            groups.append(matches[0].get("group_id", row["video_id"]) if matches else row["video_id"])
            identities.append(row["track_id"])
            bounds.append((start, end))
    return {"X": np.array(windows, dtype=np.float32).reshape(-1, frames, 51),
            "y": np.array(targets, dtype=np.int64), "groups": np.array(groups, dtype=str),
            "track_ids": np.array(identities, dtype=np.int64),
            "bounds": np.array(bounds, dtype=np.float64).reshape(-1, 2)}
