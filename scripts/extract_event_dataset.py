"""Extract causal YOLO26/RGB features from URFD, without reading depth features."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from app.config import load_config
from app.fusion.pipeline import VisionPipeline
from app.inference.appearance import AppearanceBackend
from app.inference.yolo26_pose import Yolo26PoseBackend
from app.runtime import ROOT
from app.temporal.event_features import FEATURE_NAMES


def read_labels(directory):
    labels = {}
    for filename in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        with (directory / filename).open() as file:
            for row in csv.reader(file):
                # Other columns are DEPTH measurements: never use them as inputs.
                labels[(row[0], int(row[1]))] = int(row[2])
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/urfd")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation/features.jsonl")
    parser.add_argument("--hz", type=int, default=10, choices=[5, 6, 10, 15, 30])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    cfg.pose.confidence = .1
    cfg.tracker.backend = "bytetrack"
    cfg.evidence.model = None
    backend = Yolo26PoseBackend(cfg.pose)
    backend.warmup()
    appearance = AppearanceBackend(cfg.secondary, backend.device)
    labels = read_labels(args.data)
    sequences = [f"fall-{i:02}" for i in range(1, 31)] + [f"adl-{i:02}" for i in range(1, 41)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parts = args.output.parent / "sequences"
    parts.mkdir(exist_ok=True)
    for index, seq in enumerate(sequences):
        destination = parts / f"{seq}.jsonl"
        if args.resume and destination.is_file():
            continue
        path = args.data / f"{seq}-cam0.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot decode {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        times = {int(r[0]): float(r[1])/1000 for r in csv.reader((args.data/f"{seq}-data.csv").open())}
        pipeline = VisionPipeline(cfg, backend, camera_id=seq, appearance_backend=appearance)
        rows, frame_number = [], 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            if (frame_number-1) % round(fps/args.hz):
                continue
            if frame.shape[:2] != (240, 640):
                raise ValueError(f"Unexpected preview layout {seq}: {frame.shape}")
            rgb = frame[:, 320:].copy()  # fixed source-format split, not a hand-drawn scene ROI
            time = times.get(frame_number, (frame_number-1)/fps)
            observation = pipeline.process(rgb, time, frame_number)
            label = labels.get((seq, frame_number))
            if label is None:
                continue  # Annotation unavailable, never invent ground truth.
            primary = max(observation.persons, key=lambda p: p.geometry.bbox_width*p.geometry.bbox_height,
                          default=None)
            features = pipeline.last_features.get(primary.track_id) if primary else None
            rows.append({
                "sequence": seq, "frame": frame_number, "time": time, "source_label": label,
                "target": int(seq.startswith("fall-") and label >= 0),
                "features": [float(v) if np.isfinite(v) else None for v in features] if features is not None else None,
                "track": primary.track_id if primary else None,
                "posture": primary.posture.value if primary else None,
                "coverage": primary.pose_coverage if primary else 0,
                "rgb_confidence": primary.appearance.fallen_confidence if primary else None,
                "latency_ms": observation.processing_latency_ms,
            })
        capture.release()
        destination.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
        print(f"{index+1}/70 {seq}: {len(rows)} annotated frames, "
              f"{sum(r['features'] is not None for r in rows)} complete windows", flush=True)
    with args.output.open("w", encoding="utf-8") as output:
        for seq in sequences:
            output.write((parts/f"{seq}.jsonl").read_text(encoding="utf-8"))
    metadata = {"config": cfg.model_dump(), "feature_names": FEATURE_NAMES, "sample_hz": args.hz,
                "input": "URFD official camera-0 preview, RGB right half only, 320x240",
                "target": "fall-sequence frame with source label 0 or 1; every ADL frame negative",
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "pose_sha256": hashlib.sha256((ROOT/cfg.pose.model).read_bytes()).hexdigest(),
                "rgb_sha256": hashlib.sha256((ROOT/cfg.secondary.model).read_bytes()).hexdigest()}
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
