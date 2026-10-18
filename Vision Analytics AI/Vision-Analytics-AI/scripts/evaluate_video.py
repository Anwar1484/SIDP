"""Generate an annotated MP4, per-person CSV/JSONL and measured runtime summary."""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.capture.video import VideoCapture  # noqa: E402
from app.fusion.pipeline import VisionPipeline  # noqa: E402
from app.main import config_from_args  # noqa: E402
from app.visualization.overlay import draw_overlay  # noqa: E402

COLUMNS = ("frame", "timestamp", "track_id", "posture", "fall_probability", "temporal_state",
           "bbox_ratio", "torso_angle", "vertical_velocity", "score_source", "score_calibrated")


def evaluate(path: Path, output: Path, pipeline: VisionPipeline, fallback_fps: float | None = None):
    output.mkdir(parents=True, exist_ok=True)
    writer = None
    latencies, tracks, counts, states = [], set(), Counter(), Counter()
    timing_values: dict[str, list[float]] = {}
    start = perf_counter()
    try:
        with VideoCapture(path, fallback_fps) as video, (output / "results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as csvfile, (output / "observations.jsonl").open("w", encoding="utf-8") as jsonfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=COLUMNS)
            csvwriter.writeheader()
            for item in video:
                observation = pipeline.process(item.image, item.timestamp, item.frame_id, item.wall_timestamp)
                if writer is None:
                    h, w = item.image.shape[:2]
                    writer = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             video.fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError("MP4 video writer unavailable; check OpenCV codecs")
                writer.write(draw_overlay(item.image, observation))
                jsonfile.write(observation.model_dump_json() + "\n")
                latencies.append(observation.processing_latency_ms)
                for name, value in observation.timings_ms.items():
                    timing_values.setdefault(name, []).append(value)
                for person in observation.persons:
                    tracks.add(person.track_id)
                    counts[person.posture.value] += 1
                    states[person.state.value] += 1
                    csvwriter.writerow({"frame": item.frame_id, "timestamp": item.timestamp,
                                        "track_id": person.track_id, "posture": person.posture.value,
                                        "fall_probability": person.fall_probability,
                                        "temporal_state": person.state.value,
                                        "bbox_ratio": person.motion.aspect_ratio,
                                        "torso_angle": person.motion.torso_angle,
                                        "vertical_velocity": person.motion.vertical_velocity,
                                        "score_source": person.fall_inference.model_name,
                                        "score_calibrated": person.fall_inference.calibrated})
            summary = {"video": str(path), "frames": len(latencies), "unique_tracks": len(tracks),
                       "mean_latency_ms": float(np.mean(latencies)) if latencies else None,
                       "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
                       "mean_stage_timings_ms": {k: float(np.mean(v)) for k, v in timing_values.items()},
                       "processing_fps": len(latencies) / (perf_counter()-start),
                       "posture_frame_counts": dict(counts), "state_frame_counts": dict(states),
                       "media_fps": video.fps, "timestamp_fallback_used": video.timestamp_fallback_used,
                       "accuracy_metrics": None,
                       "note": "Counts and runtime only; no accuracy claims without ground-truth labels."}
    finally:
        if writer is not None:
            writer.release()
        pipeline.close()
    if not latencies:
        raise ValueError("video contained no decodable frames")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    p.add_argument("--config")
    p.add_argument("--pose-model")
    p.add_argument("--device")
    p.add_argument("--fps", type=float, help="fallback for media without FPS metadata")
    p.add_argument("--no-temporal-model", action="store_true")
    args = p.parse_args()
    pipeline = VisionPipeline(config_from_args(args), camera_id=args.video.name)
    try:
        pipeline.pose_backend.warmup()
        pipeline.load_optional_models()
        print(json.dumps(evaluate(args.video, args.output_dir, pipeline, args.fps), indent=2))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
