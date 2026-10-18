"""Verify real YOLO26 boxes and COCO17 keypoints using a local image."""
import argparse
import sys
from pathlib import Path
from time import perf_counter

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import load_config  # noqa: E402
from app.fusion.pipeline import VisionPipeline  # noqa: E402
from app.visualization.overlay import draw_overlay  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path)
    p.add_argument("--output", type=Path, default=Path("outputs/pose-image.jpg"))
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    frame = cv2.imread(str(args.image))
    if frame is None:
        p.error("image could not be read")
    cfg = load_config()
    cfg.pose.device = args.device
    pipeline = VisionPipeline(cfg, camera_id="image")
    observation = pipeline.process(frame, perf_counter(), 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), draw_overlay(frame, observation)):
        raise RuntimeError("failed to write annotated image")
    args.output.with_suffix(".json").write_text(observation.model_dump_json(indent=2), encoding="utf-8")
    print(f"Detected {len(observation.persons)} people; 17 ordered keypoints per person. Output: {args.output}")


if __name__ == "__main__":
    main()
