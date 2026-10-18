"""Extract per-video/per-person raw and normalized JSONL sequences; no labels invented."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.capture.video import VideoCapture  # noqa: E402
from app.config import load_config  # noqa: E402
from app.inference.yolo26_pose import Yolo26PoseBackend  # noqa: E402
from app.posture.normalization import NORMALIZATION_VERSION, normalize_pose  # noqa: E402
from app.tracking.person_tracker import create_tracker  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("videos", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("data/sequences"))
    p.add_argument("--config")
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true", help="replace existing extracted track files")
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg.pose.device = args.device
    backend = Yolo26PoseBackend(cfg.pose)
    videos = sorted(x for x in args.videos.rglob("*") if x.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"))
    if not videos:
        p.error("no supported videos found")
    for path in videos:
        tracker = create_tracker(cfg.tracker)
        # Preserve relative video path, including suffix, to prevent same-stem collisions.
        relative = path.relative_to(args.videos)
        target = args.output_dir / relative
        if target.exists() and list(target.glob("track_*.jsonl")):
            if not args.overwrite:
                raise FileExistsError(f"{target} contains extracted tracks; choose a new output or --overwrite")
            # Remove only known extractor outputs, after resolving the intended directory.
            root = args.output_dir.resolve()
            resolved = target.resolve()
            if root not in resolved.parents:
                raise ValueError("extraction target escaped output directory")
            for existing in resolved.glob("track_*.jsonl"):
                existing.unlink()
        target.mkdir(parents=True, exist_ok=True)
        with VideoCapture(path) as video:
            for frame in video:
                h, w = frame.image.shape[:2]
                for track in tracker.update(backend.infer(frame.image), frame.timestamp, (w, h)):
                    norm = normalize_pose(track.pose)
                    row = {"video_id": relative.as_posix(), "track_id": track.id, "frame": frame.frame_id,
                           "timestamp": frame.timestamp, "pose_family": "YOLO26", "pose_model": cfg.pose.model,
                           "normalization": NORMALIZATION_VERSION, "keypoint_threshold": cfg.pose.keypoint_confidence,
                           "frame_size": [w, h], "bbox": track.pose.bbox.as_list(),
                           "keypoints": [k.model_dump() for k in track.pose.keypoints],
                           "normalized_xy": norm.xy.tolist(), "mask": norm.mask.astype(int).tolist(),
                           "normalization_valid": norm.valid, "scale_pixels": norm.scale_pixels}
                    with (target / f"track_{track.id}.jsonl").open("a", encoding="utf-8") as output:
                        output.write(json.dumps(row) + "\n")
        print(f"Extracted {relative}")


if __name__ == "__main__":
    main()
