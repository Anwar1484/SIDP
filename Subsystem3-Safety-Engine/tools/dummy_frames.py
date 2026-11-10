"""Stands in for the camera: keeps data/frames/<camera_id>.jpg fresh using a dummy image.

Remove once real camera frames are hooked up.
Usage: python tools/dummy_frames.py --cameras cam-04
"""
import argparse
import os
import shutil
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="assets/dummy_frame.jpg")
    parser.add_argument("--frame-dir", default="data/frames")
    parser.add_argument("--cameras", nargs="+", default=[f"cam-0{i}" for i in range(1, 7)])
    parser.add_argument("--interval-s", type=float, default=1.0)
    args = parser.parse_args()

    frame_dir = Path(args.frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing {args.image} to {frame_dir} for {args.cameras}. Ctrl+C to stop.")
    try:
        while True:
            for camera_id in args.cameras:
                temp = frame_dir / f"{camera_id}.tmp"
                shutil.copyfile(args.image, temp)
                try:
                    os.replace(temp, frame_dir / f"{camera_id}.jpg")  # atomic, as a real feed should be
                except PermissionError:
                    pass  # Windows: S3 is reading the frame right now, so try again next round
            time.sleep(args.interval_s)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
