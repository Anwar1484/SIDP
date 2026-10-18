"""Build quality-gated 30-frame posture or 60-frame fall windows with explicit human labels."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import load_config  # noqa: E402
from app.posture.normalization import NORMALIZATION_VERSION  # noqa: E402
from app.temporal.dataset import prepare_windows  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sequences", type=Path)
    p.add_argument("--labels", type=Path, help="JSON list of per-track time-interval annotations")
    p.add_argument("--task", choices=("posture", "fall"), default="fall")
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--config")
    p.add_argument("--output", type=Path, default=Path("data/fall_windows.npz"))
    args = p.parse_args()
    cfg = load_config(args.config).temporal
    frames = cfg.posture_window if args.task == "posture" else cfg.fall_window
    dataset = prepare_windows(args.sequences, args.labels, cfg, frames, args.stride, args.task)
    if not len(dataset["X"]):
        p.error("no continuous windows passed quality gates")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **dataset)
    metadata = {"normalization": NORMALIZATION_VERSION, "pose_family": "YOLO26", "features": 51,
                "frames": frames, "sample_hz": cfg.sample_hz, "task": args.task,
                "class_names": ["STANDING", "SITTING", "LYING"] if args.task == "posture" else ["NON_FALL", "FALL"],
                "unlabeled_value": -1, "windows": len(dataset["X"]),
                "labeled_windows": int((dataset["y"] >= 0).sum())}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
