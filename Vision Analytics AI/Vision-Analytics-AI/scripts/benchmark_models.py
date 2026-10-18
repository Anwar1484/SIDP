"""Measure warm inference on fixed local images. No accuracy claims from this microbenchmark."""
import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.runtime import ROOT, prepare_runtime

prepare_runtime()
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openvino", action="store_true")
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    image = cv2.imread(str(ROOT / "outputs" / "bus.jpg"))
    if image is None:
        raise FileNotFoundError("outputs/bus.jpg is required for the reproducible benchmark")
    image = cv2.resize(image, (640, 480))
    report = {"platform": platform.platform(), "torch": torch.__version__, "runs": args.runs,
              "input": "Ultralytics bus.jpg resized to 640x480 (not an accuracy test)", "results": []}
    if args.openvino:
        import openvino as ov
        core = ov.Core()
        report["devices"] = {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices}
    report["device"] = args.device
    if args.device != "cpu":
        report["gpu"] = torch.cuda.get_device_name(0)
    for filename in ("yolo26n-pose.pt", "models/yolo26s-pose.pt", "models/human-fall-detection.pt"):
        for size in (416, 640):
            for threads in ((4,) if args.openvino or args.device != "cpu" else (2, 4, 8)):
                torch.set_num_threads(threads)
                model = YOLO(str(ROOT / filename))
                if args.openvino:
                    exported = model.export(format="openvino", imgsz=size, dynamic=False, half=False,
                                            nms=False, device="cpu")
                    model = YOLO(exported)
                for _ in range(3):
                    model.predict(image, imgsz=size, verbose=False, device=args.device)
                times, people, coverage = [], [], []
                for _ in range(args.runs):
                    start = perf_counter()
                    result = model.predict(image, imgsz=size, verbose=False, device=args.device)[0]
                    times.append((perf_counter()-start)*1000)
                    people.append(len(result.boxes))
                    if result.keypoints is not None:
                        coverage.append(float((result.keypoints.conf.cpu().numpy() >= .3).mean()))
                row = {"model": filename, "backend": "openvino" if args.openvino else "torch",
                       "size": size, "threads": threads, "p50_ms": float(np.median(times)),
                       "p95_ms": float(np.percentile(times, 95)), "mean_ms": float(np.mean(times)),
                       "detections": people[-1], "coverage": coverage[-1] if coverage else None,
                       "classes": model.names}
                report["results"].append(row)
                print(json.dumps(row), flush=True)
    suffix = "openvino" if args.openvino else ("cuda" if args.device != "cpu" else "torch")
    destination = ROOT / "outputs" / f"benchmark-{suffix}.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
