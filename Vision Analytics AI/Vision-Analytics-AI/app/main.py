"""One-command webcam demo: python -m app.main (Q to exit)."""
import argparse
import json
import logging
import platform
from contextlib import nullcontext
from pathlib import Path

import cv2

from app.capture.webcam import WebcamCapture
from app.config import load_config
from app.fusion.pipeline import VisionPipeline
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--posture-config")
    p.add_argument("--camera", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--pose-model")
    p.add_argument("--device")
    p.add_argument("--confidence", type=float)
    p.add_argument("--keypoint-confidence", type=float)
    p.add_argument("--no-temporal-model", action="store_true", help="disable the optional retrained Keras manifest")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-frames", type=int, help="bounded smoke test; default runs until Q/Ctrl+C")
    p.add_argument("--jsonl", type=Path, help="optional observation output (no raw frames)")
    return p


def config_from_args(args):
    cfg = load_config(args.config, getattr(args, "posture_config", None))
    for field in ("camera", "width", "height"):
        value = getattr(args, field, None)
        if value is not None:
            setattr(cfg.camera, "index" if field == "camera" else field, value)
    for cli, field in (("pose_model", "model"), ("device", "device"), ("confidence", "confidence"),
                       ("keypoint_confidence", "keypoint_confidence")):
        value = getattr(args, cli, None)
        if value is not None:
            setattr(cfg.pose, field, value)
    if getattr(args, "no_temporal_model", False):
        cfg.temporal.model_manifest = None
    cfg.camera.id = f"webcam_{cfg.camera.index}"
    return cfg


def startup_info(pipeline: VisionPipeline, resolution: tuple[int, int]):
    import torch
    import ultralytics
    info = {"python": platform.python_version(), "pytorch": torch.__version__,
            "ultralytics": ultralytics.__version__, "device": pipeline.pose_backend.device,
            "pose_model": pipeline.config.pose.model, "webcam_resolution": resolution,
            "temporal_backend": ("retrained" if pipeline.fall_backend else "pretrained tree/fusion"
                                 if pipeline.risk_backend else "event classifier"
                                 if pipeline.event_classifier else "unavailable")}
    logger.info("startup %s", json.dumps(info))


def main(argv: list[str] | None = None):
    args = parser().parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")
    configure_logging(args.debug)
    cfg = config_from_args(args)
    pipeline = VisionPipeline(cfg)
    try:
        logger.info("warming_up_pose_model")
        pipeline.pose_backend.warmup(cfg.camera.width, cfg.camera.height)
        pipeline.load_optional_models()
        if args.jsonl:
            args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        from app.visualization.overlay import draw_overlay
        with WebcamCapture(cfg.camera) as camera, (
            args.jsonl.open("w", encoding="utf-8") if args.jsonl else nullcontext()
        ) as output:
            startup_info(pipeline, camera.resolution)
            frame_id = 0
            while args.max_frames is None or frame_id < args.max_frames:
                frame, timestamp = camera.read()
                observation = pipeline.process(frame, timestamp, frame_id)
                if output:
                    output.write(observation.model_dump_json() + "\n")
                frame_id += 1
                if not args.headless:
                    cv2.imshow("SIDP - YOLO26 Vision Posture POC", draw_overlay(frame, observation))
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                        break
                    if cv2.getWindowProperty("SIDP - YOLO26 Vision Posture POC", cv2.WND_PROP_VISIBLE) < 1:
                        break
            logger.info("demo_complete frames=%d", frame_id)
    except KeyboardInterrupt:
        logger.info("demo_interrupted")
    finally:
        try:
            pipeline.close()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
