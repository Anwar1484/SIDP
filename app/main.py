"""Unified SIDP desktop launcher: python scripts/run_demo.py (RTSP + ONVIF PTZ + Dataset + Vision AI)."""
import argparse
import json
import logging
import platform
from contextlib import nullcontext
from pathlib import Path

import cv2

from app.capture.dataset import DatasetCollector
from app.capture.ptz import PTZController
from app.capture.webcam import WebcamCapture
from app.config import load_config
from app.fusion.pipeline import VisionPipeline
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)
WINDOW_TITLE = "SIDP - Unified Subsystem 1 & 2 Monitor"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--posture-config")
    p.add_argument("--source", choices=("rtsp", "webcam"), help="override camera source (rtsp or webcam)")
    p.add_argument("--host", help="Tapo C210 IP address (default from configs/default.yaml: 172.20.10.4)")
    p.add_argument("--camera", type=int, help="local webcam index (automatically switches source to webcam)")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--pose-model")
    p.add_argument("--device")
    p.add_argument("--confidence", type=float)
    p.add_argument("--keypoint-confidence", type=float)
    p.add_argument("--no-ptz", action="store_true", help="disable ONVIF PTZ motor connection")
    p.add_argument("--no-temporal-model", action="store_true", help="disable the optional retrained Keras manifest")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-frames", type=int, help="bounded smoke test; default runs until Q/Ctrl+C")
    p.add_argument("--jsonl", type=Path, help="optional observation output (no raw frames)")
    return p


def config_from_args(args):
    cfg = load_config(args.config, getattr(args, "posture_config", None))
    if getattr(args, "source", None) is not None:
        cfg.camera.source = args.source
    if getattr(args, "host", None) is not None:
        cfg.camera.host = args.host
        cfg.camera.source = "rtsp"
    if getattr(args, "camera", None) is not None:
        cfg.camera.index = args.camera
        cfg.camera.source = "webcam"
        cfg.camera.id = f"webcam_{cfg.camera.index}"
    elif cfg.camera.source == "webcam":
        cfg.camera.id = f"webcam_{cfg.camera.index}"
    for field in ("width", "height"):
        value = getattr(args, field, None)
        if value is not None:
            setattr(cfg.camera, field, value)
    for cli, field in (("pose_model", "model"), ("device", "device"), ("confidence", "confidence"),
                       ("keypoint_confidence", "keypoint_confidence")):
        value = getattr(args, cli, None)
        if value is not None:
            setattr(cfg.pose, field, value)
    if getattr(args, "no_ptz", False):
        cfg.ptz.enabled = False
    if getattr(args, "no_temporal_model", False):
        cfg.temporal.model_manifest = None
    return cfg


def startup_info(pipeline: VisionPipeline, resolution: tuple[int, int]):
    import torch
    import ultralytics
    info = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "device": getattr(pipeline.pose_backend, "device_name", pipeline.pose_backend.device),
        "pose_model": pipeline.config.pose.model,
        "camera_source": pipeline.config.camera.display_address,
        "resolution": resolution,
        "relateanything": pipeline.scene_backend is not None,
        "temporal_backend": (
            "retrained" if pipeline.fall_backend else "pretrained tree/fusion"
            if pipeline.risk_backend else "event classifier"
            if pipeline.event_classifier else "unavailable"
        ),
    }
    logger.info("startup %s", json.dumps(info))


def main(argv: list[str] | None = None):
    args = parser().parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")
    configure_logging(args.debug)
    cfg = config_from_args(args)
    pipeline = VisionPipeline(cfg)
    ptz = PTZController(cfg.camera, cfg.ptz) if (not args.headless and args.max_frames is None) else None
    collector = DatasetCollector(cfg.dataset)
    ai_enabled = True
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
                clean_frame = frame.copy()
                collector.maybe_record(clean_frame)
                observation = pipeline.process(frame, timestamp, frame_id)
                if output:
                    output.write(observation.model_dump_json() + "\n")
                frame_id += 1
                if not args.headless:
                    rendered = draw_overlay(
                        frame,
                        observation,
                        ptz_mode=ptz.mode_label if ptz else None,
                        dataset_status=collector.status(),
                        ai_enabled=ai_enabled,
                    )
                    cv2.imshow(WINDOW_TITLE, rendered)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q")):
                        break
                    elif key in (ord("p"), ord("P")):
                        ai_enabled = not ai_enabled
                    elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                        collector.select_index(key - ord("1"))
                    elif key in (ord("c"), ord("C")):
                        saved = collector.snap(clean_frame)
                        logger.info("dataset_snap_saved path=%s", saved)
                    elif key in (ord("r"), ord("R")):
                        rec = collector.toggle_recording()
                        logger.info("dataset_auto_record enabled=%s", rec)
                    elif ptz is not None:
                        if key in (ord("z"), ord("Z")):
                            ptz.toggle_patrol()
                        elif key in (ord("a"), ord("A"), 81):
                            ptz.pan_left()
                        elif key in (ord("d"), ord("D"), 83):
                            ptz.pan_right()
                        elif key in (ord("w"), ord("W"), 82):
                            ptz.tilt_up()
                        elif key in (ord("s"), ord("S"), 84):
                            ptz.tilt_down()
                    if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                        break
            logger.info("demo_complete frames=%d", frame_id)
    except KeyboardInterrupt:
        logger.info("demo_interrupted")
    finally:
        try:
            if ptz is not None:
                ptz.close()
            pipeline.close()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
