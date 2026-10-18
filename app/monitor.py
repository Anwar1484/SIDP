"""Single-command browser POC launcher, using the unified Subsystem 1 + 2 architecture."""
import argparse
import logging
import os
import threading
import time
import webbrowser

from app.config import load_config


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="SIDP real-time vision monitor")
    parser.add_argument("--config", help="YAML settings; defaults to SIDP_CONFIG or configs/default.yaml")
    parser.add_argument("--source", choices=("rtsp", "webcam"), help="rtsp (Tapo C210) or webcam")
    parser.add_argument("--host", help="Tapo C210 IP address (default: 172.20.10.4)")
    parser.add_argument("--camera", type=int, help="Local webcam index (switches source to webcam)")
    parser.add_argument("--device", help="auto, cpu, or 0 for the NVIDIA GPU")
    parser.add_argument("--image-size", type=int, choices=(416, 512, 640))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-camera", action="store_true", help="Wait for Start monitoring in the dashboard")
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config or os.getenv("SIDP_CONFIG"))
    if args.source is not None:
        config.camera.source = args.source
    if args.host is not None:
        config.camera.host = args.host
        config.camera.source = "rtsp"
    if args.camera is not None:
        config.camera.index = args.camera
        config.camera.source = "webcam"
    if config.camera.source == "webcam" or config.camera.index != 0:
        config.camera.id = f"webcam_{config.camera.index}"
    if args.device is not None:
        config.pose.device = args.device
    if args.image_size is not None:
        config.pose.image_size = args.image_size
    config.api.camera_enabled = not args.no_camera
    import uvicorn

    from app.api.server import create_app
    url = f"http://127.0.0.1:{args.port}"
    print(f"\nSIDP Vision Monitor · {url}\nPress Ctrl+C in this terminal to stop.\n", flush=True)
    if args.open_browser:
        def open_when_ready():
            import requests
            for _ in range(90):
                try:
                    if requests.get(url + "/health", timeout=1).ok:
                        webbrowser.open(url)
                        return
                except requests.RequestException:
                    pass
                time.sleep(.5)
        threading.Thread(target=open_when_ready, name="sidp-open-browser", daemon=True).start()
    uvicorn.run(create_app(config), host="127.0.0.1", port=args.port, workers=1,
                access_log=False, log_level="info")


if __name__ == "__main__":
    main()
