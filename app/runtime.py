"""Keep model settings local to this checkout, including packaged Windows installs."""
import os
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def prepare_runtime():
    config = ROOT / "outputs" / "runtime" / "ultralytics"
    config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    warnings.filterwarnings("ignore", message=".*half.*deprecated.*", category=Warning)

