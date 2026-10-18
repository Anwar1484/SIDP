"""Pinned local artifacts. Model setup never executes a fetched install script."""
import hashlib
import json
from pathlib import Path

from app.runtime import ROOT


def model_manifest() -> dict:
    return json.loads((ROOT / "configs" / "poc_models.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(prefix: str, directory: Path | None = None) -> None:
    root = directory if directory is not None else ROOT / prefix
    for asset in model_manifest()["assets"]:
        if not asset["path"].startswith(prefix):
            continue
        path = root / Path(asset["path"]).relative_to(prefix)
        if not path.is_file() or sha256(path) != asset["sha256"]:
            raise RuntimeError(f"Missing or incomplete model artifact: {asset['path']}. "
                               "Run .venv\\Scripts\\python.exe scripts\\download_models.py")
