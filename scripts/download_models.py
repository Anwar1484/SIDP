"""Install pinned POC models and their upstream inference adapters locally."""
import sys
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.model_assets import model_manifest, sha256  # noqa: E402
from app.runtime import ROOT  # noqa: E402


def main():
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
                                                          status_forcelist=[429, 500, 502, 503, 504])))
    installed = 0
    for asset in model_manifest()["assets"]:
        destination = (ROOT / asset["path"]).resolve()
        if not destination.is_relative_to(ROOT.resolve()):
            raise ValueError("Model destination must stay inside the project")
        if destination.is_file() and sha256(destination) == asset["sha256"]:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"Downloading {asset['path']}", flush=True)
        with session.get(asset["url"], stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        if sha256(temporary) != asset["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {asset['path']}; rerun setup to retry")
        temporary.replace(destination)
        installed += 1
    print(f"All POC models verified ({installed} files installed).")


if __name__ == "__main__":
    main()
