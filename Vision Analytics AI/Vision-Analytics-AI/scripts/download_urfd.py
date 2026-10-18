"""Download the university's published RGB/depth preview videos and annotations.

Academic use only: URFD is CC BY-NC-SA 4.0. Inference uses ONLY the RGB right
half of each preview (320x240), never the depth half or supplied depth features.
Frame indices and camera-0 synchronization files align these previews to labels.
Use --sequences fall-01 adl-01 for a small reproducible download.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import requests

BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data/"


def download(name: str, destination: Path):
    path = destination / name
    if not path.exists():
        with requests.get(BASE + name, stream=True, timeout=(15, 40)) as response:
            response.raise_for_status()
            temporary = path.with_suffix(path.suffix + ".part")
            with temporary.open("wb") as file:
                for chunk in response.iter_content(65536):
                    file.write(chunk)
            temporary.replace(path)
    return {"file": name, "url": BASE + name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/urfd"))
    parser.add_argument("--sequences", nargs="*")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sequences = args.sequences or ([f"fall-{i:02}" for i in range(1, 31)]
                                   + [f"adl-{i:02}" for i in range(1, 41)])
    names = ["urfall-cam0-falls.csv", "urfall-cam0-adls.csv"]
    names += [name + suffix for name in sequences for suffix in ("-cam0.mp4", "-data.csv")]
    manifest, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download, name, args.output): name for name in names}
        for future in concurrent.futures.as_completed(futures):
            try:
                item = future.result()
                manifest.append(item)
                print(f"{len(manifest)}/{len(names)} {item['file']} {item['bytes']} bytes", flush=True)
            except Exception as exc:
                failures.append({"file": futures[future], "error": str(exc)})
                print(f"Failed: {futures[future]}: {exc}", flush=True)
    (args.output / "downloads.json").write_text(json.dumps({
        "source": "https://fenix.ur.edu.pl/~mkepski/ds/uf.html", "license": "CC-BY-NC-SA-4.0",
        "files": sorted(manifest, key=lambda x: x['file']), "failures": failures,
    }, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit("Some downloads failed; rerun to resume. Do not silently omit sequences.")


if __name__ == "__main__":
    main()
