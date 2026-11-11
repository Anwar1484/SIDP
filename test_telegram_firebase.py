#!/usr/bin/env python3
"""
Live connectivity check for Subsystem 3 (Telegram) and Subsystem 4 (Firebase RTDB).

Run:  .\\.venv\\Scripts\\python.exe test_telegram_firebase.py

1. Creates captures/YYYY-MM-DD/unsafe/test_snapshot.jpg
2. Sends it (photo + caption) to the Telegram group in TELEGRAM_CHAT_ID
3. POSTs a test record to <FIREBASE_DATABASE_URL>/incidents.json and /ACTIVITY_LOGS.json
4. Prints the HTTP status of each call (200 = OK)

Credentials are read from .env (never hardcoded here).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from subsystem1_2_3_4_master import TelemetryDispatcher  # noqa: E402


def main() -> int:
    now = datetime.now()
    out_dir = ROOT / "captures" / now.strftime("%Y-%m-%d") / "unsafe"
    os.makedirs(out_dir, exist_ok=True)
    img_path = out_dir / "test_snapshot.jpg"

    canvas = np.full((360, 640, 3), 30, dtype=np.uint8)
    cv2.putText(canvas, "SIDP TEST SNAPSHOT - SUBSYSTEM 3 & 4 ONLINE", (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, now.strftime("%Y-%m-%d %H:%M:%S"), (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.imwrite(str(img_path), canvas)
    print(f"[1/3] Test snapshot written -> {img_path.relative_to(ROOT).as_posix()}")

    dispatcher = TelemetryDispatcher(
        resident_lookup=lambda rid: {"full_name": "Test Resident", "room_number": "000"},
    )
    meta = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "resident_id": 1,
        "tier": "unsafe",
        "posture": "Bending",
        "torso_angle": 64.5,
        "snapshot_path": str(img_path),
    }
    result = dispatcher.dispatch_now(meta)
    dispatcher.shutdown(drain_timeout=1.0)

    tg = result.get("telegram")
    fb = result.get("firebase", {})
    print(f"[2/3] Telegram sendPhoto -> HTTP {tg}  {'OK' if tg == 200 else 'FAILED'}")
    print(f"[3/3] Firebase /incidents.json      -> HTTP {fb.get('incidents')}  {'OK' if fb.get('incidents') == 200 else 'FAILED'}")
    print(f"      Firebase /ACTIVITY_LOGS.json  -> HTTP {fb.get('ACTIVITY_LOGS')}  {'OK' if fb.get('ACTIVITY_LOGS') == 200 else 'FAILED'}")

    ok = tg == 200 and fb.get("incidents") == 200
    print("\nRESULT:", "ALL ENDPOINTS ONLINE" if ok else "ONE OR MORE ENDPOINTS FAILED (see warnings above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
