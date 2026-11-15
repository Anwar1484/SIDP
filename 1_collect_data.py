import csv
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# SETTINGS: Choose Camera Source
# ==========================================
# Set USE_TAPO = True for your Tapo C210 camera, or False to test with your laptop webcam
USE_TAPO = True
TAPO_URL = "rtsp://SIDPProject:SIDPUTP2026@172.20.10.4:554/stream2"
CAMERA_SOURCE = TAPO_URL if USE_TAPO else 0

CSV_FILE = "posture_dataset.csv"
POSE_MODEL = "yolo26n-pose.pt"  # If not found, change to "yolov8n-pose.pt"

CLASSES = {
    ord("1"): "standing",
    ord("2"): "sitting",
    ord("3"): "bending",
    ord("4"): "lying_down",
}


def is_whole_body_visible(kpts_xyn, kpts_conf=None, conf_thresh=0.45):
    """
    Strict Whole-Body Visibility Gate for COCO 17 keypoints.
    Requires:
      - Both shoulders (5, 6) and both hips (11, 12) within (0.02 < x < 0.98, 0.02 < y < 0.98)
      - At least one knee (13 or 14) AND at least one ankle (15 or 16) within frame boundaries
      - Confidence >= conf_thresh (0.45) for those required keypoints when kpts_conf is provided
    """
    if kpts_xyn is None or len(kpts_xyn) < 17:
        return False

    def _valid_pt(idx):
        x, y = float(kpts_xyn[idx][0]), float(kpts_xyn[idx][1])
        if not (0.02 < x < 0.98 and 0.02 < y < 0.98):
            return False
        if kpts_conf is not None and len(kpts_conf) > idx:
            if float(kpts_conf[idx]) < conf_thresh:
                return False
        return True

    shoulders_ok = _valid_pt(5) and _valid_pt(6)
    hips_ok = _valid_pt(11) and _valid_pt(12)
    knees_ok = _valid_pt(13) or _valid_pt(14)
    ankles_ok = _valid_pt(15) or _valid_pt(16)
    return bool(shoulders_ok and hips_ok and knees_ok and ankles_ok)

# Create CSV file with column headers if it doesn't exist yet
if not os.path.exists(CSV_FILE):
    # 34 position numbers + 34 movement speed numbers = 68 features
    headers = (
        ["label"]
        + [f"pos_{i}" for i in range(34)]
        + [f"vel_{i}" for i in range(34)]
    )
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(headers)


def count_samples():
    counts = {name: 0 for name in CLASSES.values()}
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row and row[0] in counts:
                    counts[row[0]] += 1
    return counts


print("[1/2] Loading YOLO Pose Model...")
model = YOLO(POSE_MODEL)

print(f"[2/2] Opening Camera ({CAMERA_SOURCE})...")
cap = cv2.VideoCapture(CAMERA_SOURCE)

current_label = "standing"
is_recording = False
prev_norm_kpts = None
last_save_time = 0
counts = count_samples()

print("\n=======================================================")
print("   STEP 1: AI POSTURE DATA COLLECTOR (4 CLASSES)")
print("=======================================================")
print(
    " Press 1: standing | 2: sitting | 3: bending | 4: lying_down"
)
print(" Press R: Turn AUTO-RECORD ON / OFF (saves 4 samples per second)")
print(" Press Q: Quit and Save")
print("=======================================================\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.01)
        continue

    results = model(frame, verbose=False)
    annotated = results[0].plot()
    features = None
    whole_body_ok = False

    # Check if a person's skeleton is detected
    if (
        results[0].keypoints is not None
        and len(results[0].keypoints.xyn) > 0
    ):
        kpts = results[0].keypoints.xyn[0].cpu().numpy()  # (17, 2)
        kpts_conf = (
            results[0].keypoints.conf[0].cpu().numpy()
            if results[0].keypoints.conf is not None
            and len(results[0].keypoints.conf) > 0
            else None
        )

        # Make sure the whole body (shoulders, hips, knee, ankle) is visible
        whole_body_ok = is_whole_body_visible(kpts, kpts_conf, conf_thresh=0.45)
        if whole_body_ok:
            # 1. Center skeleton around the middle of the hips
            mid_hip = (kpts[11] + kpts[12]) / 2.0
            centered = kpts - mid_hip

            # 2. Scale by body size so distance from camera doesn't matter
            mid_shoulder = (kpts[5] + kpts[6]) / 2.0
            body_size = np.linalg.norm(mid_shoulder - mid_hip) + 1e-6
            norm_kpts = (centered / body_size).flatten()  # 34 numbers

            # 3. Calculate movement speed from previous frame
            if prev_norm_kpts is None:
                velocity = np.zeros(34)
            else:
                velocity = norm_kpts - prev_norm_kpts
            prev_norm_kpts = norm_kpts.copy()

            features = np.concatenate([norm_kpts, velocity]).tolist()
        else:
            prev_norm_kpts = None

    # Save data if recording is ON (every 0.25 seconds = 4 samples/sec)
    now = time.time()
    if is_recording and features is not None and (now - last_save_time >= 0.25):
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([current_label] + features)
        counts[current_label] += 1
        last_save_time = now

    # Draw On-Screen Instructions
    status_text = (
        "RECORDING ON (Press R to Stop)"
        if is_recording
        else "PAUSED (Press R to Record)"
    )
    status_color = (0, 0, 255) if is_recording else (0, 255, 0)

    cv2.rectangle(annotated, (10, 10), (620, 135), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        f"Selected Class: {current_label.upper()} (Keys 1-4)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        annotated,
        f"Status: {status_text}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
    )

    summary = " | ".join([f"{k[:4]}:{v}" for k, v in counts.items()])
    cv2.putText(
        annotated,
        f"Saved Samples -> {summary}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )

    cv2.imshow("Step 1: Collect Posture Data", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key in CLASSES:
        current_label = CLASSES[key]
        is_recording = False  # Automatically pause when switching posture
        print(f"Switched to [{current_label}]. Press R when ready to record.")
    elif key == ord("r") or key == ord("R"):
        is_recording = not is_recording
        print(f"Recording {'STARTED' if is_recording else 'PAUSED'} for {current_label}")

cap.release()
cv2.destroyAllWindows()