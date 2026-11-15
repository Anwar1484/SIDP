from collections import deque
import cv2
import joblib
import numpy as np
from ultralytics import YOLO

USE_TAPO = True
TAPO_URL = "rtsp://SIDPProject:SIDPUTP2026@172.20.10.4:554/stream2"
CAMERA_SOURCE = TAPO_URL if USE_TAPO else 0

POSE_MODEL = "yolo26n-pose.pt"
AI_MODEL_FILE = "trained_posture_model.pkl"


def is_whole_body_visible(kpts_xyn, kpts_conf=None, conf_thresh=0.45):
    """
    Strict Whole-Body Visibility Gate for COCO 17 keypoints:
      - Upper Body: Nose (0), Left Shoulder (5), Right Shoulder (6)
      - Core / Torso: Left Hip (11), Right Hip (12)
      - Lower Body / Legs: Left Knee (13), Right Knee (14), Left Ankle (15), Right Ankle (16)

    Validation Rules:
      1. Both shoulders (5, 6) and both hips (11, 12) must have valid normalized
         coordinates (0.02 < x < 0.98) and (0.02 < y < 0.98).
      2. At least one knee (13 or 14) AND at least one ankle (15 or 16) must be
         detected inside the frame boundaries (0.02 < x < 0.98, 0.02 < y < 0.98).
      3. If keypoint confidence scores (kpts_conf) are available, require
         confidence >= conf_thresh (0.45) for shoulders, hips, knees, and ankles.
    """
    if kpts_xyn is None or len(kpts_xyn) < 17:
        return False

    def _is_valid_point(idx: int) -> bool:
        x = float(kpts_xyn[idx][0])
        y = float(kpts_xyn[idx][1])
        if not (0.02 < x < 0.98 and 0.02 < y < 0.98):
            return False
        if kpts_conf is not None and len(kpts_conf) > idx:
            if float(kpts_conf[idx]) < conf_thresh:
                return False
        return True

    shoulders_ok = _is_valid_point(5) and _is_valid_point(6)
    hips_ok = _is_valid_point(11) and _is_valid_point(12)
    knees_ok = _is_valid_point(13) or _is_valid_point(14)
    ankles_ok = _is_valid_point(15) or _is_valid_point(16)
    return bool(shoulders_ok and hips_ok and knees_ok and ankles_ok)


if __name__ == "__main__":
    print("Loading YOLO and your Retrained 4-Class AI Model...")
    pose_model = YOLO(POSE_MODEL)
    classifier = joblib.load(AI_MODEL_FILE)

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    prev_norm_kpts = None
    # Smooth predictions over the last 5 frames so text doesn't flicker
    prediction_history = deque(maxlen=5)

    print("Live AI Test started! Press 'Q' on the video window to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        results = pose_model(frame, verbose=False)
        annotated = results[0].plot()

        posture_text = "No Person Detected"
        hud_color = (200, 200, 200)
        whole_body_ok = True
        person_detected = False

        if (
            results[0].keypoints is not None
            and len(results[0].keypoints.xyn) > 0
        ):
            person_detected = True
            kpts = results[0].keypoints.xyn[0].cpu().numpy()
            kpts_conf = (
                results[0].keypoints.conf[0].cpu().numpy()
                if results[0].keypoints.conf is not None
                and len(results[0].keypoints.conf) > 0
                else None
            )

            whole_body_ok = is_whole_body_visible(
                kpts, kpts_conf=kpts_conf, conf_thresh=0.45
            )

            if whole_body_ok:
                mid_hip = (kpts[11] + kpts[12]) / 2.0
                centered = kpts - mid_hip
                mid_shoulder = (kpts[5] + kpts[6]) / 2.0
                body_size = np.linalg.norm(mid_shoulder - mid_hip) + 1e-6
                norm_kpts = (centered / body_size).flatten()

                if prev_norm_kpts is None:
                    velocity = np.zeros(34)
                else:
                    velocity = norm_kpts - prev_norm_kpts
                prev_norm_kpts = norm_kpts.copy()

                features = np.concatenate([norm_kpts, velocity]).reshape(1, -1)

                # Ask the trained AI model for the posture and confidence %
                pred = classifier.predict(features)[0]
                probs = classifier.predict_proba(features)[0]
                conf = float(np.max(probs)) * 100

                prediction_history.append(pred)
                # Pick the most common prediction in the last 5 frames
                smoothed_pred = max(
                    set(prediction_history), key=prediction_history.count
                )
                posture_text = f"{smoothed_pred.upper()} ({conf:.1f}%)"
                hud_color = (0, 255, 0)
            else:
                # Whole body NOT visible -> Do NOT call classifier.predict()
                prev_norm_kpts = None
                prediction_history.clear()
                posture_text = "Whole Body Not Detected (Insufficient Evidence)"
                hud_color = (0, 180, 255)  # Orange/Yellow in BGR
        else:
            prev_norm_kpts = None
            prediction_history.clear()

        # Display the AI Prediction at the top of the screen
        cv2.rectangle(annotated, (10, 10), (760, 70), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            f"AI Prediction: {posture_text}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            hud_color,
            2,
        )

        # Orange/yellow HUD status banner when full-body visibility is lacking
        if person_detected and not whole_body_ok:
            cv2.rectangle(annotated, (10, 78), (760, 120), (0, 140, 255), -1)
            cv2.putText(
                annotated,
                "WARNING: Full-body visibility required (shoulders, hips, knees & ankles)",
                (20, 106),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
            )

        cv2.imshow("Step 3: Test Trained AI Model", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()