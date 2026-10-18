"""
SIDP Unified Master Integration Script (Subsystem 1 + Subsystem 2)
RBB4013 System Integrated Design Project — Universiti Teknologi PETRONAS (UTP)

Subsystem 1: Tapo C210 Zero-Lag RTSP Ingestion, ONVIF PTZ Motor Control & Dataset Collector
Subsystem 2: YOLO26-Pose Skeletal Tracking, Geometric Posture Rule Engine & Auxiliary RGB Fall Classifier
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
from onvif import ONVIFCamera

# =====================================================================
# 1. DYNAMIC MODULE PATH RESOLUTION (SUBSYSTEM 2)
# =====================================================================
WORKSPACE_ROOT = Path(__file__).resolve().parent
SUBSYSTEM2_ROOT = WORKSPACE_ROOT / "Vision Analytics AI" / "Vision-Analytics-AI"
if not SUBSYSTEM2_ROOT.exists():
    SUBSYSTEM2_ROOT = WORKSPACE_ROOT / "Vision-Analytics-AI"

if str(SUBSYSTEM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSYSTEM2_ROOT))

from ultralytics import YOLO  # noqa: E402
from app.config import load_config  # noqa: E402
from app.inference.yolo26_pose import select_device  # noqa: E402
from app.posture.classifier import GeometricPostureClassifier  # noqa: E402
from app.posture.geometry import clip_bbox, extract_geometry  # noqa: E402
from app.schemas import Keypoint, PersonPose, Posture  # noqa: E402


# =====================================================================
# 2. SUBSYSTEM 1: CAMERA STREAM, ONVIF PTZ & PATROL CONTROLLER
# =====================================================================
class CameraController:
    """
    Subsystem 1 Production Layer:
    - Multi-threaded RTSP buffer-flushing ingestion (<200 ms latency)
    - Local ONVIF PTZ motor control on port 2020
    - Manual W/A/S/D stepping and Z continuous 35-degree patrol sweep
    """

    def __init__(self, ip: str = "172.20.10.4", user: str = "SIDPProject", password: str = "SIDPUTP2026"):
        self.ip = ip
        self.user = user
        self.password = password
        self.rtsp_url = f"rtsp://{self.user}:{self.password}@{self.ip}:554/stream2"

        self.stopped = False
        self.is_patrolling = False

        # 1. Initialize RTSP Video Capture
        print(f"[S1] Connecting to Tapo C210 RTSP feed at {self.ip}:554/stream2...")
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise ConnectionError(f"[ERROR] Failed to open RTSP stream at {self.rtsp_url}")

        self.ret, self.frame = self.cap.read()

        # 2. Bind Local ONVIF PTZ Motor Control Service (Port 2020)
        print(f"[S1] Binding ONVIF PTZ motor daemon at {self.ip}:2020...")
        try:
            self.mycam = ONVIFCamera(self.ip, 2020, self.user, self.password)
            self.media = self.mycam.create_media_service()
            self.ptz = self.mycam.create_ptz_service()
            self.profile = self.media.GetProfiles()[0]
            self.token = self.profile.token
            self.ptz_available = True
            print("[S1] ONVIF PTZ motor service online and ready.")
        except Exception as exc:
            print(f"[WARNING] ONVIF PTZ service unavailable ({exc}). Operating in video-only mode.")
            self.ptz_available = False

        # 3. Start Background Buffer-Flushing Thread
        self.worker = Thread(target=self._update_frames, daemon=True)
        self.worker.start()
        print("[S1] Background low-latency frame grabber started.")

    def _update_frames(self):
        """Continuously flushes stale RTSP buffer frames so get_frame() returns the newest frame."""
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            self.frame = frame

    def get_frame(self) -> np.ndarray | None:
        """Returns the newest decoded BGR frame."""
        return self.frame

    def move(self, pan_speed: float = 0.0, tilt_speed: float = 0.0, duration: float = 0.4):
        """Executes a non-blocking directional PTZ step for W/A/S/D controls."""
        if not self.ptz_available or self.is_patrolling:
            return

        def _actuate():
            try:
                request = self.ptz.create_type("ContinuousMove")
                request.ProfileToken = self.token
                request.Velocity = {"PanTilt": {"x": pan_speed, "y": tilt_speed}}
                self.ptz.ContinuousMove(request)
                time.sleep(duration)
                self.ptz.Stop({"ProfileToken": self.token})
            except Exception as err:
                print(f"[ERROR] PTZ step command failed: {err}")

        Thread(target=_actuate, daemon=True).start()

    def pan_left(self):
        self.move(pan_speed=-0.4, tilt_speed=0.0)

    def pan_right(self):
        self.move(pan_speed=0.4, tilt_speed=0.0)

    def tilt_up(self):
        self.move(pan_speed=0.0, tilt_speed=0.3)

    def tilt_down(self):
        self.move(pan_speed=0.0, tilt_speed=-0.3)

    def toggle_continuous_pan(self):
        """Toggles non-stop continuous ~35-degree patrol sweeping via key 'Z'."""
        if not self.ptz_available:
            print("[WARNING] PTZ service offline; cannot toggle patrol sweep.")
            return

        if self.is_patrolling:
            print("[S1] Halting continuous patrol sweep...")
            self.is_patrolling = False
            try:
                self.ptz.Stop({"ProfileToken": self.token})
            except Exception as exc:
                print(f"[ERROR] Failed to stop PTZ motor: {exc}")
        else:
            print("[S1] Continuous 35° patrol sweep ACTIVATED (Press 'Z' to stop).")
            self.is_patrolling = True
            Thread(target=self._patrol_worker, daemon=True).start()

    def _patrol_worker(self):
        """Background worker executing continuous back-and-forth 35° sweeps."""
        pan_speed = 0.4
        sweep_duration = 1.0  # ~35 degrees arc at 0.4 velocity
        direction = -1        # -1 = Left, +1 = Right

        while self.is_patrolling and not self.stopped:
            try:
                request = self.ptz.create_type("ContinuousMove")
                request.ProfileToken = self.token
                request.Velocity = {"PanTilt": {"x": direction * pan_speed, "y": 0.0}}
                self.ptz.ContinuousMove(request)

                for _ in range(int(sweep_duration / 0.1)):
                    if not self.is_patrolling or self.stopped:
                        break
                    time.sleep(0.1)

                self.ptz.Stop({"ProfileToken": self.token})

                for _ in range(4):
                    if not self.is_patrolling or self.stopped:
                        break
                    time.sleep(0.1)

                direction *= -1
            except Exception as exc:
                print(f"[ERROR] Patrol worker exception: {exc}")
                break

        try:
            self.ptz.Stop({"ProfileToken": self.token})
        except Exception:
            pass

    def release(self):
        """Cleanly halts background threads, brakes PTZ motors, and closes RTSP socket."""
        print("[S1] Releasing camera, halting threads, and braking PTZ motors...")
        self.is_patrolling = False
        self.stopped = True
        time.sleep(0.2)
        if self.ptz_available:
            try:
                self.ptz.Stop({"ProfileToken": self.token})
            except Exception:
                pass
        if self.cap.isOpened():
            self.cap.release()
        print("[S1] Subsystem 1 teardown complete.")


# =====================================================================
# 3. SUBSYSTEM 2: DEEP LEARNING VISION ANALYTICS LAYER
# =====================================================================
class VisionAnalyticsEngine:
    """
    Subsystem 2 Vision Analytics Layer:
    - Primary model: yolo26n-pose.pt (person tracking + 17 COCO keypoints)
    - Auxiliary model: models/human-fall-detection.pt (RGB Fall/Posture detector)
    - Geometric rule engine (app.posture.geometry + app.posture.classifier):
      Classifies STANDING, SITTING, BENDING (<angle> deg), and FALLING / LYING
    """

    def __init__(
        self,
        pose_weights: str | Path | None = None,
        fall_weights: str | Path | None = None,
    ):
        self.app_config = load_config(
            SUBSYSTEM2_ROOT / "configs" / "default.yaml",
            SUBSYSTEM2_ROOT / "configs" / "posture.yaml",
        )
        self.device = select_device(self.app_config.pose.device)
        self.posture_classifier = GeometricPostureClassifier(self.app_config.posture)

        # Resolve Pose Weights (check workspace root first, then Subsystem 2 directory)
        if pose_weights is None:
            candidate_root = WORKSPACE_ROOT / "yolo26n-pose.pt"
            candidate_s2 = SUBSYSTEM2_ROOT / "yolo26n-pose.pt"
            pose_path = candidate_root if candidate_root.is_file() else candidate_s2
        else:
            pose_path = Path(pose_weights)

        if not pose_path.is_file():
            raise FileNotFoundError(f"[S2] Pose weights not found at: {pose_path}")

        print(f"[S2] Loading Pose Estimation & Tracking Model: {pose_path} (device={self.device})")
        self.pose_model = YOLO(str(pose_path))

        # Resolve Auxiliary RGB Fall Detection Weights
        if fall_weights is None:
            candidate_fall_root = WORKSPACE_ROOT / "models" / "human-fall-detection.pt"
            candidate_fall_s2 = SUBSYSTEM2_ROOT / "models" / "human-fall-detection.pt"
            fall_path = candidate_fall_root if candidate_fall_root.is_file() else candidate_fall_s2
        else:
            fall_path = Path(fall_weights)

        self.fall_model = None
        self.fall_class_names: dict[int, str] = {}
        if fall_path.is_file():
            print(f"[S2] Loading Auxiliary RGB Fall Classifier: {fall_path}")
            self.fall_model = YOLO(str(fall_path))
            self.fall_class_names = {int(k): str(v).lower() for k, v in self.fall_model.names.items()}
        else:
            print(f"[WARNING] Auxiliary RGB fall model not found at {fall_path}; continuing with Pose+Geometry.")

    def _build_person_pose(
        self,
        bbox_xyxy: np.ndarray,
        det_conf: float,
        kpts_data: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> PersonPose | None:
        """Converts raw Ultralytics pose outputs into Subsystem 2's validated PersonPose schema."""
        box = clip_bbox(bbox_xyxy, frame_w, frame_h)
        if box is None or not np.isfinite(det_conf):
            return None

        kpt_conf_thresh = self.app_config.pose.keypoint_confidence
        keypoints: list[Keypoint] = []
        for idx in range(17):
            if idx < len(kpts_data):
                x, y, conf = kpts_data[idx]
                finite = bool(np.isfinite([x, y, conf]).all())
                visible = finite and 0 <= x <= frame_w and 0 <= y <= frame_h
                keypoints.append(
                    Keypoint(
                        index=idx,
                        x=float(x) if finite else 0.0,
                        y=float(y) if finite else 0.0,
                        confidence=float(np.clip(conf, 0.0, 1.0)) if finite else 0.0,
                        valid=bool(visible and conf >= kpt_conf_thresh),
                    )
                )
            else:
                keypoints.append(Keypoint(index=idx, x=0.0, y=0.0, confidence=0.0, valid=False))

        return PersonPose(
            bbox=box,
            detection_confidence=float(np.clip(det_conf, 0.0, 1.0)),
            keypoints=keypoints,
        )

    def classify_posture_geometry(
        self,
        keypoints_data: np.ndarray,
        bbox: np.ndarray,
        det_conf: float,
        frame_w: int,
        frame_h: int,
    ) -> tuple[str, float, tuple[int, int, int], str]:
        """
        Executes Subsystem 2's geometric rule engine (torso inclination angle & leg ratios)
        to classify:
        - STANDING
        - SITTING
        - BENDING (<angle> deg)
        - FALLING / LYING
        Returns (label, torso_angle_deg, bgr_color, safety_status).
        """
        x1, y1, x2, y2 = bbox
        w = max(float(x2 - x1), 1.0)
        h = max(float(y2 - y1), 1.0)
        aspect_ratio = w / h

        # Extract Subsystem 2 geometry features & classifier output
        pose_obj = self._build_person_pose(bbox, det_conf, keypoints_data, frame_w, frame_h)
        geom = extract_geometry(pose_obj, frame_h) if pose_obj is not None else None
        s2_posture = self.posture_classifier.classify(geom).label if geom is not None else Posture.UNKNOWN

        # 1. Check FALLING / LYING (aspect ratio or horizontal torso)
        if aspect_ratio > 1.15 or s2_posture == Posture.LYING:
            angle_val = geom.torso_angle if (geom and geom.torso_angle is not None) else 90.0
            return "FALLING / LYING", float(angle_val), (0, 0, 255), "CRITICAL: FALL"

        # 2. Compute torso inclination angle & leg ratios from 17 COCO keypoints
        keypoints_xy = keypoints_data[:, :2]
        l_sh, r_sh = keypoints_xy[5], keypoints_xy[6]
        l_hip, r_hip = keypoints_xy[11], keypoints_xy[12]

        if np.all(l_sh > 0) and np.all(r_sh > 0) and np.all(l_hip > 0) and np.all(r_hip > 0):
            mid_sh = (l_sh + r_sh) / 2.0
            mid_hip = (l_hip + r_hip) / 2.0
            dx = mid_sh[0] - mid_hip[0]
            dy = mid_hip[1] - mid_sh[1]
            angle_deg = (
                float(geom.torso_angle)
                if (geom is not None and geom.torso_angle is not None)
                else float(abs(np.degrees(np.arctan2(abs(dx), max(dy, 1e-5)))))
            )

            # Check BENDING (<angle> deg)
            if angle_deg > 45.0:
                if angle_deg >= 65.0 and aspect_ratio >= 1.05:
                    return "FALLING / LYING", angle_deg, (0, 0, 255), "CRITICAL: FALL"
                return f"BENDING ({angle_deg:.0f} deg)", angle_deg, (0, 165, 255), "CAUTION: BENDING"

            # Check SITTING via Subsystem 2 knee/hip angles or vertical hip-to-knee leg ratio
            if s2_posture == Posture.SITTING:
                return "SITTING", angle_deg, (255, 200, 0), "SAFE"

            l_knee, r_knee = keypoints_xy[13], keypoints_xy[14]
            if np.all(l_knee > 0) and np.all(r_knee > 0):
                mid_knee = (l_knee + r_knee) / 2.0
                torso_len = float(np.linalg.norm(mid_sh - mid_hip))
                if abs(mid_knee[1] - mid_hip[1]) < 0.65 * torso_len:
                    return "SITTING", angle_deg, (255, 200, 0), "SAFE"

            return "STANDING", angle_deg, (0, 255, 0), "SAFE"

        return "STANDING", 0.0, (0, 255, 0), "MONITORING"

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Runs YOLO26-Pose tracking, skeletal keypoint rendering, geometric posture classification,
        and auxiliary RGB fall detection on the current frame.
        """
        frame_h, frame_w = frame.shape[:2]
        results = self.pose_model.track(
            frame,
            persist=True,
            conf=self.app_config.pose.confidence,
            imgsz=self.app_config.pose.image_size,
            device=self.device,
            verbose=False,
        )[0]

        # Render skeleton connections and tracking boxes via Ultralytics plot()
        annotated = results.plot(labels=False, conf=False)

        # Run auxiliary RGB fall classifier (models/human-fall-detection.pt)
        rgb_fall_boxes: list[tuple[np.ndarray, float]] = []
        if self.fall_model is not None:
            fall_res = self.fall_model.predict(
                frame,
                conf=0.50,
                imgsz=self.app_config.secondary.image_size,
                device=self.device,
                verbose=False,
            )[0]
            if fall_res.boxes is not None and len(fall_res.boxes) > 0:
                f_boxes = fall_res.boxes.xyxy.cpu().numpy()
                f_confs = fall_res.boxes.conf.cpu().numpy()
                f_cls = fall_res.boxes.cls.cpu().numpy()
                for f_box, f_conf, cls_id in zip(f_boxes, f_confs, f_cls):
                    cls_name = self.fall_class_names.get(int(cls_id), "")
                    if cls_name == "fallen":
                        rgb_fall_boxes.append((f_box, float(f_conf)))

        # Process tracked persons & overlay persistent IDs, posture labels, and safety status tags
        if results.boxes is not None and results.keypoints is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            kpts_data = results.keypoints.data.cpu().numpy()
            track_ids = (
                results.boxes.id.int().cpu().tolist()
                if results.boxes.id is not None
                else list(range(1, len(boxes) + 1))
            )

            for idx, box in enumerate(boxes):
                det_conf = float(confs[idx]) if idx < len(confs) else 1.0
                joints = kpts_data[idx] if idx < len(kpts_data) else np.zeros((17, 3))
                label, angle_deg, color, safety_tag = self.classify_posture_geometry(
                    joints, box, det_conf, frame_w, frame_h
                )

                # Fuse with auxiliary RGB fall detection if a fallen box overlaps this person
                x1, y1, x2, y2 = map(int, box)
                for f_box, f_conf in rgb_fall_boxes:
                    fx1, fy1, fx2, fy2 = f_box
                    inter_x1, inter_y1 = max(x1, fx1), max(y1, fy1)
                    inter_x2, inter_y2 = min(x2, fx2), min(y2, fy2)
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        person_area = max((x2 - x1) * (y2 - y1), 1)
                        if (inter_area / person_area) >= 0.25:
                            label = "FALLING / LYING"
                            color = (0, 0, 255)
                            safety_tag = f"CRITICAL: FALL (RGB {f_conf:.0%})"
                            break

                # Draw bounding box & status header for each tracked person
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                tag_text = f"ID:{track_ids[idx]} | {label} | [{safety_tag}]"
                text_y = max(y1 - 10, 25)
                cv2.putText(
                    annotated,
                    tag_text,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    4,
                )
                cv2.putText(
                    annotated,
                    tag_text,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

        # Display global alert banner if RGB Fall Classifier detected a fall
        if rgb_fall_boxes:
            best_conf = max(conf for _, conf in rgb_fall_boxes)
            alert_msg = f"ALERT: RGB FALL CLASSIFIER TRIGGERED ({best_conf:.0%})!"
            cv2.putText(
                annotated,
                alert_msg,
                (20, annotated.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                4,
            )
            cv2.putText(
                annotated,
                alert_msg,
                (20, annotated.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )

        return annotated


# =====================================================================
# 4. UNIFIED MASTER EXECUTION LOOP
# =====================================================================
def main():
    classes = {
        ord("1"): "standing",
        ord("2"): "sitting",
        ord("3"): "bending",
        ord("4"): "falling_lying",
    }
    dataset_root = WORKSPACE_ROOT / "dataset"
    for folder in classes.values():
        os.makedirs(dataset_root / folder, exist_ok=True)

    controller = CameraController()
    ai_engine = VisionAnalyticsEngine()

    current_class = "standing"
    auto_record = False
    ai_enabled = True
    last_save_time = 0.0
    save_interval = 0.5  # 2 fps automated recording

    print("\n==================================================================")
    print("   SIDP UNIFIED MASTER CONSOLE (SUBSYSTEM 1 + SUBSYSTEM 2)        ")
    print("==================================================================")
    print(" [W / A / S / D] : Directional PTZ Camera Stepping")
    print(" [Z]             : Toggle Non-Stop 35° Patrol Sweeping (ON / OFF)")
    print(" [P]             : Toggle Subsystem 2 AI Vision Overlay (ON / OFF)")
    print(" [1 - 4]         : Set Dataset Class (1:Stand, 2:Sit, 3:Bend, 4:Fall)")
    print(" [C]             : Capture Single Clean Frame to dataset/<class>/")
    print(" [R]             : Toggle Automated 2 FPS Recording (ON / OFF)")
    print(" [Q]             : Clean Teardown & Quit")
    print("==================================================================\n")

    prev_time = time.time()
    window_name = "SIDP Master Feed — Subsystem 1 (PTZ/RTSP) + Subsystem 2 (Vision AI)"

    try:
        while not controller.stopped:
            frame = controller.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Keep clean copy without HUD overlay for dataset collection
            raw_frame = frame.copy()
            display_frame = ai_engine.process_frame(frame.copy()) if ai_enabled else frame.copy()

            class_dir = dataset_root / current_class
            img_count = len(os.listdir(class_dir))
            now = time.time()

            # Automated 2 FPS hands-free dataset recording
            if auto_record and (now - last_save_time >= save_interval):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filepath = class_dir / f"{current_class}_{timestamp}.jpg"
                cv2.imwrite(str(filepath), raw_frame)
                last_save_time = now
                img_count += 1
                print(f"[AUTO-REC 2FPS] Saved clean frame: {filepath} (Total: {img_count})")

            # Compute operational FPS
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            # Render Master HUD Overlay
            ptz_status = "PATROL 35° (Z)" if controller.is_patrolling else "MANUAL (WASD)"
            ai_status = "ON" if ai_enabled else "OFF"
            rec_status = "REC 2FPS [ON]" if auto_record else "STANDBY"

            cv2.putText(
                display_frame,
                f"SIDP S1+S2 | FPS: {fps:.1f} | AI(P): {ai_status} | PTZ: {ptz_status}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                f"Dataset [1-4]: {current_class.upper()} ({img_count}) | {rec_status} (C:Snap / R:Auto)",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
            )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("\n[MASTER] 'Q' pressed — initiating clean system teardown...")
                break
            elif key in (ord("p"), ord("P")):
                ai_enabled = not ai_enabled
                print(f"[S2] AI Vision Overlay {'ENABLED' if ai_enabled else 'DISABLED'}")
            elif key in classes:
                current_class = classes[key]
                print(f"[DATASET] Active target class set to: dataset/{current_class}/")
            elif key in (ord("c"), ord("C")):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filepath = class_dir / f"{current_class}_{timestamp}.jpg"
                cv2.imwrite(str(filepath), raw_frame)
                print(f"[SNAP SAVED] Clean frame saved to: {filepath}")
            elif key in (ord("r"), ord("R")):
                auto_record = not auto_record
                print(f"[RECORDER] 2 FPS Hands-Free Recording {'ENABLED' if auto_record else 'DISABLED'}")
            elif key in (ord("z"), ord("Z")):
                controller.toggle_continuous_pan()
            elif key in (ord("a"), ord("A"), 81):
                controller.pan_left()
            elif key in (ord("d"), ord("D"), 83):
                controller.pan_right()
            elif key in (ord("w"), ord("W"), 82):
                controller.tilt_up()
            elif key in (ord("s"), ord("S"), 84):
                controller.tilt_down()
    finally:
        controller.release()
        cv2.destroyAllWindows()
        print("[MASTER] All sockets, threads, motors, and GUI windows closed cleanly.")


if __name__ == "__main__":
    main()
