import cv2
import os
import time
import numpy as np
from datetime import datetime
from threading import Thread
from importlib import import_module
from onvif import ONVIFCamera

try:
    YOLO = import_module("ultralytics").YOLO
except ModuleNotFoundError:
    YOLO = None

# =====================================================================
# SUBSYSTEM 1: MULTI-THREADED RTSP INGESTION & ONVIF PTZ CONTROLLER
# =====================================================================
class CameraController:
    def __init__(self, ip="172.20.10.4", user="SIDPProject", password="SIDPUTP2026"):
        self.ip = ip
        self.user = user
        self.password = password
        self.rtsp_url = f"rtsp://{self.user}:{self.password}@{self.ip}:554/stream2"
        
        self.stopped = False
        self.is_patrolling = False

        print(f"[S1] Connecting to Tapo C210 RTSP at {self.ip}...")
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise ConnectionError(f"[ERROR] Failed to open RTSP stream at {self.rtsp_url}")
            
        self.ret, self.frame = self.cap.read()

        print(f"[S1] Binding ONVIF motor daemon at {self.ip}:2020...")
        try:
            self.mycam = ONVIFCamera(self.ip, 2020, self.user, self.password)
            self.media = self.mycam.create_media_service()
            self.ptz = self.mycam.create_ptz_service()
            self.profile = self.media.GetProfiles()[0]
            self.token = self.profile.token
            self.ptz_available = True
            print("[S1] ONVIF PTZ service online.")
        except Exception as e:
            print(f"[WARNING] PTZ offline: {e}. Running in video-only mode.")
            self.ptz_available = False

        self.worker = Thread(target=self._update_frames, daemon=True)
        self.worker.start()

    def _update_frames(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            self.frame = frame

    def get_frame(self):
        return self.frame

    def move(self, pan_speed=0.0, tilt_speed=0.0, duration=0.4):
        if not self.ptz_available or self.is_patrolling:
            return

        def _actuate():
            try:
                request = self.ptz.create_type('ContinuousMove')
                request.ProfileToken = self.token
                request.Velocity = {'PanTilt': {'x': pan_speed, 'y': tilt_speed}}
                self.ptz.ContinuousMove(request)
                time.sleep(duration)
                self.ptz.Stop({'ProfileToken': self.token})
            except Exception as err:
                print(f"[ERROR] Motor command failed: {err}")

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
        if not self.ptz_available:
            return
        if self.is_patrolling:
            print("[S1] Stopping continuous pan...")
            self.is_patrolling = False
            try:
                self.ptz.Stop({'ProfileToken': self.token})
            except Exception:
                pass
        else:
            print("[S1] Starting non-stop continuous pan (Press Z to stop)...")
            self.is_patrolling = True
            Thread(target=self._patrol_worker, daemon=True).start()

    def _patrol_worker(self):
        PAN_SPEED = 0.4
        SWEEP_DURATION = 1.0  # ~35 degrees
        direction = -1

        while self.is_patrolling and not self.stopped:
            try:
                request = self.ptz.create_type('ContinuousMove')
                request.ProfileToken = self.token
                request.Velocity = {'PanTilt': {'x': direction * PAN_SPEED, 'y': 0.0}}
                self.ptz.ContinuousMove(request)

                for _ in range(int(SWEEP_DURATION / 0.1)):
                    if not self.is_patrolling or self.stopped:
                        break
                    time.sleep(0.1)

                self.ptz.Stop({'ProfileToken': self.token})

                for _ in range(4):
                    if not self.is_patrolling or self.stopped:
                        break
                    time.sleep(0.1)

                direction *= -1
            except Exception:
                break

    def release(self):
        self.is_patrolling = False
        self.stopped = True
        time.sleep(0.2)
        if self.cap.isOpened():
            self.cap.release()


# =====================================================================
# SUBSYSTEM 2: VISION POSTURE & FALL DETECTION ENGINE
# =====================================================================
class VisionAnalyticsEngine:
    def __init__(self, pose_weights="yolo26n-pose.pt", fall_weights="models/human-fall-detection.pt"):
        if YOLO is None:
            raise ImportError("Ultralytics is required for vision analytics. Install it with: pip install ultralytics")

        print("[S2] Loading Pose Model:", pose_weights)
        self.pose_model = YOLO(pose_weights)
        
        self.fall_model = None
        if os.path.exists(fall_weights):
            print("[S2] Loading Fall Detection Model:", fall_weights)
            self.fall_model = YOLO(fall_weights)

    def classify_posture_geometry(self, keypoints_xy, bbox):
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        if (w / h) > 1.15:
            return "FALLING / LYING", 90.0, (0, 0, 255)

        l_sh, r_sh = keypoints_xy[5], keypoints_xy[6]
        l_hip, r_hip = keypoints_xy[11], keypoints_xy[12]

        if np.all(l_sh > 0) and np.all(r_sh > 0) and np.all(l_hip > 0) and np.all(r_hip > 0):
            mid_sh = (l_sh + r_sh) / 2.0
            mid_hip = (l_hip + r_hip) / 2.0
            dx = mid_sh[0] - mid_hip[0]
            dy = mid_hip[1] - mid_sh[1]
            angle_deg = abs(np.degrees(np.arctan2(abs(dx), max(dy, 1e-5))))

            if angle_deg > 45.0:
                return f"BENDING ({angle_deg:.0f} deg)", angle_deg, (0, 165, 255)

            l_knee, r_knee = keypoints_xy[13], keypoints_xy[14]
            if np.all(l_knee > 0) and np.all(r_knee > 0):
                mid_knee = (l_knee + r_knee) / 2.0
                torso_len = np.linalg.norm(mid_sh - mid_hip)
                if abs(mid_knee[1] - mid_hip[1]) < 0.65 * torso_len:
                    return f"SITTING ({angle_deg:.0f} deg)", angle_deg, (255, 200, 0)

            return f"STANDING ({angle_deg:.0f} deg)", angle_deg, (0, 255, 0)

        return "DETECTED", 0.0, (200, 200, 200)

    def process_frame(self, frame):
        results = self.pose_model.track(frame, persist=True, verbose=False)[0]
        annotated = results.plot()

        if results.boxes is not None and results.keypoints is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            kpts = results.keypoints.xy.cpu().numpy()
            track_ids = (
                results.boxes.id.int().cpu().tolist()
                if results.boxes.id is not None
                else list(range(len(boxes)))
            )

            for idx, box in enumerate(boxes):
                label, angle, color = self.classify_posture_geometry(kpts[idx], box)
                x1, y1, x2, y2 = map(int, box)
                cv2.putText(
                    annotated,
                    f"ID:{track_ids[idx]} | {label}",
                    (x1, max(y1 - 12, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        if self.fall_model is not None:
            fall_res = self.fall_model(frame, conf=0.55, verbose=False)[0]
            if fall_res.boxes is not None and len(fall_res.boxes) > 0:
                cv2.putText(
                    annotated,
                    "ALERT: RGB FALL CLASSIFIER TRIGGERED!",
                    (20, annotated.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )

        return annotated


# =====================================================================
# MAIN INTEGRATED EXECUTION LOOP
# =====================================================================
if __name__ == "__main__":
    CLASSES = {
        ord('1'): "standing",
        ord('2'): "sitting",
        ord('3'): "bending",
        ord('4'): "falling_lying"
    }
    BASE_DIR = "dataset"
    for folder in CLASSES.values():
        os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

    controller = CameraController()
    ai_engine = VisionAnalyticsEngine()

    current_class = "standing"
    auto_record = False
    ai_enabled = True
    last_save_time = 0.0
    SAVE_INTERVAL = 0.5

    print("\n=======================================================")
    print("   INTEGRATED SUBSYSTEM 1 + SUBSYSTEM 2 CONSOLE        ")
    print("=======================================================")
    print(" [W / A / S / D] : Manual Tilt / Pan")
    print(" [Z]             : Toggle Non-Stop Pan (ON / OFF)")
    print(" [P]             : Toggle Subsystem 2 AI Overlay (ON / OFF)")
    print(" [1-4]           : Select Dataset Class (Stand/Sit/Bend/Fall)")
    print(" [C] / [R]       : Single Snap [C] | Auto-Record 2fps [R]")
    print(" [Q]             : Quit")
    print("=======================================================\n")

    prev_time = time.time()

    while not controller.stopped:
        frame = controller.get_frame()
        if frame is None:
            continue

        raw_frame = frame.copy()
        display_frame = ai_engine.process_frame(frame.copy()) if ai_enabled else frame.copy()

        class_dir = os.path.join(BASE_DIR, current_class)
        img_count = len(os.listdir(class_dir))
        now = time.time()
        if auto_record and (now - last_save_time >= SAVE_INTERVAL):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filepath = os.path.join(class_dir, f"{current_class}_{timestamp}.jpg")
            cv2.imwrite(filepath, raw_frame)
            last_save_time = now
            img_count += 1
            print(f"[AUTO-SAVED] {filepath} ({img_count})")

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time

        pan_status = "PANNING (Z)" if controller.is_patrolling else "MANUAL"
        rec_status = "REC ON" if auto_record else "STANDBY"
        cv2.putText(display_frame, f"S1+S2 Live | FPS: {fps:.1f} | PTZ: {pan_status}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Dataset: [{current_class.upper()}] ({img_count}) | {rec_status}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("SIDP - Subsystem 1 & 2 Integrated Feed", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p') or key == ord('P'):
            ai_enabled = not ai_enabled
        elif key in CLASSES:
            current_class = CLASSES[key]
            print(f"[DATASET] Target folder: dataset/{current_class}/")
        elif key == ord('c') or key == ord('C'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filepath = os.path.join(class_dir, f"{current_class}_{timestamp}.jpg")
            cv2.imwrite(filepath, raw_frame)
            print(f"[SNAP SAVED] {filepath}")
        elif key == ord('r') or key == ord('R'):
            auto_record = not auto_record
            print(f"[RECORDER] Auto-Capture {'ON' if auto_record else 'OFF'}")
        elif key == ord('z') or key == ord('Z'):
            controller.toggle_continuous_pan()
        elif key == ord('a') or key == 81:
            controller.pan_left()
        elif key == ord('d') or key == 83:
            controller.pan_right()
        elif key == ord('w') or key == 82:
            controller.tilt_up()
        elif key == ord('s') or key == 84:
            controller.tilt_down()

    controller.release()
    cv2.destroyAllWindows()