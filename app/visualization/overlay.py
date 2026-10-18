"""High-contrast monochrome (black & white) boxy UI overlay for YOLO26-Pose, RelateAnything, and the posture/safety classifier."""
import cv2
import numpy as np

from app.schemas import Activity, FallState, Posture, TrackedPerson, VisionObservation

COCO_EDGES = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def confidence_text(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.0%}"


def safety_decision(person: TrackedPerson) -> str:
    """Maps person state, posture, and temporal risk to the 4 SIDP safety tiers: SAFE / WARNING / UNSAFE / ALERT."""
    if person.state in (FallState.FALLING, FallState.FALLEN) or person.activity == Activity.FALL:
        return "ALERT"
    if person.appearance.fallen_confidence is not None and person.appearance.fallen_confidence >= 0.65:
        return "ALERT"
    if person.state == FallState.POSSIBLE_FALL or person.posture == Posture.LYING:
        return "UNSAFE"
    if person.posture == Posture.BENDING or person.activity == Activity.INACTIVE:
        return "WARNING"
    return "SAFE"


def _draw_boxy_rect(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, severe: bool = False):
    """Draws a sharp monochrome bounding frame with emphasized corner brackets."""
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 4 if severe else 3)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2 if severe else 1)
    corner = max(10, min(22, (x2 - x1) // 5, (y2 - y1) // 5))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(image, (cx, cy), (cx + dx * corner, cy), (255, 255, 255), 2)
        cv2.line(image, (cx, cy), (cx, cy + dy * corner), (255, 255, 255), 2)


def _draw_label_box(image: np.ndarray, x: int, y: int, lines: list[str], inverted: bool = False):
    """Draws a crisp rectangular monochrome info box with a 1px contrasting border."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.44
    thickness = 1
    pad_x, pad_y, line_h = 8, 6, 17
    widths = [cv2.getTextSize(text, font, scale, thickness)[0][0] for text in lines]
    box_w = (max(widths) if widths else 60) + pad_x * 2
    box_h = len(lines) * line_h + pad_y * 2 - 3

    h, w = image.shape[:2]
    bx1 = max(4, min(x, w - box_w - 4))
    by1 = max(50, y - box_h - 4) if y - box_h - 4 >= 50 else min(h - box_h - 32, y + 4)
    bx2, by2 = bx1 + box_w, by1 + box_h

    bg = (248, 248, 248) if inverted else (12, 12, 12)
    fg = (12, 12, 12) if inverted else (245, 245, 245)
    border = (12, 12, 12) if inverted else (220, 220, 220)

    cv2.rectangle(image, (bx1, by1), (bx2, by2), bg, -1)
    cv2.rectangle(image, (bx1, by1), (bx2, by2), border, 1)
    for idx, text in enumerate(lines):
        ty = by1 + pad_y + 11 + idx * line_h
        cv2.putText(image, text, (bx1 + pad_x, ty), font, scale, fg, thickness, cv2.LINE_AA)


def draw_overlay(
    frame_bgr: np.ndarray,
    observation: VisionObservation,
    ptz_mode: str | None = None,
    dataset_status: dict | None = None,
    ai_enabled: bool = True,
) -> np.ndarray:
    image = frame_bgr.copy()
    h, w = image.shape[:2]

    if ai_enabled:
        for person in observation.persons:
            tier = safety_decision(person)
            severe = tier in ("UNSAFE", "ALERT")
            x1, y1, x2, y2 = map(int, person.bbox)

            # 1. Monochrome COCO-17 skeleton (black casing + crisp white bone & joint)
            for a, b in COCO_EDGES:
                ka, kb = person.keypoints[a], person.keypoints[b]
                if ka.valid and kb.valid:
                    pa, pb = (int(ka.x), int(ka.y)), (int(kb.x), int(kb.y))
                    cv2.line(image, pa, pb, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.line(image, pa, pb, (255, 255, 255), 2, cv2.LINE_AA)
            for k in person.keypoints:
                if k.valid:
                    pt = (int(k.x), int(k.y))
                    cv2.circle(image, pt, 4, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(image, pt, 2, (245, 245, 245), -1, cv2.LINE_AA)

            # 2. Sharp boxy bounding rectangle
            _draw_boxy_rect(image, x1, y1, x2, y2, severe=severe)

            # 3. Clear multi-line boxy label: ID + Decision + Posture/Activity + RelateAnything + Risk
            conf = person.activity_confidence if person.activity_confidence is not None else person.detection_confidence
            angle_str = f" {person.geometry.torso_angle:.0f}deg" if (
                person.posture == Posture.BENDING and person.geometry.torso_angle is not None
            ) else ""
            posture_tag = f"{person.posture.value}{angle_str}"
            lines = [f"P{person.track_id:02d} [{tier}] {posture_tag} | {person.movement_state} ({conf:.0%})"]

            if person.scene.label:
                sc_conf = f" {person.scene.confidence:.0%}" if person.scene.confidence is not None else ""
                lines.append(f"RelateAnything: {person.scene.label}{sc_conf}")
            elif person.activity_label and person.activity_label.upper() != person.posture.value:
                lines.append(f"Activity: {person.activity_label} ({person.activity_source})")

            if person.risk.status == "ready":
                lines.append(
                    f"Pre-fall {confidence_text(person.risk.prefall_confidence)}"
                    f" | Fall {confidence_text(person.risk.fall_confidence)}"
                )
            elif person.appearance.fallen_confidence is not None:
                lines.append(f"RGB Fall Detector: {person.appearance.fallen_confidence:.0%}")

            _draw_label_box(image, x1, y1, lines, inverted=(tier == "ALERT"))

    # 4. Top Boxy Telemetry Bar (Black & White)
    timing = observation.timings_ms
    cv2.rectangle(image, (0, 0), (w, 48), (10, 10, 10), -1)
    cv2.line(image, (0, 48), (w, 48), (220, 220, 220), 1)
    top_line = (
        f"SIDP S1+S2 MONITOR  |  FPS {observation.fps:.1f}  |  "
        f"POSE {timing['pose']:.1f}ms  TEMP {timing['temporal']:.1f}ms  TOTAL {timing['total']:.1f}ms  |  "
        f"TRACKS: {len(observation.persons)}"
    )
    cv2.putText(image, top_line, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (250, 250, 250), 1, cv2.LINE_AA)

    sub_parts = [f"CAM: {observation.camera_id}", f"AI(P): {'ON' if ai_enabled else 'OFF'}"]
    if ptz_mode is not None:
        sub_parts.append(f"PTZ(WASD/Z): {ptz_mode}")
    if dataset_status is not None:
        rec_tag = "REC 2FPS" if dataset_status.get("recording") else "STANDBY"
        cls_name = str(dataset_status.get("current_class", "standing")).upper()
        cnt = dataset_status.get("count", 0)
        sub_parts.append(f"DATA[1-4]: {cls_name} ({cnt}) [{rec_tag} C/R]")
    sub_parts.append("Q: QUIT")
    cv2.putText(image, "  |  ".join(sub_parts), (10, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (195, 195, 195), 1, cv2.LINE_AA)

    return image
