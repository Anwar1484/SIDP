"""OSD HUD overlay for YOLO26-Pose, RelateAnything, 6-Class Posture Banners, and Tri-Mode PTZ Target Reticle."""
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


def six_class_posture_banner(person: TrackedPerson) -> tuple[str, tuple[int, int, int]]:
    """
    Returns (display_tag, bgr_color) for the 6-class taxonomy:
      - STANDING, SITTING, WALKING -> Green
      - BENDING -> Orange / Yellow
      - FALLING, LYING DOWN -> Red
    """
    if person.state in (FallState.FALLING, FallState.FALLEN) or person.activity == Activity.FALL:
        return "FALLING", (35, 35, 240)
    if person.posture == Posture.LYING or person.activity == Activity.LYING:
        return "LYING DOWN", (35, 35, 240)
    if person.posture == Posture.BENDING or person.activity == Activity.BENDING:
        ang = person.geometry.torso_angle
        if ang is not None:
            warn = " [WARN >=60°]" if ang >= 60.0 else ""
            return f"BENDING ({ang:.0f} deg){warn}", (0, 175, 255)
        return "BENDING", (0, 175, 255)
    if person.activity == Activity.WALKING:
        return "WALKING", (50, 205, 50)
    if person.posture == Posture.SITTING or person.activity == Activity.SITTING:
        return "SITTING", (50, 205, 50)
    if person.posture == Posture.STANDING or person.activity == Activity.STANDING:
        return "STANDING", (50, 205, 50)
    return person.posture.value, (50, 205, 50)


def _draw_boxy_rect(image: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                    color: tuple[int, int, int], severe: bool = False):
    """Draws a sharp bounding frame with emphasized corner brackets."""
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 4 if severe else 3)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2 if severe else 1)
    corner = max(10, min(22, (x2 - x1) // 5, (y2 - y1) // 5))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(image, (cx, cy), (cx + dx * corner, cy), color, 2)
        cv2.line(image, (cx, cy), (cx, cy + dy * corner), color, 2)


def _draw_label_box(image: np.ndarray, x: int, y: int, lines: list[str],
                    banner_color: tuple[int, int, int]):
    """Draws a crisp rectangular info box with a color-coded header bar."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.44
    thickness = 1
    pad_x, pad_y, line_h = 8, 6, 17
    widths = [cv2.getTextSize(text, font, scale, thickness)[0][0] for text in lines]
    box_w = (max(widths) if widths else 60) + pad_x * 2
    box_h = len(lines) * line_h + pad_y * 2 - 3

    h, w = image.shape[:2]
    bx1 = max(4, min(x, w - box_w - 4))
    by1 = max(56, y - box_h - 4) if y - box_h - 4 >= 56 else min(h - box_h - 32, y + 4)
    bx2, by2 = bx1 + box_w, by1 + box_h

    cv2.rectangle(image, (bx1, by1), (bx2, by2), (14, 14, 14), -1)
    cv2.rectangle(image, (bx1, by1), (bx2, by1 + line_h + 4), banner_color, -1)
    cv2.rectangle(image, (bx1, by1), (bx2, by2), banner_color, 1)

    for idx, text in enumerate(lines):
        ty = by1 + pad_y + 11 + idx * line_h
        fg = (12, 12, 12) if (idx == 0 and banner_color != (35, 35, 240)) else (245, 245, 245)
        cv2.putText(image, text, (bx1 + pad_x, ty), font, scale, fg, thickness, cv2.LINE_AA)


def draw_overlay(
    frame_bgr: np.ndarray,
    observation: VisionObservation,
    ptz_mode: str | None = None,
    dataset_status: dict | None = None,
    ai_enabled: bool = True,
    auto_state=None,
) -> np.ndarray:
    image = frame_bgr.copy()
    h, w = image.shape[:2]

    if ai_enabled:
        for person in observation.persons:
            tier = safety_decision(person)
            severe = tier in ("UNSAFE", "ALERT")
            posture_tag, color = six_class_posture_banner(person)
            x1, y1, x2, y2 = map(int, person.bbox)

            # 1. COCO-17 skeleton
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
                    cv2.circle(image, pt, 2, color, -1, cv2.LINE_AA)

            # 2. Sharp bounding rectangle
            _draw_boxy_rect(image, x1, y1, x2, y2, color=color, severe=severe)

            # 3. Color-coded status banner + RelateAnything / Risk details
            conf = person.activity_confidence if person.activity_confidence is not None else person.detection_confidence
            lines = [f"P{person.track_id:02d} [{tier}] {posture_tag} ({conf:.0%})"]

            if person.scene.label:
                sc_conf = f" {person.scene.confidence:.0%}" if person.scene.confidence is not None else ""
                lines.append(f"RelateAnything: {person.scene.label}{sc_conf}")
            elif person.activity_label and person.activity_label.upper() not in posture_tag:
                lines.append(f"Activity: {person.activity_label} ({person.activity_source})")

            if person.risk.status == "ready":
                lines.append(
                    f"Pre-fall {confidence_text(person.risk.prefall_confidence)}"
                    f" | Fall {confidence_text(person.risk.fall_confidence)}"
                )
            elif person.appearance.fallen_confidence is not None:
                lines.append(f"RGB Fall Detector: {person.appearance.fallen_confidence:.0%}")

            _draw_label_box(image, x1, y1, lines, banner_color=color)

    # 4. Target Reticle when in AUTOMATIC mode
    if ptz_mode == "AUTO-TRACKING":
        cx_f, cy_f = w // 2, h // 2
        db_w, db_h = int(round(0.15 * (w / 2.0))), int(round(0.15 * (h / 2.0)))
        in_db = True if auto_state is None else getattr(auto_state, "in_deadband", True)
        db_color = (160, 220, 160) if in_db else (0, 200, 255)
        cv2.rectangle(image, (cx_f - db_w, cy_f - db_h), (cx_f + db_w, cy_f + db_h), db_color, 1, cv2.LINE_AA)
        cv2.line(image, (cx_f - 8, cy_f), (cx_f + 8, cy_f), db_color, 1, cv2.LINE_AA)
        cv2.line(image, (cx_f, cy_f - 8), (cx_f, cy_f + 8), db_color, 1, cv2.LINE_AA)

        if auto_state is not None and getattr(auto_state, "group_bbox", None) and getattr(auto_state, "collective_center", None):
            gx1, gy1, gx2, gy2 = auto_state.group_bbox
            gcx, gcy = auto_state.collective_center
            r_col = (50, 230, 50) if in_db else (0, 200, 255)
            cv2.rectangle(image, (gx1, gy1), (gx2, gy2), r_col, 1, cv2.LINE_AA)
            cv2.circle(image, (gcx, gcy), 10, r_col, 2, cv2.LINE_AA)
            cv2.line(image, (gcx - 15, gcy), (gcx + 15, gcy), r_col, 2, cv2.LINE_AA)
            cv2.line(image, (gcx, gcy - 15), (gcx, gcy + 15), r_col, 2, cv2.LINE_AA)
            cv2.arrowedLine(image, (cx_f, cy_f), (gcx, gcy), r_col, 1, cv2.LINE_AA, tipLength=0.12)

    # 5. Top OSD Telemetry Bar
    timing = observation.timings_ms
    cv2.rectangle(image, (0, 0), (w, 52), (10, 10, 10), -1)
    cv2.line(image, (0, 52), (w, 52), (220, 220, 220), 1)
    mode_tag = f"[{ptz_mode}]" if ptz_mode else "[STATIC]"
    top_line = (
        f"MODE: {mode_tag}  |  FPS: {observation.fps:.1f}  |  OCCUPANTS: {len(observation.persons)}  |  "
        f"POSE {timing['pose']:.1f}ms  TOTAL {timing['total']:.1f}ms"
    )
    cv2.putText(image, top_line, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (250, 250, 250), 1, cv2.LINE_AA)

    sub_parts = [f"CAM: {observation.camera_id}", f"AI(P): {'ON' if ai_enabled else 'OFF'}"]
    if ptz_mode is not None:
        sub_parts.append(f"PTZ(M/Tab/WASD/Z): {ptz_mode}")
    if dataset_status is not None:
        rec_tag = "REC 2FPS" if dataset_status.get("recording") else "STANDBY"
        cls_name = str(dataset_status.get("current_class", "standing")).upper()
        cnt = dataset_status.get("count", 0)
        sub_parts.append(f"DATA[1-4]: {cls_name} ({cnt}) [{rec_tag} C/R]")
    sub_parts.append("Q: QUIT")
    cv2.putText(image, "  |  ".join(sub_parts), (10, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (195, 195, 195), 1, cv2.LINE_AA)

    return image
