"""COCO17 skeleton, evidence states and actual measured processing timings."""
import cv2

from app.schemas import FallState, VisionObservation

COCO_EDGES = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
COLORS = {FallState.NORMAL: (210, 210, 210), FallState.POSSIBLE_FALL: (235, 235, 235),
          FallState.FALLING: (255, 255, 255), FallState.FALLEN: (255, 255, 255)}


def confidence_text(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.0%}"


def draw_overlay(frame_bgr, observation: VisionObservation):
    image = frame_bgr.copy()
    for person in observation.persons:
        color = COLORS[person.state]
        x1, y1, x2, y2 = map(int, person.bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        for a, b in COCO_EDGES:
            ka, kb = person.keypoints[a], person.keypoints[b]
            if ka.valid and kb.valid:
                cv2.line(image, (int(ka.x), int(ka.y)), (int(kb.x), int(kb.y)), color, 2)
        for k in person.keypoints:
            if k.valid:
                cv2.circle(image, (int(k.x), int(k.y)), 3, color, -1)
        confidence = person.activity_confidence if person.activity_confidence is not None else person.detection_confidence
        lines = [f"P{person.track_id} {person.activity_label} {confidence:.0%}"]
        if person.risk.status == "ready":
            lines.append(f"Pre-fall {confidence_text(person.risk.prefall_confidence)}"
                         f" | Fall {confidence_text(person.risk.fall_confidence)}")
        for i, text in enumerate(lines):
            y = min(image.shape[0] - 12, max(60, y1) + i*19)
            cv2.putText(image, text, (max(0, min(x1, image.shape[1]-250)), y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(image, text, (max(0, min(x1, image.shape[1]-250)), y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    timing = observation.timings_ms
    text = (f"FPS {observation.fps:.1f} | pose {timing['pose']:.1f} ms | temporal {timing['temporal']:.1f} ms"
            f" | total {timing['total']:.1f} ms")
    cv2.rectangle(image, (0, 0), (image.shape[1], 45), (25, 25, 25), -1)
    cv2.putText(image, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(image, "Experimental vision evidence | Q: quit", (6, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1)
    return image
