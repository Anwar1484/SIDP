"""YOLO26 pose, independent tracks and learned event evidence; no drawing dependencies."""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Protocol
from uuid import uuid4

import numpy as np

from app.config import AppConfig
from app.inference.appearance import match_appearance
from app.posture.classifier import GeometricPostureClassifier, PostureSmoother
from app.posture.geometry import extract_geometry
from app.posture.normalization import normalize_pose
from app.runtime import ROOT
from app.schemas import (
    Activity,
    AppearanceEvidence,
    FallInference,
    FallState,
    PersonPose,
    Posture,
    RiskAssessment,
    SceneEvidence,
    TrackedPerson,
    VisionObservation,
)
from app.temporal.buffer import PoseSample, TemporalBuffer
from app.temporal.event_features import event_features
from app.temporal.evidence_state import EvidenceStateMachine
from app.temporal.features import motion_features
from app.tracking.person_tracker import create_tracker, iou

logger = logging.getLogger(__name__)


class PoseBackend(Protocol):
    def infer(self, frame_bgr: np.ndarray) -> list[PersonPose]: ...


@dataclass
class PersonContext:
    buffer: TemporalBuffer
    smoother: PostureSmoother
    machine: EvidenceStateMachine
    appearance: AppearanceEvidence = field(default_factory=AppearanceEvidence)
    risk_track: object | None = None
    last_fall: FallInference | None = None
    last_fall_timestamp: float | None = None
    stationary_since: float | None = None


class VisionPipeline:
    def __init__(self, config: AppConfig, pose_backend: PoseBackend | None = None,
                 fall_backend=None, camera_id: str | None = None, appearance_backend=None,
                 event_classifier=None, risk_backend=None, scene_backend=None):
        if pose_backend is None:
            from app.inference.yolo26_pose import Yolo26PoseBackend
            pose_backend = Yolo26PoseBackend(config.pose)
        self.config, self.pose_backend = config, pose_backend
        self.fall_backend = fall_backend
        if self.fall_backend is None and config.temporal.model_manifest:
            from app.inference.temporal_fall import KerasTemporalBackend
            self.fall_backend = KerasTemporalBackend(config.temporal.model_manifest, config.temporal)
        self.appearance_backend = appearance_backend
        self.event_classifier = event_classifier
        self.risk_backend, self.scene_backend = risk_backend, scene_backend
        self.scene_worker = None
        self.risk_error: str | None = None
        self.scene_error: str | None = None
        self.appearance_error: str | None = None
        self.event_error: str | None = None
        self.tracker = create_tracker(config.tracker)
        self.classifier = GeometricPostureClassifier(config.posture)
        self.contexts: dict[int, PersonContext] = {}
        self.last_features: dict[int, np.ndarray] = {}
        self.session_id = str(uuid4())
        self.camera_id = camera_id or config.camera.id
        self.first_timestamp: float | None = None
        self.last_timestamp: float | None = None
        self.last_rgb_timestamp = -float("inf")

    def load_optional_models(self):
        """Explicit load-once boundary for the monitor and offline evaluator."""
        if self.config.secondary.enabled and self.appearance_backend is None:
            try:
                from app.inference.appearance import AppearanceBackend
                self.appearance_backend = AppearanceBackend(self.config.secondary, self.pose_backend.device)
                self.appearance_backend.warmup(self.config.camera.width, self.config.camera.height)
            except Exception as exc:
                self.appearance_error = str(exc)
                logger.warning("appearance_model_unavailable: %s", exc)
        if self.config.evidence.model and self.event_classifier is None and self.fall_backend is None:
            try:
                from app.inference.event_classifier import EventClassifier
                self.event_classifier = EventClassifier(ROOT / self.config.evidence.model)
            except Exception as exc:
                self.event_error = str(exc)
                logger.warning("event_model_unavailable: %s", exc)
        if self.config.risk.enabled and self.risk_backend is None and self.fall_backend is None:
            try:
                from app.inference.prefall import PrefallBackend
                self.risk_backend = PrefallBackend(self.config.risk)
            except Exception as exc:
                self.risk_error = str(exc)
                logger.warning("risk_model_unavailable: %s", exc)
        if self.config.scene.enabled and self.scene_backend is None:
            try:
                from app.inference.scene import SceneBackend
                self.scene_backend = SceneBackend(self.config.scene)
            except Exception as exc:
                self.scene_error = str(exc)
                logger.warning("scene_model_unavailable: %s", exc)

    def _new_context(self):
        return PersonContext(TemporalBuffer(self.config.temporal), PostureSmoother(self.config.posture),
                             EvidenceStateMachine(self.config.evidence),
                             risk_track=self.risk_backend.new_track() if self.risk_backend else None)

    def close(self):
        if self.scene_worker:
            self.scene_worker.close()
            self.scene_worker = None

    def reset_risk(self):
        for context in self.contexts.values():
            if context.risk_track is not None:
                context.risk_track = self.risk_backend.new_track()

    def process(self, frame_bgr: np.ndarray, timestamp: float, frame_id: int,
                wall_timestamp: datetime | None = None) -> VisionObservation:
        start = perf_counter()
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("frame timestamps must strictly increase")
        height, width = frame_bgr.shape[:2]
        poses = self.pose_backend.infer(frame_bgr)
        after_pose = perf_counter()
        tracks = self.tracker.update(poses, timestamp, (width, height))
        for identity in self.tracker.expired_ids:
            self.contexts.pop(identity, None)
        ambiguous = set()
        for n, first in enumerate(tracks):
            for second in tracks[n+1:]:
                if iou(first.pose.bbox, second.pose.bbox) >= self.config.tracker.ambiguity_iou:
                    ambiguous.update((first.id, second.id))
        after_tracking = perf_counter()
        if self.scene_backend is not None:
            if self.scene_worker is None:
                from app.inference.scene import SceneWorker
                self.scene_worker = SceneWorker(self.scene_backend)
            self.scene_worker.submit(frame_bgr, [t for t in tracks if t.id not in ambiguous], timestamp, self.session_id)
        rgb_matches, rgb_ms = {}, 0.0
        if (self.appearance_backend is not None and tracks
                and timestamp-self.last_rgb_timestamp >= self.config.secondary.interval_seconds):
            rgb_start = perf_counter()
            try:
                detections = self.appearance_backend.infer(frame_bgr)
                rgb_ms = (perf_counter()-rgb_start)*1000
                rgb_matches = match_appearance(tracks, detections, timestamp,
                                               self.config.secondary.min_iou, rgb_ms)
                self.appearance_error = None
            except Exception as exc:
                self.appearance_error = str(exc)
                logger.exception("appearance_inference_failed")
                rgb_matches = {t.id: AppearanceEvidence(status="unavailable") for t in tracks}
            self.last_rgb_timestamp = timestamp
        persons, temporal_ms = [], 0.0
        self.last_features = {}
        for track in tracks:
            context = self.contexts.get(track.id)
            if context is None:
                context = self._new_context()
                self.contexts[track.id] = context
            previous = context.buffer.samples[-1] if context.buffer.samples else None
            if track.id in ambiguous or (previous is not None and (
                timestamp-previous.timestamp > self.config.temporal.max_gap_seconds
                or previous.frame_size != (width, height)
            )):
                context = self._new_context()
                self.contexts[track.id] = context
                previous = None
            geometry = extract_geometry(track.pose, height)
            posture = context.smoother.update(timestamp, self.classifier.classify(geometry))
            sample = PoseSample(timestamp, track.pose, normalize_pose(track.pose), posture, geometry, (width, height))
            context.buffer.append(sample)
            motion = motion_features(sample, previous, self.config.temporal)
            if track.id in rgb_matches and track.id not in ambiguous:
                context.appearance = rgb_matches[track.id]
            appearance = context.appearance
            if appearance.sampled_at is not None:
                age = timestamp-appearance.sampled_at
                appearance = appearance.model_copy(update={"age_seconds": age})
                if age > self.config.secondary.max_age_seconds:
                    appearance = AppearanceEvidence(status="stale", age_seconds=age)
            if track.id in ambiguous:
                appearance = AppearanceEvidence(status="identity_uncertain")
            model_start = perf_counter()
            features, status = event_features(context.buffer, appearance, self.config.evidence.window_seconds,
                                               self.config.evidence.min_coverage)
            if features is not None:
                self.last_features[track.id] = features
            fall = FallInference(probability=None, model_name="event_classifier", window_frames=len(context.buffer.samples),
                                 valid_keypoint_ratio=geometry.confidence_coverage,
                                 status=status if self.event_classifier else "event_model_unavailable")
            if self.fall_backend is not None and track.id not in ambiguous:
                if not sample.normalized.valid or geometry.confidence_coverage < self.config.temporal.min_coverage:
                    context.last_fall = context.last_fall_timestamp = None
                    fall = fall.model_copy(update={"status": "insufficient_body_coverage"})
                elif (context.last_fall_timestamp is None or timestamp-context.last_fall_timestamp
                      >= self.config.temporal.inference_interval_seconds):
                    fall = self.fall_backend.infer(context.buffer, motion, posture.label)
                    if not fall.calibrated and fall.probability is not None:
                        fall = fall.model_copy(update={
                            "raw_score": fall.raw_score if fall.raw_score is not None else fall.probability,
                            "probability": None,
                        })
                    context.last_fall, context.last_fall_timestamp = fall, timestamp
                else:
                    fall = context.last_fall.model_copy(update={"latency_ms": 0})
            elif features is not None and self.event_classifier is not None and track.id not in ambiguous:
                try:
                    fall = self.event_classifier.infer(features, appearance.status in ("matched", "no_matched_fallen_box"),
                                                       geometry.confidence_coverage, len(context.buffer.samples))
                    self.event_error = None
                except Exception as exc:
                    self.event_error = str(exc)
                    fall = fall.model_copy(update={"status": "event_model_error"})
            elif (self.event_classifier is None and appearance.fallen_confidence is not None
                  and track.id not in ambiguous):
                # RGB confidence does not establish a temporal fall probability.
                fall = FallInference(
                    probability=None,
                    raw_score=appearance.fallen_confidence,
                    model_name="human-fall-detection_yolov11n",
                    window_frames=len(context.buffer.samples),
                    valid_keypoint_ratio=geometry.confidence_coverage,
                    calibrated=False, status="raw fallen-detector confidence",
                    target="RGB Fallen class detection, not a temporal fall probability",
                )
            temporal_ms += (perf_counter()-model_start)*1000
            state = context.machine.update(timestamp, fall)
            risk = RiskAssessment()
            if self.risk_backend is not None:
                try:
                    risk = self.risk_backend.infer(context.risk_track, track.pose, timestamp, (width, height),
                                                   track.id in ambiguous)
                    self.risk_error = None
                except Exception as exc:
                    self.risk_error = str(exc)
                    logger.exception("risk_inference_failed")
                    context.risk_track = self.risk_backend.new_track()
                    risk = RiskAssessment(status="unavailable")
                temporal_ms += risk.latency_ms
                if self.fall_backend is None:
                    fall = FallInference(
                        probability=None, raw_score=risk.fall_confidence, model_name=risk.model_name,
                        window_frames=risk.window_frames, valid_keypoint_ratio=geometry.confidence_coverage,
                        calibrated=False, status=risk.status, latency_ms=risk.latency_ms,
                        target=risk.target,
                    )
                state = {"NORMAL": FallState.NORMAL, "PRE_FALL": FallState.POSSIBLE_FALL,
                         "FALL": FallState.FALLEN}.get(risk.advisory or risk.state, FallState.NORMAL)
            movement = "UNKNOWN" if motion.joint_velocity is None else (
                "STATIONARY" if motion.joint_velocity < self.config.temporal.stationary_velocity else "MOVING")
            if movement == "STATIONARY":
                if context.stationary_since is None:
                    context.stationary_since = timestamp
                inactive_seconds = max(0.0, timestamp - context.stationary_since)
            else:
                context.stationary_since = None
                inactive_seconds = 0.0
            scene = (self.scene_worker.snapshot(track.id, timestamp, self.session_id,
                                                min_timestamp=context.buffer.samples[0].timestamp)
                     if self.scene_worker is not None and track.id not in ambiguous
                     else SceneEvidence(status="identity_uncertain" if track.id in ambiguous else "unavailable"))
            geometric_bending = posture.label == Posture.BENDING
            if appearance.posture is not None and not (geometric_bending and appearance.posture != Posture.LYING):
                posture = posture.model_copy(update={"label": appearance.posture,
                                                       "confidence": appearance.posture_confidence,
                                                       "source": "RGB posture model",
                                                       "reason": "RGB posture detector class; confidence is uncalibrated"})
            if scene.predicate and scene.predicate.startswith("sitting ") and not geometric_bending:
                posture = posture.model_copy(update={"label": Posture.SITTING,
                                                       "confidence": scene.confidence,
                                                       "source": "RelateAnything",
                                                       "reason": f"Scene relation: {scene.label}"})
            elif scene.predicate and scene.predicate.startswith("standing ") and not geometric_bending:
                posture = posture.model_copy(update={"label": Posture.STANDING,
                                                       "confidence": scene.confidence,
                                                       "source": "RelateAnything",
                                                       "reason": f"Scene relation: {scene.label}"})
            walking = (posture.label == Posture.STANDING and movement == "MOVING"
                       and motion.centroid_velocity is not None
                       and motion.centroid_velocity >= self.config.activity.walking_min_velocity)
            if state in (FallState.FALLING, FallState.FALLEN):
                activity, activity_reason = Activity.FALL, "Temporal fall state active"
            elif inactive_seconds >= self.config.activity.inactivity_seconds and posture.label != Posture.UNKNOWN:
                activity, activity_reason = Activity.INACTIVE, f"Stationary for {inactive_seconds:.0f}s"
            elif walking:
                activity, activity_reason = Activity.WALKING, "Standing posture with sustained translation"
            elif posture.label in (Posture.STANDING, Posture.SITTING, Posture.BENDING, Posture.LYING):
                activity, activity_reason = Activity(posture.label.value), posture.reason
            else:
                activity, activity_reason = Activity.UNKNOWN, posture.reason

            label, confidence, source = "Person detected", None, "pose"
            # Keep model confidence separate from the geometric fallback label.
            if posture.label == Posture.BENDING and geometry.torso_angle is not None:
                bend_text = f"Bending ({geometry.torso_angle:.0f}°)"
                if scene.label:
                    label, confidence, source = f"{bend_text} · {scene.label}", scene.confidence, "RelateAnything"
                else:
                    label, confidence, source = bend_text, None, "geometric"
            elif scene.label:
                label, confidence, source = scene.label, scene.confidence, "RelateAnything"
            elif activity == Activity.WALKING:
                label, confidence, source = "Walking", posture.confidence, posture.source
            elif appearance.posture is not None:
                label, confidence, source = appearance.posture.value.title(), appearance.posture_confidence, "RGB posture model"
            elif posture.label != Posture.UNKNOWN:
                label = posture.label.value.title()
            persons.append(TrackedPerson(
                track_id=track.id, bbox=track.pose.bbox.as_list(), detection_confidence=track.pose.detection_confidence,
                keypoints=track.pose.keypoints, posture=posture.label, posture_confidence=posture.confidence,
                posture_reason=posture.reason, posture_source=posture.source, fall_probability=fall.probability,
                fall_inference=fall, state=state, motion=motion, geometry=geometry, movement_state=movement,
                pose_coverage=geometry.confidence_coverage, appearance=appearance,
                history_seconds=timestamp-context.buffer.samples[0].timestamp, identity_uncertain=track.id in ambiguous,
                risk=risk, scene=scene, activity_label=label, activity_confidence=confidence, activity_source=source,
                activity=activity, activity_reason=activity_reason, inactive_seconds=inactive_seconds,
            ))
        end = perf_counter()
        total = (end-start)*1000
        fps = 1/(timestamp-self.last_timestamp) if self.last_timestamp is not None else 0
        self.first_timestamp = timestamp if self.first_timestamp is None else self.first_timestamp
        self.last_timestamp = timestamp
        return VisionObservation(
            timestamp=wall_timestamp or datetime.now(timezone.utc), elapsed_seconds=timestamp-self.first_timestamp,
            frame_id=frame_id, camera_id=self.camera_id, session_id=self.session_id, fps=fps,
            processing_latency_ms=total, frame_width=width, frame_height=height, persons=persons,
            detected_people=len(tracks),
            timings_ms={"pose": (after_pose-start)*1000, "tracking": (after_tracking-after_pose)*1000,
                        "features_and_state": max(0, (end-after_tracking)*1000-temporal_ms-rgb_ms),
                        "temporal": temporal_ms, "appearance": rgb_ms, "total": total},
        )
