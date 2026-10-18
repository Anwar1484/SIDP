from app.schemas import FallInference, FallState, MotionFeatures, Posture
from app.temporal.state_machine import FallStateMachine


def evidence(kind):
    if kind == "falling":
        return MotionFeatures(aspect_ratio=0.6, torso_angle=40, reliable=True, hip_vertical_velocity=0.6,
                              torso_angular_velocity=120, bbox_height_velocity=-0.7)
    return MotionFeatures(aspect_ratio=3 if kind == "lying" else 0.4,
                          torso_angle=90 if kind == "lying" else 0, reliable=True,
                          hip_vertical_velocity=0, torso_angular_velocity=0, bbox_height_velocity=0)


def inference(probability):
    return FallInference(probability=probability, model_name="test", window_frames=60,
                         valid_keypoint_ratio=1, calibrated=False, status="synthetic")


def test_static_lying_and_ordinary_sitting_do_not_declare_fall(config):
    for posture in (Posture.LYING, Posture.SITTING):
        machine = FallStateMachine(config.state)
        for i in range(80):
            assert machine.update(i/10, posture, evidence("lying" if posture == Posture.LYING else "upright"),
                                  inference(0.05)) == FallState.NORMAL


def test_sustained_fall_lying_persistence_and_recovery(config):
    machine = FallStateMachine(config.state)
    assert machine.update(0, Posture.UNKNOWN, evidence("falling"), inference(0.9)) == FallState.NORMAL
    visited = set()
    for i in range(1, 8):
        visited.add(machine.update(i/10, Posture.UNKNOWN, evidence("falling"), inference(0.9)))
    assert FallState.POSSIBLE_FALL in visited and FallState.FALLING in visited
    assert machine.update(0.8, Posture.LYING, evidence("lying"), inference(0.05)) == FallState.FALLING
    for i in range(9, 21):
        machine.update(i/10, Posture.LYING, evidence("lying"), inference(0.05))
    assert machine.state == FallState.FALLEN
    for i in range(21, 32):
        machine.update(i/10, Posture.STANDING, evidence("upright"), inference(0))
    assert machine.state == FallState.NORMAL


def test_single_frame_and_missing_evidence_cannot_complete_fall(config):
    machine = FallStateMachine(config.state)
    machine.update(0, Posture.UNKNOWN, evidence("falling"), inference(1))
    for i in range(1, 6):
        machine.update(i/10, Posture.UNKNOWN, MotionFeatures(aspect_ratio=1), inference(1))
    assert machine.state == FallState.NORMAL


def test_possible_fall_recovers_without_falling(config):
    machine = FallStateMachine(config.state)
    for i in range(4):
        machine.update(i/10, Posture.UNKNOWN, evidence("falling"), inference(0.9))
    assert machine.state == FallState.POSSIBLE_FALL
    for i in range(4, 15):
        machine.update(i/10, Posture.STANDING, evidence("upright"), inference(0))
    assert machine.state == FallState.NORMAL
