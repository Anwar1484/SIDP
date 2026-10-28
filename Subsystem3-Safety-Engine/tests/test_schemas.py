import pytest
from pydantic import ValidationError

from alert_engine.schemas import PostureEventIn, PostureRejectedError, validate_posture

VALID_PAYLOAD = {
    "event_id": "11111111-1111-1111-1111-111111111111",
    "timestamp": "2026-10-05T10:42:01.123Z",
    "camera_id": "cam-01",
    "person_id": "p-003",
    "posture": "bending",
    "confidence": 0.87,
    "bend_angle_deg": 68.0,
    "zone": "lounge",
}

ALLOWED_POSTURES = ["sitting", "standing", "walking", "bending", "lying", "fall", "unknown"]


def test_valid_event_parses():
    event = PostureEventIn.model_validate(VALID_PAYLOAD)
    assert event.person_id == "p-003"
    assert event.posture == "bending"
    assert event.confidence == 0.87


def test_optional_fields_default_to_none():
    payload = dict(VALID_PAYLOAD)
    del payload["bend_angle_deg"]
    del payload["zone"]
    event = PostureEventIn.model_validate(payload)
    assert event.bend_angle_deg is None
    assert event.zone is None


@pytest.mark.parametrize("field,bad_value", [
    ("confidence", 1.5),
    ("confidence", -0.1),
    ("timestamp", "not-a-timestamp"),
])
def test_invalid_field_rejected(field, bad_value):
    payload = dict(VALID_PAYLOAD)
    payload[field] = bad_value
    with pytest.raises(ValidationError):
        PostureEventIn.model_validate(payload)


def test_missing_required_field_rejected():
    payload = dict(VALID_PAYLOAD)
    del payload["person_id"]
    with pytest.raises(ValidationError):
        PostureEventIn.model_validate(payload)


def test_posture_in_allowed_list_passes():
    event = PostureEventIn.model_validate(VALID_PAYLOAD)
    validate_posture(event, ALLOWED_POSTURES)  # should not raise


def test_posture_not_in_allowed_list_rejected():
    payload = dict(VALID_PAYLOAD)
    payload["posture"] = "cartwheeling"
    event = PostureEventIn.model_validate(payload)
    with pytest.raises(PostureRejectedError):
        validate_posture(event, ALLOWED_POSTURES)
