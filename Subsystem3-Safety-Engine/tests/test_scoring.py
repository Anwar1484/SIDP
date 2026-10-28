import pytest

from alert_engine.config import Config
from alert_engine.scoring import score_posture


@pytest.fixture
def config() -> Config:
    return Config.model_validate({
        "mqtt": {"host": "localhost", "port": 1883, "client_id": "alert-engine"},
        "postures": ["sitting", "standing", "walking", "bending", "lying", "fall", "unknown"],
        "debounce": {
            "window_s": 1.5, "min_confirmations": 3, "min_confidence": 0.6, "resolve_after_s": 3.0,
        },
        "scoring": {
            "posture_severity": {
                "sitting": 5, "standing": 5, "walking": 10, "bending": 40, "lying": 55, "fall": 95, "unknown": 0,
            },
            "duration_weight": 30,
            "duration_cap_s": 10,
            "force_alert_postures": ["fall"],
        },
        "tiers": {"safe": 0, "warning": 30, "unsafe": 60, "alert": 80},
        "resident_profiles": {"p-003": 1.2},
        "dispatch": {"cooldown_s": 30, "status_min_interval_s": 2},
        "watchdog": {"camera_timeout_s": 5},
        "audio": {"default_device": None, "messages": {}},
        "sink": {"type": "sqlite", "sqlite_path": "data/events.db"},
    })


def test_score_formula_at_zero_duration(config):
    result = score_posture(config, "bending", duration_s=0.0, person_id="p-999")
    assert result.risk_score == 40  # base only, multiplier 1.0
    assert result.tier == "warning"


def test_score_formula_with_duration(config):
    # base 40 + 30 * (5/10) = 55
    result = score_posture(config, "bending", duration_s=5.0, person_id="p-999")
    assert result.risk_score == 55
    assert result.tier == "warning"


def test_duration_is_capped(config):
    # base 40 + 30 * min(20/10, 1) = 70, not 40 + 30*2
    result = score_posture(config, "bending", duration_s=20.0, person_id="p-999")
    assert result.risk_score == 70
    assert result.tier == "unsafe"


def test_resident_multiplier_applied(config):
    # base 40 + 0 duration, * 1.2 multiplier = 48
    result = score_posture(config, "bending", duration_s=0.0, person_id="p-003")
    assert result.risk_score == 48


def test_unknown_resident_defaults_to_multiplier_one(config):
    result = score_posture(config, "bending", duration_s=0.0, person_id="unknown-person")
    assert result.risk_score == 40


def test_score_never_exceeds_100(config):
    # base 95 (lying not in force list) but let's push via a hypothetical high multiplier resident
    config.resident_profiles["p-heavy"] = 5.0
    result = score_posture(config, "lying", duration_s=10.0, person_id="p-heavy")
    assert result.risk_score == 100


def test_force_alert_posture_is_always_alert_tier(config):
    # fall has base severity 95, but even with score not maxed, force-alert makes it "alert"
    result = score_posture(config, "fall", duration_s=0.0, person_id="p-999")
    assert result.tier == "alert"
    assert any("force-alert" in reason for reason in result.reasons)


def test_force_alert_posture_alert_even_with_multiplier_zero_duration(config):
    result = score_posture(config, "fall", duration_s=0.0, person_id="unknown-person")
    assert result.tier == "alert"


@pytest.mark.parametrize("score,expected_tier", [
    (0, "safe"),
    (29, "safe"),
    (30, "warning"),
    (59, "warning"),
    (60, "unsafe"),
    (79, "unsafe"),
    (80, "alert"),
    (100, "alert"),
])
def test_tier_boundaries(config, score, expected_tier):
    assert config.tiers.classify(score) == expected_tier


def test_reasons_explain_the_score(config):
    result = score_posture(config, "bending", duration_s=6.4, person_id="p-003")
    assert any("bending" in r and "base" in r for r in result.reasons)
    assert any("sustained" in r for r in result.reasons)
    assert any("multiplier" in r for r in result.reasons)
