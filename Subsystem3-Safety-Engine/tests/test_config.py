from alert_engine.config import load_config


def test_config_loads_and_validates():
    config = load_config("config/config.yaml")
    assert config.mqtt.host == "localhost"
    assert config.mqtt.port == 1883
    assert "bending" in config.postures
    assert config.scoring.posture_severity["fall"] == 95
    assert config.resident_multiplier("p-003") == 1.2
    assert config.resident_multiplier("p-999") == 1.0


def test_tiers_classify_by_lower_bound():
    config = load_config("config/config.yaml")
    assert config.tiers.classify(0) == "safe"
    assert config.tiers.classify(29) == "safe"
    assert config.tiers.classify(30) == "warning"
    assert config.tiers.classify(60) == "unsafe"
    assert config.tiers.classify(80) == "alert"
    assert config.tiers.classify(100) == "alert"
