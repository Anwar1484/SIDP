"""Tests for the learned risk scorer (no MQTT, audio or files outside tmp_path)."""
import numpy as np
import pytest

from alert_engine.config import load_config
from alert_engine.ml_data import generate_synthetic
from alert_engine.ml_scoring import HybridScorer, RiskModel, train


@pytest.fixture(scope="module")
def config():
    return load_config("config/config.yaml")


@pytest.fixture(scope="module")
def model(config):
    x, y = generate_synthetic(config.postures, n=3000, seed=1)
    return train(x, y, config.postures)


def test_falls_back_to_rules_without_model(config):
    scorer = HybridScorer(config, None)
    assert scorer.mode == "rules"
    assert scorer.score("walking", 1.0, "p-001").tier == "safe"


def test_model_accuracy_on_held_out_data(config, model):
    x, y = generate_synthetic(config.postures, n=1000, seed=2)
    assert (model.predict_tier_index(x) == y).mean() > 0.8


def test_model_separates_safe_from_long_lying(config, model):
    scorer = HybridScorer(config, model)
    assert scorer.mode == "model"
    safe = scorer.score("walking", 5.0, "p-001", confidence=0.9)
    risky = scorer.score("lying", 9.0, "p-001", confidence=0.9)
    assert safe.tier == "safe"
    assert risky.tier in ("unsafe", "alert")
    assert risky.risk_score > safe.risk_score


def test_fall_always_alert_even_if_model_disagrees(config, model):
    scorer = HybridScorer(config, model)
    assert scorer.score("fall", 0.0, "p-001", confidence=0.6).tier == "alert"


def test_reasons_include_model_and_rule_comparison(config, model):
    reasons = HybridScorer(config, model).score("bending", 6.0, "p-003", 0.9, 75.0).reasons
    assert any(r.startswith("model predicts") for r in reasons)
    assert any(r.startswith("rule-based check") for r in reasons)


def test_save_and_load_roundtrip(config, model, tmp_path):
    path = tmp_path / "m.npz"
    model.save(path)
    loaded = RiskModel.load(path)
    x, _ = generate_synthetic(config.postures, n=50, seed=3)
    assert np.allclose(model.predict_proba(x), loaded.predict_proba(x))


def test_mismatched_posture_list_uses_rules(config, model):
    model2 = RiskModel(["a"], model.weights, model.bias, model.mean, model.scale)
    assert HybridScorer(config, model2).mode == "rules"
