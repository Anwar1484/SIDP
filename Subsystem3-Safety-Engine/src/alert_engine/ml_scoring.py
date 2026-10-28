"""Learned risk scoring: a small softmax-regression model trained on labelled examples.

Runs alongside the rule-based scorer in scoring.py (which is untouched). The model
predicts a tier from posture, hold duration, detector confidence, bend angle and
resident risk. It uses numpy only, so there is no extra dependency and the weights
can be inspected to explain every alert.

HybridScorer is what main.py uses. It prefers the model, and falls back to the
rule-based scorer when no trained model file is available. The rule score is
always computed and recorded in `reasons` for comparison.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alert_engine.config import Config
from alert_engine.scoring import ScoreResult, score_posture

logger = logging.getLogger(__name__)

TIERS = ["safe", "warning", "unsafe", "alert"]
# Representative 0-100 score for each tier, used to turn probabilities into a score.
TIER_ANCHORS = np.array([10.0, 45.0, 70.0, 92.0])
MAX_ANGLE_DEG = 90.0


def feature_names(postures: list[str]) -> list[str]:
    """Names of the model inputs, in vector order."""
    names = [f"posture={p}" for p in postures]
    names += [f"{p} x duration" for p in postures]
    names += [f"{p} x duration^2" for p in postures]
    names += ["duration", "confidence", "bend_angle", "bend_angle_present", "resident_risk", "resident_risk x duration"]
    return names


def build_features(
    postures: list[str],
    posture: str,
    duration_s: float,
    confidence: float,
    bend_angle_deg: float | None,
    resident_multiplier: float,
    duration_cap_s: float = 10.0,
) -> np.ndarray:
    """Turn one confirmed posture into a numeric feature vector."""
    onehot = np.array([1.0 if p == posture else 0.0 for p in postures])
    dur = min(max(duration_s, 0.0) / duration_cap_s, 1.0)
    angle_present = 1.0 if bend_angle_deg is not None else 0.0
    angle = min(max(bend_angle_deg or 0.0, 0.0), MAX_ANGLE_DEG) / MAX_ANGLE_DEG
    risk = resident_multiplier - 1.0
    extras = np.array([dur, confidence, angle, angle_present, risk, risk * dur])
    return np.concatenate([onehot, onehot * dur, onehot * dur**2, extras])


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class RiskModel:
    """Multinomial logistic regression over the four tiers."""

    postures: list[str]
    weights: np.ndarray        # (n_features, 4)
    bias: np.ndarray           # (4,)
    mean: np.ndarray           # (n_features,)
    scale: np.ndarray          # (n_features,)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Tier probabilities for one vector or a batch."""
        z = ((x - self.mean) / self.scale) @ self.weights + self.bias
        return _softmax(z)

    def predict_tier_index(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=-1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, weights=self.weights, bias=self.bias, mean=self.mean, scale=self.scale,
            postures=np.array(self.postures),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RiskModel":
        data = np.load(path, allow_pickle=False)
        return cls(
            postures=[str(p) for p in data["postures"]],
            weights=data["weights"], bias=data["bias"], mean=data["mean"], scale=data["scale"],
        )


def train(
    x: np.ndarray,
    y: np.ndarray,
    postures: list[str],
    epochs: int = 600,
    lr: float = 0.5,
    l2: float = 1e-3,
) -> RiskModel:
    """Fit the model with batch gradient descent. `y` holds tier indices 0-3."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    xs = (x - mean) / scale
    n, d = xs.shape
    w = np.zeros((d, len(TIERS)))
    b = np.zeros(len(TIERS))
    targets = np.eye(len(TIERS))[y]
    for _ in range(epochs):
        p = _softmax(xs @ w + b)
        grad = (p - targets) / n
        w -= lr * (xs.T @ grad + l2 * w)
        b -= lr * grad.sum(axis=0)
    return RiskModel(postures=postures, weights=w, bias=b, mean=mean, scale=scale)


def explain(model: RiskModel, x: np.ndarray, tier_index: int, top: int = 3) -> list[str]:
    """Names of the features pushing hardest towards the predicted tier."""
    names = feature_names(model.postures)
    contrib = ((x - model.mean) / model.scale) * model.weights[:, tier_index]
    order = np.argsort(contrib)[::-1][:top]
    return [f"{names[i]} ({contrib[i]:+.2f})" for i in order if contrib[i] > 0.05]


class HybridScorer:
    """Scores with the learned model when available, otherwise with the rules.

    Safety rule kept from the rule-based design: postures in force_alert_postures
    are always tier "alert", whatever the model says.
    """

    def __init__(self, config: Config, model: RiskModel | None) -> None:
        self._config = config
        self._model = model
        if model is not None and model.postures != config.postures:
            logger.warning("Model trained on a different posture list than config; using rules instead.")
            self._model = None

    @classmethod
    def from_config(cls, config: Config) -> "HybridScorer":
        """Load the model named in config.ml if enabled and present."""
        model = None
        if config.ml.enabled:
            path = Path(config.ml.model_path)
            if path.exists():
                model = RiskModel.load(path)
                logger.info("Loaded risk model from %s", path)
            else:
                logger.warning("ML scoring enabled but %s not found; using rule-based scoring", path)
        return cls(config, model)

    @property
    def mode(self) -> str:
        return "model" if self._model is not None else "rules"

    def score(
        self,
        posture: str,
        duration_s: float,
        person_id: str,
        confidence: float = 1.0,
        bend_angle_deg: float | None = None,
    ) -> ScoreResult:
        """Score a confirmed posture; same result type as score_posture."""
        rule = score_posture(self._config, posture, duration_s, person_id)
        if self._model is None:
            return rule

        x = build_features(
            self._model.postures, posture, duration_s, confidence, bend_angle_deg,
            self._config.resident_multiplier(person_id), self._config.scoring.duration_cap_s,
        )
        proba = self._model.predict_proba(x)
        idx = int(proba.argmax())
        score = int(round(float(proba @ TIER_ANCHORS)))
        tier = TIERS[idx]

        reasons = [f"model predicts {tier} ({proba[idx]:.0%} confidence)"]
        reasons += [f"driver: {r}" for r in explain(self._model, x, idx)]
        if posture in self._config.scoring.force_alert_postures and tier != "alert":
            tier = "alert"
            score = max(score, int(TIER_ANCHORS[3]))
            reasons.append(f"{posture} is a force-alert posture")
        reasons.append(f"rule-based check: {rule.tier} ({rule.risk_score})")
        return ScoreResult(risk_score=min(score, 100), tier=tier, reasons=reasons)
