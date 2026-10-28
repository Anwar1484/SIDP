"""Risk scoring and tier classification (CLAUDE.md section 8.3).

    base      = posture_severity[posture]                 # 0 to 100, from config
    duration  = duration_weight * min(duration_s / duration_cap_s, 1)
    score     = min(100, (base + duration) * resident_multiplier)

Any posture in force_alert_postures is always tier "alert", whatever the score.
"""
from __future__ import annotations

from dataclasses import dataclass

from alert_engine.config import Config


@dataclass
class ScoreResult:
    risk_score: int
    tier: str
    reasons: list[str]


def score_posture(
    config: Config,
    posture: str,
    duration_s: float,
    person_id: str,
) -> ScoreResult:
    """Score a confirmed posture and classify it into a tier, with reasons."""
    scoring = config.scoring
    reasons: list[str] = []

    base = scoring.posture_severity.get(posture, 0.0)
    duration_component = scoring.duration_weight * min(duration_s / scoring.duration_cap_s, 1.0)
    multiplier = config.resident_multiplier(person_id)
    raw_score = (base + duration_component) * multiplier
    score = min(100.0, raw_score)

    reasons.append(f"{posture} base severity {base:.0f}")
    if duration_component > 0:
        reasons.append(f"{posture} sustained {duration_s:.1f} s (+{duration_component:.0f})")
    if multiplier != 1.0:
        reasons.append(f"resident risk multiplier {multiplier:g}")

    if posture in scoring.force_alert_postures:
        reasons.append(f"{posture} is a force-alert posture")
        return ScoreResult(risk_score=round(score), tier="alert", reasons=reasons)

    tier = config.tiers.classify(score)
    return ScoreResult(risk_score=round(score), tier=tier, reasons=reasons)
