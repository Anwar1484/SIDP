"""Labelled training examples for the risk model.

generate_synthetic() makes placeholder data from care-home style reasoning about
how dangerous each situation is. It is written independently of the scoring
formula in config, with noise added. Replace or extend it with real labelled
events from S2 using load_csv() when they are available.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from alert_engine.ml_scoring import TIERS, build_features

SAFE = {"sitting", "standing", "walking"}


def label_situation(posture: str, duration_s: float, confidence: float,
                    bend_angle_deg: float | None, resident_multiplier: float) -> int:
    """Ground-truth tier index (0 safe .. 3 alert) for one situation."""
    if posture in SAFE or posture == "unknown":
        return 0
    if posture == "fall":
        return 3
    level = 0
    if posture == "bending":
        level = 0 if duration_s < 2 else 1 if duration_s < 5 else 2
        if bend_angle_deg is not None and bend_angle_deg >= 70 and duration_s >= 2:
            level += 1
    elif posture == "lying":
        level = 1 if duration_s < 3 else 2 if duration_s < 8 else 3
    if resident_multiplier > 1.1 and duration_s >= 2:
        level += 1
    if confidence < 0.7:
        level -= 1
    return int(min(max(level, 0), 3))


def generate_synthetic(postures: list[str], n: int = 6000, seed: int = 0,
                       label_noise: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """Random situations with labels and a little label noise. Returns (X, y)."""
    rng = np.random.default_rng(seed)
    probs = {"sitting": 0.2, "standing": 0.15, "walking": 0.2, "bending": 0.2,
             "lying": 0.15, "fall": 0.05, "unknown": 0.05}
    weights = np.array([probs.get(p, 0.05) for p in postures])
    weights /= weights.sum()
    xs, ys = [], []
    for _ in range(n):
        posture = postures[rng.choice(len(postures), p=weights)]
        duration = float(rng.uniform(0, 14))
        confidence = float(rng.uniform(0.6, 1.0))
        angle = float(rng.uniform(10, 90)) if posture in ("bending", "lying") and rng.random() < 0.8 else None
        mult = float(rng.choice([1.0, 1.0, 1.0, 1.2, 1.5]))
        y = label_situation(posture, duration, confidence, angle, mult)
        if rng.random() < label_noise:
            y = int(np.clip(y + rng.choice([-1, 1]), 0, 3))
        xs.append(build_features(postures, posture, duration, confidence, angle, mult))
        ys.append(y)
    return np.array(xs), np.array(ys)


def load_csv(path: str | Path, postures: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load real labelled examples.

    Columns: posture, duration_s, confidence, bend_angle_deg (may be blank),
    resident_multiplier, tier (safe/warning/unsafe/alert).
    """
    xs, ys = [], []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            angle = float(row["bend_angle_deg"]) if row.get("bend_angle_deg") else None
            xs.append(build_features(
                postures, row["posture"], float(row["duration_s"]), float(row["confidence"]),
                angle, float(row.get("resident_multiplier") or 1.0),
            ))
            ys.append(TIERS.index(row["tier"]))
    return np.array(xs), np.array(ys)
