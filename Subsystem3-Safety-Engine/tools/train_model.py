"""Train the learned risk model and save it to the path in config.ml.model_path.

    python tools/train_model.py                      # synthetic placeholder data
    python tools/train_model.py --csv labelled.csv   # real labelled data from S2
"""
from __future__ import annotations

import argparse

import numpy as np

from alert_engine.config import load_config
from alert_engine.ml_data import generate_synthetic, load_csv
from alert_engine.ml_scoring import TIERS, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="labelled examples CSV (see ml_data.load_csv)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    postures = config.postures
    x, y = load_csv(args.csv, postures) if args.csv else generate_synthetic(postures, seed=args.seed)

    idx = np.random.default_rng(args.seed).permutation(len(x))
    cut = int(0.8 * len(x))
    tr, te = idx[:cut], idx[cut:]
    model = train(x[tr], y[tr], postures)

    pred = model.predict_tier_index(x[te])
    print(f"samples: {len(x)} (train {len(tr)}, test {len(te)})  source: {args.csv or 'synthetic'}")
    print(f"held-out accuracy: {(pred == y[te]).mean():.1%}")
    print("confusion matrix (rows = true, cols = predicted):")
    print("          " + " ".join(f"{t:>8}" for t in TIERS))
    for i, t in enumerate(TIERS):
        row = [int(((y[te] == i) & (pred == j)).sum()) for j in range(len(TIERS))]
        print(f"{t:>9} " + " ".join(f"{v:>8}" for v in row))

    model.save(config.ml.model_path)
    print(f"saved model to {config.ml.model_path}")


if __name__ == "__main__":
    main()
