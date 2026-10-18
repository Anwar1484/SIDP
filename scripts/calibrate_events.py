"""Fit train/calibration/validation videos, freeze artifact, then evaluate held-out test once."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from app.evaluation.metrics import calibration_metrics, choose_threshold, classification_metrics
from app.inference.event_classifier import sigmoid
from app.runtime import ROOT
from app.temporal.event_features import FEATURE_NAMES

VARIANTS = {"static_pose": list(range(9)), "temporal_pose": list(range(19)),
            "rgb": [19, 20], "pose_rgb": list(range(9)) + [19, 20],
            "temporal_pose_rgb": list(range(21))}


def dataset(rows, sequences):
    selected = [r for r in rows if r["sequence"] in sequences]
    features = np.array([[np.nan if v is None else v for v in r["features"]]
                         if r["features"] is not None else [np.nan]*len(FEATURE_NAMES) for r in selected])
    available = np.array([r["features"] is not None for r in selected])
    return selected, features, np.array([r["target"] for r in selected]), available


def predict(model, data):
    _, x, _, available = data
    x = x[:, model["indices"]]
    x = np.where(np.isfinite(x), x, model["medians"])
    logits = (x-model["mean"])/model["scale"] @ np.asarray(model["coef"]) + model["intercept"]
    raw = sigmoid(logits)
    calibrated = sigmoid(model["platt_coef"]*logits+model["platt_intercept"])
    raw[~available] = calibrated[~available] = np.nan
    return raw, calibrated


def score(model, data):
    rows, _, labels, available = data
    raw, p = predict(model, data)
    result = classification_metrics(labels, p, model["threshold"])
    result["raw_calibration"] = calibration_metrics(labels, raw)
    result["calibration"] = calibration_metrics(labels, p)
    if len(np.unique(labels[available])) == 2:
        result["auroc_available"] = float(roc_auc_score(labels[available], p[available]))
        result["auprc_available"] = float(average_precision_score(labels[available], p[available]))
    clips = []
    for seq in sorted({r["sequence"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["sequence"] == seq]
        high_since, previous_time, previous_track, fired = None, None, None, []
        for i in indices:
            t, track = rows[i]["time"], rows[i]["track"]
            if (previous_time is not None and t-previous_time > .5) or track != previous_track:
                high_since = None
            if np.isfinite(p[i]) and p[i] >= model["threshold"]:
                high_since = t if high_since is None else high_since
                if t-high_since >= .4:
                    fired.append(i)
            else:
                high_since = None
            previous_time, previous_track = t, track
        positive = any(labels[i] for i in indices)
        correct = [i for i in fired if labels[i] == 1]
        onset = min((rows[i]["time"] for i in indices if labels[i]), default=None)
        clips.append({"sequence": seq, "target": int(positive), "detected": bool(correct) if positive else bool(fired),
                      "false_evidence_frames": sum(labels[i] == 0 for i in fired),
                      "delay_seconds": rows[correct[0]]["time"]-onset if correct else None})
    result["clips"] = clips
    result["event_metrics"] = classification_metrics([c["target"] for c in clips],
                                                      [float(c["detected"]) for c in clips], .5)
    return result


def fit(rows, splits, metadata, destination):
    sets = {key: dataset(rows, values) for key, values in splits.items() if key != "test"}
    train, calibration, validation = (sets[k] for k in ("train", "calibration", "validation"))
    artifact = {"format": 1, "feature_names": FEATURE_NAMES, "selected": None, "models": {},
                "splits": splits, "extraction": metadata,
                "threshold_policy": "Validation only: maximize frame recall subject to FPR <=0.10; "
                                    "ties maximize precision, then threshold. Missing evidence predicts no event.",
                "target": "current fall or aftermath (URFD fall clips label >=0); all ADLs negative",
                "limitations": ["staged adult falls", "video-wise, not subject-wise", "320x240 preview RGB",
                                "unknown RGB pretraining overlap", "not a pre-fall predictor"]}
    for name, indices in VARIANTS.items():
        _, xt, yt, at = train
        _, xc, yc, ac = calibration
        xt, yt = xt[at][:, indices], yt[at]
        medians = np.nanmedian(xt, axis=0)
        if not np.isfinite(medians).all():
            raise ValueError(f"No finite training observations for {name}")
        xt = np.where(np.isfinite(xt), xt, medians)
        scaler = StandardScaler().fit(xt)
        estimator = LogisticRegression(C=1.0, max_iter=3000, random_state=20261002).fit(scaler.transform(xt), yt)
        xc = xc[ac][:, indices]
        xc = np.where(np.isfinite(xc), xc, medians)
        calibration_logits = estimator.decision_function(scaler.transform(xc))[:, None]
        platt = LogisticRegression(C=1000.0, max_iter=3000).fit(calibration_logits, yc[ac])
        model = {"indices": indices, "medians": medians.tolist(), "mean": scaler.mean_.tolist(),
                 "scale": scaler.scale_.tolist(), "coef": estimator.coef_[0].tolist(),
                 "intercept": float(estimator.intercept_[0]), "platt_coef": float(platt.coef_[0, 0]),
                 "platt_intercept": float(platt.intercept_[0]), "threshold": 1.0,
                 "n_train": len(yt), "n_calibration": int(ac.sum())}
        _, pv = predict(model, validation)
        model["threshold"], model["validation_operating_point"] = choose_threshold(validation[2], pv)
        model["validation"] = score(model, validation)
        artifact["models"][name] = model
    # Architecture selection sees validation only. Prefer simpler models on exact ties.
    artifact["selected"] = max(VARIANTS, key=lambda name: (
        artifact["models"][name]["validation"]["recall"],
        artifact["models"][name]["validation"]["precision"],
        -artifact["models"][name]["validation"]["calibration"]["brier"], -len(VARIANTS[name])))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def evaluate(rows, artifact, path, output):
    test = dataset(rows, artifact["splits"]["test"])
    report = {"artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "selected_before_test": artifact["selected"], "splits": artifact["splits"],
              "target": artifact["target"], "threshold_policy": artifact["threshold_policy"],
              "test": {name: score(model, test) for name, model in artifact["models"].items()}}
    output.mkdir(parents=True, exist_ok=True)
    (output/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = report["test"][artifact["selected"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal calibration")
    for key, label in (("raw_calibration", "Before Platt mapping"), ("calibration", "After Platt mapping")):
        bins = [b for b in result[key]["bins"] if b["n"]]
        ax.plot([b["mean_probability"] for b in bins], [b["event_frequency"] for b in bins], "o-", label=label)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed event frequency", xlim=(0, 1), ylim=(0, 1),
           title="URFD held-out videos · available windows only")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/"reliability.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"selected": artifact["selected"], "threshold": artifact["models"][artifact["selected"]]["threshold"],
                      "test": {k: v for k, v in result.items() if k not in ("clips", "raw_calibration", "calibration")},
                      "brier": result["calibration"]["brier"], "ece": result["calibration"]["ece"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT/"outputs/evaluation/features.jsonl")
    parser.add_argument("--artifact", type=Path, default=ROOT/"models/event_classifier.json")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/evaluation")
    parser.add_argument("--fit", action="store_true", help="Fit only train/calibration/validation; never read test labels")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate frozen artifact without fitting anything")
    args = parser.parse_args()
    if args.fit == args.evaluate:
        raise SystemExit("Choose exactly one of --fit or --evaluate")
    rows = [json.loads(line) for line in args.features.read_text(encoding="utf-8").splitlines()]
    if args.fit:
        splits = json.loads((ROOT/"configs/evaluation_split.json").read_text())["splits"]
        all_ids = [seq for values in splits.values() for seq in values]
        if len(all_ids) != len(set(all_ids)) or set(all_ids) != {r["sequence"] for r in rows}:
            raise ValueError("Splits overlap or dataset sequences are missing")
        metadata = json.loads(args.features.with_suffix(".metadata.json").read_text())
        train_rows = [row for row in rows if row["sequence"] not in splits["test"]]
        artifact = fit(train_rows, splits, metadata, args.artifact)
        print(f"Frozen model: {artifact['selected']}. Test videos have not been used for selection.")
    else:
        artifact = json.loads(args.artifact.read_text())
        if hashlib.sha256(args.features.read_bytes()).hexdigest() != artifact["extraction"]["sha256"]:
            raise ValueError("Features differ from the artifact's recorded dataset")
        evaluate(rows, artifact, args.artifact, args.output)


if __name__ == "__main__":
    main()
