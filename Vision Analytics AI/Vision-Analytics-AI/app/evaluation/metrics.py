"""Missing predictions count as missed positives; calibration metrics report coverage."""
import numpy as np


def classification_metrics(labels, probabilities, threshold):
    y, p = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    prediction = np.isfinite(p) & (p >= threshold)
    tp, fp = int(np.sum(prediction & (y == 1))), int(np.sum(prediction & (y == 0)))
    fn, tn = int(np.sum(~prediction & (y == 1))), int(np.sum(~prediction & (y == 0)))
    precision, recall = tp/max(tp+fp, 1), tp/max(tp+fn, 1)
    return {"precision": precision, "recall": recall,
            "f1": 2*precision*recall/max(precision+recall, 1e-12), "fpr": fp/max(fp+tn, 1),
            "confusion_matrix": [[tn, fp], [fn, tp]], "n": len(y),
            "coverage": float(np.isfinite(p).mean()) if len(p) else 0}


def choose_threshold(labels, probabilities, max_fpr=.1):
    """Validation only: maximize recall with FPR <=10%, then precision, then threshold."""
    p = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.r_[0.0, p[np.isfinite(p)], 1.0])
    feasible = []
    for threshold in candidates:
        metrics = classification_metrics(labels, p, threshold)
        if metrics["fpr"] <= max_fpr + 1e-12:
            feasible.append((metrics["recall"], metrics["precision"], float(threshold), metrics))
    _, _, threshold, metrics = max(feasible, key=lambda item: item[:3])
    return threshold, metrics


def calibration_metrics(labels, probabilities, bins=10):
    y, p = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    available = np.isfinite(p)
    y, p = y[available], p[available]
    if not len(y):
        return {"n": 0, "brier": None, "ece": None, "bins": []}
    rows, ece = [], 0.0
    for i in range(bins):
        low, high = i/bins, (i+1)/bins
        mask = (p >= low) & ((p < high) if i < bins-1 else (p <= high))
        count = int(mask.sum())
        confidence = float(p[mask].mean()) if count else None
        frequency = float(y[mask].mean()) if count else None
        if count:
            ece += count/len(y) * abs(confidence-frequency)
        rows.append({"lower": low, "upper": high, "n": count,
                     "mean_probability": confidence, "event_frequency": frequency})
    return {"n": len(y), "brier": float(np.mean((p-y)**2)), "ece": ece, "bins": rows}
