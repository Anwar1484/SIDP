"""Explicit opt-in BiLSTM training. Split by video group BEFORE fitting statistics."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.posture.normalization import NORMALIZATION_VERSION  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("models/temporal"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    import tensorflow as tf
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GroupShuffleSplit
    meta = json.loads(args.dataset.with_suffix(".json").read_text(encoding="utf-8"))
    if meta["normalization"] != NORMALIZATION_VERSION or meta["features"] != 51:
        raise ValueError("unexpected dataset contract")
    data = np.load(args.dataset, allow_pickle=False)
    labeled = data["y"] >= 0
    x, y, groups = data["X"][labeled], data["y"][labeled], data["groups"][labeled]
    if len(set(groups)) < 5:
        raise ValueError("at least five labeled video groups required for separate train/validation/test splits")
    tf.keras.utils.set_random_seed(args.seed)
    trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed).split(x, y, groups))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed).split(
        x[trainval], y[trainval], groups[trainval]))
    train, val = trainval[tr], trainval[va]
    expected = set(range(len(meta["class_names"])))
    if set(y[train]) != expected:
        raise ValueError("training split lacks one or more classes; collect more labeled videos")
    mean, std = x[train].reshape(-1, 51).mean(axis=0), x[train].reshape(-1, 51).std(axis=0)
    std = np.where(std < 1e-6, 1, std)
    scaled = (x - mean) / std
    is_fall = meta["task"] == "fall"
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(meta["frames"], 51)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Dropout(0.3), tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1 if is_fall else 3, activation="sigmoid" if is_fall else "softmax"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy" if is_fall else "sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(scaled[train], y[train], validation_data=(scaled[val], y[val]),
              epochs=args.epochs, batch_size=args.batch_size,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    output = np.asarray(model.predict(scaled[test]))
    predicted = (output[:, 0] >= 0.5).astype(int) if is_fall else output.argmax(axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = meta["task"] + ".keras"
    model.save(args.output_dir / name)
    scaler = meta["task"] + "_stats.npz"
    np.savez(args.output_dir / scaler, mean=mean, std=std)
    manifest = {k: meta[k] for k in ("normalization", "pose_family", "features", "frames", "sample_hz", "task")}
    manifest.update(model=name, scaler=scaler, calibrated=False)
    (args.output_dir / (meta["task"] + "_manifest.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = classification_report(y[test], predicted, labels=sorted(expected),
                                   target_names=meta["class_names"], output_dict=True, zero_division=0)
    report["split_video_groups"] = {name: sorted(set(groups[idx])) for name, idx in
                                    (("train", train), ("validation", val), ("test", test))}
    (args.output_dir / (meta["task"] + "_report.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
