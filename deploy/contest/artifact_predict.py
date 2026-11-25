"""Standalone prediction entry shipped inside each model artifact."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines.data import flatten
from baselines.model import Model


def load_artifact(location):
    path = Path(location)
    checkpoint = path / "checkpoint.npz" if path.is_dir() else path
    model = Model.load(checkpoint)
    config = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    if config["task"] != model.meta["task"] or config["classes"] != model.meta["classes"]:
        raise ValueError("artifact configuration does not match checkpoint")
    return model


def predict_batch(model, samples):
    if not samples:
        return []
    rows = []
    for sample in samples:
        if sample.get("task", model.meta["task"]) != model.meta["task"]:
            raise ValueError("prediction task mismatch")
        paths, values = flatten(sample["inputs"])
        if paths != model.meta["paths"]:
            raise ValueError("prediction feature schema mismatch")
        rows.append(values)
    predictions = model.predict(np.asarray(rows, dtype=np.float64))
    if not np.all(np.isfinite(predictions)):
        raise ValueError("non-finite prediction")
    if model.meta["task"] == "regression":
        return [{"outputs": {"value": float(value)}} for value in predictions]
    results = []
    for probabilities in predictions:
        order = np.argsort(-probabilities, kind="stable")
        scores = probabilities[order].copy()
        scores[1:] = np.minimum(scores[1:], np.nextafter(scores[0], 0.0))
        results.append({"outputs": [{"label": model.meta["classes"][idx], "score": float(score)}
                                    for idx, score in zip(order, scores)]})
    return results


def predict(samples, model_dir):
    return predict_batch(load_artifact(model_dir), samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?")
    parser.add_argument("--model-dir", "--model_dir", "--model-path", "--model_path", "--checkpoint", dest="location")
    args = parser.parse_args()
    model = load_artifact(args.location or args.path or Path(__file__).resolve().parent)
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        if isinstance(request, list):
            result = predict_batch(model, request)
        elif "samples" in request:
            result = predict_batch(model, request["samples"])
        else:
            result = predict_batch(model, [request])[0]
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

