"""Fit on public batch 1, select on earlier batch 2 rows, confirm on later rows."""
import copy
import json
import sys
import time
from pathlib import Path

from contest_submission import Model, read_dataset
import numpy as np

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data")
parent = Model.load(root / "checkpoint.npz")
classes = parent.meta["classes"]
ae, et = [classes.index(name) for name in ("Acetaldehyde", "Ethanol")]
train, valid = [read_dataset(root / "public" / f"{name}.jsonl", "classification", classes, parent.meta["paths"])
                for name in ("train", "validation")]
early = np.zeros(len(valid["y"]), dtype=bool)
for label in range(len(classes)):
    rows = np.flatnonzero(valid["y"] == label)
    early[rows[:len(rows)//2]] = True
pair = np.isin(train["y"], (ae, et))
y = train["y"][pair]
baseline = parent.predict(valid["x"]).argmax(axis=1)


def score(pred, mask):
    truth = valid["y"][mask]
    pred = pred[mask]
    n = len(classes)
    matrix = np.bincount(truth * n + pred, minlength=n * n).reshape(n, n)
    f1 = 2 * matrix.diagonal() / (matrix.sum(0) + matrix.sum(1)).clip(1)
    return {"macro_f1": float(f1.mean()), "acetaldehyde_f1": float(f1[ae]), "ethanol_f1": float(f1[et]),
            "acetaldehyde_recall": float(matrix[ae, ae] / matrix.sum(1)[ae]),
            "acetaldehyde_precision": float(matrix[ae, ae] / max(1, matrix.sum(0)[ae])),
            "acetaldehyde_correct": int(matrix[ae, ae]), "ethanol_false_positive": int(matrix[et, ae])}


def features(transform, source, target):
    base_scale = np.median(np.abs(source), axis=0).clip(1e-8)
    source, target = [np.arcsinh(x / base_scale) for x in (source, target)]
    if transform == "compressed":
        return source, target
    if transform == "row_direction":
        return tuple(x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8) for x in (source, target))
    if transform == "relative_features":
        return tuple(x - x.mean(axis=1, keepdims=True) for x in (source, target))
    return tuple((x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True).clip(1e-8)
                 for x in (source, target))


def distances(a, b):
    return np.maximum(0, (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T)


def candidates():
    for transform in ("compressed", "row_direction", "relative_features", "relative_standardized"):
        start = time.process_time()
        x, target = features(transform, train["x"][pair], valid["x"])
        train_dist, test_dist = distances(x, x), distances(target, x)
        scale = np.median(train_dist[train_dist > 1e-9])
        for width in (.1, .4, 1., 3.):
            k_train = np.exp(-train_dist / (width * scale))
            k_test = np.exp(-test_dist / (width * scale))
            for penalty in (.01, .1):
                # Give the two classes equal aggregate weight in the ridge loss.
                ridge = np.where(y == ae, (y == ae).mean(), (y == et).mean())
                coef = np.linalg.solve(k_train + np.diag(penalty * ridge), np.where(y == ae, 1., -1.))
                margin = k_test @ coef
                for threshold in (-.25, 0., .25, .5):
                    pred = baseline.copy()
                    eligible = (baseline == ae) | (baseline == et)
                    pred[eligible] = np.where(margin[eligible] > threshold, ae, et)
                    key = f"{transform}/width{width}/ridge{penalty}/threshold{threshold}"
                    yield key, pred, margin, time.process_time() - start


parent_metrics = score(baseline, early)
print(json.dumps(dict(kind="baseline", early=parent_metrics, held=score(baseline, ~early))))
choices = {}
for key, pred, margin, cpu in candidates():
    metric = score(pred, early)
    print(json.dumps(dict(kind="screen", key=key, cpu_s=cpu, **metric)))
    value = (metric["macro_f1"], metric["acetaldehyde_f1"])
    if metric["ethanol_f1"] >= parent_metrics["ethanol_f1"] - .02:
        goals = ["overall"]
        if metric["acetaldehyde_precision"] >= .85:
            goals.append("precision85")
        for goal in goals:
            if goal not in choices or value > choices[goal][0]:
                choices[goal] = (value, key, pred, margin)
for goal, (_, key, pred, _) in choices.items():
    print(json.dumps(dict(kind="selected", goal=goal, key=key, early=score(pred, early), held=score(pred, ~early))))

model = copy.deepcopy(parent)
for rnd in range(1, 11):
    incumb = -1.
    for _ in range(2):
        model.fit(train["x"], train["y"], 1)
        pred = model.predict(valid["x"]).argmax(axis=1)
        rank = score(pred, early)["macro_f1"]
        if rank > incumb:
            incumb, checkpoint = rank, copy.deepcopy(model)
    model = checkpoint
    pred = model.predict(valid["x"]).argmax(axis=1)
    print(json.dumps(dict(kind="round", goal="baseline", round=rnd, **score(pred, ~early))))
    for goal, (_, key, _, margin) in choices.items():
        threshold = float(key.rsplit("threshold", 1)[1])
        corrected = pred.copy()
        can_correct = (pred == ae) | (pred == et)
        corrected[can_correct] = np.where(margin[can_correct] > threshold, ae, et)
        print(json.dumps(dict(kind="round", goal=goal, round=rnd, **score(corrected, ~early))))
