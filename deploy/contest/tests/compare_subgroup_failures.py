"""Screen fixed subgroup mitigations on public train with untouched validation groups."""
import copy
import json
import sys
from pathlib import Path

from contest_submission import Model, read_dataset
from bearing_sampling import balanced_rows
from engine_calibration import CalibratedModel, fit_calibration
import numpy as np


def emit(kind, dataset, name, **values):
    print(json.dumps(dict(kind=kind, dataset=dataset, name=name, **values)), flush=True)


def load(root, dataset):
    folder = next(root.glob(f"cpu-{dataset.lower()}-*"))
    model = Model.load(folder / "checkpoint.npz")
    train, valid = [read_dataset(folder / "public" / f"{name}.jsonl", model.meta["task"],
                                 model.meta["classes"], model.meta["paths"])
                    for name in ("train", "validation")]
    return model, train, valid


def confusion(y, pred, count):
    return np.bincount(y * count + pred, minlength=count * count).reshape(count, count)


def f1(matrix):
    return 2 * matrix.diagonal() / (matrix.sum(0) + matrix.sum(1)).clip(1)


def audit_gases(parent, train, valid):
    classes = parent.meta["classes"]
    count = len(classes)
    support = np.bincount(train["y"], minlength=count)
    probability = 1 / np.sqrt(support[train["y"]])
    probability /= probability.sum()
    for index, name in enumerate(classes):
        emit("support", "A", name, train=int(support[index]), valid=int(np.sum(valid["y"] == index)))
    blocks = np.zeros(len(valid["y"]), dtype=int)
    for index in range(count):
        for block, rows in enumerate(np.array_split(np.flatnonzero(valid["y"] == index), 3)):
            blocks[rows] = block
    for seed in (42, 43, 44):
        for policy in ("baseline", "sqrt_class_balance"):
            model = copy.deepcopy(parent)
            model.meta.update(seed=seed, learning_rate=.003)
            rng = np.random.default_rng(seed)
            for _ in range(2):
                rows = slice(None)
                if policy != "baseline":
                    rows = rng.choice(len(probability), len(probability), replace=True, p=probability)
                model.fit(train["x"][rows], train["y"][rows], 1)
            predictions = model.predict(valid["x"]).argmax(axis=1)
            matrix = confusion(valid["y"], predictions, count)
            for index, name in enumerate(classes):
                mistakes = matrix[index].copy()
                mistakes[index] = 0
                rival = int(mistakes.argmax())
                emit("gas_class", "A", name, policy=policy, seed=seed,
                     n=int(matrix[index].sum()), correct=int(matrix[index, index]),
                     f1=float(f1(matrix)[index]), confused_with=classes[rival], confused_n=int(mistakes[rival]))
            for block in range(3):
                mask = blocks == block
                emit("gas_block", "A", str(block + 1), policy=policy, seed=seed,
                     macro_f1=float(f1(confusion(valid["y"][mask], predictions[mask], count)).mean()))


def group_ids(dataset, ids):
    groups = [identity.rsplit("_", 1)[0] for identity in ids]
    condition = [group.rsplit("_", 1)[0] if dataset == "B" else group.split("_", 1)[0]
                 for group in groups]
    return np.array(groups), np.array(condition)


def error_metrics(error):
    output = {"rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(np.mean(error)),
              "over15": float(np.mean(error > 15))}
    return output


def equal_unit_affine(fit_prediction, y, groups, prediction):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    center = np.average(fit_prediction, weights=weights)
    residuals = y - fit_prediction
    offset = np.average(residuals, weights=weights)
    spread = np.average((fit_prediction - center) ** 2, weights=weights)
    slope = np.average((fit_prediction - center) * residuals, weights=weights) / max(1e-8, spread)
    # Limit an extrapolated correction at both ends of the life range.
    correction = np.clip(.5 * slope, -.15, .15) * (prediction - center)
    cap = .1 * np.median(y)
    return np.maximum(0, prediction + np.clip(correction + .5 * offset, -cap, cap))


def audit_life(parent, train, valid, dataset):
    fit_groups, fit_condition = group_ids(dataset, train["ids"])
    test_groups, test_condition = group_ids(dataset, valid["ids"])
    assert not (set(fit_groups) & set(test_groups)), f"{dataset} overlaps between fit and validation"
    epochs, lr = {"B": (3, .006), "C": (1, .001)}[dataset]
    parent.meta["learning_rate"] = lr
    for _ in range(epochs):
        rows = (balanced_rows(train["ids"], parent.meta["epochs"], parent.meta["seed"], 1)
                if dataset == "B" else slice(None))
        parent.fit(train["x"][rows], train["y"][rows], 1)
    fit_prediction = parent.predict(train["x"])
    prediction = parent.predict(valid["x"])
    corrected = prediction.copy()
    residuals = fit_prediction - train["y"]
    # Independent units get equal influence. Estimate only one additive term per operating condition.
    for condition in np.unique(fit_condition):
        within = fit_condition == condition
        unit_bias = [np.mean(residuals[fit_groups == group]) for group in np.unique(fit_groups[within])]
        shift = -0.5 * np.median(unit_bias) if len(unit_bias) >= 3 else 0.0
        on_target = test_condition == condition
        corrected[on_target] = np.maximum(0, prediction[on_target] + shift)
        emit("fit", dataset, condition, units=len(unit_bias), shift=float(shift))
    alternatives = {
        "baseline": prediction,
        "group_bias_shrink": corrected,
        "equal_unit_affine": equal_unit_affine(fit_prediction, train["y"], fit_groups, prediction),
    }
    if dataset == "C":
        calibration = fit_calibration(parent, train)
        if calibration is not None:
            alternatives["sensor_calibration"] = CalibratedModel(parent, calibration).apply(valid["x"], prediction)
    for policy, estimates in alternatives.items():
        error = estimates - valid["y"]
        for name in np.unique(test_groups):
            mask = test_groups == name
            emit("unit", dataset, name, policy=policy, condition=test_condition[mask][0],
                 n=int(mask.sum()), **error_metrics(error[mask]))
        for condition in np.unique(test_condition):
            mask = test_condition == condition
            emit("condition", dataset, condition, policy=policy, n=int(mask.sum()), **error_metrics(error[mask]))
        for name, mask in (("near_end", valid["y"] < 25), ("long_life", valid["y"] >= 250),
                           ("all", np.ones(len(error), dtype=bool))):
            if mask.any():
                emit("life_bucket", dataset, name, policy=policy, n=int(mask.sum()), **error_metrics(error[mask]))


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/packages")
    audit_gases(*load(root, "A"))
    for dataset in ("B", "C"):
        audit_life(*load(root, dataset), dataset)
