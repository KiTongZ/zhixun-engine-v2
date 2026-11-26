"""Three-stage A/B/C comparison with fixed budgets and whole-unit confirmation."""
import copy
import sys
import zlib
from pathlib import Path

from compare_subgroup_failures import confusion, emit, equal_unit_affine, error_metrics, f1, group_ids, load
from bearing_sampling import balanced_rows
from engine_calibration import CalibratedModel, fit_calibration
import numpy as np


def gas_study(parent, train, valid, phase):
    classes = parent.meta["classes"]
    ae = classes.index("Acetaldehyde")
    et = classes.index("Ethanol")
    counts = np.bincount(train["y"])
    probability = 1.0 / np.sqrt(counts[train["y"]])
    probability /= probability.sum()
    folds = np.empty(len(valid["y"]), dtype=int)
    for label in range(len(classes)):
        for split, indices in enumerate(np.array_split(np.flatnonzero(valid["y"] == label), 3)):
            folds[indices] = split
    policies = ["baseline", "mask_features", "balanced"]
    if phase != "single":
        policies.append("combined")
    for seed in ((43, 44, 45) if phase == "confirm" else (42,)):
        for policy in policies:
            model = copy.deepcopy(parent)
            model.meta.update(seed=seed, learning_rate=.003)
            for epoch in range(2):
                rows = slice(None)
                if policy in ("balanced", "combined"):
                    rng = np.random.default_rng([seed, epoch, 1])
                    rows = rng.choice(len(probability), len(probability), replace=True, p=probability)
                features = train["x"][rows]
                if policy in ("mask_features", "combined"):
                    rng = np.random.default_rng([seed, epoch, 2])
                    ignore = rng.random(features.shape) < .0625
                    features = np.where(ignore, model.mean, features)
                model.fit(features, train["y"][rows], 1)
            pred = model.predict(valid["x"]).argmax(axis=1)
            for fold in ((1, 2) if phase == "confirm" else (0,)):
                keep = folds == fold
                matrix = confusion(valid["y"][keep], pred[keep], len(classes))
                scores = f1(matrix)
                emit("gas", "A", policy, phase=phase, seed=seed, fold=fold, macro_f1=float(scores.mean()),
                     acetaldehyde_correct=int(matrix[ae, ae]), ethanol_to_acetaldehyde=int(matrix[et, ae]),
                     ethanol_f1=float(scores[et]))


def near_end_correction(model, train, valid, groups, estimate):
    # Fit the warning from snapshot features available at inference.
    # Balance physical units and the binary classes in the fit.
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    close = train["y"] < 25
    weights = 1.0 / counts[inverse]
    weights[close] *= weights[~close].sum() / weights[close].sum()
    weights /= weights.sum()
    x, _ = model.forward(model.normalized(train["x"]))
    target, _ = model.forward(model.normalized(valid["x"]))
    center = np.average(x, axis=0, weights=weights)
    x -= center
    gram = x.T @ (x * weights[:, None])
    gram.flat[::len(gram) + 1] += .1
    coefficient = np.linalg.solve(gram, x.T @ (weights * (2 * close - 1)))
    fit_score = x @ coefficient
    # Set the alarm cutoff on training labels only, to bound late alarms on healthy rows.
    threshold = np.quantile(fit_score[~close], .95)
    warn = (target - center) @ coefficient > threshold
    return np.where(warn, .85 * estimate, estimate)


def regression_study(parent, train, valid, dataset, phase):
    train_groups, _ = group_ids(dataset, train["ids"])
    validation_groups, condition = group_ids(dataset, valid["ids"])
    assert not (set(train_groups) & set(validation_groups))
    if dataset == "B":
        held = condition == "xjtu_Bearing3"
    else:
        held = np.array([zlib.crc32(group.encode()) % 3 == 0 for group in validation_groups])
    scope = held if phase == "confirm" else ~held
    for seed in ((43, 44, 45) if phase == "confirm" else (42,)):
        model = copy.deepcopy(parent)
        epochs, lr = {"B": (3, .006), "C": (1, .001)}[dataset]
        model.meta.update(seed=seed, learning_rate=lr)
        for _ in range(epochs):
            rows = balanced_rows(train["ids"], model.meta["epochs"], seed, 1) if dataset == "B" else slice(None)
            model.fit(train["x"][rows], train["y"][rows], 1)
        base = model.predict(valid["x"])
        warning = near_end_correction(model, train, valid, train_groups, base)
        estimates = {"baseline": base, "near_end_warning": warning}
        if dataset == "C":
            calibrated = CalibratedModel(model, fit_calibration(model, train)).apply(valid["x"], base)
            estimates["sensor_calibration"] = calibrated
            if phase != "single":
                # Add half of the calibration to the warning policy under the same forward budget.
                estimates["combined"] = np.maximum(0, warning + .5 * (calibrated - base))
        if dataset == "B":
            affine = equal_unit_affine(model.predict(train["x"]), train["y"], train_groups, base)
            estimates["equal_unit_affine"] = affine
            if phase != "single":
                estimates["combined"] = np.maximum(0, warning + (affine - base))
        for policy, prediction in estimates.items():
            error = prediction - valid["y"]
            slices = {"all": np.ones(len(error), dtype=bool), "near": valid["y"] < 25,
                      "long": valid["y"] >= 250}
            if phase == "confirm":
                slices.update({key: condition == key for key in np.unique(condition)})
            for name, mask in slices.items():
                mask = mask & scope
                if mask.any():
                    emit("life", dataset, policy, phase=phase, seed=seed, cut=name, n=int(mask.sum()),
                         **error_metrics(error[mask]))
            if phase == "confirm":
                units = [error[validation_groups == name] for name in np.unique(validation_groups[scope])]
                rmse = [error_metrics(unit)["rmse"] for unit in units]
                emit("unit_summary", dataset, policy, phase=phase, seed=seed,
                     unit_count=len(units), equal_unit_rmse=float(np.mean(rmse)), worst_unit_rmse=float(max(rmse)))


if __name__ == "__main__":
    phase = sys.argv[1]
    assert phase in ("single", "combine", "confirm")
    packages = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/packages")
    gas_study(*load(packages, "A"), phase)
    for dataset in ("B", "C"):
        regression_study(*load(packages, dataset), dataset, phase)
