"""C snapshot calibration fitted to the current public training projection."""
import numpy as np


# Canonical flattening sorts sensors before settings.
ENGINE_PATHS = [
    f'[["key","{field}"],["index",{index}]]'
    for field, count in (("sensors", 21), ("settings", 3))
    for index in range(count)
]


def _design(x, conditions, center, scale):
    known, indices = np.nonzero(np.rint(x[:, 21, None]) == conditions)
    design = np.zeros((len(x), 21 + len(conditions)))
    design[known, :21] = np.clip((x[known, :21] - center[indices]) / scale[indices], -8.0, 8.0)
    design[known, 21 + indices] = 1.0
    return design


def fit_calibration(model, train):
    x, y = train["x"], train["y"]
    operating = np.rint(x[:, 21])
    conditions = np.unique(operating)
    if not np.isin(conditions, (0, 10, 20, 25, 35, 42)).all():
        return None
    center = np.stack([x[operating == value, :21].mean(axis=0) for value in conditions])
    scale = np.stack([x[operating == value, :21].std(axis=0) for value in conditions])
    scale[scale < 1e-8] = 1.0
    design = _design(x, conditions, center, scale)
    # Public engine identifiers are only used to weight training rows.
    groups = [identity.rsplit("_", 1)[0] for identity in train["ids"]]
    if any(group == identity or not identity.rsplit("_", 1)[-1].isdigit()
           for group, identity in zip(groups, train["ids"])):
        # A projection with opaque IDs is still usable without engine weights.
        weights = np.ones(len(y))
    else:
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        weights = 1.0 / np.sqrt(counts[inverse])
    weights *= 1 + (y >= 150)
    weighted = design.T * weights
    gram = weighted @ design
    gram.flat[::gram.shape[0] + 1] += .02 * weights.sum()
    coefficient = np.linalg.solve(gram, weighted @ (y - model.predict(x)))
    return {"conditions": conditions.tolist(), "center": center.tolist(),
            "scale": scale.tolist(), "coefficient": coefficient.tolist()}


class CalibratedModel:
    def __init__(self, parent, config):
        if parent.meta["task"] != "regression" or parent.meta["paths"] != ENGINE_PATHS:
            raise ValueError("engine calibration requires the C sensor schema")
        self.parent = parent
        self.meta = parent.meta
        self.conditions = np.asarray(config["conditions"], dtype=float)
        self.center = np.asarray(config["center"], dtype=float)
        self.scale = np.asarray(config["scale"], dtype=float)
        self.coefficient = np.asarray(config["coefficient"], dtype=float)
        count = len(self.conditions)
        if (self.conditions.shape != (count,) or not 0 < count <= 16
                or self.center.shape != (count, 21) or self.scale.shape != (count, 21)
                or self.coefficient.shape != (21 + count,)
                or len(np.unique(self.conditions)) != count or np.any(self.scale <= 0)
                or not all(np.isfinite(value).all() for value in
                           (self.conditions, self.center, self.scale, self.coefficient))):
            raise ValueError("invalid engine calibration")

    def predict(self, x):
        base = self.parent.predict(x)
        design = _design(x, self.conditions, self.center, self.scale)
        return np.maximum(0, base + design @ self.coefficient)


def validation_allows_calibration(model, corrected, validation):
    y = validation["y"]
    original = (model.predict(validation["x"]) - y) ** 2
    candidate = (corrected.predict(validation["x"]) - y) ** 2
    if float(candidate.mean()) >= float(original.mean()):
        return False
    return all(not mask.any() or float(candidate[mask].mean()) <= float(original[mask].mean())
               for mask in (y < 25, y >= 250))
