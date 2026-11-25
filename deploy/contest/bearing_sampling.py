"""Deterministic per-bearing sampling for the public XJTU training data."""
import re
import numpy as np

def bearing_groups(ids):
    matches = [re.fullmatch(r"xjtu_(Bearing\d+_\d+)_s\d+", item) for item in ids]
    if not all(matches):
        return None
    return np.array([match[1] for match in matches])

def balanced_rows(ids, epoch, seed, power, *, divisor=1):
    groups = bearing_groups(ids)
    size = max(1, round(len(ids) / divisor))
    rng = np.random.default_rng([seed, epoch, 719])
    if groups is None or power == 0:
        return slice(None) if divisor == 1 else rng.choice(len(ids), size, replace=False)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    probabilities = counts[inverse].astype(float) ** (-power)
    return rng.choice(len(ids), size, replace=True, p=probabilities / probabilities.sum())
