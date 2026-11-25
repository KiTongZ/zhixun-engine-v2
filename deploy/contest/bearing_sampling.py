"""Deterministic per-bearing sampling for the public XJTU training data."""
import re
import numpy as np

def bearing_groups(ids):
    matches = [re.fullmatch(r"xjtu_(Bearing\d+_\d+)_s\d+", item) for item in ids]
    if not all(matches):
        return None
    return np.array([match[1] for match in matches])

def balanced_rows(ids, epoch, seed, power):
    groups = bearing_groups(ids)
    if groups is None or power == 0:
        return slice(None)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    probabilities = counts[inverse].astype(float) ** (-power)
    rng = np.random.default_rng([seed, epoch, 719])
    return rng.choice(len(ids), len(ids), replace=True, p=probabilities / probabilities.sum())
