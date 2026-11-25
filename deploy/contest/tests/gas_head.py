"""Estimate a shared-covariance gas classifier in the MLP's hidden features."""
import numpy as np


def blend_gas_head(model, x, y, weight, shrinkage):
    hidden, logits = model.forward(model.normalized(x))
    count = len(model.meta["classes"])
    if len(np.unique(y)) != count:
        return False
    centers = np.stack([hidden[y == label].mean(axis=0) for label in range(count)])
    residuals = hidden - centers[y]
    cov = residuals.T @ residuals / len(y)
    scale = max(np.trace(cov) / len(cov), 1e-8)
    cov *= 1 - shrinkage
    cov.flat[::len(cov) + 1] += shrinkage * scale
    slopes = np.linalg.solve(cov, centers.T)
    bias = -.5 * (centers * slopes.T).sum(axis=1)
    # Match the typical spread of logits so subsequent Adam steps stay comparable.
    linear = hidden @ slopes + bias
    factor = np.median(np.ptp(logits, axis=1)) / max(np.median(np.ptp(linear, axis=1)), 1e-8)
    model.params["w2"] = (1-weight)*model.params["w2"] + weight*factor*slopes
    model.params["b2"] = (1-weight)*model.params["b2"] + weight*factor*bias
    model.validate()
    return True
