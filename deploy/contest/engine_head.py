"""Fit a residual correction to the engine regression head on training rows only."""
import numpy as np


def refine_head(model, x, y, penalty):
    hidden, logits = model.forward(model.normalized(x))
    design = np.column_stack((hidden, np.ones(len(hidden))))
    target = (y - model.meta["target_mean"]) / model.meta["target_scale"]
    gram = design.T @ design
    gram.flat[::gram.shape[0] + 1] += penalty * len(x)
    delta = np.linalg.solve(gram, design.T @ (target - logits[:, 0]))
    model.params["w2"][:, 0] += delta[:-1]
    model.params["b2"][0] += delta[-1]
    model.validate()
