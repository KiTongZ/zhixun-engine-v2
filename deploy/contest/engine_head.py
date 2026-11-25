"""Fit a residual correction to the engine regression head on training rows only."""
import numpy as np
import copy


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


def _validation_risk(model, x, y):
    error = model.predict(x) - y
    penalty = np.maximum(error / 10, -error / 13)
    peak = float(penalty.max())
    # log(1 + mean asymmetric penalty), stable for large errors.
    log_penalty = peak + np.log(np.exp(penalty - peak).mean())
    return float(np.sqrt(np.mean(error**2))), float(log_penalty)


def moderate_update(parent, candidate, x, y):
    """Compare positive update fractions using validation only for selection."""
    reference_rmse, reference_tail = _validation_risk(parent, x, y)
    best_rank = None
    for fraction in (1.0, .5, .125):
        proposal = candidate
        if fraction < 1:
            proposal = copy.deepcopy(candidate)
            for key in proposal.params:
                proposal.params[key] = parent.params[key] + fraction * (candidate.params[key] - parent.params[key])
        rmse, tail = _validation_risk(proposal, x, y)
        excess = max(0., rmse / max(reference_rmse, 1e-12) - 1.)
        excess += max(0., tail - reference_tail - np.log(1.05))
        rank = (excess, rmse)
        if best_rank is None or rank < best_rank:
            best_model, best_fraction, best_rank = proposal, fraction, rank
        if fraction == 1 and excess == 0:
            break
    return best_model, best_fraction
