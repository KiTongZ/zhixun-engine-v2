"""Apply train-frequency logit adjustment to the final linear bias of a candidate."""
import numpy as np


def adjust_gas_bias(model, labels, strength):
    counts = np.bincount(labels, minlength=len(model.meta["classes"]))
    if counts.min() == 0:
        return False
    log_counts = np.log(counts)
    model.params["b2"] -= strength * (log_counts - log_counts.mean())
    return True
