"""Training-only resampling and response-amplitude augmentation for gas features."""
import numpy as np


def gas_training_rows(model, x, y, spread, balancing, seed_offset=0):
    rng = np.random.default_rng([model.meta["seed"] + seed_offset, model.meta["epochs"], 224])
    if balancing:
        counts = np.bincount(y)
        weights = counts[y].astype(float) ** (-balancing)
        indices = rng.choice(len(y), len(y), replace=True, p=weights/weights.sum())
        x, y = x[indices], y[indices]
    if spread:
        factor = np.exp(rng.uniform(-spread, spread, size=(len(y), 1)))
        x = x * factor
    return x, y

def dropout_gas_sensors(model, x, y, probability, mode, seed_offset=0):
    rng = np.random.default_rng([model.meta["seed"]+seed_offset,model.meta["epochs"],312])
    block = 8 if mode=="sensor" else 1
    mask = rng.random((len(x),x.shape[1]//block)) < probability
    return np.where(np.repeat(mask,block,axis=1),model.mean,x), y
