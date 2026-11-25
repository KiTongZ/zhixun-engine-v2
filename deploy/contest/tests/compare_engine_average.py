"""Compare final-round parameter averaging on public engine data."""
import argparse
import copy
import hashlib
import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import swanlab

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contest_submission import Model, read_dataset
from engine_head import refine_head


def scores(model, data, mask):
    err = model.predict(data["x"][mask]) - data["y"][mask]
    groups = np.array([name.split("_")[0] for name in data["ids"]])[mask]
    values = {"rmse": np.sqrt(np.mean(err**2)), "mae": np.mean(abs(err)),
              "asymmetric_mean": np.expm1(np.maximum(err / 10, -err / 13)).mean(),
              "overestimate_p95": np.percentile(np.maximum(err, 0), 95)}
    for group in np.unique(groups):
        values[group] = np.sqrt(np.mean(err[groups == group]**2))
    return {k: float(v) for k, v in values.items()}


def compare(folder, split, run):
    model = Model.load(folder / "checkpoint.npz")
    train, val = [read_dataset(folder / "public" / (s + ".jsonl"), "regression", [], model.meta["paths"])
                  for s in ("train", "validation")]
    groups = [key.rsplit("_", 1)[0] for key in val["ids"]]
    held = np.array([int.from_bytes(hashlib.sha256(g.encode()).digest()[:4], "big") % 3 == 0 for g in groups])
    selection = ~held if split else np.ones(len(groups), dtype=bool)
    report = held if split else selection
    label = "held_engines" if split else "full_public"
    for rnd in range(1, 11):
        model.meta["learning_rate"] = .001 if rnd == 1 else .002 if rnd <= 5 else .004
        snapshots = [copy.deepcopy(model)]
        ranks = []
        for epoch in range(1 if rnd == 1 else 2):
            loss = model.fit(train["x"], train["y"], 1)[0]
            snapshots.append(copy.deepcopy(model))
            stats = scores(model, val, selection)
            run.log({f"{label}/training/{k}": v for k,v in dict(stats, loss=loss, round=rnd, epoch=epoch+1, lr=model.meta["learning_rate"]).items()})
            ranks.append(stats["rmse"])
        model = snapshots[1 + int(np.argmin(ranks))]
    smoothed = copy.deepcopy(snapshots[-1])
    for key in smoothed.params:
        smoothed.params[key] = np.mean([s.params[key] for s in snapshots], axis=0)
    smoothed.validate()
    candidates = [("baseline", model), ("average", smoothed)]
    for penalty in (.001, .01, .1):
        candidate = copy.deepcopy(model)
        refine_head(candidate, train["x"], train["y"], penalty)
        candidates.append((f"ridge_{penalty}", candidate))
    for name, candidate in candidates:
        result = scores(candidate, val, report)
        run.log({f"{label}/{name}/{k}": v for k,v in result.items()})
        print(label, name, len(set(np.array(groups)[report])), result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    run = swanlab.init(workspace="KiTongZ", project="zhixun-engine-v2", mode="online",
                       on_online_error="raise", experiment_name="C-final-head-ridge",
                       config={"head_penalties":[.001,.01,.1],"rates":[.001,.002,.004],"epochs":[1]+[2]*9,"seed":42,
                               "holdout":"sha256(engine) modulo 3 == 0","average":"parent plus every final-round epoch"})
    for split in (True, False):
        compare(args.package, split, run)
    run.finish()

