"""Compare a validation-triggered half step with the current continuation policy."""
import argparse
import copy
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import swanlab

CONTEST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTEST))
from contest_submission import Model, _metrics, balanced_rows, read_dataset, refine_head


def replay(parent, training, selection, report, dataset, seed, schedule, threshold, run, group):
    model = copy.deepcopy(parent)
    model.meta["seed"] = seed
    key = "macro_f1" if dataset == "A" else "rmse"
    direction = 1 if dataset == "A" else -1
    prefix = f"{dataset}/{group}/s{seed}/{'baseline' if threshold is None else f'half_{threshold:g}'}"
    times, trajectory = [], []
    damped = 0
    parent_score = direction * _metrics(model, selection)[key]
    for index, table in enumerate(schedule, 1):
        start = time.process_time()
        train = training[table]
        epochs, lr = {
            "A": (2, .003),
            "B": (3, .006) if index == 1 else (1, .0015),
            "C": (1, .001) if index == 1 else (2, .002 if index <= 5 else .004),
        }[dataset]
        model.meta["learning_rate"] = lr
        anchor = {name: array.copy() for name, array in model.params.items()} if threshold is not None else None
        best_score = -float("inf")
        for _ in range(epochs):
            rows = slice(None)
            if dataset == "B":
                rows = balanced_rows(train["ids"], model.meta["epochs"], seed, 1,
                                     divisor=32 if index > 1 else 1)
            model.fit(train["x"][rows], train["y"][rows], 1)
            score = direction * _metrics(model, selection)[key]
            if score > best_score:
                best_model, best_score = copy.deepcopy(model), score
        model = best_model
        if dataset == "C" and index == 10:
            proposal = copy.deepcopy(model)
            refine_head(proposal, train["x"], train["y"], .001)
            score = direction * _metrics(proposal, selection)[key]
            if score > best_score:
                model, best_score = proposal, score
        if threshold is not None and best_score < parent_score - threshold * abs(parent_score):
            proposal = copy.deepcopy(model)
            for name, value in proposal.params.items():
                value[:] = anchor[name] + .5 * (value - anchor[name])
            score = direction * _metrics(proposal, selection)[key]
            if score > best_score:
                model, best_score = proposal, score
                damped += 1
        parent_score = best_score
        times.append(time.process_time() - start)
        metrics = _metrics(model, report)
        trajectory.append(metrics[key])
        run.log({f"{prefix}/{k}": v for k, v in dict(round=index, damped=damped, **metrics).items()})
    stats = dict(average=float(np.mean(trajectory)),
                 worst=(min if direction == 1 else max)(trajectory),
                 last=trajectory[-1], seconds=sum(times), damped=damped)
    print(dataset, group, seed, prefix.rsplit("/", 1)[-1], stats, flush=True)
    run.log({f"{prefix}/summary_{k}": v for k, v in stats.items()})


def split(data, mask):
    return {"x": data["x"][mask], "y": data["y"][mask],
            "ids": [item for item, keep in zip(data["ids"], mask) if keep]}


def main(packages, datasets, thresholds, run):
    for dataset in datasets:
        package = next(packages.glob(f"cpu-{dataset.lower()}-*"))
        model = Model.load(package / "checkpoint.npz")
        task, classes, paths = (model.meta[k] for k in ("task", "classes", "paths"))
        train, valid = [read_dataset(package / "public" / f"{s}.jsonl", task, classes, paths)
                        for s in ("train", "validation")]
        if dataset == "A":
            held = np.zeros(len(valid["y"]), dtype=bool)
            for label in np.unique(valid["y"]):
                ids = np.flatnonzero(valid["y"] == label)
                held[ids[len(ids) // 2:]] = True
        else:
            separator = "_" if dataset == "C" else "_s"
            groups = [item.rsplit(separator, 1)[0] for item in valid["ids"]]
            held = np.array([int.from_bytes(hashlib.sha256(group.encode()).digest()[:4], "big") % 3 == 0
                             for group in groups])
        partitions = {"full": train}
        scenarios = {"public": (["full"] * 10, valid, valid)}
        if held.any() and (~held).any():
            select, test = split(valid, ~held), split(valid, held)
            scenarios["held"] = (["full"] * 10, select, test)
            if dataset == "C":
                groups = np.array([item.split("_")[0] for item in train["ids"]])
                conditions = sorted(set(groups))
                partitions.update({g: split(train, groups == g) for g in conditions})
                for label, order in (("shift_forward", conditions), ("shift_reverse", conditions[::-1])):
                    scenarios[label] = ([g for g in [*order, order[0]] for _ in range(2)], select, test)
        for group, (schedule, select, test) in scenarios.items():
            for seed in (42, 43, 44):
                for threshold in (None, *thresholds):
                    replay(model, partitions, select, test, dataset, seed, schedule, threshold, run, group)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", type=Path)
    parser.add_argument("--datasets", default="ABC")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0])
    args = parser.parse_args()
    with swanlab.init(workspace="KiTongZ", project="zhixun-engine-v2", mode="online", on_online_error="raise",
                      experiment_name="feature-cache-validation-triggered-half-steps",
                      config={"parent": "ba319e8", "rounds": 10, "fraction": .5,
                              "thresholds": args.thresholds}) as run:
        main(args.packages, args.datasets, args.thresholds, run)
