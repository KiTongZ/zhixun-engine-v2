"""Compare B training policies across whole-device public splits."""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import swanlab

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contest_submission import Model, read_dataset
from bearing_sampling import balanced_rows, bearing_groups

PLANS = [(0, 0), (.01, .25), (.01, .5), (.01, 1), (.1, .25), (.1, .5), (.1, 1), (1, .5)]
SAMPLING_PLANS = [(32, 0), (32, 1), (64, 0), (64, 1)]
TAPER_PLANS = [(32, 10), (64, 10), (64, 4), (64, 7), (64, 9)]


def stratified_rows(ids, targets, epoch, seed, divisor):
    size = max(1, round(len(ids) / divisor))
    groups = bearing_groups(ids)
    rng = np.random.default_rng([seed, epoch, 719])
    if groups is None:
        order = np.argsort(targets, kind="stable")
        cdf = np.arange(1, len(ids) + 1, dtype=float) / len(ids)
    else:
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        order = np.lexsort((targets, inverse))
        cdf = np.cumsum(1 / counts[inverse[order]])
        cdf /= cdf[-1]
    positions = (np.arange(size) + rng.random(size)) / size
    return order[np.searchsorted(cdf, positions, side="right")]


def refine_head(model, x, y, penalty, weights, fraction):
    hidden, logits = model.forward(model.normalized(x))
    design = np.column_stack((hidden, np.ones(len(hidden))))
    target = (y - model.meta["target_mean"]) / model.meta["target_scale"]
    weighted_design = weights[:, None] * design
    gram = weighted_design.T @ design
    gram.flat[::gram.shape[0] + 1] += penalty * weights.sum()
    delta = np.linalg.solve(gram, weighted_design.T @ (target - logits[:, 0]))
    model.params["w2"][:, 0] += fraction * delta[:-1]
    model.params["b2"][0] += fraction * delta[-1]
    model.validate()


def replay(parent, x, y, ids, vx, vy, select, report, key, seed, mode, run):
    groups = bearing_groups(ids)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    plans = {"head": PLANS, "sampling": SAMPLING_PLANS, "taper": TAPER_PLANS}
    for plan in plans[mode]:
        penalty, fraction = plan if mode == "head" else (0, 0)
        divisor, stratified = plan if mode == "sampling" else (32, False)
        until = 10
        if mode == "taper":
            divisor, until = plan
        model = copy.deepcopy(parent)
        model.meta["seed"] = seed
        prefix = f"{key}/seed{seed}/{mode}_{plan[0]:g}_{plan[1]:g}"
        trajectory = []
        accepted = False
        first = time.process_time()
        for index in range(1, 11):
            model.meta["learning_rate"] = .006 if index == 1 else .0015
            best_mse = float("inf")
            for _ in range(3 if index == 1 else 1):
                epoch = model.meta["epochs"]
                if index > 1 and stratified:
                    rows = stratified_rows(ids, y, epoch, seed, divisor)
                else:
                    current_divisor = divisor if index <= until else 32
                    rows = balanced_rows(ids, epoch, seed, 1, divisor=1 if index == 1 else current_divisor)
                model.fit(x[rows], y[rows], 1)
                error = (model.predict(vx) - vy) ** 2
                mse = error[select].mean()
                if mse < best_mse:
                    best_mse, best_error, best_model = mse, error, copy.deepcopy(model)
            model = best_model
            if index == 1 and fraction:
                refined = copy.deepcopy(model)
                refine_head(refined, x, y, penalty, weights=weights, fraction=fraction)
                error = (refined.predict(vx) - vy) ** 2
                if error[select].mean() < best_mse:
                    model, best_error = refined, error
                    accepted = True
            rmse = float(best_error[report].mean() ** .5)
            trajectory.append(rmse)
            run.log({prefix+"/round":index, prefix+"/rmse":rmse, prefix+"/used_head":int(accepted)})
        summary = {"mean":float(np.mean(trajectory)), "worst":max(trajectory), "final":trajectory[-1],
                   "first":trajectory[0], "head":int(accepted), "cpu_s":time.process_time()-first}
        run.log({prefix+"/"+k:v for k,v in summary.items()})
        print("RESULT",key,seed,*plan,summary,flush=True)


def main(package, mode, run):
    parent = Model.load(package / "checkpoint.npz")
    train, valid = [read_dataset(package / "public" / f"{s}.jsonl", "regression", [], parent.meta["paths"])
                    for s in ("train", "validation")]
    vx, vy = valid["x"], valid["y"]
    all_rows = np.ones(len(vy), dtype=bool)
    vgroups = bearing_groups(valid["ids"])
    for seed in (42,43,44):
        replay(parent, train["x"], train["y"], train["ids"], vx, vy, all_rows, all_rows, "public", seed, mode, run)
        for group in np.unique(vgroups):
            held = vgroups == group
            replay(parent, train["x"], train["y"], train["ids"], vx, vy, ~held, held, group, seed, mode, run)
        if mode != "head":
            anonymous = [f"public-{index}" for index in range(len(train["y"]))]
            replay(parent, train["x"], train["y"], anonymous, vx, vy, all_rows, all_rows, "anonymous", seed, mode, run)
    groups = bearing_groups(train["ids"])
    ids = np.array(train["ids"])
    for group in np.unique(groups):
        keep = groups != group
        x, y = train["x"][keep], train["y"][keep]
        # Exclude the held bearing from the initial model as well as continuation.
        model = Model("regression", train["paths"], [], x, y, hidden=parent.meta["hidden"],
                      batch_size=parent.meta["batch_size"], seed=42, learning_rate=.003)
        for index, loss in enumerate(model.fit(x, y, 15), 1):
            run.log({f"held_{group}/pretrain_epoch": index, f"held_{group}/pretrain_loss": loss})
        report_x = np.concatenate((vx, train["x"][~keep]))
        report_y = np.concatenate((vy, train["y"][~keep]))
        select = np.arange(len(report_y)) < len(vy)
        replay(model, x, y, ids[keep], report_x, report_y, select, ~select, "held_"+group, 42, mode, run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--taper", action="store_true")
    args = parser.parse_args()
    mode = "taper" if args.taper else "sampling" if args.sampling else "head"
    with swanlab.init(workspace="KiTongZ", project="zhixun-engine-v2", mode="online", on_online_error="raise",
                      experiment_name=f"B-device-holdout-{mode}",
                      config={"baseline":"a8a19d1", "mode":mode, "rounds":10}) as run:
        main(args.package, mode, run)
