"""Screen early-round batch sizes, then confirm one choice with paired shuffles."""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import swanlab

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contest_submission import Model, _metrics, read_dataset


def replay(parent, train, select, report, batch, rounds, seed, run, phase):
    model = copy.deepcopy(parent)
    model.meta["seed"] = seed
    initial_steps = model.meta["steps"]
    standard_batch = parent.meta["batch_size"]
    trace = []
    steps = 0
    cpu_start = time.process_time()
    prefix = f"{phase}/s{seed}_b{batch}_r{rounds}"
    for index in range(1, 11):
        model.meta["batch_size"] = batch if index <= rounds else standard_batch
        model.meta["learning_rate"] = .003
        best_f1 = -1
        for _ in range(2):
            previous = model.meta["steps"]
            loss = model.fit(train["x"], train["y"], 1)[0]
            steps += model.meta["steps"] - previous
            f1 = _metrics(model, select)["macro_f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model)
                best_loss = loss
        model = best_model
        measured = _metrics(model, report)
        trace.append(measured["macro_f1"])
        run.log({f"{prefix}/{k}": float(v) for k, v in
                 dict(measured, round=index, loss=best_loss, selected_f1=best_f1,
                      gradient_steps=steps).items()})
    result = dict(mean=float(np.mean(trace)), worst=min(trace), final=trace[-1],
                  first=trace[0], gradient_steps=steps, selected_steps=model.meta["steps"]-initial_steps,
                  cpu_s=round(time.process_time()-cpu_start, 3))
    run.log({f"{prefix}/summary_{k}": v for k, v in result.items()})
    print(prefix, result, flush=True)
    return result


def main(folder):
    parent = Model.load(folder / "checkpoint.npz")
    train, valid = [read_dataset(folder / "public" / (s+".jsonl"), "classification",
                                 parent.meta["classes"], parent.meta["paths"])
                    for s in ("train", "validation")]
    early = np.zeros(len(valid["y"]), dtype=bool)
    for label in np.unique(valid["y"]):
        rows = np.flatnonzero(valid["y"] == label)
        early[rows[:(len(rows)+1)//2]] = True
    select, confirm = [{k: valid[k][mask] for k in ("x", "y")} for mask in (early, ~early)]
    run = swanlab.init(workspace="KiTongZ", project="zhixun-engine-v2", mode="online",
                       on_online_error="raise", experiment_name="A-early-minibatch-paired-replay",
                       config={"rounds":10, "epochs":2, "screen":"first half within class",
                               "confirm":"later half within class", "paired_seeds":[42,43,44],
                               "rank":"mean ten-round macro F1, then worst-round macro F1"})
    control = (parent.meta["batch_size"], 0)
    cases = [control] + [(b, r) for b in (8, 16, 32, 64) for r in (1, 3, 10)]
    try:
        trials = {c: replay(parent, train, select, select, *c, 42, run, "screen") for c in cases}
        best = max(cases, key=lambda c:(trials[c]["mean"], trials[c]["worst"]))
        print("screen winner", best, flush=True)
        for phase, selection, report in (("confirm", select, confirm), ("full", valid, valid)):
            for seed in (42, 43, 44):
                for case in dict.fromkeys((control, best)):
                    replay(parent, train, selection, report, *case, seed, run, phase)
    except Exception as exc:
        run.finish(state="crashed", error=str(exc))
        raise
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    main(parser.parse_args().package)
