"""Focused B-task budget check: 1/32 vs 1/16 balanced sampling.

Replays the production schedule (round 1: 3 epochs at 2x lr; later rounds: 1
epoch at 0.5x lr) on device-disjoint folds and the full public split, then
compares mean/worst RMSE, MAE and CPU cost.

Usage:
    python deploy/contest/tests/compare_b_budget_tail.py <cpu-b-rul package>
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contest_submission import Model, read_dataset  # noqa: E402
from bearing_sampling import balanced_rows, bearing_groups  # noqa: E402

PKG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"C:\Users\Z\Desktop\赛题九交接\02_项目源码与文档\AI+项目\vendor\baselines\cpu-b-rul-20261110-v1")
PLANS = {"tiny_1_32": (1, .0015, 1 / 32), "sixteenth_1_16": (1, .0015, 1 / 16)}


def rmse(model, x, y):
    return float(np.sqrt(np.mean((model.predict(x) - y) ** 2)))


def mae(model, x, y):
    return float(np.mean(np.abs(model.predict(x) - y)))


def replay(parent, train, valid, select, report, policy):
    epochs, rate, fraction = PLANS[policy]
    model = copy.deepcopy(parent)
    trace, samples, cpu = [], 0, 0.0
    for rnd in range(1, 11):
        first = rnd == 1
        model.meta["learning_rate"] = .006 if first else rate
        best, best_score = None, float("inf")
        for _ in range(3 if first else epochs):
            start = time.process_time()
            rows = balanced_rows(train["ids"], model.meta["epochs"], model.meta["seed"], 1)
            if not first:
                rows = rows[:max(1, round(len(rows) * fraction))]
            model.fit(train["x"][rows], train["y"][rows], 1)
            cpu += time.process_time() - start
            samples += len(rows)
            score = rmse(model, valid["x"][select], valid["y"][select])
            if score < best_score:
                best, best_score = copy.deepcopy(model), score
        model = best
        trace.append(rmse(model, valid["x"][report], valid["y"][report]))
    return {"mean": float(np.mean(trace)), "worst": float(np.max(trace)),
            "final": trace[-1], "first": trace[0], "samples": samples,
            "cpu_s": round(cpu, 2), "mae": mae(model, valid["x"][report], valid["y"][report])}


def main() -> None:
    t0 = time.time()
    parent = Model.load(PKG / "checkpoint.npz")
    train = read_dataset(PKG / "public" / "train.jsonl", "regression", [], parent.meta["paths"])
    valid = read_dataset(PKG / "public" / "validation.jsonl", "regression", [], train["paths"])
    groups = bearing_groups(valid["ids"])
    devices = sorted(set(groups))
    folds = []
    for bearing in devices[:3]:
        mask = groups == bearing
        if mask.sum() >= 25:
            folds.append((bearing, mask))
    folds.append(("full", np.ones(len(groups), dtype=bool)))
    out = {}
    for name, report in folds:
        select = ~report if name != "full" else report
        out[name] = {}
        for policy in PLANS:
            result = replay(parent, train, valid, select, report, policy)
            out[name][policy] = result
            print(name, policy, result, flush=True)
    summary = {}
    for policy in PLANS:
        rows = [out[f][policy] for f in out]
        summary[policy] = {
            "mean_rmse": float(np.mean([r["mean"] for r in rows])),
            "worst_rmse": float(np.mean([r["worst"] for r in rows])),
            "final_mae": float(np.mean([r["mae"] for r in rows])),
            "samples": int(np.mean([r["samples"] for r in rows])),
            "cpu_s": float(np.mean([r["cpu_s"] for r in rows])),
        }
    print(json.dumps({"folds": out, "summary": summary, "seconds": round(time.time() - t0, 1)},
                     ensure_ascii=False, indent=2), flush=True)
    Path(__file__).with_name("b_budget_tail_results.json").write_text(
        json.dumps({"folds": out, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
