"""Focused C-task tail-risk check: production replay vs moderated update at round 10.

Runs on the official C public train/validation split with a held-out group of
engines. Compares four round-10 policies on the same parent:

  parent      : round-9 model, no further update
  base        : round-10 training, best epoch by selection RMSE
  refine      : base + official head refinement (ships today)
  moderate    : base + moderated update fraction (candidate for next image)

Usage:
    python deploy/contest/tests/compare_engine_tail.py <cpu-c-guide package>
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contest_submission import Model, read_dataset  # noqa: E402
from engine_head import moderate_update, refine_head  # noqa: E402

PKG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"C:\Users\Z\Desktop\赛题九交接\02_项目源码与文档\AI+项目\vendor\baselines\cpu-c-guide-20261110-v1")


def metrics(model, x, y):
    err = model.predict(x) - y
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "asym": float(np.expm1(np.maximum(err / 10, -err / 13)).mean()),
        "p95_over": float(np.percentile(np.maximum(err, 0), 95)),
    }


def rmse(model, x, y):
    return float(np.sqrt(np.mean((model.predict(x) - y) ** 2)))


def main() -> None:
    t0 = time.time()
    parent = Model.load(PKG / "checkpoint.npz")
    train = read_dataset(PKG / "public" / "train.jsonl", "regression", [], parent.meta["paths"])
    valid = read_dataset(PKG / "public" / "validation.jsonl", "regression", [], train["paths"])
    engines = np.array([key.rsplit("_", 1)[0] for key in valid["ids"]])
    held = np.array([int.from_bytes(hashlib.sha256(k.encode()).digest()[:4], "big") % 3 == 0
                     for k in engines])
    select = ~held
    report = held
    print(f"train={len(train['ids'])} val={len(valid['ids'])} held_engines={len(set(engines[held]))}", flush=True)

    model = copy.deepcopy(parent)
    trace = []
    for rnd in range(1, 10):
        lr = 0.001 if rnd == 1 else (0.004 if rnd >= 6 else 0.002)
        epochs = 1 if rnd == 1 else 2
        model.meta["learning_rate"] = lr
        best, best_score = None, float("inf")
        for _ in range(epochs):
            model.fit(train["x"], train["y"], 1)
            score = rmse(model, valid["x"][select], valid["y"][select])
            if score < best_score:
                best, best_score = copy.deepcopy(model), score
        model = best
        trace.append(best_score)
        print(f"round {rnd}: select_rmse={best_score:.4f} ({time.time()-t0:.0f}s)", flush=True)

    r9 = copy.deepcopy(model)
    model.meta["learning_rate"] = 0.004
    best, best_score = None, float("inf")
    for _ in range(2):
        model.fit(train["x"], train["y"], 1)
        score = rmse(model, valid["x"][select], valid["y"][select])
        if score < best_score:
            best, best_score = copy.deepcopy(model), score
    base = best

    refined = copy.deepcopy(base)
    refine_head(refined, train["x"], train["y"], penalty=.001)
    refined_ok = rmse(refined, valid["x"][select], valid["y"][select]) < best_score

    moderated, fraction = moderate_update(r9, base, valid["x"][select], valid["y"][select])

    results = {
        "parent": metrics(r9, valid["x"][report], valid["y"][report]),
        "base": metrics(base, valid["x"][report], valid["y"][report]),
        "refine": metrics(refined, valid["x"][report], valid["y"][report]),
        "moderate": metrics(moderated, valid["x"][report], valid["y"][report]),
        "refine_accepted": bool(refined_ok),
        "moderate_fraction": fraction,
        "select_rmse": {"r9": trace[-1], "base": best_score},
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    out = Path(__file__).with_name("c_tail_results.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
