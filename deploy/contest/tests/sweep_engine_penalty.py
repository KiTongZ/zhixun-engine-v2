"""Sweep the engine head-refinement penalty on the held-out engine split."""
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
from engine_head import refine_head  # noqa: E402

PKG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"C:\Users\Z\Desktop\赛题九交接\02_项目源码与文档\AI+项目\vendor\baselines\cpu-c-guide-20261110-v1")
PENALTIES = [0.0, 0.00000001, 0.0000001, 0.000001]


def stats(model, x, y):
    err = model.predict(x) - y
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "asym": float(np.expm1(np.maximum(err / 10, -err / 13)).mean())}


def main() -> None:
    t0 = time.time()
    parent = Model.load(PKG / "checkpoint.npz")
    train = read_dataset(PKG / "public" / "train.jsonl", "regression", [], parent.meta["paths"])
    valid = read_dataset(PKG / "public" / "validation.jsonl", "regression", [], train["paths"])
    engines = np.array([k.rsplit("_", 1)[0] for k in valid["ids"]])
    held = np.array([int.from_bytes(hashlib.sha256(k.encode()).digest()[:4], "big") % 3 == 0
                     for k in engines])
    select, report = ~held, held

    model = copy.deepcopy(parent)
    for rnd in range(1, 10):
        lr = 0.001 if rnd == 1 else (0.004 if rnd >= 6 else 0.002)
        model.meta["learning_rate"] = lr
        best, best_score = None, float("inf")
        for _ in range(1 if rnd == 1 else 2):
            model.fit(train["x"], train["y"], 1)
            score = float(np.sqrt(np.mean((model.predict(valid["x"][select]) - valid["y"][select]) ** 2)))
            if score < best_score:
                best, best_score = copy.deepcopy(model), score
        model = best
    r9 = copy.deepcopy(model)
    model.meta["learning_rate"] = 0.004
    best, best_score = None, float("inf")
    for _ in range(2):
        model.fit(train["x"], train["y"], 1)
        score = float(np.sqrt(np.mean((model.predict(valid["x"][select]) - valid["y"][select]) ** 2)))
        if score < best_score:
            best, best_score = copy.deepcopy(model), score
    base = best

    out = {"base_select_rmse": best_score, "base_held": stats(base, valid["x"][report], valid["y"][report]),
           "penalties": {}}
    for penalty in PENALTIES:
        refined = copy.deepcopy(base)
        refine_head(refined, train["x"], train["y"], penalty=penalty)
        sel = float(np.sqrt(np.mean((refined.predict(valid["x"][select]) - valid["y"][select]) ** 2)))
        out["penalties"][str(penalty)] = {
            "select_rmse": sel,
            "accepted": bool(sel < best_score),
            "held": stats(refined, valid["x"][report], valid["y"][report]),
        }
        print(penalty, out["penalties"][str(penalty)], flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    Path(__file__).with_name("engine_penalty_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()



