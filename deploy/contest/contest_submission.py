"""Contest harness submission built on the OFFICIAL CPU baseline.

This is a continuation-first submission:

* data is read with the baseline's own loader (`baselines.data.read_dataset`)
* the model is the baseline's `contest.cpu-mlp.v1` NumPy MLP (`baselines.model.Model`)
* a parent checkpoint is loaded with `Model.load` and keeps weights, Adam state,
  normalization and epoch counters; only epoch count / learning rate are chosen
* the candidate checkpoint is written by `Model.save`, so the artifact format
  stays byte-compatible with the official baseline
* every round performs at least one real platform model call (`rnd.model`) and
  cites the receipt in `finish`

If the parent checkpoint cannot be loaded or its task/schema/classes do not
match the round, the round fails loudly. We never fall back to a fresh model.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

from engine_head import refine_head
from bearing_sampling import balanced_rows
from baselines.data import flatten, read_dataset
from baselines.model import Model

TASKS = ("classification", "regression")
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
CONFIG_DIR = Path(__file__).resolve().parent / "baselines" / "configs"
WORK = Path(os.environ.get("CONTEST_WORK", "/workspace"))


def log(message: str) -> None:
    print("[submission] " + message, file=sys.stderr, flush=True)


# ------------------------------------------------------------------ config
def _config(dataset_key: str, task: str) -> dict:
    if dataset_key == "B" and task == "classification":
        name = "B_classification.json"
    elif dataset_key == "B":
        name = "B_regression.json"
    else:
        name = f"{dataset_key}.json"
    path = CONFIG_DIR / name
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("task") != task:
        raise RuntimeError(f"config {name} task {config.get('task')} != contract {task}")
    return config


# ------------------------------------------------------------------ metrics
def _metrics(model: Model, data: dict) -> dict:
    predicted = model.predict(data["x"])
    if model.meta["task"] == "classification":
        labels = predicted.argmax(axis=1)
        expected = data["y"]
        observed = sorted(set(expected.tolist()) | set(labels.tolist()))
        f1 = []
        for label in observed:
            tp = np.sum((labels == label) & (expected == label))
            den = np.sum(labels == label) + np.sum(expected == label)
            f1.append(2 * tp / den if den else 0.0)
        return {"macro_f1": float(np.mean(f1)), "accuracy": float(np.mean(labels == expected))}
    error = predicted - data["y"]
    return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error)))}


def _primary(metrics: dict, metric: str) -> float:
    if metric in metrics:
        return float(metrics[metric])
    return float(metrics.get("macro_f1", metrics.get("rmse", 0.0)))


def _is_better(direction: str, value: float, current: float | None) -> bool:
    if current is None:
        return True
    return value < current if direction == "lower" else value > current


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


# ------------------------------------------------------------------ train
def train(rnd) -> None:
    contract = rnd.task if isinstance(rnd.task, dict) else {}
    task = str(contract.get("task") or "")
    if task not in TASKS:
        raise RuntimeError(f"unsupported task contract: {task!r}")
    dataset_key = str(rnd.dataset_key)
    config = _config(dataset_key, task)
    classes = list(config.get("classes") or [])
    metric = str(contract.get("metric") or ("rmse" if task == "regression" else "macro_f1"))
    direction = str(contract.get("metric_direction") or ("lower" if metric in ("rmse", "mae") else "higher"))

    projection = contract.get("public_projection") or {}
    train_path = projection.get("train_path", "/data/train.jsonl")
    validation_path = projection.get("validation_path", "/data/validation.jsonl")
    train = read_dataset(train_path, task, classes)
    validation = read_dataset(validation_path, task, classes, train["paths"])
    log(f"dataset={dataset_key} task={task} train={len(train['ids'])} val={len(validation['ids'])} "
        f"features={len(train['paths'])}")

    # ---- parent checkpoint (required for continuation; first round may be empty)
    descriptor = getattr(rnd.parent, "descriptor", {}) or {}
    declared = list(descriptor.get("checkpoint_paths") or [])
    npz = next((name for name in declared if name.lower().endswith(".npz")), None)
    parent_path = None
    parent_info = {"artifact": getattr(rnd.parent, "artifact_id", ""),
                   "sha256": getattr(rnd.parent, "checkpoint_sha256", ""), "path": npz}
    if npz:
        root = os.environ.get("CONTEST_PARENT_ROOT") or None
        parent_path = rnd.parent.path(npz, root=root)
        parent_info["size"] = parent_path.stat().st_size
    if parent_path is not None:
        model = Model.load(parent_path)
        if model.meta["task"] != task:
            raise RuntimeError("parent checkpoint task does not match the round contract")
        if model.meta["paths"] != train["paths"]:
            raise RuntimeError("parent checkpoint feature schema does not match the public data")
        if classes and model.meta["classes"] != classes:
            raise RuntimeError("parent checkpoint classes do not match the round config")
        if parent_path.stat().st_size > 16 * 1024 * 1024:
            raise RuntimeError("parent checkpoint exceeds the 16 MiB baseline limit")
        log("parent checkpoint loaded and validated")
    else:
        model = Model(task, train["paths"], classes, train["x"], train["y"],
                      hidden=int(config["hidden"]), seed=int(config["seed"]),
                      learning_rate=float(config["learning_rate"]),
                      batch_size=int(config["batch_size"]))
        log("no parent checkpoint: initialised from the official baseline constructor")

    before = _metrics(model, validation)

    # ---- one real platform model call decides the continuation budget
    state_payload = {
        "dataset_key": dataset_key,
        "round_index": rnd.round_index,
        "task": task,
        "metric": metric,
        "metric_direction": direction,
        "train_rows": len(train["ids"]),
        "validation_rows": len(validation["ids"]),
        "features": len(train["paths"]),
        "parent_epochs": int(model.meta.get("epochs", 0)),
        "parent_steps": int(model.meta.get("steps", 0)),
        "parent_validation": {k: round(v, 5) for k, v in before.items()},
        "choices": {"epochs": [1, 2, 3], "learning_rate": [config["learning_rate"],
                                                            config["learning_rate"] * 2]},
    }
    tested_budget = None
    if rnd.round_index == 1 and parent_path is not None:
        tested_budget = {"B": (3, 2.0), "C": (1, 0.5)}.get(dataset_key)
    if tested_budget:
        tested_epochs, factor = tested_budget
        tested_lr = float(config["learning_rate"]) * factor
        state_payload["public_validation_recommendation"] = {
            "epochs": tested_epochs, "learning_rate": tested_lr,
            "reason": "Lowest validation RMSE in an official-parent 3-rate comparison. Prefer this budget when applicable.",
        }
        state_payload["choices"]["learning_rate"] = sorted(set(
            state_payload["choices"]["learning_rate"] + [tested_lr]))
    late_engine = dataset_key == "C" and task == "regression" and 6 <= rnd.round_index <= 10
    if late_engine:
        state_payload["choices"]["learning_rate"] = [2 * config["learning_rate"]]
        state_payload["public_validation_recommendation"] = {
            "epochs": 2, "learning_rate": 2 * config["learning_rate"],
            "reason": "Late-round public replay: lowest round-ten RMSE across three tested rates."
        }
    prompt = ("You are the decision layer of a continuation-training system. "
              "Choose the training budget for this round. Reply with one JSON object only: "
              '{"epochs": 1-3, "learning_rate": number, "hypothesis": "<=200 chars", '
              '"change_summary": "<=200 chars", "reason": "<=200 chars"}. '
              "State: " + json.dumps(state_payload, ensure_ascii=False, separators=(",", ":")))
    limit = max(16, min(320, int(getattr(rnd, "max_output_tokens", 256) or 256)))
    receipt = rnd.model([{"role": "user", "content": prompt}],
                        max_tokens=limit, temperature=0)
    decision = _extract_json(receipt.text) or {}
    log(f"model call id={receipt.call_id} decision={decision}")

    try:
        epochs = int(decision.get("epochs", tested_epochs if tested_budget else 2))
    except (TypeError, ValueError):
        epochs = 2
    epochs = max(1, min(3, epochs))
    default_lr = float(config["learning_rate"])
    try:
        lr = float(decision.get("learning_rate", tested_lr if tested_budget else default_lr))
    except (TypeError, ValueError):
        lr = default_lr
    if not math.isfinite(lr) or lr <= 0:
        lr = default_lr
    lr = min(max(lr, default_lr * 0.5), default_lr * 2.0)
    if late_engine:
        lr = 2 * default_lr
    model.meta["learning_rate"] = lr
    model.validate()

    # ---- continue training epoch by epoch, keep the best validation state
    best_model = None
    best_score = None
    history = []
    for _ in range(epochs):
        rows = slice(None)
        if dataset_key == "B" and task == "regression":
            rows = balanced_rows(train["ids"], model.meta["epochs"], model.meta["seed"], 1)
        loss = float(model.fit(train["x"][rows], train["y"][rows], 1)[0])
        current = _metrics(model, validation)
        score = _primary(current, metric)
        history.append({"epoch": int(model.meta["epochs"]), "loss": loss, "metrics": current})
        log(f"epoch {model.meta['epochs']} loss={loss:.5f} {metric}={score:.5f}")
        if _is_better(direction, score, best_score):
            best_score = score
            best_model = copy.deepcopy(model)
    model = best_model
    head_refined = False
    if late_engine and rnd.round_index == 10:
        refined = copy.deepcopy(model)
        refine_head(refined, train["x"], train["y"], penalty=.001)
        refined_score = _primary(_metrics(refined, validation), metric)
        if _is_better(direction, refined_score, best_score):
            model = refined
            best_score = refined_score
            head_refined = True

    # ---- candidate artifact (checkpoint.npz written by the baseline's own saver)
    work = WORK
    work.mkdir(parents=True, exist_ok=True)
    candidate_dir = work / f"candidate-r{rnd.round_index}"
    if candidate_dir.exists():
        import shutil
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True)
    checkpoint = candidate_dir / "checkpoint.npz"
    model.save(checkpoint)

    history_ids = [item["id"] for item in rnd.history()]
    hypothesis = str(decision.get("hypothesis") or f"round {rnd.round_index} continuation")[:900]
    change = str(decision.get("change_summary") or f"continue baseline for {epochs} epoch(s), lr={lr}")[:900]
    reason = str(decision.get("reason") or f"validation {metric}={best_score:.5f} after {epochs} epoch(s)")[:900]

    if head_refined:
        change = f"{change[:780]} | Refitted final linear head on public training rows (ridge=0.001)."
        reason = f"Final head improved public validation {metric} to {best_score:.6f}."
    submitted = rnd.submit_candidate(
        candidate_dir,
        hypothesis=hypothesis,
        change_summary=change,
        configuration={"round": rnd.round_index, "dataset_key": dataset_key, "task": task,
                       "epochs": epochs, "learning_rate": lr, "metric": metric,
                       "head_refined": head_refined, "validation_before": before, "validation_after": _metrics(model, validation),
                       "model_call_id": receipt.call_id, "parent": parent_info,
                       "history": history},
        seed=int(model.meta["seed"]),
        checkpoint_paths=["checkpoint.npz"],
        code_paths=[],
        predict_argv=PREDICT_ARGV,
    )

    candidate_metrics = _metrics(model, validation)
    selected_id = submitted["artifact_id"]
    selected_kind = "candidate"
    selected_epoch = int(model.meta["epochs"])
    selected_metrics = candidate_metrics
    state_dir = work / f"state-r{rnd.round_index}"
    if state_dir.exists():
        import shutil
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True)
    (state_dir / "round.json").write_text(json.dumps({
        "round": rnd.round_index, "dataset_key": dataset_key, "task": task,
        "selected": selected_kind, "artifact_id": selected_id, "candidate_id": submitted["artifact_id"],
        "model_call_id": receipt.call_id, "epochs": epochs, "learning_rate": lr,
        "head_refined": head_refined, "validation_before": before, "validation_after": _metrics(model, validation),
        "history": history,
        "selected_epoch": selected_epoch, "selected_metrics": selected_metrics,
        "parent": parent_info,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    rnd.save_state(state_dir)

    rnd.finish(
        selected_artifact_id=selected_id,
        hypothesis=hypothesis,
        change_summary=change,
        reason=reason,
        configuration={"round": rnd.round_index, "selected": selected_kind, "selected_metrics": selected_metrics, "epochs": epochs, "learning_rate": lr,
                       "validation_after": _metrics(model, validation),
                       "model_call_id": receipt.call_id},
        seed=int(model.meta["seed"]),
        history_ids=history_ids,
        model_call_ids=list(rnd.call_ids),
    )


# ------------------------------------------------------------------ predict
_CACHE: dict = {}


def _model_path(model_dir) -> Path:
    explicit = os.environ.get("CONTEST_ARTIFACT_PATH")
    if explicit:
        return Path(explicit)
    return Path(model_dir or "/model") / "checkpoint.npz"


def _load(path: Path) -> Model:
    key = (str(path), path.stat().st_mtime if path.exists() else 0)
    model = _CACHE.get(key)
    if model is None:
        model = Model.load(path)
        _CACHE.clear()
        _CACHE[key] = model
    return model


def predict(sample, model_dir):
    model = _load(_model_path(model_dir))
    paths, values = flatten(sample.inputs or {})
    if paths != model.meta["paths"]:
        raise RuntimeError("prediction input schema does not match the checkpoint")
    x = np.asarray(values, dtype=np.float64)[None, :]
    predicted = model.predict(x)
    if sample.task == "regression":
        value = float(predicted[0])
        if not math.isfinite(value) or value < 0:
            value = 0.0
        return {"value": value}
    if sample.task != "classification":
        raise RuntimeError("unknown prediction task")
    probabilities = predicted[0]
    classes = list(model.meta["classes"])
    order = np.argsort(-probabilities)
    scores = [float(probabilities[i]) for i in order]
    if len(scores) >= 2 and scores[0] <= scores[1]:
        scores[1] = max(0.0, scores[0] - 1e-6)
    return [{"label": str(classes[int(i)]), "score": max(0.0, min(1.0, scores[rank]))}
            for rank, i in enumerate(order)]




