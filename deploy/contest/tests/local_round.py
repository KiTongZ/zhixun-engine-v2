"""Local end-to-end round against the SDK host stand-in, using the OFFICIAL baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from tempfile import mkdtemp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]           # AI+项目
SDK = ROOT / "vendor" / "contest-harness-sdk-1.0.4"
PACKAGES = {"A": "cpu-a-guide", "B": "cpu-b-rul", "C": "cpu-c-guide"}
CONTEST = ROOT / "deploy" / "contest"
sys.path.insert(0, str(SDK))

from tests.support import Host  # noqa: E402

PROMPT = "official prompt fixture"


def run_round(dataset: str, budget: int, run) -> None:
    from local_predict import run_predict, check_outputs

    pkg = Path(os.environ.get("CONTEST_BASELINE_PACKAGES", ROOT / "vendor" / "baselines")) / f"{PACKAGES[dataset]}-20261110-v1"
    output = ROOT / "out" / "contest_round"
    output.mkdir(parents=True, exist_ok=True)
    work = Path(mkdtemp(prefix=f"{dataset}-", dir=output))
    task = "classification" if dataset == "A" else "regression"
    metric = "macro_f1" if dataset == "A" else "rmse"
    baseline = json.loads((pkg / "baseline-info.json").read_text(encoding="utf-8"))
    baseline_id = baseline["id"]

    parent_dir = work / "parent"
    parent_dir.mkdir()
    shutil.copyfile(pkg / "checkpoint.npz", parent_dir / "checkpoint.npz")
    digest = "sha256:" + hashlib.sha256((parent_dir / "checkpoint.npz").read_bytes()).hexdigest()
    assert digest == baseline["checkpoint_sha256"], baseline_id
    run.config["parent_checkpoint_sha256"] = digest

    start = {
        "protocol": "contest.harness.workspace.v2",
        "run_id": "run-local",
        "dataset_key": dataset,
        "round_index": 1,
        "stage_id": "local",
        "prompt": PROMPT,
        "prompt_version": "industrial-workspace-flexible-v3",
        "rounds_per_dataset": 10,
        "parent_checkpoint_sha256": digest,
        "task_contract": {
            "dataset_key": dataset,
            "task": task,
            "metric": metric,
            "metric_direction": "higher" if task == "classification" else "lower",
            "public_projection": {
                "train_path": str(pkg / "public" / "train.jsonl"),
                "validation_path": str(pkg / "public" / "validation.jsonl"),
                "row_fields": ["id", "inputs", "answer"],
                "source_metadata": False,
            },
        },
        "model_name": "base:test-model",
        "parent_artifact": {
            "artifact_id": baseline_id,
            "artifact_sha256": digest,
            "artifact": {
                "checkpoint_paths": ["checkpoint.npz"],
                "code_paths": [],
                "predict_argv": ["python3", "/app/harness.py", "predict"],
            },
            "files": [{"path": "checkpoint.npz", "sha256": digest,
                       "size_bytes": (parent_dir / "checkpoint.npz").stat().st_size}],
        },
        "artifacts": [{"artifact_id": baseline_id}],
        "state_sha256": "",
        "state_path": None,
        "max_output_tokens": 128,
        "resource_limits": {
            "artifact_upload_protocol": "chunked-v1",
            "artifact_bytes": 512 * 1024 * 1024,
            "state_bytes": 512 * 1024 * 1024,
            "candidate_upload_bytes_per_round": 1024 * 1024 * 1024,
            "state_upload_bytes_per_round": 512 * 1024 * 1024,
        },
        "current_validation": {},
        "wall_seconds": 300,
    }

    def stub_model(messages, options):
        assert messages[0]["role"] == "system"
        return {
            "request_id": "call-local-1",
            "id": "chatcmpl-local",
            "choices": [{"message": {"role": "assistant", "content": json.dumps({
                "epochs": budget, "learning_rate": 0.002 if dataset == "C" else 0.003,
                "hypothesis": "验证官方检查点续训协议",
                "change_summary": f"在官方 baseline 检查点上续训 {budget} 个 epoch",
                "reason": "验证集用于比较，先跑通协议"})}}],
        }

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(SDK), str(CONTEST)))
    env["CONTEST_SUBMISSION"] = "contest_submission"
    env["CONTEST_WORK"] = str(work / "workspace")
    env["CONTEST_PARENT_ROOT"] = str(parent_dir)
    env["PYTHONIOENCODING"] = "utf-8"

    host = Host(start, stub_model)
    stderr_path = work / "train.stderr.log"
    with stderr_path.open("w", encoding="utf-8") as stderr:
        with subprocess.Popen(
            [sys.executable, str(CONTEST / "harness.py"), "train"],
            cwd=CONTEST, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr, text=True, encoding="utf-8",
        ) as proc:
            thread = host.serve(proc)
            try:
                proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill()
                raise
            finally:
                thread.join(timeout=10)
    if host.failure:
        raise RuntimeError(f"host failure: {host.failure}; stderr: {stderr_path}")
    if proc.returncode:
        raise RuntimeError(f"Harness exited with {proc.returncode}; stderr: {stderr_path}")
    record = host.record
    (work / "finish.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    assert record["selected_artifact_id"] == "model-test-1", record
    assert record["model_call_ids"] == ["call-local-1"], record

    state = json.loads((work / "workspace/state-r1/round.json").read_text(encoding="utf-8"))
    run.config.update({"learning_rate": state["learning_rate"], "selected_epoch": state["selected_epoch"],
                       "model_call_id": state["model_call_id"]})
    run.log({f"parent/{key}": value for key, value in state["validation_before"].items()}, step=0)
    for step, entry in enumerate(state["history"], start=1):
        run.log({"train/epoch": entry["epoch"], "train/loss": entry["loss"],
                 **{f"validation/{key}": value for key, value in entry["metrics"].items()}}, step=step)
    run.log({f"selected/{key}": value for key, value in state["validation_after"].items()}, step=budget)

    sys.path.insert(0, str(CONTEST))
    from baselines.model import Model

    checkpoint = work / "workspace/candidate-r1/checkpoint.npz"
    parent = Model.load(parent_dir / "checkpoint.npz")
    candidate = Model.load(checkpoint)
    assert len(state["history"]) == budget
    assert candidate.meta["epochs"] == state["selected_epoch"] > parent.meta["epochs"]
    assert candidate.meta["steps"] > parent.meta["steps"]
    assert (candidate.mean == parent.mean).all() and (candidate.scale == parent.scale).all()
    with (pkg / "public/validation.jsonl").open(encoding="utf-8") as source:
        samples = [{"task": task, "inputs": json.loads(next(source))["inputs"]} for _ in range(2)]
    check_outputs(run_predict(checkpoint, task, samples), task, len(samples))
    print(f"LOCAL ROUND OK {dataset}: {metric} {state['validation_before'][metric]:.6f} -> "
          f"{state['validation_after'][metric]:.6f} | {work}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=PACKAGES, default="A")
    parser.add_argument("--epochs", choices=(1, 2, 3), type=int, default=1)
    args = parser.parse_args()

    import swanlab

    run = swanlab.init(
        workspace="KiTongZ", project="zhixun-engine-v2", mode="online",
        on_online_error="raise", name=f"official-{args.dataset}-local-{args.epochs}ep",
        log_dir=str(ROOT / "out" / "swanlog"),
        config={"dataset": args.dataset, "requested_epochs": args.epochs,
                "model_gateway": "SDK host stub", "evaluation": "official public validation"},
    )
    try:
        run_round(args.dataset, args.epochs, run)
    except Exception as exc:
        run.finish(state="crashed", error=str(exc))
        raise
    else:
        run.finish()


if __name__ == "__main__":
    main()

