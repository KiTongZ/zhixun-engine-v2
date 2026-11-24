"""Local prediction smoke test using the official baseline checkpoint."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SDK = ROOT / "vendor" / "contest-harness-sdk-1.0.4"
PKG = ROOT / "vendor" / "baselines" / "cpu-a-guide-20261110-v1"
CONTEST = ROOT / "deploy" / "contest"

def run_predict(checkpoint: Path, task: str, samples: list[dict]) -> list[dict]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(SDK), str(CONTEST)))
    env["CONTEST_ARTIFACT_PATH"] = str(checkpoint)
    env["CONTEST_TASK"] = task
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, str(CONTEST / "harness.py"), "predict"],
        input="\n".join(json.dumps(s) for s in samples) + "\n",
        text=True, capture_output=True, env=env, cwd=str(CONTEST), timeout=120)
    if proc.returncode != 0:
        print(proc.stderr[-3000:])
        raise SystemExit("predict failed")
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def check_outputs(out: list[dict], task: str, count: int) -> None:
    assert len(out) == count
    for row in out:
        outputs = row["outputs"]
        if task == "regression":
            assert math.isfinite(outputs["value"]) and outputs["value"] >= 0
            continue
        scores = [item["score"] for item in outputs]
        assert len(outputs) == 6
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert scores.count(max(scores)) == 1
        labels = {item["label"] for item in outputs}
        assert labels == {"Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"}


def main() -> None:
    with (PKG / "public" / "train.jsonl").open(encoding="utf-8") as handle:
        sample = json.loads(handle.readline())
    inputs = [{"task": "classification", "inputs": sample["inputs"]}] * 2
    check_outputs(run_predict(PKG / "checkpoint.npz", "classification", inputs), "classification", len(inputs))
    print("PREDICT OK")


if __name__ == "__main__":
    main()
