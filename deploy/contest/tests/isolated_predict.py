"""Exercise the submitted argv with only the read-only model directory mounted."""
import argparse
import json
import subprocess
import sys

from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifact_predict import load_artifact
from contest_submission import PREDICT_ARGV, PREDICTION_FILES
from baselines.data import flatten


def check(package, key, runtime, artifact):
    artifact = artifact.resolve()
    model = load_artifact(artifact)
    task = model.meta["task"]
    files = ["checkpoint.npz", *PREDICTION_FILES]
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    if "engine_calibration" in config:
        files.append("engine_calibration.py")
    assert sorted(str(p.relative_to(artifact).as_posix()) for p in artifact.rglob("*") if p.is_file()) == sorted(files)
    with (package / "public/validation.jsonl").open(encoding="utf-8") as f:
        samples = [{"task":task, "inputs":json.loads(next(f))["inputs"]} for _ in range(5)]
    ordered = [samples[i] for i in (4, 1, 0, 2, 1)]
    expected = model.predict(np.array([flatten(row["inputs"])[1] for row in ordered]))
    command = ["docker", "run", "--rm", "-i", "--cpus", "4", "--memory", "4g",
               "--memory-swap", "4g", "--network", "none", "--read-only",
               "--user", "65534:65534", "--mount", f"type=bind,source={artifact},target=/model,readonly",
               runtime, *PREDICT_ARGV]
    for extra in ([], ["--checkpoint", "/model/checkpoint.npz"], ["--model-dir", "/model"]):
        requests = [ordered, {"samples":ordered}, [], *ordered]
        output = subprocess.run(command + extra,input="".join(json.dumps(r)+"\n" for r in requests),
                                text=True,capture_output=True,timeout=90)
        if output.returncode:
            raise RuntimeError(output.stderr)
        lines = [json.loads(line) for line in output.stdout.splitlines()]
        assert len(lines) == len(requests) and lines[2] == []
        for rows in (lines[0], lines[1], lines[3:]):
            assert len(rows) == len(ordered)
            for actual, reference in zip(rows, expected):
                if task == "regression":
                    np.testing.assert_allclose(actual["outputs"]["value"],reference,rtol=1e-10,atol=1e-9)
                else:
                    scores = {s["label"]:s["score"] for s in actual["outputs"]}
                    np.testing.assert_allclose([scores[k] for k in model.meta["classes"]], reference,rtol=1e-10,atol=1e-10)
    print(key, "OK: ordered batches, repeats, empty batch, JSONL, explicit paths", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("packages",type=Path)
    parser.add_argument("--runtime", default="zhixun-predict-check:bare")
    parser.add_argument("artifacts", nargs=3, type=Path, help="training-produced candidate directories in A, B, C order")
    args = parser.parse_args()
    for key, artifact in zip("ABC", args.artifacts):
        check(next(args.packages.glob(f"cpu-{key.lower()}-*")), key, args.runtime, artifact)
