"""Replay both entries on public data with the same budgets and compare checkpoints."""
import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from types import SimpleNamespace

import swanlab

CONTEST = Path(__file__).resolve().parents[1]
BASELINE = "59fff6c"
sys.path.insert(0, str(CONTEST))
import contest_submission as current


def load_baseline(relative):
    name = "deploy/contest/" + relative
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{name}"], cwd=CONTEST)
    module = types.ModuleType("baseline_" + Path(relative).stem)
    module.__file__ = str(CONTEST / relative)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def replay(entry, package, dataset, work, round_index, checkpoint, run, label):
    entry.WORK = work
    task = "classification" if dataset == "A" else "regression"
    epochs, rate = {
        "A": (2, .003),
        "B": (3, .006) if round_index == 1 else (1, .0015),
        "C": (1, .001) if round_index == 1 else (2, .002 if round_index <= 5 else .004),
    }[dataset]
    recorded = {}
    call_ids = []
    artifact_id = f"candidate-{round_index}"

    def model(messages, **options):
        assert not call_ids
        call_ids.append("local-call")
        recorded["prompt_chars"] = len(messages[0]["content"])
        recorded["output_cap"] = options["max_tokens"]
        if label == "candidate":
            assert options["enable_thinking"] is False and options["max_tokens"] == 64
            # Exercise invalid, truncated and valid compact responses. Fallback
            # matches the paired budget at the chosen rounds.
            if round_index == 1:
                response = '{"e":1e999,"l":null}'
            elif round_index == 2:
                response = '{"e":'
            else:
                response = json.dumps({"e": epochs, "l": rate})
        else:
            response = json.dumps({"epochs": epochs, "learning_rate": rate})
        return SimpleNamespace(text=response, call_id=call_ids[0])

    rnd = SimpleNamespace(
        dataset_key=dataset, round_index=round_index, max_output_tokens=320, model=model,
        task={"task": task, "public_projection": {
            f"{s}_path": package / "public" / f"{s}.jsonl" for s in ("train", "validation")
        }},
        parent=SimpleNamespace(artifact_id=f"candidate-{round_index - 1}",
                               descriptor={"checkpoint_paths": ["checkpoint.npz"]},
                               path=lambda name, root: checkpoint),
        call_ids=call_ids, history=lambda: [],
        submit_candidate=lambda *args, **kwargs: {"artifact_id": artifact_id},
        save_state=lambda path: None, finish=lambda **fields: recorded.update(fields),
    )
    begin = time.process_time()
    entry.train(rnd)
    recorded["cpu_s"] = time.process_time() - begin
    assert recorded["selected_artifact_id"] == artifact_id
    assert recorded["model_call_ids"] == call_ids == ["local-call"]
    config = recorded["configuration"]
    assert (config["epochs"], config["learning_rate"]) == (epochs, rate)
    run.log({f"{dataset}/{label}/{k}": v for k, v in {
        "round": round_index, "cpu_s": recorded["cpu_s"], "output_cap": recorded["output_cap"],
        "prompt_chars": recorded["prompt_chars"], **config["selected_metrics"],
    }.items()})
    return work / f"candidate-r{round_index}" / "checkpoint.npz", recorded


def compare(packages, work, run):
    prior = load_baseline("contest_submission.py")
    prior.Model = load_baseline("baselines/model.py").Model
    prior.read_dataset = load_baseline("baselines/data.py").read_dataset
    for dataset in "ABC":
        package = next(packages.glob(f"cpu-{dataset.lower()}-*"))
        checkpoints = {label: package / "checkpoint.npz" for label in ("baseline", "candidate")}
        cpu = {label: [] for label in checkpoints}
        chars = {label: [] for label in checkpoints}
        for round_index in range(1, 11):
            records = {}
            order = ("baseline", "candidate") if round_index % 2 else ("candidate", "baseline")
            for label in order:
                entry = prior if label == "baseline" else current
                checkpoints[label], rec = replay(entry, package, dataset, work / dataset / label,
                                                 round_index, checkpoints[label], run, label)
                records[label] = rec["configuration"]["selected_metrics"]
                cpu[label].append(rec["cpu_s"])
                chars[label].append(rec["prompt_chars"])
            assert records["baseline"] == records["candidate"], (dataset, round_index, records)
            assert checkpoints["baseline"].read_bytes() == checkpoints["candidate"].read_bytes(), (dataset, round_index)
        print(dataset, "10 byte-identical checkpoints;", "CPU seconds", {k: round(sum(v), 3) for k, v in cpu.items()},
              "median prompt chars", {k: statistics.median(v) for k, v in chars.items()}, records["candidate"], flush=True)
        run.log({f"{dataset}/{k}/total_cpu_s": sum(v) for k, v in cpu.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", type=Path)
    args = parser.parse_args()
    with swanlab.init(workspace="KiTongZ", project="zhixun-engine-v2", mode="online", on_online_error="raise",
                      experiment_name="feature-cache-efficient-abc-paired-replay",
                      config={"baseline": BASELINE, "rounds": 10, "model_gateway": "local stub"}) as run:
        with tempfile.TemporaryDirectory() as folder:
            compare(args.packages, Path(folder), run)
