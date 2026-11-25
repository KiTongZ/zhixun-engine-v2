"""Replay task B with recorded budgets and the production training entry point."""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import swanlab

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contest_submission as entry

def replay(package, work):
    entry.WORK = work
    checkpoint = package / "checkpoint.npz"
    parent_id = "official-B"
    run = swanlab.init(workspace="KiTongZ", project="zhixun-engine-v2", mode="online",
                       on_online_error="raise", experiment_name="B-balanced-round-replay",
                       config={"split":"official public validation","decision_source":"recorded budgets"})
    for index in range(1, 11):
        budget = {"epochs": 3 if index == 1 else 2, "learning_rate": .006 if index == 1 else .003}
        current = f"round-{index}"
        record = {}
        rnd = SimpleNamespace(dataset_key="B", round_index=index, call_ids=["local"],
            task={"task":"regression","public_projection":{f"{s}_path":package/"public"/(s+".jsonl") for s in ("train","validation")}},
            parent=SimpleNamespace(artifact_id=parent_id, descriptor={"checkpoint_paths":["checkpoint.npz"]},
                                   path=lambda name,root: checkpoint),
            model=lambda *a,**k: SimpleNamespace(call_id="local",text=json.dumps(budget)),
            history=lambda:[], submit_candidate=lambda *a,**k: {"artifact_id":current},
            save_state=lambda path:None, finish=lambda **fields:record.update(fields))
        entry.train(rnd)
        state = json.loads((work/f"state-r{index}/round.json").read_text(encoding="utf8"))
        for epoch in state["history"]:
            run.log(dict(epoch["metrics"], loss=epoch["loss"], round=index, **budget))
        run.log({"selected_rmse":state["selected_metrics"]["rmse"],"round":index})
        print("SELECT",index,record["selected_artifact_id"],state["selected_metrics"],flush=True)
        if record["selected_artifact_id"] == current:
            checkpoint = work/f"candidate-r{index}/checkpoint.npz"
            parent_id = current
        else:
            assert index == 10 and record["selected_artifact_id"] == parent_id
    run.finish()
    assert parent_id == "round-10"

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as work:
        replay(Path(sys.argv[1]).resolve(),Path(work))

