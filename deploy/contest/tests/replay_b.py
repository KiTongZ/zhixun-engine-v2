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

def replay(package, work, dataset="B"):
    if dataset not in ("B", "C"):
        raise ValueError("expected B or C")
    entry.WORK = work
    checkpoint = package / "checkpoint.npz"
    parent_id = "official-" + dataset
    run = swanlab.init(workspace="KiTongZ", project="zhixun-engine-v2", mode="online",
                       on_online_error="raise", experiment_name=("B-small-draw-round-replay" if dataset == "B" else "C-head-round-replay"),
                       config={"split":"official public validation","decision_source":"recorded budgets"})
    for index in range(1, 11):
        budget = {"epochs": 3 if index == 1 else 2, "learning_rate": .006 if index == 1 else .003}
        if dataset == "C":
            budget = {"epochs": 1 if index == 1 else 2, "learning_rate": .001 if index == 1 else .002 if index <= 5 else .004}
        current = f"round-{index}"
        record = {}
        rnd = SimpleNamespace(dataset_key=dataset, round_index=index, call_ids=["local"], max_output_tokens=320,
            task={"task":"regression","public_projection":{f"{s}_path":package/"public"/(s+".jsonl") for s in ("train","validation")}},
            parent=SimpleNamespace(artifact_id=parent_id, descriptor={"checkpoint_paths":["checkpoint.npz"]},
                                   path=lambda name,root: checkpoint),
            model=lambda *a,**k: SimpleNamespace(call_id="local",text=json.dumps(budget)),
            history=lambda:[], submit_candidate=lambda *a,**k: {"artifact_id":current},
            save_state=lambda path:None, finish=lambda **fields:record.update(fields))
        entry.train(rnd)
        state = json.loads((work/f"state-r{index}/round.json").read_text(encoding="utf8"))
        for epoch in state["history"]:
            run.log(dict(epoch["metrics"], loss=epoch["loss"], round=index, epochs=state["epochs"], learning_rate=state["learning_rate"], training_rows=epoch["training_rows"]))
        run.log({"selected_rmse":state["selected_metrics"]["rmse"],"round":index})
        print("SELECT",index,record["selected_artifact_id"],state["selected_metrics"],flush=True)
        if record["selected_artifact_id"] == current:
            checkpoint = work/f"candidate-r{index}/checkpoint.npz"
            parent_id = current
        else:
            assert index == 10 and record["selected_artifact_id"] == parent_id
    if dataset == "C":
        final = entry.Model.load(checkpoint)
        val = entry.read_dataset(package/"public/validation.jsonl", "regression", [], final.meta["paths"])
        assert abs(entry._metrics(final, val)["rmse"] - state["selected_metrics"]["rmse"]) < 1e-8
        run.log({"final_head_refined": int(state["head_refined"])})
    run.finish()
    assert parent_id == "round-10"

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as work:
        replay(Path(sys.argv[1]).resolve(),Path(work), sys.argv[2] if len(sys.argv) > 2 else "B")




