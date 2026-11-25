"""Compare continuation learning rates on identical public folds and parents."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import swanlab

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contest_submission import Model, _metrics, read_dataset

parser = argparse.ArgumentParser()
parser.add_argument("packages", type=Path)
args = parser.parse_args()
run = swanlab.init(workspace="KiTongZ", project="zhixun-engine-v2",
    experiment_name="public-parent-lr-comparison", mode="online", on_online_error="raise",
    config={"factors":[0.5,1,2], "epochs_per_branch":3, "split":"official public validation"})
for key in "ABC":
    folder = next(args.packages.glob("cpu-" + key.lower() + "-*"))
    parent = Model.load(folder / "checkpoint.npz")
    meta = parent.meta
    train, val = [read_dataset(folder / "public" / (split + ".jsonl"), meta["task"], meta["classes"], meta["paths"]) for split in ("train","validation")]
    lr = json.loads((folder / "baselines/configs" / (key + ".json")).read_text())["learning_rate"]
    for factor in (0.5,1,2):
        model = Model.load(folder / "checkpoint.npz")
        model.meta["learning_rate"] = lr * factor
        for epoch in range(1,4):
            loss = model.fit(train["x"],train["y"],1)[0]
            metrics = _metrics(model,val)
            if key == "C":
                err = model.predict(val["x"]) - val["y"]
                metrics["asymmetric_proxy"] = float(np.expm1(np.maximum(err/10,-err/13)).mean())
            run.log({f"{key}/lr{factor}/{k}":v for k,v in dict(metrics,loss=loss,epoch=epoch,learning_rate=lr*factor).items()})
            print("RESULT",key,factor,epoch,json.dumps(metrics),flush=True)
run.finish()
