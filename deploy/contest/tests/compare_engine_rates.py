"""Replay C budgets on public data and compare late-round rates by operating subset."""
import argparse
import copy
import os
import sys
import time
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("ENGINE_BLAS_THREADS", "1")
import numpy as np
import swanlab

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contest_submission import Model, read_dataset


def replay(model, rounds, rate, branch, train, validation, run):
    groups = np.array([item.split("_")[0] for item in validation["ids"]])
    for index in rounds:
        model.meta["learning_rate"] = .001 if index == 1 else rate
        best = None
        for epoch in range(1 if index == 1 else 2):
            start = time.perf_counter()
            loss = model.fit(train["x"], train["y"], 1)[0]
            err = model.predict(validation["x"]) - validation["y"]
            metrics = {"rmse": np.sqrt(np.mean(err**2)), "mae": np.mean(abs(err)),
                       "asymmetric_mean": np.mean(np.expm1(np.maximum(err/10, -err/13))),
                       "overestimate_p95": np.percentile(np.maximum(err, 0), 95)}
            for group in np.unique(groups):
                metrics[group] = np.sqrt(np.mean(err[groups == group]**2))
            run.log({f"{branch}/{k}": float(v) for k,v in dict(metrics,loss=loss,round=index,epoch=epoch+1,lr=model.meta["learning_rate"],seconds=time.perf_counter()-start).items()})
            if best is None or metrics["rmse"] < best[0]["rmse"]:
                best = metrics, copy.deepcopy(model)
        metrics, model = best
        print(branch,index,{k:round(float(v),3) for k,v in metrics.items()},flush=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package",type=Path)
    folder = parser.parse_args().package
    model = Model.load(folder/"checkpoint.npz")
    train,val = [read_dataset(folder/"public"/(s+".jsonl"),"regression",[],model.meta["paths"]) for s in ("train","validation")]
    run = swanlab.init(workspace="KiTongZ",project="zhixun-engine-v2",experiment_name="C-late-rate-public-replay",mode="online",on_online_error="raise",config={"source":"public baseline","blas_threads":os.environ["OPENBLAS_NUM_THREADS"],"shared_rounds":5,"rates":[.001,.002,.004],"seed":model.meta["seed"]})
    run.log({"platform/B_hidden_rmse":217.029778,"platform/B_hidden_mae":141.506738,"platform/A_hidden_macro_f1":.87553,"platform/C_hidden_rmse":56.369377,"platform/input_tokens":16644,"platform/output_tokens":9600,"platform/predictions":14773})
    parent = replay(model,range(1,6),.002,"shared",train,val,run)
    for rate in (.001,.002,.004):
        replay(copy.deepcopy(parent),range(6,11),rate,f"lr{rate}",train,val,run)
    run.finish()


if __name__ == "__main__":
    main()

