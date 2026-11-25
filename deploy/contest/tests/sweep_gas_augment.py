"""Screen on alternating class-stratified rows; reserve the other rows for confirmation."""
import argparse
import copy
import os
from pathlib import Path
import sys
import numpy as np
import swanlab
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contest_submission import Model, read_dataset
from gas_augment import gas_training_rows, dropout_gas_sensors
from sweep_gas_head import metrics


def replay(parent, tr, val, select, spread, balancing, run, phase, seed_offset=0):
    model = copy.deepcopy(parent)
    branch = f"{phase}/amp{spread}_bal{balancing}_seed{seed_offset}"
    trace = []
    for rnd in range(1,11):
        model.meta["learning_rate"] = .003
        best = -1.
        for ep in range(2):
            transform = dropout_gas_sensors if isinstance(balancing,str) else gas_training_rows
            x,y=transform(model,tr["x"],tr["y"],spread,balancing,seed_offset)
            loss=model.fit(x,y,1)[0]
            score=metrics(model,val["x"][select],val["y"][select])["f1"]
            run.log({branch+"/loss":loss,branch+"/select_f1":score})
            if score > best:
                best,candidate=score,copy.deepcopy(model)
        model=candidate
        mask = ~select if phase=="confirm" else select
        result=metrics(model,val["x"][mask],val["y"][mask])
        run.log({branch+"/"+k:float(v) for k,v in dict(result,round=rnd).items()})
        trace.append(result["f1"])
    summary=[round(float(i),5) for i in (np.mean(trace),min(trace),trace[-1],trace[0])]
    print(branch,summary,{k:round(float(v),3) for k,v in result.items()},flush=True)
    return np.mean(trace)


def main(folder, dropout=False):
    model=Model.load(folder/"checkpoint.npz")
    tr,va=[read_dataset(folder/"public"/(s+".jsonl"),"classification",model.meta["classes"],model.meta["paths"]) for s in ("train","validation")]
    select=np.ones(len(va["y"]),dtype=bool)
    for k in range(len(model.meta["classes"])):
        rows=np.flatnonzero(va["y"]==k)
        select[rows[(len(rows)+1)//2:]]=False
    run=swanlab.init(workspace="KiTongZ",project="zhixun-engine-v2",mode="online",on_online_error="raise",
                     experiment_name="A-gas-sensor-dropout" if dropout else "A-gas-amplitude-class-balance",config={"epochs_per_round":2,"rounds":10,"screen":"earlier half within each class","confirmation":"later half within each class","seed":42})
    candidates=[(0,0)]+[(w,s) for w in (0,.4,1,1.6,2.2) for s in (0,.5,1) if w or s]
    if dropout:
        candidates=[(0,0)]+[(p,m) for p in (.05,.15,.3,.5,.7) for m in ("sensor","feature")]
    winner=max(candidates,key=lambda p:replay(model,tr,va,select,*p,run,"screen"))
    print("winner",winner,flush=True)
    for phase,mask in (("confirm",select),("full",np.ones_like(select))):
        for params in dict.fromkeys([(0,0),winner]):
            for seed in ((0,1,2) if params!=(0,0) else (0,)):
                replay(model,tr,va,mask,*params,run,phase,seed)
    run.finish()


if __name__=="__main__":
    main(Path(sys.argv[1]), "--dropout" in sys.argv[2:])
