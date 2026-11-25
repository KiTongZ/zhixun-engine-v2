"""Screen on the first half of each class; reserve the other rows for confirmation."""
import argparse
import copy
import os
from pathlib import Path
import sys
import numpy as np
import swanlab
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contest_submission import Model, read_dataset
from gas_bias import adjust_gas_bias
from sweep_gas_head import metrics


def replay(parent, tr, val, select, strength, adjust_rounds, run, phase):
    model = copy.deepcopy(parent)
    branch = f"{phase}/strength{strength}_rounds{adjust_rounds}"
    trace = []
    for rnd in range(1,11):
        model.meta["learning_rate"] = .003
        best = -1.
        for ep in range(2):
            loss=model.fit(tr["x"],tr["y"],1)[0]
            score=metrics(model,val["x"][select],val["y"][select])["f1"]
            run.log({branch+"/loss":loss,branch+"/select_f1":score})
            if score > best:
                best,candidate=score,copy.deepcopy(model)
        model=candidate
        if strength and rnd <= adjust_rounds:
            adjusted=copy.deepcopy(model)
            adjust_gas_bias(adjusted,tr["y"],strength)
            if metrics(adjusted,val["x"][select],val["y"][select])["f1"]>best:
                model=adjusted
        mask = ~select if phase=="confirm" else select
        result=metrics(model,val["x"][mask],val["y"][mask])
        run.log({branch+"/"+k:float(v) for k,v in dict(result,round=rnd).items()})
        trace.append(result["f1"])
    summary=[round(float(i),5) for i in (np.mean(trace),min(trace),trace[-1],trace[0])]
    print(branch,summary,{k:round(float(v),3) for k,v in result.items()},flush=True)
    return np.mean(trace)


def main(folder):
    model=Model.load(folder/"checkpoint.npz")
    tr,va=[read_dataset(folder/"public"/(s+".jsonl"),"classification",model.meta["classes"],model.meta["paths"]) for s in ("train","validation")]
    select=np.ones(len(va["y"]),dtype=bool)
    for k in range(len(model.meta["classes"])):
        rows=np.flatnonzero(va["y"]==k)
        select[rows[(len(rows)+1)//2:]]=False
    run=swanlab.init(workspace="KiTongZ",project="zhixun-engine-v2",mode="online",on_online_error="raise",
                     experiment_name="A-training-prior-adjustment",config={"epochs_per_round":2,"rounds":10,"screen":"earlier half within each class","confirmation":"later half within each class","seed":42})
    candidates=[(0,0)]+[(w,n) for w in (1,2,4,6,8,12) for n in (1,2)]
    winner=max(candidates,key=lambda p:replay(model,tr,va,select,*p,run,"screen"))
    print("winner",winner,flush=True)
    for phase,mask in (("confirm",select),("full",np.ones_like(select))):
        for params in dict.fromkeys([(0,0),winner]):

            replay(model,tr,va,mask,*params,run,phase)
    run.finish()


if __name__=="__main__":
    main(Path(sys.argv[1]))
