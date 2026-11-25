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
from gas_head import blend_gas_head


def metrics(model, x, y):
    labels = model.predict(x).argmax(axis=1)
    scores = []
    for i in range(len(model.meta["classes"])):
        den = np.sum(labels == i) + np.sum(y == i)
        scores.append(2*np.sum((labels == i) & (y == i))/max(den,1))
    return dict(f1=np.mean(scores), accuracy=np.mean(labels==y), **dict(zip(model.meta["classes"],scores)))


def replay(parent, tr, val, select, weight, shrinkage, run, phase):
    model = copy.deepcopy(parent)
    branch = f"{phase}/w{weight}_s{shrinkage}"
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
        if rnd==1 and weight:
            other=copy.deepcopy(model)
            blend_gas_head(other,tr["x"],tr["y"],weight,shrinkage)
            if metrics(other,val["x"][select],val["y"][select])["f1"]>best:
                model=other
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
        select[np.flatnonzero(va["y"]==k)[1::2]]=False
    run=swanlab.init(workspace="KiTongZ",project="zhixun-engine-v2",mode="online",on_online_error="raise",
                     experiment_name="A-first-round-gas-head",config={"epochs_per_round":2,"rounds":10,"screen":"even within each class","confirmation":"odd within each class","seed":42})
    candidates=[(0,0)]+[(w,s) for w in (.25,.5,.75,1) for s in (.05,.25,.75)]
    winner=max(candidates,key=lambda p:replay(model,tr,va,select,*p,run,"screen"))
    print("winner",winner,flush=True)
    for phase,mask in (("confirm",select),("full",np.ones_like(select))):
        for params in dict.fromkeys([(0,0),winner]):
            replay(model,tr,va,mask,*params,run,phase)
    run.finish()


if __name__=="__main__":
    main(Path(sys.argv[1]))
