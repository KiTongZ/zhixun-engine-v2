"""Compare sampling policies on whole unseen bearings, then check official validation."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
import sys
import copy
import json
from pathlib import Path
import numpy as np
import swanlab
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contest_submission import Model,read_dataset
from bearing_sampling import bearing_groups,balanced_rows
folder=Path(sys.argv[1])
train=read_dataset(folder/"public/train.jsonl","regression")
val=read_dataset(folder/"public/validation.jsonl","regression",[],train["paths"])
groups=bearing_groups(train["ids"])
ids=np.asarray(train["ids"])
x,y=train["x"],train["y"]
run=swanlab.init(workspace="KiTongZ",project="zhixun-engine-v2",mode="online",on_online_error="raise",experiment_name="B-whole-bearing-sampling",config={"powers":[0,.5,1],"warmup":15,"continuation_epochs":20,"seed":42})
results=[]
def compare(parent,xx,yy,names,vx,vy,fold):
    scores=[]
    for power in (0,.5,1):
        m=copy.deepcopy(parent)
        for i in range(20):
            idx=balanced_rows(names,m.meta["epochs"],42,power)
            loss=m.fit(xx[idx],yy[idx],1)[0]
            error=m.predict(vx)-vy
            rmse=float(np.sqrt(np.mean(error**2)))
            mae=float(np.mean(abs(error)))
            run.log({f"{fold}/{power}/loss":loss,f"{fold}/{power}/rmse":rmse,f"{fold}/{power}/mae":mae})
        scores.append([rmse,mae])
    return np.array(scores)
for group in np.unique(groups):
    keep=groups!=group
    m=Model("regression",train["paths"],[],x[keep],y[keep],hidden=24,seed=42,learning_rate=.003,batch_size=128)
    for i in range(15):
        run.log({group+"/warmup_loss":m.fit(x[keep],y[keep],1)[0]})
    scores=compare(m,x[keep],y[keep],ids[keep],x[~keep],y[~keep],group)
    results.append(scores)
    print(group,scores.round(2).tolist(),flush=True)
a=np.array(results)
print("SUMMARY",a.mean(axis=0), "RMSE WINS",(a[:,:,0]<a[:,:1,0]).sum(axis=0),flush=True)
m=Model.load(folder/"checkpoint.npz")
print("OFFICIAL",compare(m,x,y,ids,val["x"],val["y"],"official"),flush=True)
run.finish()

