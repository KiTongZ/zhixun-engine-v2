"""Ten-round B budget sweeps with device-disjoint selection and reporting."""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import swanlab

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contest_submission import Model, read_dataset
from bearing_sampling import balanced_rows, bearing_groups

# epochs after round one, later LR, and fraction of a balanced epoch
PLANS = {"current": (2, .003, 1), "one": (1, .003, 1),
         "one_low": (1, .0015, 1), "half": (1, .003, .5),
         "half_low": (1, .0015, .5), "quarter": (1, .003, .25)}
SMALL_PLANS = {"current": PLANS["current"], "half_low": PLANS["half_low"],
               "eighth_low": (1,.0015,.125), "sixteenth": (1,.003,.0625),
               "sixteenth_low": (1,.0015,.0625), "tiny_low": (1,.0015,.03125)}


def compare(parent, train, val, select, report, policy, fold, run, seed_offset):
    model = copy.deepcopy(parent)
    epochs, rate, fraction = PLANS[policy]
    x, y, ids = train
    vx, vy = val
    def rmse(pred, mask):
        return float(np.sqrt(np.mean((pred[mask] - vy[mask])**2)))
    base = rmse(model.predict(vx), report)
    errors = []
    samples = 0
    cpu = 0.
    for rnd in range(1, 11):
        first = rnd == 1
        model.meta["learning_rate"] = .006 if first else rate
        best = None
        for _ in range(3 if first else epochs):
            start = time.process_time()
            rows = balanced_rows(ids, model.meta["epochs"], model.meta["seed"]+seed_offset, 1)
            if not first:
                rows = rows[:max(1, round(len(rows)*fraction))]
            loss = model.fit(x[rows], y[rows], 1)[0]
            pred = model.predict(vx)
            score = rmse(pred, select)
            if best is None or score < best[0]:
                best = score, copy.deepcopy(model), rmse(pred, report)
            cpu += time.process_time()-start
            samples += len(rows)
            run.log({f"{fold}/{policy}/loss":loss, f"{fold}/{policy}/selection_rmse":score})
        _, model, error = best
        errors.append(error)
        run.log({f"{fold}/{policy}/held_rmse":error,f"{fold}/{policy}/round":rnd})
    path = np.array(errors)
    # Count consecutive rounds below the best performance seen so far.
    streak = longest = 0
    peak = base
    for value in errors:
        if value <= peak:
            peak, streak = value, 0
        else:
            streak += 1
            longest = max(longest, streak)
    gains = 100*(base-path)/base
    result = dict(mean_gain=float(gains.mean()),worst_gain=float(gains.min()),
                  recovery_wait=longest, change=float(np.abs(np.diff(np.r_[base,path])).mean()/base*100),
                  final_rmse=errors[-1],samples=samples,cpu_s=cpu)
    run.log({f"{fold}/{policy}/{k}":v for k,v in result.items()})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--confirm", action="store_true", help="confirm two shortlisted budgets on nine fresh device folds")
    parser.add_argument("--screen-small", action="store_true")
    parser.add_argument("--confirm-plan", choices=SMALL_PLANS, default="half_low")
    args=parser.parse_args()
    if args.screen_small or args.confirm:
        PLANS.update(SMALL_PLANS)
    parent=Model.load(args.package/"checkpoint.npz")
    tr,va=[read_dataset(args.package/"public"/(s+".jsonl"),"regression",[],parent.meta["paths"]) for s in ("train","validation")]
    ids=bearing_groups(va["ids"])
    run=swanlab.init(workspace="KiTongZ",project="zhixun-engine-v2",mode="online",on_online_error="raise",
        experiment_name="B-budget-nine-device-confirmation" if args.confirm else "B-budget-small-draws" if args.screen_small else "B-budget-device-holdout",config={"plans":PLANS,"selected_confirmation":args.confirm_plan,"fold":"full public validation" if args.screen_small else "withhold one train bearing, select on public validation" if args.confirm else "select on other two public bearings","sampling_seed_offsets":[0] if args.confirm else [0,1,2] if args.screen_small else [0,1],"rounds":10})
    records=[]
    if args.screen_small:
        all_rows=np.ones(len(ids),dtype=bool)
        for offset in (0,1,2):
            for policy in SMALL_PLANS:
                result=compare(parent,(tr["x"],tr["y"],tr["ids"]),(va["x"],va["y"]),all_rows,all_rows,policy,f"full_s{offset}",run,offset)
                records.append(dict(fold=f"full_s{offset}",policy=policy,**result))
    if args.confirm:
        train_ids=np.asarray(tr["ids"])
        devices=bearing_groups(train_ids)
        for device in np.unique(devices):
            kept=devices!=device
            x,y=tr["x"][kept],tr["y"][kept]
            meta=parent.meta
            fresh=Model("regression",tr["paths"],[],x,y,hidden=meta["hidden"],seed=meta["seed"],
                        learning_rate=.003,batch_size=meta["batch_size"])
            for step in range(15):
                run.log({device+"/warmup_loss":fresh.fit(x,y,1)[0]})
            vx=np.concatenate([va["x"],tr["x"][~kept]])
            vy=np.concatenate([va["y"],tr["y"][~kept]])
            select=np.arange(len(vy))<len(va["y"])
            for policy in ("current",args.confirm_plan):
                result=compare(fresh,(x,y,train_ids[kept]),(vx,vy),select,~select,policy,device,run,0)
                records.append(dict(fold=device,policy=policy,**result))
            print("confirmed",device,flush=True)
    for offset in (() if args.confirm or args.screen_small else (0,1)):
        for group in np.unique(ids):
            mask=ids==group
            fold=f"{group}_s{offset}"
            for policy in PLANS:
                result=compare(parent,(tr["x"],tr["y"],tr["ids"]),(va["x"],va["y"]),~mask,mask,policy,fold,run,offset)
                records.append(dict(fold=fold,policy=policy,**result))
            print("finished",fold,flush=True)
    args.output.write_text(json.dumps(records,indent=2),encoding="utf8")
    for policy in sorted({r["policy"] for r in records}):
        rows=[r for r in records if r["policy"]==policy]
        print(policy,{k:round(np.mean([r[k] for r in rows]),3) for k in result},flush=True)
    run.finish()


if __name__=="__main__":
    main()


