"""Evaluate engine mini-batch and tail weight averages with an engine-level split."""
import argparse
import copy
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import swanlab

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contest_submission import Model, read_dataset
from engine_head import moderate_update, refine_head


def evaluate(errors, mask):
    err = errors[mask]
    return dict(rmse=float(np.mean(err**2)**.5), mae=float(np.mean(abs(err))),
                asymmetric_mean=float(np.expm1(np.maximum(err/10, -err/13)).mean()))


def replay(parent, train, valid, select, report, batch, average, seed, run, phase,
           warm_rounds=10, head_each_round=False, stages=None, moderated=False):
    model = copy.deepcopy(parent)
    model.meta.update(batch_size=batch, seed=seed)
    prefix = f"{phase}/s{seed}_b{batch}_avg{int(average)}_warm{warm_rounds}_head{int(head_each_round)}"
    trace = []
    steps = 0
    start = time.perf_counter()
    for rnd in range(1, 11):
        current_train = train if stages is None else stages[rnd-1]
        anchor = copy.deepcopy(model) if moderated else None
        model.meta["batch_size"] = batch if rnd <= warm_rounds else parent.meta["batch_size"]
        model.meta["learning_rate"] = .001 if rnd == 1 else .002 if rnd <= 5 else .004
        best_rmse = float("inf")
        for epoch in range(1 if rnd == 1 else 2):
            before = model.meta["steps"]
            loss = model.fit(current_train["x"], current_train["y"], 1, average_tail=average and rnd <= warm_rounds)[0]
            steps += model.meta["steps"] - before
            errors = model.predict(valid["x"]) - valid["y"]
            score = evaluate(errors, select)["rmse"]
            run.log({prefix+"/epoch_loss":loss, prefix+"/selection_rmse":score})
            if score < best_rmse:
                best_model, best_rmse, best_errors = copy.deepcopy(model), score, errors
        model = best_model
        if rnd == 10 or head_each_round:
            refined = copy.deepcopy(model)
            refine_head(refined, current_train["x"], current_train["y"], .001)
            errors = refined.predict(valid["x"]) - valid["y"]
            if evaluate(errors, select)["rmse"] < best_rmse:
                model, best_errors = refined, errors
        fraction = 1.
        if moderated:
            model, fraction = moderate_update(anchor, model, valid["x"][select], valid["y"][select])
            best_errors = model.predict(valid["x"]) - valid["y"]
        metrics = evaluate(best_errors, report)
        trace.append(metrics["rmse"])
        run.log({prefix+"/"+k:v for k,v in dict(metrics, round=rnd, gradient_steps=steps, fraction=fraction).items()})
    result = dict(mean=float(np.mean(trace)), worst=max(trace), final=trace[-1], first=trace[0],
                  grad_steps=steps, seconds=round(time.perf_counter()-start,2), final_mae=metrics["mae"],
                  final_asymmetric=metrics["asymmetric_mean"])
    run.log({prefix+"/summary_"+k:v for k,v in result.items()})
    print(prefix, result, flush=True)
    return result


def main(folder, hybrid, shift_rounds, head_each, guarded):
    parent = Model.load(folder / "checkpoint.npz")
    train, valid = [read_dataset(folder/"public"/(s+".jsonl"), "regression", [], parent.meta["paths"])
                    for s in ("train","validation")]
    engines = [key.rsplit("_",1)[0] for key in valid["ids"]]
    held = np.array([int.from_bytes(hashlib.sha256(k.encode()).digest()[:4],"big") % 3 == 0 for k in engines])
    full = np.ones(len(held), dtype=bool)
    base = parent.meta["batch_size"]
    cases = [(base*m, avg) for m in (1,2,4,8) for avg in (False,True)]
    schedules = [(base,False,10,False)] + [(512,True,w,h) for w in (3,5,7,9,10) for h in (False,True)]
    if not hybrid:
        schedules = [(b,a,10,False) for b,a in cases]
    run = swanlab.init(workspace="KiTongZ",project="zhixun-engine-v2",mode="online",on_online_error="raise",
                       experiment_name=("C-moderated-update-paired" if guarded else "C-operating-subset-shifts" if shift_rounds else
                                        "C-early-average-late-finetune" if hybrid else "C-batch-tail-average-engine-holdout"),
                       config={"held_engines":len(set(np.array(engines)[held])), "tail":"up to 8 equally spaced snapshots in second half of epoch",
                               "rank":"lowest mean RMSE, then worst round","baseline_batch_size":base,
                               "final_screen_guard":hybrid, "shift_warm_rounds":shift_rounds, "shift_head_each":head_each,
                               "moderate":guarded, "guard_fractions":[1,.5,.125]})
    try:
        if shift_rounds:
            groups = np.array([key.split("_")[0] for key in train["ids"]])
            partitions = [{k:train[k][groups==g] for k in ("x","y")} for g in sorted(set(groups))]
            assert len(partitions) == 4
            for sequence in ((0,0,1,1,2,2,3,3,0,0),(3,3,2,2,1,1,0,0,3,3)):
                stages = [partitions[g] for g in sequence]
                for seed in (42,43,44):
                    for b,a,w,h in ((base,False,10,False),(512,True,shift_rounds,head_each)):
                        replay(parent,train,valid,~held,held,b,a,seed,run,
                               "shift"+"".join(map(str,sequence)),w,h,stages,guarded and a)
            if guarded:
                for phase, select, report in (("confirm", ~held, held), ("full",full,full)):
                    for seed in (42,43,44):
                        for b,a,w,h in ((base,False,10,False),(512,True,shift_rounds,head_each)):
                            replay(parent,train,valid,select,report,b,a,seed,run,phase,w,h,moderated=a)
            run.finish()
            return
        def measure(case, seed, select, report, phase):
            batch, average, warm_rounds, head_each = case
            return replay(parent,train,valid,select,report,batch,average,seed,run,phase,warm_rounds,head_each)
        trials = {c:measure(c,42,~held,~held,"screen") for c in schedules}
        control = schedules[0]
        eligible = [c for c in schedules if not hybrid or trials[c]["final"] <= trials[control]["final"]]
        best = min(eligible,key=lambda c:(trials[c]["mean"],trials[c]["worst"]))
        print("screen winner",best,flush=True)
        for phase, select, report in (("confirm",~held,held),("full",full,full)):
            for seed in (42,43,44):
                for case in dict.fromkeys((control,best)):
                    measure(case,seed,select,report,phase)
    except Exception as exc:
        run.finish(state="crashed",error=str(exc))
        raise
    run.finish()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package",type=Path)
    parser.add_argument("--hybrid",action="store_true",help="Require final screening RMSE to match or improve baseline.")
    parser.add_argument("--shift-rounds",type=int,choices=range(1,11),help="Stress test one fixed schedule under two shifts.")
    parser.add_argument("--head-each", action="store_true")
    parser.add_argument("--guarded", action="store_true",help="Test positive update fractions when RMSE or tail loss regresses.")
    args = parser.parse_args()
    main(args.package, args.hybrid, args.shift_rounds, args.head_each, args.guarded)
