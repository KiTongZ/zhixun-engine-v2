"""Compare candidate data loading with the previously submitted implementation."""
import argparse
import statistics
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
import numpy as np
import swanlab
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines import data
from baselines.model import Model

def main(folder, revision):
    repo=Path(__file__).resolve().parents[3]
    prior=types.ModuleType("prior_data")
    code=subprocess.check_output(["git","show",f"{revision}:deploy/contest/baselines/data.py"],cwd=repo)
    exec(compile(code,"prior_data.py","exec"),prior.__dict__)
    invalid=[{"true_RUL":1},{"x":True},{"x":float("inf")},{"x":None},[]]
    for value in invalid:
        for loader in (prior,data):
            try: loader.flatten(value)
            except ValueError: continue
            raise AssertionError(value)
    for raw in ('{"x":1,"x":2}', '{"x":{"y":1,"y":2}}', '[NaN]', '[Infinity]', '[0] []'):
        for loader in (prior,data):
            try: loader.json_value(raw)
            except ValueError: continue
            raise AssertionError(raw)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "chunk-boundary.jsonl"
        # Cut the first multibyte ID exactly in the middle of a UTF-8 sequence.
        prefix = b'{"id":"'
        padding = b" " * (1024 * 1024 - len(prefix) - 1)
        row = '气","inputs":{"x":2},"answer":3}\r\n'.encode()
        path.write_bytes(padding + prefix + row + b'\r{"id":"end","inputs":{"x":5},"answer":7}')
        before = prior.read_dataset(path, "regression")
        after = data.read_dataset(path, "regression")
        for key in ("ids", "paths", "sha256"):
            assert before[key] == after[key]
        for key in ("x", "y"):
            np.testing.assert_array_equal(before[key], after[key])
    run=swanlab.init(workspace="KiTongZ",project="zhixun-engine-v2",mode="online",on_online_error="raise",
                     experiment_name="ABC-feature-path-cache-parity",config={"baseline":revision,"repeats":3,"scope":"JSONL loader CPU time"})
    for key in "ABC":
        pack=next(folder.glob("cpu-"+key.lower()+"-*"))
        model=Model.load(pack/"checkpoint.npz")
        for split in ("train","validation"):
            times={"old":[],"new":[]}
            for rep in range(3):
                outputs={}
                for label in (("old","new") if rep%2==0 else ("new","old")):
                    start=time.process_time()
                    loader=prior if label=="old" else data
                    outputs[label]=loader.read_dataset(pack/"public"/(split+".jsonl"),model.meta["task"],model.meta["classes"],model.meta["paths"])
                    times[label].append(time.process_time()-start)
                a,b=outputs["old"],outputs["new"]
                for k in ("ids","paths","sha256"): assert a[k]==b[k]
                for k in ("x","y"): np.testing.assert_array_equal(a[k],b[k])
                np.testing.assert_array_equal(model.predict(a["x"]),model.predict(b["x"]))
                run.log({f"{key}/{split}/{label}_cpu_s":t[-1] for label,t in times.items()})
            print(key,split,len(a["ids"]),{k:round(statistics.median(v),3) for k,v in times.items()},flush=True)
    run.finish()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("packages",type=Path)
    parser.add_argument("--revision",default="6ea97e2")
    args=parser.parse_args()
    main(args.packages,args.revision)
