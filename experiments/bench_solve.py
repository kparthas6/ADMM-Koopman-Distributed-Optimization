import os, sys, copy, json, time, tempfile, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import keepout_reference as KR
import koopman_admm as KA, dist_horizon as DH
from torque_sim import TorqueSim
from scenarios import WALL

REPS, CALLS = 15, 60

if __name__ == "__main__":
    calls, solve = [], DH.distributed_horizon
    def capture(*a, **k):
        pre = copy.deepcopy(k["state"]); r = solve(*a, **k)
        calls.append((a, {x: y for x, y in k.items() if x != "state"}, pre)); return r
    KA.DH.distributed_horizon = capture
    KA.run(TorqueSim(control_dt=0.005, obstacles=WALL), KA.learn_operators(), "trot", seconds=4.2, vx=0.3, gait_T=0.25,
           verbose=False, obstacles=WALL, steer=False)
    KA.DH.distributed_horizon = solve
    assert len(calls) >= 500 + 5*CALLS, f"captured {len(calls)} solver calls, need {500 + 5*CALLS}"
    sel = calls[500:500 + 5*CALLS:5]                          # 2.5 to 4 s: held at the wall
    n_bod = len(KA.KEEPOUT_CFG["bodies"])
    conf = [("current code, no obstacles", solve, 0), (f"current code, 1 obstacle ({n_bod} rows/step)", solve, 1),
            (f"current code, 8 obstacles ({8*n_bod} rows/step)", solve, 8)]
    tmp = tempfile.mkdtemp()
    if KR.extract_reference(tmp):
        spec = importlib.util.spec_from_file_location("dist_horizon_ref", os.path.join(tmp, "controller", "dist_horizon.py"))
        ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
        conf.insert(0, ("before the keep-out, no obstacles", ref.distributed_horizon, 0))

    def kwargs(k, n_obs):
        k = dict(k)
        if n_obs == 0:
            k.pop("keepout", None)
        else:
            k["keepout"] = dict(k["keepout"], obstacles=KA.obstacle_array(WALL if n_obs == 8 else WALL[3:4]))
        return k

    def state(pre, n_obs):
        st = copy.deepcopy(pre)
        if n_obs == 0 and st.get("h"):
            st["h"] = {x: y for x, y in st["h"].items() if x in ("f", "l", "act", "iters")}
        return st
    for name, fn, n in conf:                                   # compile
        a, k, pre = sel[0]; fn(*a, state=state(pre, n), **kwargs(k, n))
    res = {name: [] for name, _, _ in conf}
    for a, k, pre in sel:
        per = {name: [] for name, _, _ in conf}
        for _ in range(REPS):
            for name, fn, n in conf:
                kk, st = kwargs(k, n), state(pre, n)
                t0 = time.perf_counter(); fn(*a, state=st, **kk); per[name].append(1000*(time.perf_counter() - t0))
        for name in per:
            res[name].append(float(np.median(per[name])))
    active = int(np.median([c[2]["h"]["on"].sum() for c in sel if c[2].get("h") and "on" in c[2]["h"]]))
    os.makedirs(os.path.join(KR.ROOT, "results"), exist_ok=True)
    json.dump(dict(ms=res, active_rows=active), open(os.path.join(KR.ROOT, "results", "bench.json"), "w"), indent=1)
    for name, v in res.items():
        print(f"{name:40s} median {np.median(v):.3f} ms | p90 {np.percentile(v, 90):.3f} ms")
    print("active keep-out rows (wall, median):", active)
