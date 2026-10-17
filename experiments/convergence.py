import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import keepout_reference as KR

ITERS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

if __name__ == "__main__":
    out = {"iters": ITERS, "cases": {}}
    for name, (seed, obs, bodies) in KR.CASES.items():
        P = KR.instance(seed, obs, bodies)
        Wq, z, act = KR.dense_qp(P, KR.reference_rows(P))
        Gr, hr = np.array([r[1] for r in act]), np.array([r[2] for r in act])
        warm = {}                                     # converge on a nearby state, keep primal and duals only
        P2 = dict(P, x0=P["x0"] + np.r_[0.02, -0.01, 0, 0, 0, 0.02])
        KR.solve(P2, 3000, warm)
        warm["h"] = {k: v for k, v in warm["h"].items() if k != "xp"}   # same linearization point as the QP
        res = {"cold": [], "warm": []}
        for it in ITERS:
            for mode in ("cold", "warm"):
                W, _ = KR.solve(P, it, None if mode == "cold" else {"h": dict(warm["h"])})
                res[mode].append([float(np.abs(W - Wq).max()), float((Gr @ W.reshape(-1) - hr).max()) if len(hr) else 0.0])
        out["cases"][name] = dict(res, active_rows=len(act), max_multiplier=float(z.max()) if len(z) else 0.0)
        print(f"{name:30s} rows {len(act):3d} | |W - W_qp| at 8 iterations: cold {res['cold'][3][0]:.2e} N, "
              f"warm {res['warm'][3][0]:.2e} N | at 2048: {res['cold'][-1][0]:.1e} N")
    k = np.arange(KR.N)
    out["tube"] = dict(step=(k + 1).tolist(), margin=0.02,     # run()'s KEEPOUT_CFG: sigma_f 20 N, margin 2 cm
                       force=(20.0*KR.DT**2*(k + 1)**2/(2*KR.M)).tolist())
    os.makedirs(os.path.join(KR.ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(KR.ROOT, "results", "convergence.json"), "w"), indent=1)
