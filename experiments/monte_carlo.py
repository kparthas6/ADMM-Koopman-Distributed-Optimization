"""Monte Carlo of the obstacle scenarios on MJX: per scenario and variant, a batch of robots with a randomized
start pose and obstacle placement, all in one jitted closed-loop program (controller/jax_loop.py). Use
batches in the hundreds or thousands on a GPU; keep them small on a CPU.

    python monte_carlo.py --batch 1024                          # every scenario, rows+steer
    python monte_carlo.py headon wall --batch 64 --variants none rows+steer --jitter 0.05
    python monte_carlo.py --batch 1024 --ls-iterations 8        # MJX line search capped: faster batches

Randomized per robot: start x in +-dx, y in +-dy, heading in +-dyaw, and each obstacle's xy by a normal
of std jitter (m). A run succeeds when the robot stays upright, never touches an obstacle and ends 0.3 m
past the farthest one (the closed-loop test's rule). Writes per-robot arrays to <out>/<scenario>_<variant>.npz
('+' in the variant becomes '_'; <out> defaults to ../results/monte_carlo) and merges the summaries into
<out>/summary.json, keyed by scenario and variant. Wall times include compilation; bench_gpu.py measures
throughput."""
import os, sys, json, time, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "common")); sys.path.insert(0, os.path.join(ROOT, "controller"))
import backend
import numpy as np
import jax_loop as JL
from scenarios import SCENARIOS, VARIANTS, RESULTS


def run_mc(name, variant, batch, seed=0, dx=0.05, dy=0.05, dyaw=0.1, jitter=0.03, legs=None, ls_iterations=None):
    """Returns (summary dict, per-robot arrays)."""
    sc = SCENARIOS[name]
    rng = np.random.default_rng(seed)
    lay = [[(o[0] + rng.normal(0, jitter), o[1] + rng.normal(0, jitter)) + tuple(o[2:]) for o in sc["obst"]]
           for _ in range(batch)]
    roll = JL.Rollout(legs, obstacles=sc["obst"], gait_T=sc["gait_T"], keepout=variant != "none",
                      steer=variant == "rows+steer", ls_iterations=ls_iterations)
    kw = dict(vx=sc["vx"], vy=sc.get("vy", 0.0), x0=rng.uniform(-dx, dx, batch), y0=rng.uniform(-dy, dy, batch),
              yaw0=rng.uniform(-dyaw, dyaw, batch), obstacles=lay)
    t0 = time.time(); out = roll.run(sc["seconds"], **kw); t1 = time.time()
    passed = out["end_xy"][:, 0] > np.array([max(o[0] for o in l) for l in lay]) + 0.3
    free = (out["contact_ticks"] == 0) & (out["min_clear"] > 0)
    ok = out["upright"] & free & passed
    pct = lambda q: float(np.percentile(out["min_clear"], q))
    summary = dict(scenario=name, variant=variant, batch=batch, seconds=sc["seconds"],
                   mjx_iterations=int(roll.m.opt.iterations), mjx_ls_iterations=int(roll.m.opt.ls_iterations),
                   success=float(ok.mean()),
                   upright=float(out["upright"].mean()), collision_free=float(free.mean()), passed=float(passed.mean()),
                   min_clear_min=pct(0), min_clear_p5=pct(5), min_clear_median=pct(50),
                   max_tilt_max=float(out["max_tilt"].max()), wall_s_incl_compile=t1 - t0, backend=backend.info())
    return summary, dict(out, success=ok, passed=passed, obstacles=np.array([JL.KA.obstacle_array(l) for l in lay]),
                         x0=kw["x0"], y0=kw["y0"], yaw0=kw["yaw0"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--variants", nargs="*", default=["rows+steer"], choices=VARIANTS)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dx", type=float, default=0.05)
    ap.add_argument("--dy", type=float, default=0.05)
    ap.add_argument("--dyaw", type=float, default=0.1)
    ap.add_argument("--jitter", type=float, default=0.03)
    ap.add_argument("--ls-iterations", type=int, default=None, help="cap MJX's line search (default: the model's 50)")
    ap.add_argument("--out", default=os.path.join(RESULTS, "monte_carlo"))
    a = ap.parse_args()
    print(backend.info())
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "summary.json")
    legs, rows = JL.KA.learn_operators(), (json.load(open(path)) if os.path.exists(path) else [])
    for name in a.scenarios:
        for v in a.variants:
            s, arr = run_mc(name, v, a.batch, a.seed, a.dx, a.dy, a.dyaw, a.jitter, legs, a.ls_iterations)
            np.savez(os.path.join(a.out, f"{name}_{v.replace('+', '_')}.npz"), summary=json.dumps(s), **arr)
            rows = [r for r in rows if (r["scenario"], r["variant"]) != (name, v)] + [s]
            json.dump(rows, open(path, "w"), indent=1)
            pc = {k: 100*s[f"min_clear_{k}"] for k in ("min", "p5", "median")}
            print(f"{name:9s} {v:11s} | B {a.batch:5d} | success {100*s['success']:5.1f}% | "
                  f"upright {100*s['upright']:5.1f}% | collision-free {100*s['collision_free']:5.1f}% | "
                  f"passed {100*s['passed']:5.1f}% | min clearance min {pc['min']:+5.1f} / p5 {pc['p5']:+5.1f} / "
                  f"median {pc['median']:+5.1f} cm | {s['wall_s_incl_compile']:.0f} s incl. compile")
