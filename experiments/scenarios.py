"""Obstacle scenarios for the keep-out, run in MuJoCo without rendering.

    python scenarios.py                       # every scenario x variant -> ../results/scenarios/
    python scenarios.py headon wall           # selected scenarios
    python scenarios.py --variants rows+steer

Variants: none (obstacles only in the scene), rows (keep-out rows, no steering), rows+steer.
Each run saves its log, with a summary, as <scenario>_<variant>.npz; plot_results.py tabulates them."""
import os, sys, json, time, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "common")); sys.path.insert(0, os.path.join(ROOT, "controller"))
import numpy as np
import koopman_admm as KA
from torque_sim import TorqueSim

RESULTS = os.path.join(ROOT, "results")
WALL = [(1.2, y, 0.15) for y in (-1.05, -0.75, -0.45, -0.15, 0.15, 0.45, 0.75, 1.05)]
SCENARIOS = {   # obstacles: (x, y, r) vertical cylinders, (x, y, z, r) spheres; trot commands
    "headon":   dict(title="Cylinder head-on, 0.3 m/s", obst=[(1.2, 0.0, 0.15)], vx=0.3, gait_T=0.25, seconds=9),
    "fast":     dict(title="Cylinder head-on, 0.4 m/s, gait_T 0.30", obst=[(1.2, 0.0, 0.15)], vx=0.4, gait_T=0.30, seconds=8),
    "sphere":   dict(title="Sphere at trunk height", obst=[(1.2, 0.0, 0.32, 0.18)], vx=0.3, gait_T=0.25, seconds=9),
    "gap":      dict(title="50 cm gap", obst=[(1.2, 0.36, 0.10), (1.2, -0.36, 0.10)], vx=0.3, gait_T=0.25, seconds=9),
    "wall":     dict(title="8-cylinder wall", obst=WALL, vx=0.3, gait_T=0.25, seconds=12),
    "staggered": dict(title="Three staggered cylinders", obst=[(1.0, 0.08, 0.12), (2.0, -0.12, 0.12), (3.0, 0.08, 0.12)],
                     vx=0.3, gait_T=0.25, seconds=14),
    "diagonal": dict(title="Diagonal walk (vy 0.2) past a cylinder", obst=[(1.30, 0.62, 0.10)], vx=0.3, vy=0.2,
                     gait_T=0.25, seconds=8),
}
VARIANTS = ("none", "rows", "rows+steer")


def run_scenario(name, variant, legs=None, sim=None, verbose=False):
    """One closed-loop run. Returns (summary dict, log). For 'none' the controller ignores the obstacles,
    so its clearance and contacts are computed here from the trajectory and the simulator."""
    sc = SCENARIOS[name]
    legs = KA.learn_operators() if legs is None else legs
    sim = TorqueSim(control_dt=0.005, obstacles=sc["obst"]) if sim is None else sim
    kw = dict(mode="trot", seconds=sc["seconds"], vx=sc["vx"], vy=sc.get("vy", 0.0), gait_T=sc["gait_T"], verbose=verbose)
    hits = [0]
    if variant == "none":
        step = sim.step
        def counted(tau):
            out = step(tau); hits[0] += sim.obstacle_contacts() > 0; return out
        sim.step = counted
    else:
        kw.update(obstacles=sc["obst"], steer=(variant == "rows+steer"))
    t0 = time.time()
    L, up = KA.run(sim, legs, **kw)
    wall = time.time() - t0
    if variant == "none":
        obs, bod = KA.obstacle_array(sc["obst"]), np.asarray(KA.KEEPOUT_CFG["bodies"], float)
        L["clear"] = np.array([KA.keepout_clearance(q[None, :3] + bod[:, :3] @ KA.quat2R(q[3:7]).T, obs, bod)
                               for q in L["qpos"]])
        L["xy"] = L["qpos"][:, :2]
        contact = hits[0]
    else:
        contact = int((L["hits"] > 0).sum())
    tilt = np.maximum(np.abs(L["roll"]), np.abs(L["pitch"]))
    summary = dict(scenario=name, title=sc["title"], variant=variant, upright=bool(up and tilt.max() < 45),
                   min_clearance=float(L["clear"].min()), contact_ticks=int(contact), max_tilt=float(tilt.max()),
                   end_xy=[float(v) for v in L["xy"][-1]], max_abs_y=float(np.abs(L["xy"][:, 1]).max()),
                   ticks=int(len(L["roll"])), wall_s=wall)
    return summary, L


def save(summary, L, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{summary['scenario']}_{summary['variant'].replace('+', '_')}.npz"),
             summary=json.dumps(summary), **{k: np.asarray(v) for k, v in L.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--out", default=os.path.join(RESULTS, "scenarios"))
    a = ap.parse_args()
    legs = KA.learn_operators()
    for name in a.scenarios:
        for v in a.variants:
            s, L = run_scenario(name, v, legs)
            save(s, L, a.out)
            print(f"{name:9s} {v:11s} | up {s['upright']!s:5s} | min clearance {100*s['min_clearance']:+6.1f} cm | "
                  f"contact ticks {s['contact_ticks']:5d} | tilt {s['max_tilt']:5.1f} deg | "
                  f"end ({s['end_xy'][0]:+.2f}, {s['end_xy'][1]:+.2f}) | {s['wall_s']:.0f} s")
