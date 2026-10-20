"""Figures and the results table from ../results (scenarios.py, convergence.py, bench_solve.py outputs).

    python plot_results.py                      # -> ../results/figures/
    python plot_results.py --out ../docs/figures

Needs matplotlib (requirements-dev.txt)."""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scenarios import SCENARIOS, RESULTS

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL = {"rows+steer": "#2a78d6", "rows": "#eb6834", "none": "#1baf7a"}          # one colour per variant, everywhere
NAME = {"rows+steer": "keep-out rows + steering", "rows": "keep-out rows only", "none": "no keep-out"}
SOLID = dict(frameon=True, facecolor=SURF, edgecolor="none", framealpha=0.92)      # legends that sit on data
plt.rcParams.update({"font.size": 10, "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "text.color": INK, "lines.linewidth": 2, "lines.solid_capstyle": "round"})


def load_runs(path):
    runs = {}
    for f in sorted(glob.glob(os.path.join(path, "*.npz"))):
        d = dict(np.load(f)); s = json.loads(str(d.pop("summary"))); d["s"] = s
        runs[(s["scenario"], s["variant"])] = d
    return runs


def obstacle_patches(ax, name):
    for o in SCENARIOS[name]["obst"]:
        ax.add_patch(Circle((o[0], o[1]), o[2] if len(o) == 3 else o[3], color=AXIS, lw=0, zorder=2))


def fig_paths(runs, out):
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.6)); axs = axs.ravel()
    for ax, name in zip(axs, SCENARIOS):
        for v in ("none", "rows", "rows+steer"):
            d = runs.get((name, v))
            if d is None:
                continue
            xy = d["xy"]
            ax.plot(xy[:, 0], xy[:, 1], color=COL[v], label=NAME[v], zorder=3)
            ax.plot(*xy[-1], "o", ms=8, color=COL[v], mec=SURF, mew=2, zorder=4)
            if not d["s"]["upright"]:
                ax.plot(*xy[-1], "x", ms=10, color=INK, mew=2, zorder=5)
        obstacle_patches(ax, name)
        ax.set_title(SCENARIOS[name]["title"], loc="left", fontsize=10); ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    axs[-1].set_visible(False)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, "Trunk path seen from above. Grey discs: obstacles. Dot: end of run; x: fell. The keep-out acts on "
             "three 17 cm body spheres at the trunk origin and 19 cm ahead of and behind it.", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95)); fig.savefig(os.path.join(out, "01_paths_top_down.png"), dpi=150); plt.close(fig)


def fig_clearance(runs, out):
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    for ax, name in zip(axs, ("headon", "wall")):
        for v in ("none", "rows", "rows+steer"):
            d = runs.get((name, v))
            if d is None:
                continue
            c = 100*d["clear"]; t = np.arange(len(c))*0.005; i = int(np.argmin(c))
            ax.plot(t, np.clip(c, -15, 60), color=COL[v], label=f"{NAME[v]}: min {c.min():+.1f} cm")
            ax.plot(t[i], max(c[i], -15), "o", ms=7, color=COL[v], mec=SURF, mew=2)
        ax.axhline(0, color=INK2, lw=1); ax.axhspan(-15, 0, color="#f0efec", zorder=0)
        ax.text(ax.get_xlim()[1], -13.5, "inside the keep-out (contact) ", ha="right", color=INK2, fontsize=9)
        ax.set_title(SCENARIOS[name]["title"], loc="left"); ax.set_xlabel("time (s)"); ax.set_ylim(-15, 60)
        ax.legend(loc="upper right", fontsize=9, **SOLID)
    axs[0].set_ylabel("body-sphere clearance (cm)")
    fig.text(0.01, 0.01, "Clearance: smallest distance between a body sphere and an obstacle surface (clipped to -15..60 cm).",
             color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(out, "02_clearance_vs_time.png"), dpi=150); plt.close(fig)


def fig_internals(runs, out):
    d, dr = runs.get(("headon", "rows+steer")), runs.get(("headon", "rows"))
    if d is None:
        return
    fig, axs = plt.subplots(4, 1, figsize=(11, 9.6), sharex=True)
    t = np.arange(len(d["steer"]))*0.005
    axs[0].plot(t, np.degrees(d["steer"]), color=COL["rows+steer"])
    axs[0].set_ylabel("heading offset (deg)")
    axs[0].set_title("Inside the controller: cylinder head-on (heading offset: rows + steering)", loc="left")
    for dd, v in ((dr, "rows"), (d, "rows+steer")):
        if dd is None:
            continue
        tt = np.arange(len(dd["rows"]))*0.005
        axs[1].step(tt, dd["rows"], where="post", color=COL[v], lw=1.5, label=NAME[v])
        axs[2].plot(tt, np.maximum(dd["lam"], 1e-1), color=COL[v], lw=1.5, label=NAME[v])
        axs[3].plot(tt, 100*dd["plan_clear"], color=COL[v], lw=1.5, label=NAME[v] + ": planned (min over 160 ms)")
    axs[1].set_ylabel("active keep-out rows"); axs[1].legend(loc="upper right", fontsize=9, **SOLID)
    axs[2].set_yscale("log"); axs[2].set_ylabel("largest row multiplier\n(cost units, log)")
    axs[2].text(0.02, 0.85, "0.1 = no row binding", transform=axs[2].transAxes, color=INK2, fontsize=9)
    axs[3].plot(np.arange(len(d["clear"]))*0.005, 100*d["clear"], color=INK2, lw=1, label="rows + steering: realized")
    axs[3].axhline(0, color=INK2, lw=1); axs[3].set_ylim(-2, 60); axs[3].set_ylabel("clearance (cm)")
    axs[3].legend(loc="upper right", fontsize=9, **SOLID); axs[3].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "03_controller_internals.png"), dpi=150); plt.close(fig)


def fig_convergence(out):
    p = os.path.join(RESULTS, "convergence.json")
    if not os.path.exists(p):
        return
    cv = json.load(open(p)); its = np.array(cv["iters"])
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for j, mode in enumerate(("cold", "warm")):
        for c, (name, r) in zip(cols, cv["cases"].items()):
            axs[j].loglog(its, np.maximum(np.array(r[mode])[:, 0], 1e-13), color=c, marker="o", ms=4, label=name)
        axs[j].axvline(8, color=INK2, lw=1); axs[j].text(8.6, 1e-6, "8 per tick", color=INK2, fontsize=9)
        axs[j].set_title("Cold start" if mode == "cold" else "Warm start (previous solution)", loc="left")
        axs[j].set_xlabel("ADMM iterations"); axs[j].set_ylim(1e-13, 3e2)
    axs[0].set_ylabel("max |W - W_QP| (N or N m)"); axs[0].legend(loc="lower left", fontsize=8)
    tb = cv["tube"]; k = np.array(tb["step"]); fp = 100*np.array(tb["force"])
    axs[2].bar(k, fp, width=0.5, color="#2a78d6", label="wrench-disturbance tube (20 N, open loop)")
    axs[2].bar(k, 100*tb["margin"]*np.ones_like(fp), width=0.5, bottom=fp, color=AXIS, label="fixed margin 2 cm")
    for kk, v in zip(k, fp + 100*tb["margin"]):
        axs[2].text(kk, v + 0.08, f"{v:.1f}", ha="center", color=INK2, fontsize=8)
    axs[2].set_title("Tightening per horizon step (position rows)", loc="left"); axs[2].set_xlabel("horizon step (20 ms each)")
    axs[2].set_ylabel("tightening (cm)"); axs[2].legend(loc="upper left", fontsize=8); axs[2].grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "04_convergence_and_tube.png"), dpi=150); plt.close(fig)


def fig_bench(out):
    p = os.path.join(RESULTS, "bench.json")
    if not os.path.exists(p):
        return
    bj = json.load(open(p)); ms = bj["ms"]; names = list(ms)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    for y, n in enumerate(names):
        v = np.array(ms[n]); med, lo, hi = np.median(v), np.percentile(v, 10), np.percentile(v, 90)
        col = MUTED if n.startswith("before") else "#2a78d6"
        ax.plot([lo, hi], [y, y], color=col); ax.plot(med, y, "o", ms=9, color=col, mec=SURF, mew=2)
        ax.text(hi + 0.04, y, f"{med:.2f} ms", va="center", color=INK2, fontsize=9)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, color=INK2); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(np.percentile(ms[n], 90) for n in names) + 0.6)
    ax.set_xlabel("layer-1 force solve per tick (ms): median (dot) and 10th-90th percentile")
    ax.set_title("Solve-time cost of the keep-out rows", loc="left")
    fig.text(0.01, 0.01, f"60 replayed ticks held at the 8-cylinder wall ({bj['active_rows']} rows active); "
             "compare ratios, not absolute times.", color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(out, "05_solve_time.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(RESULTS, "scenarios"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "figures"))
    a = ap.parse_args()
    runs = load_runs(a.runs)
    if not runs:
        sys.exit(f"no scenario runs in {a.runs}: run scenarios.py first")
    os.makedirs(a.out, exist_ok=True)
    fig_paths(runs, a.out); fig_clearance(runs, a.out); fig_internals(runs, a.out); fig_convergence(a.out); fig_bench(a.out)
    table = [d["s"] for d in runs.values()]
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(table, open(os.path.join(RESULTS, "scenarios.json"), "w"), indent=1)
    print("| scenario | variant | upright | min clearance (cm) | contact ticks | max tilt (deg) |\n|---|---|---|---|---|---|")
    for s in sorted(table, key=lambda s: (list(SCENARIOS).index(s["scenario"]), s["variant"])):
        print(f"| {s['title']} | {NAME[s['variant']]} | {'yes' if s['upright'] else 'fell'} | {100*s['min_clearance']:+.1f} | "
              f"{s['contact_ticks']} | {s['max_tilt']:.1f} |")
