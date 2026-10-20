"""Speed of the batched paths on this machine's JAX backend (set ADMM_PLATFORM / ADMM_PRECISION):

1. layer 1: one _h_core call (the per-tick force solve, without distributed_horizon's host-side packing,
   which bench_solve.py includes), and horizon_batch for B problems at once: trot problems with one
   cylinder (3 rows per step), 8 ADMM iterations;
2. closed loop, per batch size B (MJX rollouts, controller/jax_loop.py, compile excluded):
   - in the scan: the average wall time per control tick of the whole batch and per robot-tick, from the
     median of repeated runs with the 200-tick PD settle (timed by a settle-only run) subtracted, and the
     throughput in robot-ticks and simulated robot-seconds per second; nothing is dispatched per tick, and
     the scan also keeps run()'s statistics;
   - one tick at a time from Python (Rollout.tick_fns, walking after the settle), each call blocked on and
     including its Python dispatch: the Controller step on the sensor readings, the physics (5 substeps of
     1 ms) on the full MJX state, and sensors, controller and physics fused in one program.
   Per-call rows give mean / median / p99 with the call count n; with fewer than 100 calls the tail is the
   maximum and is labelled so.

    python bench_gpu.py                                    # default sizes
    python bench_gpu.py --solve 1 64 1024 8192 --rollout 1 64 1024 --seconds 2 --ticks 200 --reps 20

Writes ../results/bench_gpu.json. Numbers are for the backend it runs on: a CPU run is a baseline, not a
GPU estimate."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import keepout_reference as KR
import backend, jax, jax.numpy as jnp
import dist_horizon as DH, jax_loop as JL


def problems(n, seed=0):
    """n packed trot problems with one cylinder ahead, randomized state and obstacle."""
    rng, out = np.random.default_rng(seed), []
    trot = [[0, 3] if (k // 4) % 2 == 0 else [1, 2] for k in range(KR.N)]
    for i in range(n):
        obs = [[0.55 + 0.2*rng.random(), rng.uniform(-0.2, 0.2), 0.0, 0.15, 1.0]]
        P = KR.instance(i, obs, JL.KA.KEEPOUT_CFG["bodies"], sched=trot)
        out.append(DH.pack_horizon(P["x0"], P["vd"], P["wd"], P["Gs"], P["f_nom"], KR.Q, KR.M, KR.I_W, KR.DT, KR.N,
                                   KR.LAM, KR.MU, KR.M*9.81, state={}, keepout=P["ko"])[0])
    return np.stack(out)


def times(f, reps):
    """Wall times (s) of reps blocked calls of f, after one warm-up call."""
    jax.block_until_ready(f())
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); jax.block_until_ready(f()); ts.append(time.perf_counter() - t)
    return ts


def stats(ts):
    ms = 1e3*np.asarray(ts)
    return dict(n=len(ms), mean_ms=float(ms.mean()), p50_ms=float(np.median(ms)),
                p99_ms=float(np.percentile(ms, 99)), max_ms=float(ms.max()))


def fmt(s):
    tail = f"p99 {s['p99_ms']:.3f}" if s["n"] >= 100 else f"max {s['max_ms']:.3f}"
    return f"mean {s['mean_ms']:.3f} / median {s['p50_ms']:.3f} / {tail} ms (n={s['n']})"


def per_tick(roll, B, ticks):
    """Per-call times of one tick, batch B, over `ticks` walking ticks (vx 0.3) after the settle."""
    f = roll.tick_fns()
    pose = jnp.asarray(np.stack([np.zeros(B), np.linspace(-0.05, 0.05, B), np.zeros(B)], 1), backend.DTYPE)
    cmd = jnp.asarray(np.tile([0.3, 0.0, 0.0], (B, 1)), backend.DTYPE)
    obs = jnp.asarray(np.broadcast_to(JL.KA.obstacle_array(roll.template), (B, len(roll.template), 5)), backend.DTYPE)
    clks = [roll.ctrl.clock(k) for k in range(ticks)]
    d, s = jax.block_until_ready(f["start"](pose))
    sn = jax.block_until_ready(f["sensors"](d))                                 # compile every piece
    s1, tau = jax.block_until_ready(f["control"](sn, s, cmd, obs, clks[0]))
    jax.block_until_ready((f["physics"](d, tau), f["tick"](d, s, cmd, obs, clks[0])))
    tc, tp, tt = [], [], []
    for k in range(ticks):
        sn = jax.block_until_ready(f["sensors"](d))                             # outside the timed calls
        t0 = time.perf_counter(); s2, tau = jax.block_until_ready(f["control"](sn, s, cmd, obs, clks[k]))
        t1 = time.perf_counter(); d2 = jax.block_until_ready(f["physics"](d, tau))
        t2 = time.perf_counter(); r = jax.block_until_ready(f["tick"](d, s, cmd, obs, clks[k]))
        t3 = time.perf_counter()
        del r                                                                    # freed outside the window
        tt.append(t3 - t2); tp.append(t2 - t1); tc.append(t1 - t0)
        d, s = d2, s2
    return dict(control=stats(tc), physics=stats(tp), tick=stats(tt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solve", type=int, nargs="*", default=[1, 8, 64, 512, 4096])
    ap.add_argument("--rollout", type=int, nargs="*", default=[1, 4, 16])
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--ticks", type=int, default=100, help="ticks timed one call at a time")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    if a.ticks < 1 or a.reps < 1 or int(a.seconds/0.005) < 1:
        ap.error("--ticks and --reps must be >= 1 and --seconds at least one 5 ms control tick")
    print(backend.info())
    res = dict(backend=backend.info(), args=vars(a), solve={}, rollout={})
    base = jnp.asarray(problems(64), backend.DTYPE)
    p0, it = base[0], jnp.int32(8)
    single = stats(times(lambda: DH._h_core(p0, it, KR.N, 4, 1, 3), 200))
    res["solve"]["single_call"] = single
    print(f"layer 1, one _h_core call: {fmt(single)}")
    for B in a.solve:
        pk = jnp.tile(base, (B // len(base) + 1, 1))[:B]
        s = stats(times(lambda: DH.horizon_batch(pk, it, KR.N, 4, 1, 3), a.reps))
        t = s["p50_ms"]/1e3
        res["solve"][str(B)] = dict(s, us_per_problem=1e6*t/B, problems_per_s=B/t)
        print(f"layer 1, batch {B:6d}: {fmt(s)} | {1e6*t/B:8.2f} us per problem | {B/t:10.0f} solves/s "
              f"({single['p50_ms']/1e3*B/t:.1f}x one-at-a-time)")
    roll = JL.Rollout(obstacles=[(1.2, 0.0, 0.15)], gait_T=0.25)
    nst = int(a.seconds/roll.dt)
    for B in a.rollout:
        kw = dict(vx=np.full(B, 0.3), y0=np.linspace(-0.05, 0.05, B))
        t0 = time.perf_counter(); roll.run(a.seconds, **kw); first = time.perf_counter() - t0
        t = float(np.median(times(lambda: roll.run(a.seconds, **kw)["ticks"], max(1, a.reps // 2))))
        t_settle = float(np.median(times(lambda: roll.run(0.0, **kw)["ticks"], max(1, a.reps // 2))))   # reset + settle
        if t - t_settle > 0.1*t_settle:                                   # one control tick of the whole batch
            tick_ms = 1e3*(t - t_settle)/nst
        else:
            tick_ms = float("nan")
            print(f"   batch {B}: the run is within noise of the settle-only run; raise --seconds or --reps")
        rate = B/(tick_ms/1e3)
        one = per_tick(roll, B, a.ticks)
        res["rollout"][str(B)] = dict(compile_and_run_s=first, run_s=t, settle_s=t_settle, scan_ms_per_tick=tick_ms,
                                      scan_us_per_robot_tick=1e3*tick_ms/B, robot_ticks_per_s=rate,
                                      robot_seconds_per_s=rate*roll.dt, per_call=one)
        print(f"closed loop, batch {B:5d}: first call {first:6.1f} s, then {t:7.2f} s ({t_settle:.2f} s settle) | "
              f"in the scan {tick_ms:.3f} ms per tick, {1e3*tick_ms/B:.1f} us per robot-tick | {rate:9.0f} robot-ticks/s "
              f"| {rate*roll.dt:7.2f} simulated robot-s per s")
        print(f"   one call per tick: controller {fmt(one['control'])}\n"
              f"                      physics    {fmt(one['physics'])}\n"
              f"                      fused tick {fmt(one['tick'])}")
    os.makedirs(os.path.join(KR.ROOT, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(KR.ROOT, "results", "bench_gpu.json"), "w"), indent=1)
