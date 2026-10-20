"""controller/jax_loop.py (needs mujoco-mjx): the jax.numpy steering and controller step against the numpy code,
in this process's precision and in a float32 process; the MJX scene; and (slow) the ported closed loop against
run() on the same CPU physics, and batched MJX rollouts."""
import os
import numpy as np
import pytest
pytest.importorskip("mujoco.mjx")
import backend, jax, jax.numpy as jnp
import koopman_admm as KA, jax_loop as JL
from torque_sim import TorqueSim
from test_gpu_paths import run_python

PATHS = "import sys; sys.path[:0] = [%r, %r]\n" % (os.path.dirname(JL.__file__),
                                                  os.path.join(os.path.dirname(JL.__file__), "..", "common"))
STEER = PATHS + """
import json, numpy as np, jax, jax.numpy as jnp, backend, koopman_admm as KA, jax_loop as JL
rng, f, err = np.random.default_rng(0), jax.jit(JL.steer_offset), 0.0
for _ in range(150):
    xy = lambda: (rng.uniform(0.3, 1.5), rng.uniform(-0.6, 0.6))
    obs = KA.obstacle_array([xy() + (rng.uniform(0.08, 0.3),) if rng.random() < 0.7               # cylinders and spheres
                             else xy() + (rng.uniform(0.1, 0.6), rng.uniform(0.1, 0.3)) for _ in range(rng.integers(1, 5))])
    pos = np.r_[rng.normal(0, 0.1, 2), 0.27]
    yaw, v, prev = rng.uniform(-np.pi, np.pi), np.r_[rng.uniform(0, 0.4), rng.uniform(-0.2, 0.2)], rng.uniform(-1, 1)
    j = lambda x: jnp.asarray(x, backend.DTYPE)
    err = max(err, abs(float(f(j(pos), yaw, j(v), j(obs), prev)) - KA.steer_offset(pos, yaw, v, obs, prev)))
print(json.dumps(dict(x64=backend.X64, err=err)))
"""
PARITY = PATHS + """
import json, numpy as np, backend, koopman_admm as KA, jax_loop as JL
from torque_sim import TorqueSim
legs, out = KA.learn_operators(), []
for obst, kw in {cases!r}:
    sim, taus = TorqueSim(obstacles=obst), []
    step = sim.step
    sim.step = lambda tau: (taus.append(np.array(tau, float)), step(tau))[1]
    L0, up0 = KA.run(sim, legs, "trot", obstacles=obst, verbose=False, **kw)
    L1, up1 = JL.run_host(TorqueSim(obstacles=obst), legs, "trot", obstacles=obst, **kw)
    ok = bool(up0 and up1 and len(L0["z"]) == len(L1["z"]))
    out.append(dict(ok=ok, tau=float(np.abs(np.array(taus[200:]) - L1["tau"]).max()) if ok else None,
                    qpos=float(np.abs(L0["qpos"] - L1["qpos"]).max()) if ok else None,
                    steer=float(np.abs(L0["steer"] - L1["steer"]).max()) if ok and obst else 0.0))
print(json.dumps(dict(x64=backend.X64, cases=out)))
"""


def test_steer_offset_matches_numpy():
    """150 random scenes; the same offset as KA.steer_offset in float64 and, in a float32 process, to rounding."""
    here = run_python(STEER, ADMM_PRECISION="64" if backend.X64 else "32")
    f32 = run_python(STEER, ADMM_PRECISION="32")
    assert not f32["x64"] and f32["err"] < 1e-5, f32
    assert here["err"] < (1e-9 if here["x64"] else 1e-5), here


def test_controller_step_jits_and_vmaps():
    """A cold step from the reset (keyframe) pose and a warm-started second one, three commands at once, against
    the unbatched step."""
    sim = TorqueSim(control_dt=0.005); sim.reset()
    legs = KA.learn_operators()
    c = JL.Controller(legs, sim.mass, sim.control_dt, "trot", 0.25, n_obs=1)
    sens = dict(q=sim.d.qpos[sim.qadr], qd=sim.d.qvel[sim.vadr], pos=sim.d.qpos[:3], quat=sim.d.qpos[3:7],
                tw=sim.d.qvel[:6], feet=sim.foot_pos_world(), bias=sim.bias_torque())
    sens = {k: jnp.asarray(v, backend.DTYPE) for k, v in sens.items()}
    cmd = jnp.asarray([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.2, 0.1, 0.3]], backend.DTYPE)
    obs = jnp.asarray(KA.obstacle_array([(0.6, 0.0, 0.15)]), backend.DTYPE)
    tol = 1e-4 if backend.X64 else 5e-4                          # N m; qpax is float32 in both
    one, many = jax.jit(c.step), jax.jit(jax.vmap(c.step, in_axes=(0, None, 0, None, None)))
    s = jax.tree.map(lambda x: jnp.stack([x]*3), c.init())
    for stp in range(2):
        clk = c.clock(stp)
        s2, tau, info = many(s, sens, cmd, obs, clk)
        for b in range(3):
            _, tau1, _ = one(jax.tree.map(lambda x: x[b], s), sens, cmd[b], obs, clk)
            assert np.all(np.isfinite(np.asarray(tau1))) and np.abs(np.asarray(tau1) - np.asarray(tau[b])).max() < tol
        s = s2
    assert bool(s["has_prev"].all()) and float(info["rows"][1]) > 0     # the cylinder 0.6 m ahead puts rows on


def test_gait_clock():
    """Stance, phase, flight and the horizon pattern at hand-checked ticks (the closed-loop match with run()
    is test_ported_loop_matches_run)."""
    c = JL.gait_clock(57, 0.005, 0.25, 0.8, 8, 0.02)                  # t = 0.285: (1, 2) in stance at s = 0.14
    assert list(c["stance"]) == [False, True, True, False] and abs(c["s_ph"] - 0.14) < 1e-12 and not c["push"]
    c = JL.gait_clock(42, 0.005, 0.25, 0.8, 8, 0.02)                  # t = 0.21: s = 0.84, flight
    assert not c["stance"].any() and list(c["lifted"]) == [True, False, False, True]
    assert not c["sched"][0].any() and list(c["sched"][3]) == [False, True, True, False]   # tk = 0.27
    assert c["tdo"][0] == 0.0 and abs(c["tdo"][3] - 0.04) < 1e-12
    s = JL.gait_clock(0, 0.005, 0.25, 1.0, 8, 0.02, "stand")
    assert s["stance"].all() and s["sched"].all() and not s["lifted"].any()


def test_mjx_scene():
    import mujoco
    m = JL.mjx_model([(1.2, 0.0, 0.15), (2.0, 0.0, 0.3, 0.2)])
    g = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle{j}") for j in range(2)]
    assert [m.geom_type[i] for i in g] == [mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE]
    coll = (m.geom_contype | m.geom_conaffinity) > 0
    robot, world = coll & (m.geom_bodyid > 0), coll & (m.geom_bodyid == 0)
    pairs = lambda a, b: ((m.geom_contype[a][:, None] & m.geom_conaffinity[b][None]) |
                          (m.geom_conaffinity[a][:, None] & m.geom_contype[b][None])) > 0
    assert not pairs(robot, robot).any()                          # no self-collision
    assert pairs(robot, world).all()                              # every robot geom against floor and obstacles


@pytest.mark.slow
@pytest.mark.parametrize("prec", ["here", "32"])
def test_ported_loop_matches_run(prec):
    """Controller (jax.numpy) and run() (numpy) on the same CPU MuJoCo physics: head-on cylinder with steering,
    the default gait period, and long strides with flight and strafing. In float32 (a separate process) both
    run their float32 parts, and the obstacle sits 10 cm off the axis: dead ahead, the left and right detours
    tie and rounding may pick either."""
    if prec == "here" and not backend.X64:
        pytest.skip("float64 comparison; the float32 case covers this precision")
    f64 = prec == "here"
    cases = [([(1.2, 0.0 if f64 else 0.1, 0.15)], dict(vx=0.3, gait_T=0.25, seconds=4.5 if f64 else 2.0)),
             (None, dict(vx=0.2, seconds=1.5)),
             (None, dict(vx=0.4, vy=0.1, gait_T=0.30, duty=0.8, seconds=1.5))]
    r = run_python(PARITY.format(cases=cases), ADMM_PRECISION="64" if f64 else "32")
    tol = (1e-4, 1e-6) if r["x64"] else (2e-2, 1e-4)             # N m, qpos; float32 swing MPC in both
    for c in r["cases"]:
        assert c["ok"] and c["tau"] < tol[0] and c["qpos"] < tol[1] and c["steer"] < 10*tol[1], (r["x64"], c)


@pytest.mark.slow
def test_mjx_rollouts():
    """Two robots in one MJX batch: with rows and steering both pass the cylinder untouched; without the
    keep-out, a pole in the path is hit while the same pole moved aside is not."""
    roll = JL.Rollout(obstacles=[(1.2, 0.0, 0.15)], gait_T=0.25)
    out = roll.run(9.0, vx=0.3, y0=np.array([-0.03, 0.03]))
    assert out["upright"].all() and (out["contact_ticks"] == 0).all() and (out["min_clear"] > 0).all(), out
    assert (out["end_xy"][:, 0] > 1.5).all(), out["end_xy"]
    bare = JL.Rollout(obstacles=[(0.6, 0.0, 0.1)], gait_T=0.25, keepout=False)
    out = bare.run(3.0, vx=0.3, obstacles=[[(0.6, 0.0, 0.1)], [(0.6, 1.5, 0.1)]])
    assert out["contact_ticks"][0] > 0 and out["contact_ticks"][1] == 0, out["contact_ticks"]
