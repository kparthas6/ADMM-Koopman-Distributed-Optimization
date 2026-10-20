"""The GPU-ready solver paths, on whatever backend runs the tests: the precision switch, the batched layer-1
solve, and float32 layer 1 (on this machine's default device, the GPU if there is one) against float64 on
the CPU. The jax.numpy controller and the MJX loop are in test_jax_loop.py."""
import os, sys, json, subprocess
import numpy as np
import backend, jax.numpy as jnp
import keepout_reference as KR
import dist_horizon as DH, koopman_admm as KA

TROT = [[0, 3] if (k // 4) % 2 == 0 else [1, 2] for k in range(KR.N)]
F32 = """
import sys, json; sys.path.insert(0, {exp!r})
import numpy as np, keepout_reference as KR, backend
out = {{}}
for name, (seed, obs, bodies) in KR.CASES.items():
    for sched in (None, {trot!r}):
        P = KR.instance(seed, obs, bodies, sched=sched)
        out[name + str(sched is None)] = np.asarray(KR.solve(P, 3000)[0]).tolist()
print(json.dumps(dict(x64=backend.X64, W=out)))
"""


def run_python(code, **env):
    """Run code in a fresh interpreter with the ADMM_* variables replaced by env; returns its last stdout line
    parsed as JSON."""
    e = {k: v for k, v in os.environ.items() if not k.startswith("ADMM_")}
    p = subprocess.run([sys.executable, "-B", "-c", code], env=dict(e, **env), capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def _pack(P, ko=True):
    return DH.pack_horizon(P["x0"], P["vd"], P["wd"], P["Gs"], P["f_nom"], KR.Q, KR.M, KR.I_W, KR.DT, KR.N, KR.LAM, KR.MU,
                           KR.M*9.81, state={}, keepout=P["ko"] if ko else None)[0]


def test_auto_precision_follows_the_platform():
    """auto: float64 on the CPU, float32 on a GPU; 32 and 64 force it."""
    code = ("import sys, json; sys.path.insert(0, %r); import backend; "
            "print(json.dumps([backend.PLATFORM, backend.X64, str(backend.DTYPE.dtype)]))" % os.path.dirname(DH.__file__))
    plat, x64, dt = run_python(code)
    assert x64 == (plat == "cpu") and dt == ("float64" if x64 else "float32")
    assert run_python(code, ADMM_PRECISION="32")[1:] == [False, "float32"]
    assert run_python(code, ADMM_PRECISION="64", ADMM_PLATFORM="cpu") == ["cpu", True, "float64"]


def test_horizon_batch_matches_single_solves():
    obs, bod = [[0.6, 0.1, 0.0, 0.2, 1.0]], KA.KEEPOUT_CFG["bodies"]
    pk = np.stack([_pack(KR.instance(s, obs, bod, sched=TROT if s % 2 else None)) for s in range(5)])
    one = np.stack([np.asarray(DH._h_core(jnp.asarray(p, backend.DTYPE), jnp.int32(8), KR.N, 4, 1, 3)) for p in pk])
    many = np.asarray(DH.horizon_batch(jnp.asarray(pk, backend.DTYPE), jnp.int32(8), KR.N, 4, 1, 3))
    for b in range(len(pk)):
        r1, r2 = DH.unpack_horizon(one[b], KR.N, 4, 3), DH.unpack_horizon(many[b], KR.N, 4, 3)
        assert np.abs(r1["W"] - r2["W"]).max() < (1e-8 if backend.X64 else 1e-2)
        assert np.array_equal(r1["on"], r2["on"])


def test_float32_layer1_matches_float64():
    """The reference problems solved in float32 on the default device and in float64 on the CPU: the per-foot
    solver's tolerances follow the dtype."""
    code = F32.format(exp=os.path.join(KR.ROOT, "experiments"), trot=TROT)
    r32, r64 = run_python(code, ADMM_PRECISION="32"), run_python(code, ADMM_PRECISION="64", ADMM_PLATFORM="cpu")
    assert r64["x64"] and not r32["x64"]
    err = max(np.abs(np.array(r32["W"][k]) - np.array(r64["W"][k])).max() for k in r64["W"])
    assert err < 2e-2, err                                        # N, on wrenches of about 130 N
