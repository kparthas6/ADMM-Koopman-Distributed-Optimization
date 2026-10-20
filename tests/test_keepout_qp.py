"""The keep-out ADMM in controller/dist_horizon.py against an independent reference: rows, tube and activity
rebuilt with numpy loops, and the dense QP they define solved by qpax (experiments/keepout_reference.py)."""
import numpy as np
import pytest
import backend
import keepout_reference as KR
import dist_horizon as DH

float64_only = pytest.mark.skipif(not backend.X64, reason="exact fixed-point checks are float64; float32 is held "
                                                          "to float64 in test_gpu_paths.py")


@float64_only
@pytest.mark.parametrize("name", list(KR.CASES))
def test_fixed_point_matches_dense_qp(name):
    seed, obs, bodies = KR.CASES[name]
    P = KR.instance(seed, obs, bodies)
    rows = KR.reference_rows(P)
    W_qp, z, act = KR.dense_qp(P, rows)
    W, st = KR.solve(P, 3000)
    on_ref = np.array([r[3] for r in rows]).reshape(KR.N, -1)
    assert np.array_equal(on_ref, st["h"]["on"]), "active rows differ from the reference"
    assert np.abs(W - W_qp).max() < 1e-8
    lam = st["h"]["lamc"].reshape(-1)[on_ref.reshape(-1) > 0]
    assert np.abs(lam - z).max() < 1e-6*max(1.0, np.abs(z).max())
    assert (np.array([r[1] for r in act]) @ W.reshape(-1) - np.array([r[2] for r in act])).max() < 1e-9


@pytest.mark.parametrize("name", list(KR.CASES))
def test_every_case_binds(name):
    """Each case has a binding row, so the multiplier check is not vacuous."""
    seed, obs, bodies = KR.CASES[name]
    _, z, _ = KR.dense_qp(KR.instance(seed, obs, bodies), KR.reference_rows(KR.instance(seed, obs, bodies)))
    assert z.max() > 1.0


def test_a_binding_row_uses_the_attitude_terms():
    """At least one binding row has roll/pitch coefficients, so the lever arm and torque tube are held to the reference."""
    seed, obs, bodies = KR.CASES["3 spheres, raised sphere"]
    P = KR.instance(seed, obs, bodies)
    _, z, act = KR.dense_qp(P, KR.reference_rows(P))
    assert any(zi > 1.0 and np.abs(r[4][3:5]).max() > 1e-3 for zi, r in zip(z, act))


@float64_only
def test_warm_start_is_a_fixed_point_and_helps():
    seed, obs, bodies = KR.CASES["cylinder, 1 sphere"]
    P = KR.instance(seed, obs, bodies)
    W_qp, _, _ = KR.dense_qp(P, KR.reference_rows(P))
    st = {}
    KR.solve(P, 3000, st)
    st["h"] = {k: v for k, v in st["h"].items() if k != "xp"}       # keep the reference's linearization point
    W1, _ = KR.solve(P, 1, {"h": dict(st["h"])})                  # a converged state stays put
    assert np.abs(W1 - W_qp).max() < 1e-8
    near = {}                                                     # warm from a nearby solution beats cold at 8
    KR.solve(dict(P, x0=P["x0"] + np.r_[0.02, -0.01, 0, 0, 0, 0.02]), 3000, near)
    near["h"] = {k: v for k, v in near["h"].items() if k != "xp"}
    warm, _ = KR.solve(P, 8, near)
    cold, _ = KR.solve(P, 8)
    assert np.abs(warm - W_qp).max() < 0.25*np.abs(cold - W_qp).max()


def test_inactive_rows_leave_the_solution_unchanged():
    """Obstacles far away: no row is active and the forces equal the solve without a keep-out."""
    P = KR.instance(1, [[5.0, 5.0, 0.0, 0.2, 1.0]], [[0, 0, 0, 0.2]])
    W_ko, st = KR.solve(P, 8)
    assert st["h"]["on"].sum() == 0
    W = DH.distributed_horizon(P["x0"], P["vd"], P["wd"], P["Gs"], P["f_nom"], KR.Q, KR.M, KR.I_W, KR.DT, KR.N, KR.LAM,
                               KR.MU, KR.M*9.81, iters=8, state={})[0]
    assert np.abs(W - W_ko).max() < (1e-9 if backend.X64 else 1e-3)


def test_reference_tube_closed_form():
    """For a cylinder row on the trunk origin the gpu_sls tube reduces to sigma_f dt^2 (k+1)^2 / (2m) + margin
    (checks the reference that the fixed-point test holds the controller to)."""
    P = KR.instance(2, [[0.9, 0.0, 0.0, 0.2, 1.0]], [[0, 0, 0, 0.2]])
    ko = dict(P["ko"], sigma_f=20.0, sigma_t=0.0, sigma_p0=0.0, sigma_v0=0.0, margin=0.02, act_dist=10.0)
    tight = KR.reference_rows(dict(P, ko=ko))
    loose = KR.reference_rows(dict(P, ko=dict(ko, sigma_f=0.0, margin=0.0)))
    got = np.array([a[2] - b[2] for a, b in zip(loose, tight)])
    assert np.allclose(got, 20.0*KR.DT**2*(np.arange(KR.N) + 1)**2/(2*KR.M) + 0.02, atol=1e-12)
