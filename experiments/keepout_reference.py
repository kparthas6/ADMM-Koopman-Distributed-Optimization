import os, sys, subprocess, tarfile, io
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "common")); sys.path.insert(0, os.path.join(ROOT, "controller"))
import numpy as np
import jax.numpy as jnp, qpax
import dist_horizon as DH, koopman_admm as KA, leg_kinematics as kin

M, N, DT, MU, LAM = 12.74, 8, 0.02, 0.7, 0.005
I_W = np.diag([0.1268, 0.394, 0.4229])
Q = np.diag([400., 400., 400., 30., 30., 15.])
KO = dict(sigma_f=20.0, sigma_t=3.0, sigma_p0=0.002, sigma_v0=0.01, margin=0.01, act_dist=0.3, k0=False, lmax=1e7)

CASES = {   # seed, obstacles [cx, cy, cz, r, cylinder], body spheres [bx, by, bz, r]; each has a binding row
    "cylinder, 1 sphere": (0, [[0.66, 0.10, 0.0, 0.30, 1.0]], [[0, 0, 0, 0.30]]),
    "cylinder, 2 spheres": (1, [[0.55, -0.05, 0.0, 0.15, 1.0]], [[0.19, 0, 0, 0.17], [-0.19, 0, 0, 0.17]]),
    "sphere + cylinder": (2, [[0.60, 0.25, 0.30, 0.25, 0.0], [0.70, -0.40, 0., 0.2, 1.0]], [[0, 0, 0, 0.30]]),
    # a sphere above the front body sphere: the binding row has a roll/pitch lever arm and a torque tube
    "3 spheres, raised sphere": (4, [[0.52, 0.14, 0.45, 0.15, 0.0]], KA.KEEPOUT_CFG["bodies"]),
}


def instance(seed, obs, bodies, vx=0.4, sched=None):
    """A standing or trotting horizon problem at the home pose with a keep-out dict."""
    rng = np.random.default_rng(seed)
    com = np.array([0., 0., 0.27])
    feet = {i: np.r_[(com + kin.fk_leg(kin.HOME_LEG, kin.HIP_OFFSETS[i], kin.THIGH_Y[i]))[:2] + rng.normal(0, 0.01, 2), 0.0]
            for i in range(4)}
    sched = sched or [[0, 1, 2, 3]]*N
    return dict(x0=np.r_[vx, 0.0, rng.normal(0, 0.02), rng.normal(0, 0.05, 3)],
                vd=[np.array([vx + 0.2, 0., 0.]) for _ in range(N)], wd=[np.zeros(3) for _ in range(N)],
                Gs=[KA.grasp_maps(feet, com, st) for st in sched], f_nom={i: np.array([0., 0., M*9.81/4]) for i in range(4)},
                ko=dict(KO, pose0=np.r_[com, rng.normal(0, 0.02, 2), 0.3], obstacles=obs, bodies=bodies))


def solve(P, iters, state=None):
    """The ADMM under test."""
    st = {} if state is None else state
    W, F = DH.distributed_horizon(P["x0"], P["vd"], P["wd"], P["Gs"], P["f_nom"], Q, M, I_W, DT, N, LAM, MU, M*9.81,
                                  iters=iters, state=st, keepout=P["ko"])
    return W, st


def reference_rows(P):
    """Rows (k, coefficient on the stacked wrench, limit, active) rebuilt by brute force: rollout of the
    trapezoid pose map, the linearized keep-out at the constant-twist guess, and the gpu_sls tube."""
    x0, ko = P["x0"], P["ko"]
    Bd = np.zeros((6, 6)); Bd[:3, :3] = DT/M*np.eye(3); Bd[3:, 3:] = DT*np.linalg.inv(I_W)
    cd = np.r_[-DT*9.81*np.array([0, 0, 1.]), np.zeros(3)]
    Sx, xb = np.zeros((N, 12, 6*N)), np.zeros((N, 12))
    tw_b, tw_S, po_b, po_S = x0.copy(), np.zeros((6, 6*N)), np.asarray(ko["pose0"], float).copy(), np.zeros((6, 6*N))
    for k in range(N):
        tw_S1 = tw_S.copy(); tw_S1[:, 6*k:6*k+6] += Bd; tw_b1 = tw_b + cd
        po_S, po_b = po_S + DT*0.5*(tw_S + tw_S1), po_b + DT*0.5*(tw_b + tw_b1)
        tw_S, tw_b = tw_S1, tw_b1
        Sx[k, :6], Sx[k, 6:], xb[k, :6], xb[k, 6:] = po_S, tw_S, po_b, tw_b
    xbar = np.hstack([ko["pose0"][None] + DT*np.arange(1, N + 1)[:, None]*x0[None], np.tile(x0, (N, 1))])
    EW = np.r_[ko["sigma_f"]*np.ones(3), ko["sigma_t"]*np.ones(3)]
    E0 = np.r_[ko["sigma_p0"]*np.ones(6), ko["sigma_v0"]*np.ones(6)]
    rows = []
    for k in range(N):
        psi = xbar[k, 5]; Rz = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1.]])
        Psi0 = np.zeros((12, 12)); Psi0[:6, :6] = np.eye(6); Psi0[:6, 6:] = DT*(k + 1)*np.eye(6); Psi0[6:, 6:] = np.eye(6)
        for o in np.asarray(ko["obstacles"], float):
            for b in np.asarray(ko["bodies"], float):
                rw = Rz @ b[:3]; msk = np.array([1, 1, 0.]) if o[4] > 0.5 else np.ones(3)
                dv = (xbar[k, :3] + rw - o[:3])*msk; n = dv/(np.linalg.norm(dv) + 1e-9)
                arm = np.cross(rw, n)*np.array([1., 1., 0.])           # roll/pitch only, heading held
                c = np.r_[-n, -arm, np.zeros(6)]
                h = -n @ o[:3] - o[3] - b[3] + n @ rw
                tube = sum(np.linalg.norm((c @ Sx[k][:, 6*j:6*j+6])*EW) for j in range(N))
                tube += np.linalg.norm((c @ Psi0)*E0) + ko["margin"]
                on = float((h - tube - c @ xbar[k]) < ko["act_dist"])*(k > 0 or ko["k0"])
                rows.append((k, c @ Sx[k], h - tube - c @ xb[k], on, c))
    return rows


def dense_qp(P, rows):
    """The horizon QP over all stance forces with the pyramid and the active rows (hard), by qpax."""
    idx = [(k, i) for k in range(N) for i in P["Gs"][k]]; nv = 3*len(idx)
    S_w, e_base = DH.build_horizon(P["x0"], P["vd"], P["wd"], M, I_W, DT, N)
    Gb, F0 = np.zeros((6*N, nv)), np.zeros(nv)
    for c, (k, i) in enumerate(idx):
        Gb[6*k:6*k+6, 3*c:3*c+3] = P["Gs"][k][i]; F0[3*c:3*c+3] = P["f_nom"][i]
    Qb = np.kron(np.eye(N), Q)
    H = Gb.T @ S_w.T @ Qb @ S_w @ Gb + LAM*np.eye(nv); q = Gb.T @ S_w.T @ Qb @ e_base - LAM*F0
    Ap = np.array([[1, 0, -MU], [-1, 0, -MU], [0, 1, -MU], [0, -1, -MU], [0, 0, 1.], [0, 0, -1.]])
    Gi, hi = np.kron(np.eye(len(idx)), Ap), np.tile([0, 0, 0, 0, M*9.81, 0], len(idx))
    act = [r for r in rows if r[3] > 0]
    Gr = np.array([r[1] @ Gb for r in act]).reshape(-1, nv); hr = np.array([r[2] for r in act])
    x, s, z, y, conv, it = qpax.solve_qp(jnp.array(0.5*(H + H.T)), jnp.array(q), jnp.zeros((0, nv)), jnp.zeros(0),
                                         jnp.array(np.vstack([Gi, Gr])), jnp.array(np.r_[hi, hr]), solver_tol=1e-10)
    F = np.asarray(x)
    return (Gb @ F).reshape(N, 6), np.asarray(z)[len(hi):], act


def extract_reference(dest, ref=None):
    """Write the controller as it was before the keep-out (or at `ref`) into dest; returns its root or None.
    The default is the parent of the commit that introduced KEEPOUT_DEFAULTS."""
    try:
        if ref is None:
            log = subprocess.run(["git", "-C", ROOT, "log", "--format=%H", "--reverse", "-S", "KEEPOUT_DEFAULTS", "--",
                                  "controller/dist_horizon.py"], capture_output=True, text=True, check=True).stdout.split()
            if not log:
                return None
            ref = log[0] + "^"
        data = subprocess.run(["git", "-C", ROOT, "archive", "--format=tar", ref], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    tarfile.open(fileobj=io.BytesIO(data)).extractall(dest, filter="data")
    cache = os.path.join(ROOT, "controller", "leg_ops.npz")
    if os.path.exists(cache):
        import shutil
        shutil.copy(cache, os.path.join(dest, "controller", "leg_ops.npz"))
    return dest
