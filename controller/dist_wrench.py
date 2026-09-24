import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

_SETS1 = [[0], [1], [2], [3], [4]]                     # one active bound
_SETS2 = [[0, 2], [0, 3], [1, 2], [1, 3],              # two facets, or facet + cap
          [0, 4], [1, 4], [2, 4], [3, 4]]
_SETS3 = [[0, 2, 4], [0, 3, 4], [1, 2, 4], [1, 3, 4]]  # cap-face vertices


def _inv1(M):
    return 1.0/M


def _inv2(M):
    a, b = M[..., 0, 0], M[..., 0, 1]
    c, d = M[..., 1, 0], M[..., 1, 1]
    det = a*d - b*c
    return jnp.stack([jnp.stack([d, -b], -1), jnp.stack([-c, a], -1)], -2)/det[..., None, None]


def _inv3(M):
    """Batched 3x3 adjugate inverse; unlike a LAPACK call it fuses into the XLA graph."""
    a, b, c = M[..., 0, 0], M[..., 0, 1], M[..., 0, 2]
    d, e, f = M[..., 1, 0], M[..., 1, 1], M[..., 1, 2]
    g, h, i = M[..., 2, 0], M[..., 2, 1], M[..., 2, 2]
    det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    r0 = jnp.stack([e*i - f*h, c*h - b*i, b*f - c*e], -1)
    r1 = jnp.stack([f*g - d*i, a*i - c*g, c*d - a*f], -1)
    r2 = jnp.stack([d*h - e*g, b*g - a*h, a*e - b*d], -1)
    return jnp.stack([r0, r1, r2], -2)/det[..., None, None]



def _sub_pre(P, mu, fz_max):
    """Per-tick factorizations for the exact per-leg subproblem
        min 1/2 f'Pf - g'f   s.t.  |fx|,|fy| <= mu fz, 0 <= fz <= fz_max.
    P is fixed across ADMM iterations, so every candidate active set's KKT matrix
    is inverted once here; each iteration then costs one small matvec per
    candidate. 19 candidates (unconstrained, 5 single bounds, 8 edges, 4 cap
    vertices, apex) cover the pyramid exactly. fz_max may be a scalar or a
    per-subproblem (S,) vector."""
    A = jnp.array([[1., 0., -mu], [-1., 0., -mu], [0., 1., -mu], [0., -1., -mu],
                   [0., 0., 1.], [0., 0., -1.]])
    S = P.shape[0]
    fzv = jnp.broadcast_to(jnp.asarray(fz_max, P.dtype), (S,))
    hS = jnp.zeros((S, 6)).at[:, 4].set(fzv)                     # only the cap row varies
    Pinv = _inv3(P)
    pre = dict(P=P, Pinv=Pinv, A=A, h=hS, K=[], Ah=[])
    for sets in (_SETS1, _SETS2, _SETS3):
        r = len(sets[0])
        Ag = jnp.stack([A[jnp.array(cs)] for cs in sets])            # (C, r, 3)
        hg = jnp.stack([hS[:, jnp.array(cs)] for cs in sets], axis=1)  # (S, C, r)
        PiAt = jnp.einsum('sjk,crk->scjr', Pinv, Ag)                 # P^-1 A'
        Sc = jnp.einsum('crj,scjq->scrq', Ag, PiAt)                  # A P^-1 A' (Schur)
        Sinv = {1: _inv1, 2: _inv2, 3: _inv3}[r](Sc)
        TR = jnp.einsum('scjr,scrq->scjq', PiAt, Sinv)               # P^-1 A' S^-1
        TL = Pinv[:, None] - jnp.einsum('scjq,sckq->scjk', TR, PiAt)
        Kinv = jnp.zeros((S, len(sets), 3 + r, 3 + r))
        Kinv = Kinv.at[:, :, :3, :3].set(TL)
        Kinv = Kinv.at[:, :, :3, 3:].set(TR)
        Kinv = Kinv.at[:, :, 3:, :3].set(jnp.swapaxes(TR, 2, 3))
        Kinv = Kinv.at[:, :, 3:, 3:].set(-Sinv)
        pre['K'].append(Kinv); pre['Ah'].append((Ag, hg))
    return pre


def _sub_solve(pre, g):
    """Exact solve of the per-leg subproblem for every row of g (S, 3):
    evaluate all candidate active sets, keep the KKT-consistent feasible ones,
    take the lowest objective."""
    A, hS = pre['A'], pre['h']
    S = g.shape[0]
    fs = [jnp.einsum('sab,sb->sa', pre['Pinv'], g)[:, None]]         # unconstrained
    ok = [jnp.ones((S, 1), bool)]
    for Kinv, (Ag, hg) in zip(pre['K'], pre['Ah']):
        C, r = Ag.shape[0], Ag.shape[1]
        rhs = jnp.concatenate([jnp.broadcast_to(g[:, None], (S, C, 3)), hg], axis=2)
        x = jnp.einsum('scab,scb->sca', Kinv, rhs)
        fs.append(x[..., :3]); ok.append((x[..., 3:] >= -1e-7).all(-1))
    fs.append(jnp.zeros((S, 1, 3))); ok.append(jnp.ones((S, 1), bool))   # pyramid apex
    F = jnp.concatenate(fs, axis=1)                                   # (S, 19, 3)
    OK = jnp.concatenate(ok, axis=1)
    feas = (jnp.einsum('ab,scb->sca', A, F) <= hS[:, None] + 1e-7).all(-1)
    J = 0.5*jnp.einsum('sca,sab,scb->sc', F, pre['P'], F) - jnp.einsum('sca,sa->sc', F, g)
    J = jnp.where(OK & feas, J, jnp.inf)
    best = jnp.argmin(J, axis=1)
    return jnp.take_along_axis(F, best[:, None, None], axis=1)[:, 0]


@jax.jit
def _bt_core(G4, act, fnom, e0, Dw, Q, lam, mu, rho, relax, caps, f0, l0, iters):
    QW = Dw.T @ Q @ Dw; qW = Dw.T @ Q @ e0
    n = act.sum()
    Md = jnp.linalg.inv(n*QW + rho*jnp.eye(6))
    GtG = jnp.einsum('lab,lac->lbc', G4, G4)
    P = lam*jnp.eye(3)[None] + rho*GtG
    P = jnp.where(act[:, None, None] > 0, P, jnp.eye(3)[None])
    pre = _sub_pre(P, mu, caps)
    m_ = act[:, None]

    def body(_, carry):
        f, l, s = carry
        g = (lam*fnom + rho*jnp.einsum('lab,la->lb', G4, s - l)) * m_
        f = _sub_solve(pre, g) * m_
        Gf = jnp.einsum('lab,lb->la', G4, f)
        Gf_r = (relax*Gf + (1.0 - relax)*s) * m_                     # over-relaxation, same fixed point
        a = (Gf_r + l) * m_
        d = Md @ (QW @ a.sum(0) + qW)
        s = (a - d[None]) * m_
        l = (l + Gf_r - s) * m_
        return f, l, s

    s0 = jnp.einsum('lab,lb->la', G4, f0) * m_
    f, l, s = jax.lax.fori_loop(0, iters, body, (f0, l0, s0))
    W = (jnp.einsum('lab,lb->la', G4, f) * m_).sum(0)
    res = jnp.linalg.norm(s.sum(0) - W)
    return W, f, l, res


def distributed_body_track(e0, D_w, Q, Gs, f_nom, lam, mu, fz_max, rho=None, iters=40, warm=None,
                           state=None, relax=1.6):
    """Instantaneous allocation for the stance feet in Gs. Returns the aggregate wrench, the
    forces {i: f_i} and the final consensus residual. `state` carries primal and dual across
    ticks; `warm` ({i: f_i}) seeds legs without stored state."""
    feet = sorted(Gs.keys())
    act = np.zeros(4); G4 = np.zeros((4, 6, 3)); fn = np.zeros((4, 3))
    for i in feet:
        act[i] = 1.0; G4[i] = Gs[i]; fn[i] = f_nom[i]
    if rho is None:
        rho = 0.1*np.trace(D_w.T @ Q @ D_w)/6.0                   # penalty at the curvature's scale
    prev = state.get('bt') if isinstance(state, dict) else None
    f0 = np.zeros((4, 3)); l0 = np.zeros((4, 6))
    for i in feet:
        if prev is not None and prev['act'][i] > 0:
            f0[i] = prev['f'][i]; l0[i] = prev['l'][i]
        elif warm and i in warm:
            f0[i] = warm[i]
        else:
            f0[i] = fn[i]
    W, f, l, res = _bt_core(jnp.array(G4), jnp.array(act), jnp.array(fn), jnp.array(e0), jnp.array(D_w),
                            jnp.array(Q), float(lam), float(mu), float(rho), float(relax),
                            jnp.array(np.full(4, float(fz_max))), jnp.array(f0), jnp.array(l0), jnp.int32(iters))
    W = np.asarray(W); f = np.asarray(f)
    if isinstance(state, dict):
        state['bt'] = dict(f=f.copy(), l=np.asarray(l), act=act.copy())
    return W, {i: f[i] for i in feet}, float(res)


def body_tracking_terms(state, ref, m, I_body, dt, gains, T_h=0.08, damp=(0.0, 0.0)):
    """e_next = e0 + D_w W: the trunk velocity/rate error T_h ahead, affine in the aggregate
    wrench W. state: dict(v, w, e_p, e_th), world frame; ref: dict(v_ref, w_ref). Position and
    attitude errors are folded into the desired velocity and rate by `gains`, with optional
    rate damping `damp`. dt is unused (kept for the pre-fix baseline's call)."""
    Kp_v, Kp_w = gains
    Kd_v, Kd_w = damp
    Kp_w = np.asarray(Kp_w) if np.ndim(Kp_w) else np.array([Kp_w, Kp_w, Kp_w])
    v_des = ref["v_ref"] - Kp_v * state["e_p"] - Kd_v * (state["v"] - ref["v_ref"])
    w_des = ref["w_ref"] - Kp_w * state["e_th"] - Kd_w * (state["w"] - ref["w_ref"])
    g = np.array([0., 0., 9.81])
    e0 = np.concatenate([state["v"] - T_h*g - v_des, state["w"] - w_des])
    Iinv = np.linalg.inv(I_body)
    D_w = np.zeros((6, 6))
    D_w[0:3, 0:3] = T_h/m * np.eye(3)
    D_w[3:6, 3:6] = T_h * Iinv
    return e0, D_w


def project_cone(f, mu, fz_max):
    """Clip to the pyramid (projected-gradient reference solvers)."""
    fz = min(max(f[2], 0.0), fz_max); lim = mu*fz
    return np.array([np.clip(f[0], -lim, lim), np.clip(f[1], -lim, lim), fz])


def centralized_body_track(e0, D_w, Q, Gs, f_nom, lam, mu, fz_max):
    """Centralized reference for validation (same problem, one solve)."""
    feet = list(Gs.keys())
    G = np.hstack([Gs[i] for i in feet]); F0 = np.concatenate([f_nom[i] for i in feet])
    QW = D_w.T @ Q @ D_w; qW = D_w.T @ Q @ e0
    F = F0.copy()
    H = G.T @ QW @ G + lam*np.eye(len(F))
    step = 1.0/np.linalg.eigvalsh(H).max()
    for _ in range(4000):
        grad = G.T @ (QW @ (G @ F) + qW) + lam*(F - F0)
        F = F - step*grad
        for k, i in enumerate(feet):
            F[3*k:3*k+3] = project_cone(F[3*k:3*k+3], mu, fz_max)
    fd = {i: F[3*k:3*k+3] for k, i in enumerate(feet)}
    return G @ F, fd
