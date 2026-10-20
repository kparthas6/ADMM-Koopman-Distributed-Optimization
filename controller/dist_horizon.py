import numpy as np
import backend                                                  # device and precision: before any JAX use
import jax, jax.numpy as jnp
from functools import partial
from dist_wrench import _sub_pre, _sub_solve, project_cone


def build_horizon(x0, v_des_seq, w_des_seq, m, I_body, dt, N):
    """Stacked trunk velocity/rate prediction error over N steps: e(Wbar) = S_w Wbar + e_base."""
    g = np.array([0., 0., 9.81]); Iinv = np.linalg.inv(I_body)
    Bd = np.zeros((6, 6)); Bd[0:3, 0:3] = dt/m*np.eye(3); Bd[3:6, 3:6] = dt*Iinv
    cd = np.concatenate([-dt*g, np.zeros(3)])
    S_w = np.kron(np.tril(np.ones((N, N))), Bd)
    base = np.concatenate([x0 + (k+1)*cd for k in range(N)])
    ref = np.concatenate([np.concatenate([v_des_seq[k], w_des_seq[k]]) for k in range(N)])
    return S_w, base - ref


KEEPOUT_DEFAULTS = dict(sigma_f=20.0, sigma_t=3.0, sigma_p0=0.0, sigma_v0=0.0, margin=0.0, act_dist=0.30,
                        k0=False, lmax=1e5)


def _h_impl(packed, iters, N, L, NO=0, NB=0):
    """One layer-1 solve on a flat problem vector (pack_horizon's layout); returns a flat output row
    (unpack_horizon's layout). Traceable: _h_core jits it, horizon_batch vmaps it."""
    o = [0]
    def take(n, shape=None):
        v = jax.lax.dynamic_slice(packed, (o[0],), (n,)); o[0] += n
        return v.reshape(shape) if shape else v
    x0 = take(6); vref = take(3*N, (N, 3)); wref = take(3*N, (N, 3))
    I_body = take(9, (3, 3)); Q = take(36, (6, 6)); caps = take(L)
    fnom = take(3*L, (L, 3)); act = take(L*N, (L, N)); G = take(18*L*N, (L, N, 6, 3))
    f0 = take(3*L*N, (L, N, 3)); l0 = take(6*L*N, (L, N, 6))
    m, dt, lam, mu, fz_max, rho_in, relax, tol = [take(1)[0] for _ in range(8)]
    g = jnp.array([0., 0., 9.81])
    Bd = jnp.zeros((6, 6)).at[0:3, 0:3].set(dt/m*jnp.eye(3)).at[3:6, 3:6].set(dt*jnp.linalg.inv(I_body))
    cd = jnp.concatenate([-dt*g, jnp.zeros(3)])
    S_w = jnp.kron(jnp.tril(jnp.ones((N, N))), Bd)
    base = (x0[None] + (jnp.arange(1, N+1)[:, None])*cd[None]).reshape(-1)
    ref = jnp.concatenate([vref, wref], axis=1).reshape(-1)
    Qbar = jnp.kron(jnp.eye(N), Q)
    QW = S_w.T @ Qbar @ S_w; qW = S_w.T @ Qbar @ (base - ref)
    rho = jnp.where(rho_in > 0, rho_in, 0.1*jnp.trace(QW)/(6*N))
    R = NO*NB
    if R:                                                         # keep-out rows (NO obstacles x NB body spheres)
        pose0 = take(6); xbar = take(12*N, (N, 12)); obs = take(5*NO, (NO, 5)); bod = take(4*NB, (NB, 4))
        sF, sT, sp0, sv0, margin, act_d, k0, lmax = [take(1)[0] for _ in range(8)]
        lamc0 = take(N*R, (N, R))
        kk = jnp.arange(N)
        S_p = dt*jnp.kron(jnp.tril(kk[:, None] - kk[None, :] + 0.5), Bd)   # pose = trapezoid of the ZOH twist
        tri = jnp.kron(jnp.tril(jnp.ones((N, N))) - 0.5*jnp.eye(N), jnp.eye(6))
        pb = jnp.tile(pose0, N) + dt*(0.5*jnp.tile(x0, N) + tri @ base)
        S_x = jnp.concatenate([S_p.reshape(N, 6, 6*N), S_w.reshape(N, 6, 6*N)], axis=1)   # x_{k+1} = S_x[k] W + xb[k]
        xb = jnp.concatenate([pb.reshape(N, 6), base.reshape(N, 6)], axis=1)
        # g = r_obs + r_body - ||p + R r_body - c|| <= 0, linearized at the predicted pose xbar (tangent
        # half-space: conservative for any normal, in body-point space). Roll and pitch enter through
        # R ~ (I + skew([Theta_x, Theta_y, 0])) Rz(yaw_bar), about zero; yaw is held at the predicted
        # yaw_bar, so the plan cannot turn the trunk to creep closer (it did, and fell, held at a wall).
        cps, sps, z_ = jnp.cos(xbar[:, 5]), jnp.sin(xbar[:, 5]), jnp.zeros(N)
        Rz = jnp.stack([jnp.stack([cps, -sps, z_], -1), jnp.stack([sps, cps, z_], -1),
                        jnp.stack([z_, z_, z_ + 1.0], -1)], -2)
        r_w = jnp.einsum('kab,jb->kja', Rz, bod[:, 0:3])[:, None]           # (N, 1, NB, 3) body sphere offsets
        mask = jnp.where(obs[:, 4:5] > 0.5, jnp.array([1., 1., 0.]), 1.0)[None, :, None]   # cylinders: xy only
        dvec = (xbar[:, None, None, 0:3] + r_w - obs[None, :, None, 0:3])*mask
        n = dvec/(jnp.sqrt((dvec*dvec).sum(-1)) + 1e-9)[..., None]
        arm = jnp.cross(r_w, n)*jnp.array([1., 1., 0.])                   # roll/pitch lever arms only
        Crow = jnp.concatenate([-n, -arm, jnp.zeros(n.shape[:-1] + (6,))], -1).reshape(N, R, 12)
        hrow = (-(n*obs[None, :, None, 0:3]).sum(-1) - obs[None, :, None, 3] - bod[None, None, :, 3]
                + (n*r_w).sum(-1)).reshape(N, R)
        # GPU-SLS tightening (get_betas / get_constraint_tightenings) with the open-loop response:
        # h = sum_j ||row . dx_{k+1}/dW_j . E_W||_2 + ||row . dx_{k+1}/dx_0 . E_0||_2, ||w_j||_2 <= 1
        EW = jnp.concatenate([sF*jnp.ones(3), sT*jnp.ones(3)])
        CS = jnp.einsum('krx,kxjw->krjw', Crow, S_x.reshape(N, 12, N, 6))*EW
        Cp, Cv = Crow[..., 0:6], Crow[..., 6:12]
        C0 = jnp.concatenate([sp0*Cp, sv0*(dt*(kk[:, None, None] + 1.0)*Cp + Cv)], -1)
        h_ct = jnp.sqrt((CS*CS).sum(-1)).sum(-1) + jnp.sqrt((C0*C0).sum(-1)) + margin
        slack = hrow - h_ct - (Crow*xbar[:, None, :]).sum(-1)
        on = (slack < act_d).astype(packed.dtype).at[0].multiply(k0)       # far rows and (by default) k = 0 off
        C_W = jnp.einsum('krx,kxw->krw', Crow, S_x).reshape(N*R, 6*N)
        hW = (hrow - h_ct - jnp.einsum('krx,kx->kr', Crow, xb)).reshape(-1)
        onv = on.reshape(-1)
        rc = onv*rho/jnp.maximum((C_W*C_W).sum(1), 1e-30)                # row penalties at the curvature's scale
        rcs = jnp.where(rc > 0, rc, 1.0); kap = lmax/rcs
        CtR = C_W.T*rc[None, :]
        QWc = QW + C_W.T @ (rc[:, None]*C_W)
    else:
        QWc = QW
    n_k = act.sum(0)
    Nk = jnp.kron(jnp.diag(n_k), jnp.eye(6))
    Md = jnp.linalg.inv(QWc @ Nk + rho*jnp.eye(6*N))                # QW Nk, not Nk QW
    GtG = jnp.einsum('lkab,lkac->lkbc', G, G)
    P = lam*jnp.eye(3)[None, None] + rho*GtG
    P = jnp.where(act[:, :, None, None] > 0, P, jnp.eye(3)[None, None])
    capsSN = jnp.repeat(caps[:, None], N, axis=1).reshape(-1)
    pre = _sub_pre(P.reshape(-1, 3, 3), mu, capsSN)
    m_ = act[:, :, None]

    def body(_, carry):
        f, l, s = carry[:3]
        gv = (lam*fnom[:, None, :] + rho*jnp.einsum('lkab,lka->lkb', G, s - l)) * m_
        f = _sub_solve(pre, gv.reshape(-1, 3)).reshape(f.shape) * m_
        Gf = jnp.einsum('lkab,lkb->lka', G, f)
        Gf_r = (relax*Gf + (1.0 - relax)*s) * m_                     # over-relaxation, same fixed point
        a = (Gf_r + l) * m_
        rhs = QWc @ a.sum(0).reshape(-1) + qW
        if R:                                                     # row copies y, updated with f (one block)
            muc, zc = carry[3:]
            v_ = zc + muc
            y = jnp.minimum(v_, hW) + jnp.maximum(v_ - hW - kap, 0.0)      # prox of the exact penalty lmax*max(0, .)
            y_r = relax*y + (1.0 - relax)*zc
            rhs = rhs + CtR @ (muc - y_r)
        d = (Md @ rhs).reshape(N, 6)
        s = (a - d[None]) * m_
        l = (l + Gf_r - s) * m_
        if R:
            zc = C_W @ s.sum(0).reshape(-1)
            return f, l, s, (muc + zc - y_r)*onv, zc
        return f, l, s

    def moving(c):
        return (c[0] < iters) & (c[-1] > tol)

    def step(c):
        n, inner = c[0], c[1:-1]
        new = body(n, inner)
        (f, l, s), (f1, l1, s1) = inner[:3], new[:3]
        res = jnp.maximum(jnp.abs(f1 - f).max(), jnp.maximum(jnp.abs(s1 - s).max(), jnp.abs(l1 - l).max()))
        if R:
            res = jnp.maximum(res, jnp.abs(CtR @ (new[3] - inner[3])).max()/rho)
        return (n + 1, *new, res)

    s0 = jnp.einsum('lkab,lkb->lka', G, f0) * m_
    extra = (jnp.where(rc > 0, lamc0.reshape(-1)/rcs, 0.0), C_W @ s0.sum(0).reshape(-1)) if R else ()
    n, f, l, s, *rest = jax.lax.while_loop(moving, step, (jnp.int32(0), f0, l0, s0, *extra,
                                                          jnp.asarray(jnp.inf, packed.dtype)))
    W_traj = (jnp.einsum('lkab,lkb->lka', G, f) * m_).sum(0)
    if R:
        xp = jnp.einsum('kxw,w->kx', S_x, W_traj.reshape(-1)) + xb
        return jnp.concatenate([W_traj.reshape(-1), f.reshape(-1), l.reshape(-1), rc*rest[0], onv,
                                xp.reshape(-1), n.astype(packed.dtype)[None]])
    return jnp.concatenate([W_traj.reshape(-1), f.reshape(-1), l.reshape(-1), n.astype(packed.dtype)[None]])


_h_core = partial(jax.jit, static_argnums=(2, 3, 4, 5))(_h_impl)


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def horizon_batch(packed, iters, N, L, NO=0, NB=0):
    """B layer-1 solves in one call: packed (B, P) rows from pack_horizon, all with the same N, legs,
    obstacle and body-sphere counts (pad obstacles with far-away ones: their rows stay off). On a GPU
    the batch is one program; with tol > 0 every problem runs until the slowest converges (each stops
    updating at its own tolerance). Returns (B, out) rows for unpack_horizon."""
    return jax.vmap(lambda p: _h_impl(p, iters, N, L, NO, NB))(packed)


def unpack_horizon(out, N, L, R=0):
    """Split a solver output row (numpy or jax): W (N, 6), f (L, N, 3), l (L, N, 6), and for R keep-out
    rows lamc and on (N, R) and xp (N, 12); the iteration count is the last entry."""
    nf = 6*N + 3*L*N; nl = nf + 6*L*N
    res = dict(W=out[:6*N].reshape(N, 6), f=out[6*N:nf].reshape(L, N, 3), l=out[nf:nl].reshape(L, N, 6), iters=out[-1])
    if R:
        res.update(lamc=out[nl:nl + N*R].reshape(N, R), on=out[nl + N*R:nl + 2*N*R].reshape(N, R),
                   xp=out[nl + 2*N*R:nl + 2*N*R + 12*N].reshape(N, 12))
    return res


def distributed_horizon(x0, v_des_seq, w_des_seq, Gs_seq, f_nom, Q, m, I_body, dt, N, lam, mu, fz_max,
                        rho=None, iters=40, state=None, relax=1.6, shift=0.0, tol=0.0, n_legs=4, keepout=None):
    """Gs_seq[k] = {i: G_{i,k}} for the contributors in stance at step k. Returns the aggregate
    wrench trajectory (N, 6) and {i: force trajectory (N, 3)}. `state` (a dict the caller
    keeps) carries primal and dual across ticks; `shift` > 0 (time since the previous solve,
    in horizon steps) lets a step warm-start from the next old step once a contact switch has
    moved into it. state['h']['iters'] reports the iterations used.

    keepout (optional): dict(pose0, obstacles, bodies, ...) adds the constraints
    ||p_k + R_k b - c|| >= r_obs + r_b for every step, obstacle (rows [cx, cy, cz, r, cylinder]) and body
    sphere (rows [bx, by, bz, r_b], trunk frame). pose0 = [p, Theta] (world position, small-angle
    attitude with absolute yaw). Each row is linearized at the previous tick's predicted pose and
    tightened by the open-loop tube of a wrench disturbance (sigma_f N, sigma_t N m per step) and an
    initial-state error (sigma_p0, sigma_v0), plus `margin`; rows clear of their tube by more than
    act_dist m, and the k = 0 rows unless k0, are off. Rows are soft with an exact penalty lmax, so
    they hold at ADMM convergence when feasible. state['h'] then also holds lamc (row multipliers),
    on (active rows) and xp (N, 12) predicted [p, Theta, v, w]."""
    packed, meta = pack_horizon(x0, v_des_seq, w_des_seq, Gs_seq, f_nom, Q, m, I_body, dt, N, lam, mu, fz_max,
                                rho, state, relax, shift, tol, n_legs, keepout)
    L, NO, NB = n_legs, meta['NO'], meta['NB']
    out = np.asarray(_h_core(jnp.asarray(packed, backend.DTYPE), jnp.int32(iters), N, L, NO, NB))
    r = unpack_horizon(out, N, L, NO*NB)
    if isinstance(state, dict):
        state['h'] = dict(f=r['f'], l=r['l'], act=meta['act'], iters=int(r['iters']))
        if NO*NB:
            state['h'].update(lamc=r['lamc'], on=r['on'], xp=r['xp'])
    return r['W'], {i: r['f'][i] for i in meta['legs']}


def pack_horizon(x0, v_des_seq, w_des_seq, Gs_seq, f_nom, Q, m, I_body, dt, N, lam, mu, fz_max,
                 rho=None, state=None, relax=1.6, shift=0.0, tol=0.0, n_legs=4, keepout=None):
    """distributed_horizon's problem, warm start included, as one flat float vector for _h_core and
    horizon_batch. Returns (packed, meta): meta holds the contact pattern act (L, N), the legs in stance
    anywhere in the horizon, and the obstacle and body-sphere counts NO, NB."""
    L = n_legs
    legs = sorted({i for k in range(N) for i in Gs_seq[k]})
    act = np.zeros((L, N)); G = np.zeros((L, N, 6, 3)); fn = np.zeros((L, 3))
    for k in range(N):
        for i in Gs_seq[k]:
            act[i, k] = 1.0; G[i, k] = Gs_seq[k][i]
    for i in legs:
        fn[i] = f_nom[i]
    prev = state.get('h') if isinstance(state, dict) else None
    fnT = np.repeat(fn[:, None, :], N, axis=1)
    src = None
    if prev is not None and prev['f'].shape == (L, N, 3):
        pa, k = prev['act'], np.arange(N)
        nx = np.minimum(k + int(np.ceil(shift)), N - 1)
        src = np.where((pa == act).all(0), k, np.where((pa[:, nx] == act).all(0), nx, k))
        keep = (act*pa[:, src])[:, :, None]
        f0 = act[:, :, None]*(keep*prev['f'][:, src] + (1 - keep)*fnT)
        l0 = keep*prev['l'][:, src]
    else:
        f0 = act[:, :, None]*fnT
        l0 = np.zeros((L, N, 6))
    packed = np.concatenate([np.asarray(x0, float).ravel(), np.asarray(v_des_seq, float).ravel(),
                             np.asarray(w_des_seq, float).ravel(), np.asarray(I_body, float).ravel(),
                             np.asarray(Q, float).ravel(), np.full(L, float(fz_max)), fn.ravel(), act.ravel(),
                             G.ravel(), f0.ravel(), l0.ravel(),
                             [m, dt, lam, mu, fz_max, rho if rho is not None else -1.0, relax, tol]])
    NO = NB = 0
    if keepout is not None:
        ko = {**KEEPOUT_DEFAULTS, **keepout}
        obs = np.asarray(ko['obstacles'], float).reshape(-1, 5); bod = np.asarray(ko['bodies'], float).reshape(-1, 4)
        NO, NB = len(obs), len(bod); R = NO*NB
        pose0 = np.asarray(ko['pose0'], float); x0f = np.asarray(x0, float)
        xp = prev.get('xp') if prev is not None else None
        if xp is not None and xp.shape == (N, 12):               # previous plan, moved on by `shift` steps
            ext = np.vstack([xp, 2*xp[-1] - xp[-2]]) if N > 1 else np.vstack([xp, xp])
            t_ = np.minimum(np.arange(N) + shift, N); i0 = np.minimum(np.floor(t_).astype(int), N - 1)
            fr = (t_ - i0)[:, None]
            xbar = (1 - fr)*ext[i0] + fr*ext[i0 + 1]
        else:                                                    # constant twist from the measured state
            xbar = np.hstack([pose0[None] + dt*np.arange(1, N + 1)[:, None]*x0f[None], np.tile(x0f, (N, 1))])
        lamc0 = np.zeros((N, R))
        if src is not None and prev.get('lamc') is not None and prev['lamc'].shape == (N, R):
            lamc0 = prev['lamc'][src]
        packed = np.concatenate([packed, pose0, xbar.ravel(), obs.ravel(), bod.ravel(),
                                 [ko['sigma_f'], ko['sigma_t'], ko['sigma_p0'], ko['sigma_v0'], ko['margin'],
                                  ko['act_dist'], float(ko['k0']), ko['lmax']], lamc0.ravel()])
    return packed, dict(act=act, legs=legs, NO=NO, NB=NB)


def centralized_horizon(x0, v_des_seq, w_des_seq, Gs_seq, f_nom, Q, m, I_body, dt, N, lam, mu, fz_max, iters=6000):
    """Centralized reference: same horizon QP over ALL leg force trajectories, projected gradient."""
    legs = sorted({i for k in range(N) for i in Gs_seq[k]})
    S_w, e_base = build_horizon(x0, v_des_seq, w_des_seq, m, I_body, dt, N)
    Qbar = np.kron(np.eye(N), Q); QW = S_w.T @ Qbar @ S_w; qW = S_w.T @ Qbar @ e_base
    idx = []; cols = 0
    for k in range(N):
        for i in Gs_seq[k]:
            idx.append((k, i, cols)); cols += 3
    Gbig = np.zeros((6*N, cols)); F0 = np.zeros(cols)
    for k, i, c in idx:
        Gbig[6*k:6*(k+1), c:c+3] = Gs_seq[k][i]; F0[c:c+3] = f_nom[i]
    H = Gbig.T @ QW @ Gbig + lam*np.eye(cols); step = 1.0/np.linalg.eigvalsh(H).max()
    F = F0.copy()
    for _ in range(iters):
        F = F - step*(Gbig.T @ (QW @ (Gbig @ F) + qW) + lam*(F - F0))
        for k, i, c in idx:
            F[c:c+3] = project_cone(F[c:c+3], mu, fz_max)
    f = {i: np.zeros((N, 3)) for i in legs}
    for k, i, c in idx: f[i][k] = F[c:c+3]
    W_traj = (Gbig @ F).reshape(N, 6)
    return W_traj, f
