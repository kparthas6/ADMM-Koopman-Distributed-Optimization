import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
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


@partial(jax.jit, static_argnums=(2, 3))
def _h_core(packed, iters, N, L):
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
    n_k = act.sum(0)
    Nk = jnp.kron(jnp.diag(n_k), jnp.eye(6))
    Md = jnp.linalg.inv(QW @ Nk + rho*jnp.eye(6*N))                 # QW Nk, not Nk QW
    GtG = jnp.einsum('lkab,lkac->lkbc', G, G)
    P = lam*jnp.eye(3)[None, None] + rho*GtG
    P = jnp.where(act[:, :, None, None] > 0, P, jnp.eye(3)[None, None])
    capsSN = jnp.repeat(caps[:, None], N, axis=1).reshape(-1)
    pre = _sub_pre(P.reshape(-1, 3, 3), mu, capsSN)
    m_ = act[:, :, None]

    def body(_, carry):
        f, l, s = carry
        gv = (lam*fnom[:, None, :] + rho*jnp.einsum('lkab,lka->lkb', G, s - l)) * m_
        f = _sub_solve(pre, gv.reshape(-1, 3)).reshape(f.shape) * m_
        Gf = jnp.einsum('lkab,lkb->lka', G, f)
        Gf_r = (relax*Gf + (1.0 - relax)*s) * m_                     # over-relaxation, same fixed point
        a = (Gf_r + l) * m_
        d = (Md @ (QW @ a.sum(0).reshape(-1) + qW)).reshape(N, 6)
        s = (a - d[None]) * m_
        l = (l + Gf_r - s) * m_
        return f, l, s

    def moving(c):
        return (c[0] < iters) & (c[4] > tol)

    def step(c):
        n, f, l, s, _ = c
        f1, l1, s1 = body(n, (f, l, s))
        res = jnp.maximum(jnp.abs(f1 - f).max(), jnp.maximum(jnp.abs(s1 - s).max(), jnp.abs(l1 - l).max()))
        return n + 1, f1, l1, s1, res

    s0 = jnp.einsum('lkab,lkb->lka', G, f0) * m_
    n, f, l, s, _ = jax.lax.while_loop(moving, step, (jnp.int32(0), f0, l0, s0,
                                                      jnp.asarray(jnp.inf, packed.dtype)))
    W_traj = (jnp.einsum('lkab,lkb->lka', G, f) * m_).sum(0)
    return jnp.concatenate([W_traj.reshape(-1), f.reshape(-1), l.reshape(-1), n.astype(packed.dtype)[None]])


def distributed_horizon(x0, v_des_seq, w_des_seq, Gs_seq, f_nom, Q, m, I_body, dt, N, lam, mu, fz_max,
                        rho=None, iters=40, state=None, relax=1.6, shift=0.0, tol=0.0, n_legs=4):
    """Gs_seq[k] = {i: G_{i,k}} for the contributors in stance at step k. Returns the aggregate
    wrench trajectory (N, 6) and {i: force trajectory (N, 3)}. `state` (a dict the caller
    keeps) carries primal and dual across ticks; `shift` > 0 (time since the previous solve,
    in horizon steps) lets a step warm-start from the next old step once a contact switch has
    moved into it. state['h']['iters'] reports the iterations used."""
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
    out = np.asarray(_h_core(jnp.asarray(packed), jnp.int32(iters), N, L))
    nf = 6*N + 3*L*N
    W = out[:6*N].reshape(N, 6); f = out[6*N:nf].reshape(L, N, 3); l = out[nf:-1].reshape(L, N, 6)
    if isinstance(state, dict):
        state['h'] = dict(f=f, l=l, act=act, iters=int(out[-1]))
    return W, {i: f[i] for i in legs}


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
