"""run() as one JAX program, for many robots at once on a GPU.

The dhmpc controller of koopman_admm.run() (gait clock, predicted footholds, the layer-1 horizon solve
with the keep-out rows, steering, swing IK and the Koopman swing MPC, stance torques) as a pure
jax.numpy step, against MJX physics, under lax.scan over control ticks and vmap over robots: a batch of
closed-loop rollouts is one compiled program.

    import jax_loop as JL
    roll = JL.Rollout(obstacles=[(1.2, 0.0, 0.15)], gait_T=0.25)          # the obstacle types are fixed here
    out = roll.run(9.0, vx=np.full(256, 0.3), obstacles=layouts)        # layouts (256, 1, 3): one per robot
    out["upright"], out["min_clear"], out["contact_ticks"]              # (256,) each

run_host() runs the same jitted step against the CPU MuJoCo TorqueSim, one robot, which is how the port
is checked against run() (tests/test_jax_loop.py::test_ported_loop_matches_run, --runslow).

The gait clock (phase, stance, horizon contact pattern, time to touchdown) is computed on the host in
Python floats, exactly as run() computes it, and passed to the step (gait_clock): under jit XLA turns t/T
into t*(1/T), which switches phases a tick off at some boundaries (at gait_T 0.10 in float64, and at most
boundaries in float32).

Scope: body dhmpc, modes trot and stand (duty < 1 flight included), no contact gating, constant commands
per robot. The MJX scene keeps robot-world contacts only (floor and obstacles, no self-collision: 96
contact slots instead of 1,112 without obstacles, 56 more per obstacle either way), and obstacle
cylinders are capsules there, since MJX's JAX implementation has no cylinder-box collision; a capsule of
the same radius and length has the same profile below 0.5 m. MJX is not bit-identical to MuJoCo's CPU
physics."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import backend                                                  # device and precision: before any JAX use
import numpy as np
import jax, jax.numpy as jnp, qpax
import mujoco
from mujoco import mjx
import leg_kinematics as kin, koopman_admm as KA, dist_horizon as DH, torque_sim

FT = backend.DTYPE
DIAGM = np.array([[1, 0, 0, 1], [0, 1, 1, 0]], bool)          # KA.DIAG as stance masks
HIP, TY = np.asarray(kin.HIP_OFFSETS, float), np.asarray(kin.THIGH_Y, float)
HOME = np.array([kin.fk_leg(kin.HOME_LEG, HIP[i], TY[i]) for i in range(4)])
I_BODY = np.diag([0.1268, 0.394, 0.4229])                     # as run()
Q_TRK = np.diag([400., 400., 400., 30., 30., 15.])
MU, Z0, CLR, LAM = 0.7, 0.27, 0.06, 0.005


def fk_leg(q, hip, ty):
    """kin.fk_leg in jax.numpy."""
    c1, s1 = jnp.cos(q[0]), jnp.sin(q[0])
    s2, c2 = jnp.sin(q[1]), jnp.cos(q[1])
    s23, c23 = jnp.sin(q[1] + q[2]), jnp.cos(q[1] + q[2])
    L = kin.LINK
    return (hip + jnp.stack([jnp.zeros_like(c1), ty*c1, ty*s1]) + jnp.stack([-L*s2, L*c2*s1, -L*c2*c1])
            + jnp.stack([-L*s23, L*c23*s1, -L*c23*c1]))


def jac_leg(q, hip, ty):
    """kin.jac_leg: forward differences (1e-6) in float64, as on the CPU; forward-mode autodiff in float32,
    where a 1e-6 difference is rounding noise."""
    if FT == jnp.float64:
        f0 = fk_leg(q, hip, ty)
        return jnp.stack([(fk_leg(q + 1e-6*e, hip, ty) - f0)/1e-6 for e in jnp.eye(3)], -1)
    return jax.jacfwd(fk_leg)(q, hip, ty)


def ik_leg(target, q0, hip, ty, iters=10):
    """KA.ik_leg: damped Newton steps clipped to 0.4 rad, frozen once the error is below 1e-5 m."""
    def body(_, c):
        q, done = c
        e = target - fk_leg(q, hip, ty)
        done = done | (e @ e < 1e-10)
        dq = jnp.clip(jnp.linalg.solve(jac_leg(q, hip, ty) + 1e-8*jnp.eye(3), e), -0.4, 0.4)
        return jnp.where(done, q, q + dq), done
    return jax.lax.fori_loop(0, iters, body, (q0, jnp.asarray(False)))[0]


def phi(q, qd):
    """KA.phi, the 17-feature lift."""
    s, c = jnp.sin(q), jnp.cos(q)
    sh = q[1] + q[2]
    return jnp.concatenate([q, qd, s, c, jnp.stack([jnp.sin(sh), jnp.cos(sh)]), qd*qd])


def quat2R(q):
    w, x, y, z = q
    return jnp.stack([jnp.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)]),
                      jnp.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)]),
                      jnp.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)])])


def rpy(q):
    """TorqueSim.trunk_rpy."""
    w, x, y, z = q
    return jnp.stack([jnp.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)), jnp.arcsin(jnp.clip(2*(w*y - z*x), -1, 1)),
                      jnp.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))])


def skew(r):
    z = jnp.zeros_like(r[..., 0])
    return jnp.stack([jnp.stack([z, -r[..., 2], r[..., 1]], -1), jnp.stack([r[..., 2], z, -r[..., 0]], -1),
                      jnp.stack([-r[..., 1], r[..., 0], z], -1)], -2)


def _seg_dist(c, a, b):
    ab = b - a
    t = jnp.clip(((c - a)*ab).sum(-1)/jnp.maximum((ab*ab).sum(-1), 1e-12), 0.0, 1.0)
    return jnp.linalg.norm(a + t[..., None]*ab - c, axis=-1)


def steer_offset(pos, yaw_cmd, v_cmd, obs, prev, cfg=None):
    """KA.steer_offset in jax.numpy (same candidates, sweep and choice)."""
    c = KA.KEEPOUT_CFG["steer"] if cfg is None else cfg
    cy, sy = jnp.cos(yaw_cmd), jnp.sin(yaw_cmd)
    vc = jnp.stack([cy*v_cmd[0] - sy*v_cmd[1], sy*v_cmd[0] + cy*v_cmd[1]])
    bod = np.asarray(KA.KEEPOUT_CFG["bodies"], float)
    xf, xr, rb = bod[:, 0].max(), bod[:, 0].min(), bod[:, 3].max()
    off = np.linspace(-c["max_off"], c["max_off"], c["n_cand"])
    a = jnp.arctan2(vc[1], vc[0]) + off
    ab = a - jnp.arctan2(v_cmd[1], v_cmd[0])
    u = jnp.stack([jnp.cos(a), jnp.sin(a)], -1)[:, None]
    e = jnp.stack([jnp.cos(ab), jnp.sin(ab)], -1)[:, None]
    p0, p1, du = pos[:2] + xr*e, pos[:2] + xf*e, c["look"]*u
    o = obs[None, :, :2]
    dist = jnp.minimum(jnp.minimum(_seg_dist(o, p0, p1), _seg_dist(o, p0 + du, p1 + du)),
                       jnp.minimum(_seg_dist(o, p0, p0 + du), _seg_dist(o, p1, p1 + du)))
    m1, m2, q = p1 - p0, du, o - p0
    det = m1[..., 0]*m2[..., 1] - m1[..., 1]*m2[..., 0]
    if FT == jnp.float64:
        ok = jnp.abs(det) > 1e-9                               # as the numpy code
    else:                                                      # float32: a parallel sweep leaves |det| ~ 1e-8 of noise
        ok = jnp.abs(det) > 1e-6*jnp.linalg.norm(m1, axis=-1)*jnp.linalg.norm(m2, axis=-1)
    dd = jnp.where(ok, det, 1.0)
    al = (q[..., 0]*m2[..., 1] - q[..., 1]*m2[..., 0])/dd; be = (m1[..., 0]*q[..., 1] - m1[..., 1]*q[..., 0])/dd
    dist = jnp.where(ok & (al >= 0) & (al <= 1) & (be >= 0) & (be <= 1), 0.0, dist)
    dz = jnp.where(obs[:, 4] > 0.5, 0.0, obs[:, 2] - pos[2])
    reach = (obs[:, 3] + rb + c["clear"])**2 - dz**2
    rr = jnp.where(reach > 0.0, jnp.sqrt(jnp.maximum(reach, 0.0)), -jnp.inf)
    score = (dist - rr[None]).min(1)
    ok = jnp.where(score.max() > 0.0, score > 0.0, score >= score.max() - 1e-3)
    cost = jnp.abs(off) + c["hyst"]*jnp.abs(off - prev)
    pick = jnp.asarray(off, FT)[jnp.argmin(jnp.where(ok, cost, jnp.inf))]
    return jnp.where((jnp.linalg.norm(vc) < 1e-3) | (score[c["n_cand"]//2] > 0.0), 0.0, pick)


def keepout_clearance(pts, obs, bod):
    """KA.keepout_clearance: smallest surface distance between body spheres at pts (..., NB, 3) and obs."""
    dv = (pts[..., :, None, :] - obs[:, :3])*jnp.where(obs[:, 4:5] > 0.5, jnp.array([1., 1., 0.]), 1.)
    return (jnp.linalg.norm(dv, axis=-1) - obs[:, 3] - bod[:, None, 3]).min()


def body_points(pos, yaw, bod):
    """Body-sphere centres for trunk positions (..., 3) and headings (...,), level trunk."""
    cp, sp = jnp.cos(yaw)[..., None], jnp.sin(yaw)[..., None]
    return pos[..., None, :] + jnp.stack([cp*bod[:, 0] - sp*bod[:, 1], sp*bod[:, 0] + cp*bod[:, 1],
                                          jnp.broadcast_to(bod[:, 2], cp.shape[:-1] + bod[:, 2].shape)], -1)


def gait_clock(stp, dt, T, duty, N, DTH, mode="trot"):
    """run()'s gait clock at tick stp, in Python floats as run() computes it: stance and lifted masks (4,),
    swing phase s_ph, the horizon's contact pattern sched (N, 4), the time to each step's touchdown tdo (N,)
    and whether the push window (1.0 to 1.15 s) is on."""
    t = stp*dt
    if mode != "trot":
        return dict(stance=np.ones(4, bool), lifted=np.zeros(4, bool), s_ph=0.0, sched=np.ones((N, 4), bool),
                    tdo=np.zeros(N), push=1.0 <= t < 1.15)
    ph = int(t/T) % 2; s_ph = (t % T)/T
    return dict(stance=DIAGM[ph] & (s_ph < duty), lifted=DIAGM[ph] & (duty < 1.0 and s_ph >= duty), s_ph=s_ph,
                sched=np.array([DIAGM[int((t + k*DTH)/T) % 2] & (((t + k*DTH) % T)/T < duty) for k in range(N)]),
                tdo=np.array([max(0.0, int((t + k*DTH)/T)*T - t) for k in range(N)]), push=1.0 <= t < 1.15)


class Controller:
    """The dhmpc branch of KA.run() as pure functions: init(yaw0) -> state, step(state, sensors, cmd, obs,
    clk) -> (state, tau (12,), info). sensors: dict(q, qd (12,), pos (3,), quat (4,), tw (6,) [world linear,
    body angular velocity, as qvel], feet (4, 3), bias (12,)); cmd: (vx, vy, wz); obs: (NO, 5) rows
    [cx, cy, cz, r, cylinder] (ignored when keepout is False); clk: clock(stp) for the tick (host-side,
    see gait_clock). init and step jit and vmap."""
    def __init__(self, legs, mass, dt, mode="trot", gait_T=0.10, duty=1.0, k_cap=0.10, n_obs=0, steer=True,
                 keepout=True):
        if mode not in ("trot", "stand"):
            raise ValueError("mode must be 'trot' or 'stand'")
        self.mode, self.T, self.duty, self.k_cap, self.dt = mode, float(gait_T), float(duty), float(k_cap), float(dt)
        self.m, self.mg = float(mass), float(mass)*9.81
        self.N, self.DTH, self.NIT, self.TOL = (KA.DHMPC_CFG[k] for k in ("N", "dt", "iters", "tol"))
        self.bod = np.asarray(KA.KEEPOUT_CFG["bodies"], float)
        self.NO = int(n_obs) if keepout else 0
        self.NB = len(self.bod) if self.NO else 0
        self.R, self.steer = self.NO*self.NB, bool(steer) and self.NO > 0
        ko = {**DH.KEEPOUT_DEFAULTS, **KA.KEEPOUT_CFG["rows"]}
        self.ko_scal = np.array([ko['sigma_f'], ko['sigma_t'], ko['sigma_p0'], ko['sigma_v0'], ko['margin'],
                                 ko['act_dist'], float(ko['k0']), ko['lmax']])
        self.f_nom = np.tile([0., 0., self.mg/4], (4, 1))
        lg = legs if legs is not None else KA.learn_operators()
        self.SuTQSz = np.stack([l.SuTQSz for l in lg]); self.SuTR = np.stack([l.SuTR for l in lg])
        self.H = jnp.stack([l.H for l in lg])                          # float32, as KoopmanLeg
        self.QP = (lg[0].Aeq, lg[0].beq, lg[0]._G, lg[0]._h)
        self.tau_lim = lg[0].tau_lim

    def clock(self, stp):
        """gait_clock for tick stp, as backend arrays."""
        c = gait_clock(stp, self.dt, self.T, self.duty, self.N, self.DTH, self.mode)
        return {k: jnp.asarray(v, FT) if k in ("s_ph", "tdo") else jnp.asarray(v) for k, v in c.items()}

    def clock_table(self, n):
        """gait_clock for ticks 0..n-1, stacked: lax.scan inputs."""
        c0 = gait_clock(0, self.dt, self.T, self.duty, self.N, self.DTH, self.mode)
        cs = [gait_clock(i, self.dt, self.T, self.duty, self.N, self.DTH, self.mode) for i in range(n)]
        tab = {k: np.array([c[k] for c in cs]).reshape((n,) + np.shape(c0[k])).astype(np.asarray(c0[k]).dtype) for k in c0}
        return {k: jnp.asarray(v, FT if k in ("s_ph", "tdo") else None) for k, v in tab.items()}

    def init(self, yaw0=0.0):
        N, R = self.N, max(self.R, 1)
        z = lambda *s: jnp.zeros(s, FT)
        y = jnp.asarray(yaw0, FT)
        return dict(stp=jnp.int32(0), yaw_ref=y, I_yaw=z(), yaw_off=z(), yaw_un=y,
                    swing_q=jnp.tile(jnp.asarray(kin.HOME_LEG, FT), (4, 1)), hf=z(4, N, 3), hl=z(4, N, 6),
                    hact=z(4, N), hlamc=z(N, R), hxp=z(N, 12), has_prev=jnp.asarray(False))

    def swing_torque(self, q, qd, qt):
        """The four legs' swing MPCs (KoopmanLeg.swing_torque), batched: (4, 3) each."""
        c = jnp.einsum('lij,lj->li', self.SuTQSz, jax.vmap(phi)(q, qd)) - jnp.einsum('lij,lj->li', self.SuTR, qt)
        Aeq, beq, G, h = self.QP
        x = jax.vmap(lambda H, cc: qpax.solve_qp(H, cc, Aeq, beq, G, h)[0])(self.H, c.astype(jnp.float32))
        return jnp.clip(jnp.nan_to_num(x[:, :3]).astype(FT), -self.tau_lim, self.tau_lim)

    def step(self, s, sens, cmd, obs, clk):
        N, DTH, T, dt, L, NO, NB, R = self.N, self.DTH, self.T, self.dt, 4, self.NO, self.NB, self.R
        vx, vy, wz = cmd[0], cmd[1], cmd[2]
        q4, qd4, pos, tw, feet = sens["q"].reshape(4, 3), sens["qd"].reshape(4, 3), sens["pos"], sens["tw"], sens["feet"]
        bias4 = sens["bias"].reshape(4, 3)
        ang = rpy(sens["quat"]); Rm = quat2R(sens["quat"]); com = pos
        stance, lifted, s_ph, sched = clk["stance"], clk["lifted"], clk["s_ph"], clk["sched"]
        ez_ = Z0 - pos[2]
        yaw_ref = s["yaw_ref"] + wz*dt
        wz_ff, yaw_off = jnp.asarray(0.0, FT), s["yaw_off"]
        if self.steer:                                         # walk around: offset the heading reference
            lim = KA.KEEPOUT_CFG["steer"]["rate"]*dt
            wz_ff = jnp.clip(steer_offset(pos, yaw_ref, (vx, vy), obs, yaw_off) - yaw_off, -lim, lim)/dt
            yaw_off = yaw_off + wz_ff*dt
        e_yaw = jnp.mod(ang[2] - yaw_ref - yaw_off + jnp.pi, 2*jnp.pi) - jnp.pi
        I_yaw = jnp.clip(s["I_yaw"] + e_yaw*dt, -0.4, 0.4)
        e_yaw = e_yaw + 1.5*I_yaw
        cy_, sy_ = jnp.cos(ang[2]), jnp.sin(ang[2]); o_, l_ = jnp.zeros_like(cy_), jnp.ones_like(cy_)
        Rz = jnp.stack([jnp.stack([cy_, -sy_, o_]), jnp.stack([sy_, cy_, o_]), jnp.stack([o_, o_, l_])])
        w_w = Rm @ tw[3:6]
        e_th_w = Rz @ jnp.stack([ang[0], ang[1], o_]) + jnp.stack([o_, o_, e_yaw])
        v_ref_w = Rz @ jnp.stack([vx, vy, o_])
        I_W = Rz @ I_BODY @ Rz.T
        kk = jnp.arange(N)
        if self.mode == "trot":                                # feet due to land advance with the command
            ns = stance.sum()
            gz = jnp.where(ns > 0, jnp.where(stance, feet[:, 2], 0.0).sum()/jnp.maximum(ns, 1), 0.0)
            com_k = com[None] + v_ref_w[None]*(kk*DTH)[:, None]
            tdo = clk["tdo"][:, None, None]
            moved = jnp.concatenate([(feet[None] + v_ref_w*tdo)[..., :2], jnp.broadcast_to(gz, (N, 4, 1))], -1)
            fk = jnp.where(tdo > 0.0, moved, feet[None])
        else:
            com_k = jnp.broadcast_to(com, (N, 3)); fk = jnp.broadcast_to(feet, (N, 4, 3))
        act = sched.T.astype(FT)                                                    # (L, N)
        eye = jnp.broadcast_to(jnp.eye(3, dtype=FT), (N, 4, 3, 3))
        G = jnp.swapaxes(jnp.concatenate([eye, skew(fk - com_k[:, None])], -2), 0, 1)   # (L, N, 6, 3)
        G = jnp.where(act[..., None, None] > 0, G, 0.0)
        vd = jnp.tile(v_ref_w - 25.0*jnp.stack([o_, o_, -ez_]), (N, 1))
        wd = jnp.tile(jnp.stack([o_, o_, wz + wz_ff]) - jnp.array([40., 40., 3.5])*e_th_w, (N, 1))
        # pack_horizon, warm start included, in jax.numpy
        x0 = jnp.concatenate([tw[0:3], w_w])
        fn = jnp.where(sched.any(0)[:, None], jnp.asarray(self.f_nom, FT), 0.0)
        fnT = jnp.repeat(fn[:, None, :], N, axis=1)
        pa, nx = s["hact"], np.minimum(np.arange(N) + int(np.ceil(dt/DTH)), N - 1)
        src = jnp.where((pa == act).all(0), kk, jnp.where((pa[:, nx] == act).all(0), nx, kk))
        keep = (act*pa[:, src])[:, :, None]
        hp = s["has_prev"]
        f0 = jnp.where(hp, act[:, :, None]*(keep*s["hf"][:, src] + (1 - keep)*fnT), act[:, :, None]*fnT)
        l0 = jnp.where(hp, keep*s["hl"][:, src], 0.0)
        parts = [x0, vd.ravel(), wd.ravel(), I_W.ravel(), jnp.asarray(Q_TRK, FT).ravel(), jnp.full(L, self.mg, FT),
                 fn.ravel(), act.ravel(), G.ravel(), f0.ravel(), l0.ravel(),
                 jnp.array([self.m, DTH, LAM, MU, self.mg, -1.0, 1.6, self.TOL], FT)]
        yaw_un = s["yaw_un"]
        if NO:                                                 # pose [p, Theta] with unwrapped yaw
            yaw_un = yaw_un + (jnp.mod(ang[2] - yaw_un + jnp.pi, 2*jnp.pi) - jnp.pi)
            pose0 = jnp.concatenate([pos, Rz @ jnp.stack([ang[0], ang[1], o_]) + jnp.stack([o_, o_, yaw_un])])
            xp = s["hxp"]; shift = dt/DTH
            ext = jnp.concatenate([xp, (2*xp[-1] - xp[-2])[None]]) if N > 1 else jnp.concatenate([xp, xp])
            t_ = np.minimum(np.arange(N) + shift, N); i0 = np.minimum(np.floor(t_).astype(int), N - 1)
            fr = (t_ - i0)[:, None]
            xbar_prev = (1 - fr)*ext[i0] + fr*ext[i0 + 1]
            xbar_cold = jnp.concatenate([pose0[None] + DTH*jnp.arange(1, N + 1)[:, None]*x0[None], jnp.tile(x0, (N, 1))], 1)
            xbar = jnp.where(hp, xbar_prev, xbar_cold)
            lamc0 = jnp.where(hp, s["hlamc"][src], 0.0)
            parts += [pose0, xbar.ravel(), obs.ravel(), jnp.asarray(self.bod, FT).ravel(),
                      jnp.asarray(self.ko_scal, FT), lamc0.ravel()]
        out = DH._h_impl(jnp.concatenate(parts), jnp.int32(self.NIT), N, L, NO, NB)
        r = DH.unpack_horizon(out, N, L, R)
        F = r["f"][:, 0]
        # layer 2: stance J^T, swing IK + Koopman MPC
        J = jax.vmap(jac_leg)(q4, HIP, TY)
        tau_st = bias4 - jnp.einsum('lji,lj->li', J, jnp.einsum('ji,lj->li', Rm, F))
        v_b = Rm.T @ tw[0:3]
        v_fwd = jnp.clip(v_b[0], -0.2, 1.2)
        stride = v_fwd*(2*T)*0.5; stride_y = vy*(2*T)*0.5
        cap_x = jnp.clip(self.k_cap*(v_fwd - vx), -0.06, 0.06)
        dpsi = (wz + wz_ff)*(2*T)*(s_ph - 0.5); cz, sz = jnp.cos(dpsi), jnp.sin(dpsi)
        hf = HOME
        frh = jnp.stack([cz*hf[:, 0] - sz*hf[:, 1], sz*hf[:, 0] + cz*hf[:, 1], jnp.broadcast_to(hf[:, 2], (4,))], -1)
        ft_lift = frh + jnp.stack([-0.45*stride, self.k_cap*v_b[1], jnp.asarray(0.04, FT)])
        ft_walk = frh + jnp.stack([stride*(s_ph - 0.5) + cap_x, stride_y*(s_ph - 0.5) + self.k_cap*v_b[1],
                                   CLR*jnp.sin(jnp.pi*s_ph)])
        ft = jnp.where(lifted[:, None], ft_lift, ft_walk)
        q_ik = jax.vmap(ik_leg)(ft, s["swing_q"], HIP, TY)
        tau_sw = bias4 + self.swing_torque(q4, qd4, q_ik)
        tau = jnp.where(stance[:, None], tau_st, tau_sw).reshape(12)
        s2 = dict(stp=s["stp"] + 1, yaw_ref=yaw_ref, I_yaw=I_yaw, yaw_off=yaw_off, yaw_un=yaw_un,
                  swing_q=jnp.where(stance[:, None], s["swing_q"], q_ik), hf=r["f"], hl=r["l"], hact=act,
                  hlamc=r["lamc"] if R else s["hlamc"], hxp=r["xp"] if R else s["hxp"], has_prev=jnp.asarray(True))
        info = dict(rpy=ang, steer=yaw_off, rows=r["on"].sum() if R else jnp.asarray(0.0, FT),
                    lam=r["lamc"].max() if R else jnp.asarray(0.0, FT))
        if R:
            info["plan_clear"] = keepout_clearance(body_points(r["xp"][:, :3], r["xp"][:, 5], self.bod), obs, self.bod)
        return s2, tau, info


def mjx_model(obstacles=(), contacts="world", sim_dt=0.001):
    """The TorqueSim scene for MJX: obstacles as world geoms (x, y, r) capsules 0.5 m long (MJX's JAX
    implementation has no cylinder-box collision) and (x, y, z, r) spheres; contacts 'world' keeps
    robot-world pairs only, 'all' every pair the CPU scene has (with self-collision; about 5x slower in MJX
    without obstacles)."""
    spec = mujoco.MjSpec.from_file(os.path.abspath(torque_sim._SCENE))
    for j, o in enumerate(obstacles):
        g = spec.worldbody.add_geom(); g.name = f"obstacle{j}"; g.rgba = [0.85, 0.35, 0.2, 1.0]
        if len(o) == 3:
            g.type = mujoco.mjtGeom.mjGEOM_CAPSULE; g.size = [o[2], 0.25, 0.0]; g.pos = [o[0], o[1], 0.25]
        else:
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE; g.size = [o[3], 0.0, 0.0]; g.pos = [o[0], o[1], o[2]]
    m = spec.compile(); m.opt.timestep = sim_dt
    if contacts == "world":
        world = m.geom_bodyid == 0; coll = (m.geom_contype | m.geom_conaffinity) > 0
        m.geom_contype[coll & ~world], m.geom_conaffinity[coll & ~world] = 2, 0
        m.geom_contype[coll & world], m.geom_conaffinity[coll & world] = 0, 2
    elif contacts != "all":
        raise ValueError("contacts must be 'world' or 'all'")
    return m


def _geom_rows(o):
    """Per-obstacle MJX geom fields (pos, size, rbound, aabb) for one (x, y, r) or (x, y, z, r)."""
    if len(o) == 3:
        x, y, r = o; return [x, y, 0.25], [r, 0.25, 0.0], r + 0.25, [0, 0, 0, r, r, 0.25 + r]
    x, y, z, r = o; return [x, y, z], [r, 0.0, 0.0], r, [0, 0, 0, r, r, r]


class Rollout:
    """Batched closed-loop rollouts: the Controller against MJX, vmapped over robots. One instance per
    obstacle template (count and types); each run() compiles once per (batch, ticks, log) and then reuses
    the program."""
    def __init__(self, legs=None, obstacles=None, mode="trot", gait_T=0.10, duty=1.0, k_cap=0.10, steer=True,
                 keepout=True, control_dt=0.005, sim_dt=0.001, contacts="world"):
        self.template = [tuple(float(v) for v in o) for o in obstacles] if obstacles is not None else []
        self.kinds = [len(o) for o in self.template]
        self.m = mjx_model(self.template, contacts, sim_dt)
        self.mx = mjx.put_model(self.m)
        oid = lambda kind, n: mujoco.mj_name2id(self.m, kind, n)
        jn = [oid(mujoco.mjtObj.mjOBJ_JOINT, f"{l}_{j}_joint") for l in torque_sim.LEGS for j in ("hip", "thigh", "calf")]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jn]); self.vadr = np.array([self.m.jnt_dofadr[j] for j in jn])
        self.aadr = np.array([oid(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{l}_{j}") for l in torque_sim.LEGS
                              for j in ("hip", "thigh", "calf")])
        self.foot_gids = np.array([oid(mujoco.mjtObj.mjOBJ_GEOM, l) for l in torque_sim.LEGS])
        self.obst_gids = np.array([oid(mujoco.mjtObj.mjOBJ_GEOM, f"obstacle{j}") for j in range(len(self.template))], int)
        self.trunk_bid = oid(mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.tau_lo, self.tau_hi = self.m.actuator_ctrlrange[self.aadr, 0], self.m.actuator_ctrlrange[self.aadr, 1]
        self.q_home = self.m.key_qpos[0][self.qadr]
        self.substeps = max(1, int(round(control_dt/sim_dt))); self.dt = self.substeps*sim_dt
        for g, o in zip(self.obst_gids, self.template):           # the batched fields match the compiled ones
            p, sz, rb, bb = _geom_rows(o)
            assert np.allclose(self.m.geom_pos[g], p) and np.allclose(self.m.geom_size[g], sz)
            assert np.isclose(self.m.geom_rbound[g], rb) and np.allclose(self.m.geom_aabb[g], bb)
        self.ctrl = Controller(legs, float(np.sum(self.m.body_mass)), self.dt, mode, gait_T, duty, k_cap,
                               len(self.template), steer, keepout)
        self.bod = np.asarray(KA.KEEPOUT_CFG["bodies"], float)
        self._fns, self._clk = {}, {}

    def _sensors(self, d):
        return dict(q=d.qpos[self.qadr], qd=d.qvel[self.vadr], pos=d.qpos[0:3], quat=d.qpos[3:7], tw=d.qvel[0:6],
                    feet=d.geom_xpos[self.foot_gids], bias=d.qfrc_bias[self.vadr])

    def _physics(self, mx, d, tau, xfrc):
        d = d.replace(ctrl=d.ctrl.at[self.aadr].set(jnp.clip(tau, self.tau_lo, self.tau_hi)), xfrc_applied=xfrc)
        return jax.lax.fori_loop(0, self.substeps, lambda _, dd: mjx.step(mx, dd), d)

    def _start(self, mx, d0, pose):
        """Reset one robot at pose (x, y, yaw) and settle it 200 ticks at home, as run() does."""
        qpos = jnp.asarray(self.m.key_qpos[0], FT).at[0:2].set(pose[0:2])
        qpos = qpos.at[3:7].set(jnp.stack([jnp.cos(pose[2]/2), 0.0*pose[2], 0.0*pose[2], jnp.sin(pose[2]/2)]))
        d = mjx.forward(mx, d0.replace(qpos=qpos, qvel=jnp.zeros_like(d0.qvel)))
        zero_x = jnp.zeros_like(d.xfrc_applied)

        def settle(d, _):
            sn = self._sensors(d)
            return self._physics(mx, d, 80*(self.q_home - sn["q"]) - 2*sn["qd"] + sn["bias"], zero_x), None
        return jax.lax.scan(settle, d, None, length=200)[0]

    def _one(self, mx, d0, cmd, push, pose, obs, clk, nst, log):
        """One robot: reset at pose (x, y, yaw), settle 200 ticks at home, nst controller ticks."""
        c = self.ctrl
        d = self._start(mx, d0, pose)
        zero_x = jnp.zeros_like(d.xfrc_applied)
        og = jnp.asarray(self.obst_gids)
        inf = jnp.asarray(jnp.inf, FT)
        stats0 = dict(alive=jnp.asarray(True), ticks=jnp.int32(0), max_tilt=jnp.asarray(0.0, FT), min_clear=inf,
                      contact_ticks=jnp.int32(0), max_abs_y=jnp.asarray(0.0, FT), plan_clear=inf,
                      max_rows=jnp.asarray(0.0, FT), max_lam=jnp.asarray(0.0, FT))

        def tick(carry, ck):
            d, s, st = carry
            sn = self._sensors(d)
            s2, tau, info = c.step(s, sn, cmd, obs, ck)
            on = (push > 0) & ck["push"]
            xfrc = zero_x.at[self.trunk_bid, :3].set(jnp.where(on, jnp.stack([push, push*0.5, 0.0*push]), 0.0))
            d2 = self._physics(mx, d, tau, xfrc)
            a = st["alive"]
            tilt = jnp.degrees(jnp.maximum(jnp.abs(info["rpy"][0]), jnp.abs(info["rpy"][1])))
            if len(self.template):
                Rm = quat2R(sn["quat"])
                clear = keepout_clearance(sn["pos"][None] + jnp.asarray(self.bod[:, :3], FT) @ Rm.T, obs, self.bod)
                con = d2._impl.contact if hasattr(d2, "_impl") else d2.contact
                g1, g2 = con.geom[:, 0], con.geom[:, 1]
                hits = (((g1[:, None] == og) | (g2[:, None] == og)).any(1) & (con.dist < con.includemargin)).sum()
            else:
                clear, hits = inf, jnp.int32(0)
            upd = lambda old, new: jnp.where(a, new, old)
            st2 = dict(alive=a & (d2.qpos[2] >= 0.12), ticks=st["ticks"] + a.astype(jnp.int32),
                       max_tilt=upd(st["max_tilt"], jnp.maximum(st["max_tilt"], tilt)),
                       min_clear=upd(st["min_clear"], jnp.minimum(st["min_clear"], clear)),
                       contact_ticks=st["contact_ticks"] + (a & (hits > 0)).astype(jnp.int32),
                       max_abs_y=upd(st["max_abs_y"], jnp.maximum(st["max_abs_y"], jnp.abs(sn["pos"][1]))),
                       plan_clear=upd(st["plan_clear"], jnp.minimum(st["plan_clear"], info.get("plan_clear", inf))),
                       max_rows=upd(st["max_rows"], jnp.maximum(st["max_rows"], info["rows"])),
                       max_lam=upd(st["max_lam"], jnp.maximum(st["max_lam"], info["lam"])))
            frz = lambda new, old: jax.tree.map(lambda x, y: jnp.where(a, x, y), new, old)   # stop at a fall
            ys = dict(xy=sn["pos"][:2], z=d2.qpos[2], rpy=jnp.degrees(info["rpy"]), clear=clear, hits=hits,
                      rows=info["rows"], steer=info["steer"], alive=a) if log else None
            return (frz(d2, d), frz(s2, s), st2), ys
        (d, s, st), ys = jax.lax.scan(tick, (d, c.init(pose[2]), stats0), clk)
        st.update(end_xy=d.qpos[0:2], end_z=d.qpos[2])
        return st, ys

    def tick_fns(self):
        """One control tick in jitted, vmapped pieces, to time a tick from Python (experiments/bench_gpu.py), on
        the template obstacles: start(pose (B, 3)) -> (data, state), reset and settled; sensors(data) -> the
        readings the controller takes; control(sensors, state, cmd (B, 3), obs (B, NO, 5), clk) -> (state,
        tau (B, 12)), the Controller step; physics(data, tau) -> data, the substeps; tick(data, state, cmd,
        obs, clk) -> (data, state), sensors, controller and physics in one program. clk is Controller.clock(k),
        shared by the batch. These are the pieces run() scans, without its statistics, push force and
        freezing at a fall."""
        c, mx = self.ctrl, self.mx
        zero_x = jnp.zeros((self.m.nbody, 6), FT)

        def control(sn, s, cmd, obs, clk):
            s2, tau, _ = c.step(s, sn, cmd, obs, clk)
            return s2, tau

        def tick(d, s, cmd, obs, clk):
            s2, tau = control(self._sensors(d), s, cmd, obs, clk)
            return self._physics(mx, d, tau, zero_x), s2
        v = lambda f, ax: jax.jit(jax.vmap(f, in_axes=ax))
        return dict(start=v(lambda pose: (self._start(mx, mjx.make_data(mx), pose), c.init(pose[2])), 0),
                    sensors=v(self._sensors, 0), control=v(control, (0, 0, 0, 0, None)),
                    physics=v(lambda d, tau: self._physics(mx, d, tau, zero_x), 0), tick=v(tick, (0, 0, 0, 0, None)))

    def run(self, seconds, vx=0.0, vy=0.0, wz=0.0, push=0.0, x0=0.0, y0=0.0, yaw0=0.0, obstacles=None, batch=None,
            log=False):
        """Closed loop for a batch of robots. vx, vy, wz, push (N, as KA.run), x0, y0, yaw0 (start pose) are
        scalars or (B,) arrays; obstacles is None (the template for every robot) or B layouts, each NO
        tuples of the template's types ((B, NO, 3) or (B, NO, 4) arrays work). Returns numpy arrays (B, ...):
        upright, ticks, max_tilt (deg), min_clear (m), contact_ticks, max_abs_y, end_xy, end_z, plan_clear,
        max_rows, max_lam, and with log=True per tick xy, z, rpy (deg), clear, hits, rows, steer, alive
        (B, ticks, ...)."""
        nst = int(seconds/self.dt)
        args = [np.asarray(v, float) for v in (vx, vy, wz, push, x0, y0, yaw0)]
        shapes = [a.shape for a in args] + ([(len(obstacles),)] if obstacles is not None else [])
        B = batch or (np.broadcast_shapes(*shapes) or (1,))[0]
        vx, vy, wz, push, x0, y0, yaw0 = [np.broadcast_to(a, (B,)) for a in args]
        lay = [self.template]*B if obstacles is None else [[tuple(float(v) for v in o) for o in l] for l in obstacles]
        if len(lay) != B or any([len(o) for o in l] != self.kinds for l in lay):
            raise ValueError(f"obstacles: {B} layouts of {len(self.template)} obstacles with lengths {self.kinds}")
        obs = np.stack([KA.obstacle_array(l) for l in lay]) if len(self.template) else np.zeros((B, 0, 5))
        mx, axes = self.mx, None
        if obstacles is not None and len(self.template):           # per-robot obstacle geoms
            rows = [[_geom_rows(o) for o in l] for l in lay]
            fields = {}
            for f, j in (("geom_pos", 0), ("geom_size", 1), ("geom_rbound", 2), ("geom_aabb", 3)):
                base = np.broadcast_to(np.asarray(getattr(self.m, f)), (B,) + np.shape(getattr(self.m, f))).copy()
                for b in range(B):
                    base[b, self.obst_gids] = [r[j] for r in rows[b]]
                fields[f] = jnp.asarray(base, FT)
            mx = self.mx.tree_replace(fields)
            axes = jax.tree.map(lambda _: None, self.mx).tree_replace({f: 0 for f in fields})
        key = (B, nst, log, axes is not None)
        if key not in self._fns:
            one = lambda m_, cmd, p, pose, ob, clk: self._one(m_, mjx.make_data(self.mx), cmd, p, pose, ob, clk, nst, log)
            self._fns[key] = jax.jit(jax.vmap(one, in_axes=(axes, 0, 0, 0, 0, None)))
        cmd = jnp.asarray(np.stack([vx, vy, wz], 1), FT)
        pose = jnp.asarray(np.stack([x0, y0, yaw0], 1), FT)
        if nst not in self._clk:                                    # the host-side gait clock, once per length
            self._clk[nst] = self.ctrl.clock_table(nst)
        st, ys = self._fns[key](mx, cmd, jnp.asarray(push, FT), pose, jnp.asarray(obs, FT), self._clk[nst])
        out = {k: np.asarray(v) for k, v in st.items()}
        out["upright"] = out.pop("alive") & (out["end_z"] > 0.12) & (out["max_tilt"] < 45)
        if log:
            out.update({k: np.asarray(v) for k, v in ys.items()})
        return out


def run_host(sim, legs=None, mode="trot", seconds=10.0, vx=0.0, vy=0.0, wz=0.0, gait_T=0.10, duty=1.0,
             k_cap=0.10, obstacles=None, steer=True, push=0.0, verbose=False):
    """The jitted Controller step against a CPU TorqueSim, one robot, ticked from Python like KA.run():
    the same settle, push and fall rule, and the log keys run() keeps (with obstacles: xy, clear, hits,
    rows, steer). Returns (log, upright)."""
    obs = KA.obstacle_array(obstacles) if obstacles is not None and len(obstacles) else np.zeros((0, 5))
    ctrl = Controller(legs, sim.mass, sim.control_dt, mode, gait_T, duty, k_cap, len(obs), steer)
    yaw0 = float(sim.trunk_rpy()[2])                           # as run(): read before the reset
    sim.reset(); dt = sim.control_dt
    for _ in range(200):
        q, qd = sim.leg_state(); sim.step(80*(sim.q_home-q) - 2*qd + sim.bias_torque())
    step = jax.jit(ctrl.step)
    s, cmd, ob = ctrl.init(yaw0), jnp.asarray([vx, vy, wz], FT), jnp.asarray(obs, FT)
    bod = np.asarray(KA.KEEPOUT_CFG["bodies"], float)
    log = {k: [] for k in ["roll", "pitch", "yaw", "z", "wz", "vx", "qpos", "tau"] +
           (["xy", "clear", "hits", "rows", "steer"] if len(obs) else [])}
    for stp in range(int(seconds/dt)):
        t = stp*dt
        pos, tw = sim.d.qpos[0:3].copy(), sim.d.qvel[0:6].copy()
        sens = dict(q=sim.d.qpos[sim.qadr], qd=sim.d.qvel[sim.vadr], pos=pos, quat=sim.d.qpos[3:7], tw=tw,
                    feet=sim.foot_pos_world(), bias=sim.bias_torque())
        s, tau, info = step(s, {k: jnp.asarray(v, FT) for k, v in sens.items()}, cmd, ob, ctrl.clock(stp))
        tau = np.asarray(tau, float)
        if push > 0 and 1.0 <= t < 1.15:
            sim.d.xfrc_applied[sim.trunk_bid, :3] = np.array([push, push*0.5, 0.])
        else:
            sim.d.xfrc_applied[sim.trunk_bid, :6] = 0.
        R = KA.quat2R(sim.d.qpos[3:7])
        sim.step(tau)
        for k, v in zip(("roll", "pitch", "yaw"), np.asarray(info["rpy"])):
            log[k].append(np.degrees(v))
        log["z"].append(sim.trunk_z()); log["wz"].append(tw[5]); log["vx"].append(tw[0])
        log["qpos"].append(sim.d.qpos.copy()); log["tau"].append(tau)
        if len(obs):
            log["xy"].append(pos[:2]); log["hits"].append(sim.obstacle_contacts())
            log["clear"].append(KA.keepout_clearance(pos[None] + bod[:, :3] @ R.T, obs, bod))
            log["rows"].append(float(info["rows"])); log["steer"].append(float(info["steer"]))
        if sim.trunk_z() < 0.12:
            if verbose: print(f"  FELL at t={t:.2f}s")
            break
    L = {k: np.array(v) for k, v in log.items()}
    return L, sim.trunk_z() > 0.12
