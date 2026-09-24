import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
import numpy as np
import jax, jax.numpy as jnp, qpax
jax.config.update("jax_enable_x64", True)
import mujoco
from torque_sim import TorqueSim
import leg_kinematics as kin
import dist_wrench as DW, dist_horizon as DH

DIAG = {0: [0, 3], 1: [1, 2]}                                  # trot diagonals
DHMPC_CFG = {"N": 8, "dt": 0.020, "iters": 8, "tol": 0.0}      # horizon steps and step length, ADMM
                                                               # iterations per tick, early-stop tolerance
NZ = 17
C_Q = np.zeros((3, NZ)); C_Q[:, 0:3] = np.eye(3)
C_QD = np.zeros((3, NZ)); C_QD[:, 3:6] = np.eye(3)


def phi(q, qd):
    """Koopman lift of one leg's joint state; the state itself is kept linearly."""
    s, c = np.sin(q), np.cos(q)
    sh = q[1] + q[2]
    return np.concatenate([q, qd, s, c, [np.sin(sh), np.cos(sh)], qd*qd])


class KoopmanLeg:
    """Lifted leg model z+ = A z + B tau and the swing MPC built on it: 6 steps, torque box,
    solved by qpax in float32 (the precision the operators were fit at)."""
    def __init__(self, A, B, hip, ty, N=6, tau_lim=(23.7, 23.7, 35.55)):
        self.A, self.B, self.hip, self.ty, self.N = A, B, hip, ty, N
        Apow = [np.eye(NZ)]
        for _ in range(N):
            Apow.append(Apow[-1] @ A)
        Sz = np.vstack([Apow[k] for k in range(1, N+1)])
        AB = [Apow[p] @ B for p in range(N)]
        Su = np.zeros((NZ*N, 3*N))
        for k in range(N):
            for j in range(k+1):
                Su[NZ*k:NZ*(k+1), 3*j:3*j+3] = AB[k-j]
        wq = np.ones(N); wq[-1] = 8.0; wd, wr = 0.01, 5e-5
        Qbar = np.zeros((NZ*N, NZ*N)); Rmap = np.zeros((NZ*N, 3)); Rt = np.zeros((NZ*N, 3))
        for k in range(N):
            Qbar[NZ*k:NZ*(k+1), NZ*k:NZ*(k+1)] = wq[k]*C_Q.T@C_Q + wd*C_QD.T@C_QD
            Rmap[NZ*k:NZ*(k+1), :] = wq[k]*C_Q.T
            Rt[NZ*k:NZ*(k+1), :] = wq[k]*(k+1)*0.0075*C_Q.T
        H = Su.T @ Qbar @ Su + wr*np.eye(3*N)
        self.SuTQSz, self.SuTR, self.SuTRt = Su.T @ Qbar @ Sz, Su.T @ Rmap, Su.T @ Rt
        SelN = np.zeros((3, NZ*N)); SelN[:, NZ*(N-1):NZ*N] = C_QD
        self.Gthr = SelN @ Su
        self.tau_lim = np.asarray(tau_lim, float)
        f32 = jnp.float32
        self.H = jnp.array(0.5*(H+H.T), f32)
        self.Aeq, self.beq = jnp.zeros((0, 3*N), f32), jnp.zeros((0,), f32)
        self._G = jnp.array(np.vstack([np.eye(3*N), -np.eye(3*N)]), f32)
        self._h = jnp.array(np.tile(self.tau_lim, 2*N), f32)
        self._solve = jax.jit(lambda c: qpax.solve_qp(self.H, c, self.Aeq, self.beq, self._G, self._h)[0])

    def thrust_torque(self, qd_dir):
        """Bang-bang torque maximizing the model's end-of-horizon joint velocity along qd_dir."""
        g = self.Gthr.T @ np.asarray(qd_dir, float)
        return (np.tile(self.tau_lim, self.N)*np.sign(g))[:3]

    def swing_torque(self, q, qd, q_target, qd_target=None):
        """First torque of the swing MPC tracking q_target (qd_target: optional rate reference)."""
        c = self.SuTQSz @ phi(q, qd) - self.SuTR @ q_target
        if qd_target is not None:
            c = c - self.SuTRt @ qd_target
        tau = np.nan_to_num(np.array(self._solve(jnp.asarray(c, jnp.float32)))[:3])   # qpax can NaN at extreme states
        return np.clip(tau, -self.tau_lim, self.tau_lim)


def learn_operators(samples=9000, cache="leg_ops.npz"):
    """Per-leg Koopman operators: fit once on simulated swing data, then loaded from the cache."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cache)
    if not os.path.exists(path):
        print("  [operators] no cache: fitting per-leg Koopman operators on %d ticks of swing data" % samples)
        A, B = _fit_operators(TorqueSim(control_dt=0.005, view=False), samples)
        np.savez(path, **{f"A{i}": A[i] for i in range(4)}, **{f"B{i}": B[i] for i in range(4)})
    d = np.load(path)
    return [KoopmanLeg(d[f"A{i}"], d[f"B{i}"], kin.HIP_OFFSETS[i], kin.THIGH_Y[i]) for i in range(4)]


def _fit_operators(sim, samples):
    """Ridge regression z+ = A z + B tau per leg on randomized, torque-excited swing motion
    with the trunk pinned in the air; tau is the torque on top of gravity compensation."""
    Z, U, Zp = [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)]
    sim.reset()
    pin = np.array([0., 0., 0.30, 1., 0., 0., 0.])             # feet ~3.5 cm above the floor
    home = [kin.fk_leg(kin.HOME_LEG, kin.HIP_OFFSETS[i], kin.THIGH_Y[i]) for i in range(4)]
    qsw = [kin.HOME_LEG.copy() for _ in range(4)]
    noise = np.zeros(12); rng = np.random.default_rng(0)
    amp = rng.uniform([0.05, 0.02, 0.04], [0.18, 0.08, 0.09], (4, 3)); per = float(rng.choice([0.20, 0.74]))
    for t in range(samples + 200):
        ph = t * sim.control_dt
        if t % 100 == 0:                                        # new random swing every 0.5 s
            amp = rng.uniform([0.05, 0.02, 0.04], [0.24, 0.10, 0.11], (4, 3))
            per = float(rng.uniform(0.15, 0.80))
            if rng.random() < 0.4:
                amp[:, 0] = rng.uniform(0.16, 0.26, 4)
        q_all, qd_all = sim.leg_state(); bias = sim.bias_torque()
        u = np.zeros(12)
        for i in range(4):
            s = (ph / per + 0.25*i) % 1.0
            ax, ay, az = amp[i]
            ft = home[i] + np.array([ax*(s-0.5), ay*(s-0.5), az*np.sin(np.pi*s)])
            qsw[i] = ik_leg(ft, qsw[i], kin.HIP_OFFSETS[i], kin.THIGH_Y[i])
            u[3*i:3*i+3] = 60.0*(qsw[i]-q_all[3*i:3*i+3]) - 2.0*qd_all[3*i:3*i+3]
        noise += 0.2*(rng.standard_normal(12)*3.0 - noise)
        u = np.clip(u + noise, -20, 20)
        sim.step(u + bias)
        sim.d.qpos[0:7] = pin; sim.d.qvel[0:6] = 0.0
        mujoco.mj_forward(sim.m, sim.d)
        qn, qdn = sim.leg_state()
        if t >= 200:
            for i in range(4):
                Z[i].append(phi(q_all[3*i:3*i+3], qd_all[3*i:3*i+3]))
                U[i].append(u[3*i:3*i+3]); Zp[i].append(phi(qn[3*i:3*i+3], qdn[3*i:3*i+3]))
    A, B = [], []
    for i in range(4):
        P = np.vstack([np.array(Z[i]).T, np.array(U[i]).T])
        AB = np.array(Zp[i]).T @ P.T @ np.linalg.inv(P @ P.T + 1e-5*np.eye(NZ+3))
        A.append(AB[:, :NZ]); B.append(AB[:, NZ:])
    return A, B


def skew(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def grasp_maps(feet, com, stance):
    """Per-foot wrench maps G_i = [I3; skew(r_i)] for the stance feet."""
    return {i: np.vstack([np.eye(3), skew(feet[i] - com)]) for i in stance}


def quat2R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def ik_leg(target, q0, hip, ty, iters=10):
    q = np.array(q0, float)
    for _ in range(iters):
        e = target - kin.fk_leg(q, hip, ty)
        if e@e < 1e-10:
            break
        q = q + np.clip(np.linalg.solve(kin.jac_leg(q, hip, ty)+1e-8*np.eye(3), e), -0.4, 0.4)
    return q


def run(sim, legs, mode="stand", seconds=10.0, vx=0.0, wz=0.0, push=0.0, vy=0.0, iters=30,
        body="dhmpc", verbose=True, contact_gate=False, gait_T=0.10, duty=1.0, k_cap=0.10, stop=None):
    """Closed loop. mode 'stand' | 'trot' (diagonal pairs swap every gait_T, duty < 1 adds
    flight); body 'dhmpc' (horizon, DHMPC_CFG) | 'dist' (instantaneous, `iters` iterations);
    k_cap is the foothold capture gain (0 disables capture).
    vx, vy, wz may be callables of time; `stop()` is polled every tick. Returns (log, upright)."""
    if body not in ("dist", "dhmpc"):
        raise ValueError("body must be 'dist' or 'dhmpc'")
    NH, DTH, NIT, TOL = DHMPC_CFG["N"], DHMPC_CFG["dt"], DHMPC_CFG["iters"], DHMPC_CFG["tol"]
    yaw_ref = float(sim.trunk_rpy()[2]); I_yaw = 0.0
    m_tot = sim.mass
    I_BODY = np.diag([0.1268, 0.394, 0.4229])                 # composite inertia at home stance (roll, pitch, yaw)
    if legs is None:
        legs = learn_operators()
    sim.reset(); dt = sim.control_dt
    mg = sim.mass*9.81; MU = 0.7; z0 = 0.27; T = float(gait_T); CLR = 0.06
    home = {i: kin.fk_leg(kin.HOME_LEG, kin.HIP_OFFSETS[i], kin.THIGH_Y[i]) for i in range(4)}
    f_nom = {i: np.array([0., 0., mg/4]) for i in range(4)}
    Q_trk = np.diag([400., 400., 400., 30., 30., 15.])
    for _ in range(200):                                      # settle at the home pose
        q, qd = sim.leg_state(); sim.step(80*(sim.q_home-q) - 2*qd + sim.bias_torque())
    nst = int(seconds/dt); cmd = (vx, vy, wz); live = any(callable(c) for c in cmd)
    bt_state, h_state = {}, {}
    log = {k: [] for k in ["roll", "pitch", "yaw", "z", "wz", "vx", "qpos"]}
    swing_q = {i: kin.HOME_LEG.copy() for i in range(4)}; t0 = time.time()
    for stp in range(nst):
        t = stp*dt
        if live:
            vx, vy, wz = (c(t) if callable(c) else c for c in cmd)
        if stop is not None and stop():
            break
        q_all, qd_all = sim.leg_state(); pos, tw = sim.trunk_pose()[0][:3], sim.trunk_pose()[1]
        rpy = sim.trunk_rpy(); R = quat2R(sim.d.qpos[3:7]); com = pos.copy()
        feet = sim.foot_pos_world(); bias = sim.bias_torque()
        if mode == "trot":
            ph = int(t/T) % 2; s_ph = (t % T)/T
            stance = DIAG[ph] if s_ph < duty else []
            lifted = DIAG[ph] if (duty < 1.0 and s_ph >= duty) else []
        else:
            stance = [0, 1, 2, 3]; s_ph = 0.0; lifted = []
        reach = set()
        if contact_gate and mode == "trot":                   # force only to feet that are down:
            cvec = sim.contacts(); sched_now = set(stance)    # late feet reach, early feet join
            reach = {i for i in sched_now if cvec[i] < 0.5}
            early = {i for i in range(4) if i not in sched_now and cvec[i] > 0.5 and s_ph > 0.6}
            gated = sorted((sched_now - reach) | early)
            stance = gated if gated else sorted(sched_now)
            if not gated: reach = set()
        ez_ = z0 - pos[2]
        yaw_ref += wz*dt
        e_yaw = (rpy[2] - yaw_ref + np.pi) % (2*np.pi) - np.pi
        I_yaw = np.clip(I_yaw + e_yaw*dt, -0.4, 0.4)
        e_yaw = e_yaw + 1.5*I_yaw
        cy_, sy_ = np.cos(rpy[2]), np.sin(rpy[2])            # world axes throughout: grasp-map moments are world-frame
        Rz = np.array([[cy_, -sy_, 0.], [sy_, cy_, 0.], [0., 0., 1.]])
        w_w = R @ tw[3:6]
        e_th_w = Rz @ np.array([rpy[0], rpy[1], 0.]) + np.array([0., 0., e_yaw])
        v_ref_w = Rz @ np.array([vx, vy, 0.])
        I_W = Rz @ I_BODY @ Rz.T
        if body == "dhmpc":
            sched = ([[0, 1, 2, 3]]*NH if mode != "trot"
                     else [DIAG[int((t + k*DTH)/T) % 2] if ((t + k*DTH) % T)/T < duty else [] for k in range(NH)])
            if mode == "trot":
                if contact_gate:
                    sched[0] = list(stance)
                gz = float(np.mean([feet[i][2] for i in stance])) if stance else 0.0
                v_cmd_w = Rz @ np.array([vx, vy, 0.])
                Gs_seq = []
                for k, st in enumerate(sched):                # feet due to land advance with the command
                    com_k = com + v_cmd_w*(k*DTH)             # until their touchdown, then hold
                    fk = {}
                    for i in st:
                        tdo = max(0.0, int((t + k*DTH)/T)*T - t)
                        fk[i] = np.r_[(feet[i] + v_cmd_w*tdo)[:2], gz] if tdo > 0.0 else feet[i]
                    Gs_seq.append(grasp_maps(fk, com_k, st))
            else:
                Gs_seq = [grasp_maps(feet, com, st) for st in sched]
            vd = [v_ref_w - 25.0*np.array([0., 0., -ez_]) for _ in range(NH)]
            wd = [np.array([0., 0., wz]) - np.array([40., 40., 3.5])*e_th_w for _ in range(NH)]
            _, Ftr = DH.distributed_horizon(np.concatenate([tw[0:3], w_w]), vd, wd, Gs_seq, f_nom, Q_trk, m_tot,
                                            I_W, DTH, NH, 0.005, MU, mg, iters=NIT, state=h_state,
                                            shift=dt/DTH, tol=TOL)
            F = {i: (Ftr[i][0] if i in Ftr else f_nom[i]) for i in stance}
        else:
            e0, D_w = DW.body_tracking_terms(dict(v=tw[0:3], w=w_w, e_p=np.array([0., 0., -ez_]), e_th=e_th_w),
                                             dict(v_ref=v_ref_w, w_ref=np.array([0., 0., wz])), m_tot, I_W, dt,
                                             gains=(25.0, np.array([40., 40., 3.5])), damp=(0.6, 0.5))
            _, F, _ = DW.distributed_body_track(e0, D_w, Q_trk, grasp_maps(feet, com, stance), f_nom, 0.005, MU, mg,
                                                iters=iters, state=bt_state)
        tau = np.zeros(12)
        v_b = R.T @ tw[0:3]
        v_fwd = float(np.clip(v_b[0], -0.2, 1.2))            # Raibert footholds from MEASURED speed
        stride = v_fwd*(2*T)*0.5; stride_y = vy*(2*T)*0.5
        cap_x = float(np.clip(k_cap*(v_fwd - vx), -0.06, 0.06))
        for i in range(4):
            qi = q_all[3*i:3*i+3]; qdi = qd_all[3*i:3*i+3]
            if i in stance:
                J = kin.jac_leg(qi, kin.HIP_OFFSETS[i], kin.THIGH_Y[i])
                tau[3*i:3*i+3] = bias[3*i:3*i+3] - J.T @ (R.T @ F[i])
            else:
                dpsi = wz*(2*T)*(s_ph-0.5); cz, sz = np.cos(dpsi), np.sin(dpsi); hf = home[i]
                fr = np.array([cz*hf[0]-sz*hf[1], sz*hf[0]+cz*hf[1], hf[2]])
                if i in lifted:                               # flight: tuck back, ready to sweep
                    ft = fr + np.array([-0.45*stride, k_cap*v_b[1], 0.04])
                elif i in reach:                              # late touchdown: reach down at the foothold
                    ft = fr + np.array([stride*0.5, stride_y*0.5 + k_cap*v_b[1], -0.03])
                else:
                    ft = fr + np.array([stride*(s_ph-0.5) + cap_x, stride_y*(s_ph-0.5) + k_cap*v_b[1],
                                        CLR*np.sin(np.pi*s_ph)])
                swing_q[i] = ik_leg(ft, swing_q[i], kin.HIP_OFFSETS[i], kin.THIGH_Y[i])
                tau[3*i:3*i+3] = bias[3*i:3*i+3] + legs[i].swing_torque(qi, qdi, swing_q[i])
        if push > 0 and 1.0 <= t < 1.15:
            sim.d.xfrc_applied[sim.trunk_bid, :3] = np.array([push, push*0.5, 0.])
        else:
            sim.d.xfrc_applied[sim.trunk_bid, :6] = 0.
        sim.step(tau)
        for k, v in (("roll", rpy[0]), ("pitch", rpy[1]), ("yaw", rpy[2])):
            log[k].append(np.degrees(v))
        log["z"].append(sim.trunk_z()); log["wz"].append(tw[5]); log["vx"].append(tw[0])
        log["qpos"].append(sim.d.qpos.copy())
        if sim.trunk_z() < 0.12:
            if verbose: print(f"  FELL at t={t:.2f}s")
            break
    L = {k: np.array(v) for k, v in log.items()}; up = sim.trunk_z() > 0.12
    if verbose:
        tilt = np.maximum(np.abs(L["roll"]), np.abs(L["pitch"])).max()
        turn = f" | turn {L['yaw'][-1] - L['yaw'][0]:+.0f} deg" if wz else ""
        print(f"  {mode} ({body}) | tilt max {tilt:.1f} deg | z {L['z'][-1]:.3f}{turn} | "
              f"{nst/(time.time() - t0):.0f} steps/s | {'UP' if up else 'FELL'}")
    return L, up


if __name__ == "__main__":
    env = os.environ.get
    legs = learn_operators()
    sim = TorqueSim(control_dt=0.005)
    run(sim, legs, env("MODE", "trot"), float(env("SECONDS", "6")), vx=float(env("VX", "0")),
        wz=float(env("WZ", "0")), push=float(env("PUSH", "0")), vy=float(env("VY", "0")),
        iters=int(env("ITERS", "30")), body=env("BODY", "dhmpc"), gait_T=float(env("GAIT_T", "0.10")))
    sim.hold()
