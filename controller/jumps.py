import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import numpy as np
import mujoco
from torque_sim import TorqueSim
import leg_kinematics as kin
import koopman_admm as KA, dist_wrench as DW, dist_horizon as DH

G = 9.81
ALL = [0, 1, 2, 3]
JUMP_CFG = {"z_stand": 0.27, "t_settle": 0.4, "T_abs": 0.12, "T_brake": 0.30, "rest": 0.25, "k_land": 0.25,
            "flight_wd": 0.002}    # stand height, first-jump delay, vertical / horizontal landing decelerations (s),
                                   # pause between jumps (s), landing foothold gain, flight-leg rate weight


class Jump:
    """One jump in trunk heights: crouch to z_crouch, push with constant acceleration to z_to,
    take off at (vx_to, vz_to) with vz_to = sqrt(2 g h_apex), land with the trunk at z_land."""
    def __init__(self, h_apex=0.10, vx_to=0.0, z_crouch=0.20, z_to=0.34, z_land=0.32, t_crouch=0.35, t_hold=0.08):
        self.h, self.vx, self.zc, self.zto, self.zl = h_apex, vx_to, z_crouch, z_to, z_land
        self.t_crouch, self.t_hold = t_crouch, t_hold
        self.vz = np.sqrt(2*G*h_apex)
        self.Tp = 2*(z_to - z_crouch)/self.vz                               # push duration
        self.az, self.ax = self.vz/self.Tp, vx_to/self.Tp
        self.dx_back = 0.25*vx_to*self.Tp                                   # crouch back to centre the push
        self.Tf = (self.vz + np.sqrt(self.vz**2 + 2*G*(z_to - z_land)))/G    # planned flight

    def shifted(self, off):
        """The same jump with every height lowered by off (trunk heights -> CoM heights)."""
        return Jump(self.h, self.vx, self.zc - off, self.zto - off, self.zl - off, self.t_crouch, self.t_hold)


_FLIGHT = {}


def flight_legs(legs, wd=None):
    """The swing MPCs with a lighter joint-rate weight (the trot's damps the fast flight
    motions), built once per set of operators."""
    wd = JUMP_CFG["flight_wd"] if wd is None else wd
    key = (id(legs[0]), wd)
    if key not in _FLIGHT:
        _FLIGHT[key] = [KA.KoopmanLeg(l.A, l.B, l.hip, l.ty, wd=wd) for l in legs]
    return _FLIGHT[key]


def stance_torques(sim, F, M):
    """Joint torques whose contact forces are exactly F = {foot: world force on the robot}, with
    those feet pinned at their contact points and the other legs' joint accelerations zero: the
    base rows of M qacc + bias = sum J_i^T F_i and J_i qacc + Jdot_i qvel = 0 fix qacc."""
    m, d = sim.m, sim.d
    mujoco.mj_fullM(m, d, M)
    feet = sorted(F); nv = m.nv
    U = list(range(6)) + [int(v) for i in feet for v in sim.vadr[3*i:3*i+3]]
    rad = m.geom_size[sim.foot_gids[0]][0]
    Jr, Jd, Jdr = np.zeros((3, nv)), np.zeros((3, nv)), np.zeros((3, nv))
    Js, JdQ = [], []
    for i in feet:
        g = sim.foot_gids[i]; b = m.geom_bodyid[g]; pt = d.geom_xpos[g] - np.array([0., 0., rad])
        J = np.zeros((3, nv)); mujoco.mj_jac(m, d, J, Jr, pt, b); mujoco.mj_jacDot(m, d, Jd, Jdr, pt, b)
        Js.append(J); JdQ.append(Jd @ d.qvel)
    n = len(U); A = np.zeros((n, n)); b_ = np.zeros(n)
    A[0:6] = M[0:6][:, U]
    b_[0:6] = -d.qfrc_bias[0:6] + sum(J[:, 0:6].T @ F[i] for J, i in zip(Js, feet))
    for k, J in enumerate(Js):
        A[6+3*k:9+3*k] = J[:, U]; b_[6+3*k:9+3*k] = -JdQ[k]
    qa = np.zeros(nv); qa[U] = np.linalg.solve(A, b_)
    gen = M @ qa + d.qfrc_bias - sum(J.T @ F[i] for J, i in zip(Js, feet))
    tau = np.zeros(12)
    for i in feet:
        tau[3*i:3*i+3] = gen[sim.vadr[3*i:3*i+3]]
    return tau


def flight_foot(J, Jc, hf, tf, v_b):
    """Foot target (leg frame) tf after takeoff: tuck by 30% of the flight, extend to the landing
    depth by 70%, and swing from behind the hip (after the push) to k_land*v*T_brake ahead;
    v_b is the trunk velocity in the body frame."""
    s = np.clip(tf/Jc.Tf, 0, 1)
    d_to, d_tuck, d_land = J.zto - 0.02, min(0.20, J.zc), J.zl - 0.005
    if s < 0.30:
        w = 0.5 - 0.5*np.cos(np.pi*s/0.30); depth = d_to + (d_tuck - d_to)*w
    else:
        w = 0.5 - 0.5*np.cos(np.pi*min(1.0, (s - 0.30)/0.40)); depth = d_tuck + (d_land - d_tuck)*w
    x_land, x_to = JUMP_CFG["k_land"]*v_b[0]*JUMP_CFG["T_brake"], -0.25*J.vx*J.Tp
    u = 0.5 - 0.5*np.cos(np.pi*min(1.0, s/0.7))
    return np.array([hf[0] + x_to + (x_land - x_to)*u, hf[1] + 0.10*v_b[1], -depth])


            if t - t_mode >= Jc.Tp or (n_off >= 2 and t - t_mode > 0.3*Jc.Tp):
                mode, t_mode, airborne = "flight", t, False; to = (x_h, com[2]); events.append(("takeoff", t))
        if mode == "flight":
            airborne = airborne or cvec.sum() == 0
            tf = t - t_mode
            if cvec.sum() > 0 and (airborne or tf > Jc.Tf) and tf > 0.3*Jc.Tf:
                mode, t_mode = "land", t; events.append(("touchdown", t))
                land = dict(x=x_h, z=com[2], vx=v_h[0], vz=v_h[2], down=set(np.flatnonzero(cvec > 0.5)))
        if mode == "land" and t - t_mode >= max(T_abs, T_brake):
            ta = T_abs; z_rest0 = land["z"] + land["vz"]*(ta - ta*ta/(2*T_abs))
            mode, t_mode = "rest", t; events.append(("rest", t))

        def push_ref(s):
            if s < Jc.Tp:
                return (np.array([x_feet - Jc.dx_back + 0.5*Jc.ax*s*s, Jc.zc + 0.5*Jc.az*s*s]),
                        np.array([Jc.ax*s, Jc.az*s]), True)
            u = s - Jc.Tp
            return (np.array([x_feet - Jc.dx_back + 0.5*Jc.vx*Jc.Tp + Jc.vx*u, Jc.zto + Jc.vz*u - 0.5*G*u*u]),
                    np.array([Jc.vx, Jc.vz - G*u]), False)

        def ref(tau):
            """Planned CoM (x, z), velocity and contact, tau ahead of now (heading frame)."""
            s = t - t_mode + tau
            if mode in ("stand", "rest"):
                zr = z_stand if mode == "stand" else z_rest0 + (z_stand - z_rest0)*min(1.0, s/0.3)
                return np.array([x_feet, zr]), np.zeros(2), True
            if mode == "crouch":
                xb = x_feet - Jc.dx_back
                if s < Jc.t_crouch:
                    u = s/Jc.t_crouch; w = 0.5 - 0.5*np.cos(np.pi*u); dw = 0.5*np.pi*np.sin(np.pi*u)/Jc.t_crouch
                    return (np.array([x_c0 + (xb - x_c0)*w, z_c0 + (Jc.zc - z_c0)*w]),
                            np.array([(xb - x_c0)*dw, (Jc.zc - z_c0)*dw]), True)
                if s < Jc.t_crouch + Jc.t_hold:
                    return np.array([xb, Jc.zc]), np.zeros(2), True
                return push_ref(s - Jc.t_crouch - Jc.t_hold)
            if mode == "push":
                return push_ref(s)
            if mode == "flight":
                return np.array([to[0] + Jc.vx*s, to[1] + Jc.vz*s - 0.5*G*s*s]), np.array([Jc.vx, Jc.vz - G*s]), False
            ta, tb = min(s, T_abs), min(s, T_brake)            # land: brake the touchdown velocity to zero
            return (np.array([land["x"] + land["vx"]*(tb - tb*tb/(2*T_brake)),
                              land["z"] + land["vz"]*(ta - ta*ta/(2*T_abs))]),
                    np.array([land["vx"]*(1 - tb/T_brake), land["vz"]*(1 - ta/T_abs)]), True)

        p_ref, v_ref, _ = ref(0.0)
        if mode == "flight":
            stance = []
        elif mode == "land":                                  # feet join as they touch down
            land["down"] |= set(np.flatnonzero(cvec > 0.5))
            stance = sorted(land["down"] if t - t_mode <= 0.06 else ALL)
        else:
            stance = list(ALL)
        e_yaw = (rpy[2] - yaw_ref + np.pi) % (2*np.pi) - np.pi
        I_yaw = np.clip(I_yaw + e_yaw*dt, -0.4, 0.4); e_yaw = e_yaw + 1.5*I_yaw
        I_W = Rz @ I_BODY @ Rz.T
        w_c = np.linalg.solve(I_W, sim.d.subtree_angmom[bid])  # whole-body angular momentum, as a rate
        e_th_w = Rz @ np.array([rpy[0], rpy[1], 0.]) + np.array([0., 0., e_yaw])
        fb = np.array([0.0 if mode == "push" else 3.0*(p_ref[0] - x_h), 25.0*(p_ref[1] - com[2])])
        F = {}
        if stance and body == "dhmpc":
            dth = DTH
            if mode == "push":                                # takeoff on a step boundary
                T_r = Jc.Tp - (t - t_mode)
                if 0 < T_r < NH*DTH:
                    dth = T_r/max(1, int(round(T_r/DTH)))
            vd, wd, Gs_seq = [], [], []
            for k in range(NH):
                p_k, _, st_k = ref(k*dth); _, v_k, _ = ref((k + 1)*dth)
                st = (list(stance) if k == 0 else list(ALL)) if st_k else []
                vd.append(Rz @ np.array([v_k[0] + fb[0], 0., v_k[1] + fb[1]]))
                wd.append(-K_TH*e_th_w if st else np.zeros(3))   # in the air: arrive with zero rate
                Gs_seq.append(KA.grasp_maps(feet, com + Rz @ np.array([p_k[0] - p_ref[0], 0., p_k[1] - p_ref[1]]), st))
            _, Ftr = DH.distributed_horizon(np.concatenate([vcom, w_c]), vd, wd, Gs_seq, f_nom, Q_trk, m, I_W, dth, NH,
                                            0.005, MU, mg, iters=NIT, state=h_state, shift=dt/dth)
            F = {i: (Ftr[i][0] if i in Ftr else f_nom[i]) for i in stance}
        elif stance:                                          # instantaneous: 80 ms look-ahead, blind to the flight
            _, v_la, st_la = ref(0.08)
            if not st_la:
                v_la = np.array([Jc.vx, Jc.vz])
            v_la_w = Rz @ np.array([v_la[0] + fb[0], 0., v_la[1]])
            e0, D_w = DW.body_tracking_terms(dict(v=vcom, w=w_c, e_p=np.array([0., 0., -fb[1]/25.0]), e_th=e_th_w),
                                             dict(v_ref=v_la_w, w_ref=np.zeros(3)), m, I_W, dt,
                                             gains=(25.0, K_TH), damp=(0.6, 0.5))
            _, F, _ = DW.distributed_body_track(e0, D_w, Q_trk, KA.grasp_maps(feet, com, stance), f_nom, 0.005, MU, mg,
                                                iters=30, state=bt_state)
        tau = np.zeros(12)
        tau_id = stance_torques(sim, F, M) if (exact_torques and F) else None
        v_b = R.T @ tw[0:3]
        for i in ALL:
            qi, qdi = q_all[3*i:3*i+3], qd_all[3*i:3*i+3]
            if i in stance:
                if tau_id is not None:
                    tau[3*i:3*i+3] = tau_id[3*i:3*i+3]
                else:
                    Jl = kin.jac_leg(qi, kin.HIP_OFFSETS[i], kin.THIGH_Y[i])
                    tau[3*i:3*i+3] = bias[3*i:3*i+3] - Jl.T @ (R.T @ F[i])
                tau[3*i:3*i+3] += damp[3*i:3*i+3]*qdi + fric[3*i:3*i+3]*np.tanh(qdi/0.05)   # passive losses
                swing_q[i] = qi.copy()
            elif mode == "flight":                            # tuck and reach, with a rate reference along the path
                tf = t - t_mode
                hip, ty = kin.HIP_OFFSETS[i], kin.THIGH_Y[i]
                q2 = KA.ik_leg(flight_foot(J, Jc, home[i], tf + 0.01, v_b), swing_q[i], hip, ty)
                swing_q[i] = KA.ik_leg(flight_foot(J, Jc, home[i], tf, v_b), swing_q[i], hip, ty)
                tau[3*i:3*i+3] = bias[3*i:3*i+3] + fl[i].swing_torque(qi, qdi, swing_q[i], (q2 - swing_q[i])/0.01)
            else:                                             # late foot while landing: reach down
                hf = home[i]; ft = np.array([hf[0] + C["k_land"]*land["vx"]*T_brake, hf[1], -(J.zl + 0.03)])
                swing_q[i] = KA.ik_leg(ft, swing_q[i], kin.HIP_OFFSETS[i], kin.THIGH_Y[i])
                tau[3*i:3*i+3] = bias[3*i:3*i+3] + legs[i].swing_torque(qi, qdi, swing_q[i])
        sim.step(tau)
        Fa = np.zeros((4, 3))
        for i in F: Fa[i] = F[i]
        for k, v in (("roll", rpy[0]), ("pitch", rpy[1]), ("yaw", rpy[2])):
            log[k].append(np.degrees(v))
        for k, v in (("t", t), ("phase", mode), ("z", sim.trunk_z()), ("com", com), ("vcom", vcom), ("F", Fa),
                     ("contact", cvec), ("qpos", sim.d.qpos.copy())):
            log[k].append(v)
        if sim.trunk_z() < 0.12 or abs(rpy[0]) > 1.2 or abs(rpy[1]) > 1.2:
            up = False; events.append(("fell", t))
            if verbose: print(f"  FELL at t={t:.2f}s")
            break
        if mode == "rest" and ji + 1 >= len(jumps) and t - t_mode > t_after:
            break
    L = {k: (np.array(v) if k != "phase" else np.array(v, dtype=object)) for k, v in log.items()}
    L["events"], L["jumps"] = events, jumps
    up = up and sim.trunk_z() > 0.12
    if verbose:
        for n, r in enumerate(jump_report(L)):
            print(f"  jump {n+1} ({body}) | apex +{100*r['apex']:.1f} cm (target {100*r['h']:.0f}) | takeoff "
                  f"vx {r['vx_to']:.2f} m/s | travel {r['travel']:.2f} m | tilt max {r['tilt']:.1f} deg")
        print(f"  {len(jumps)} jump(s) | {len(L['t'])/(time.time() - t0):.0f} steps/s | {'UP' if up else 'FELL'}")
    return L, up


def jump_report(L):
    """Per jump: CoM takeoff velocity, apex above the takeoff height, CoM travel from crouch start
    to the next start (or the end), peak |roll| or |pitch| over the same span."""
    t, ev, out = L["t"], L["events"], []
    idx = lambda tt: min(int(np.searchsorted(t, tt)), len(t) - 1)
    starts = [e[1] for e in ev if e[0] == "start"]
    for j, ts in enumerate(starts):
        t_next = starts[j + 1] if j + 1 < len(starts) else t[-1]
        pick = lambda name: next((e[1] for e in ev if e[0] == name and ts <= e[1] <= t_next), None)
        tto, ttd = pick("takeoff"), pick("touchdown")
        if tto is None:
            break
        i_s, i_to, i_n = idx(ts), idx(tto), idx(t_next); i_td = idx(ttd) if ttd else len(t) - 1
        out.append(dict(h=L["jumps"][j].h, vx=L["jumps"][j].vx, vx_to=L["vcom"][i_to, 0], vz_to=L["vcom"][i_to, 2],
                        apex=L["com"][i_to:i_td + 1, 2].max() - L["com"][i_to, 2],
                        travel=L["com"][i_n, 0] - L["com"][i_s, 0],
                        tilt=np.abs(np.c_[L["roll"], L["pitch"]][i_s:i_n + 1]).max()))
    return out


if __name__ == "__main__":
    env = os.environ.get
    legs = KA.learn_operators()
    sim = TorqueSim(control_dt=0.005)
    jumps = [Jump(h_apex=float(env("HEIGHT", "0.10")), vx_to=float(env("VX", "0")))
             for _ in range(int(env("COUNT", "1")))]
    run_jumps(sim, legs, jumps, body=env("BODY", "dhmpc"), exact_torques=env("EXACT", "1") != "0")
    sim.hold()
