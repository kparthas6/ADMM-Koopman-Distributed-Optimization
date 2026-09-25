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


def run_jumps(sim, legs, jumps, body="dhmpc", exact_torques=True, t_after=0.8, verbose=True):
    """Closed loop for a list of Jump objects, executed back to back from a stand. body 'dhmpc'
    (horizon, KA.DHMPC_CFG) | 'dist' (instantaneous); exact_torques: stance torques by

