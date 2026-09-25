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
    solved by qpax in float32 (the precision the operators were fit at). Stage weights: joint
    angle 1 (wterm on the last step), joint rate wd, torque wr."""
    def __init__(self, A, B, hip, ty, N=6, tau_lim=(23.7, 23.7, 35.55), wd=0.01, wterm=8.0, wr=5e-5):
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
        wq = np.ones(N); wq[-1] = wterm
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

