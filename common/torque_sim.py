import os, sys, time
import numpy as np
import mujoco

LEGS = ["FR", "FL", "RR", "RL"]
_SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "scene_flat.xml")


class TorqueSim:
    def __init__(self, control_dt=0.005, sim_dt=0.001, view=None, key_callback=None):
        self.m = mujoco.MjModel.from_xml_path(os.path.abspath(_SCENE))
        self.m.opt.timestep = sim_dt
        self.d = mujoco.MjData(self.m)
        self.substeps = max(1, int(round(control_dt / sim_dt)))
        self.control_dt = self.substeps * sim_dt
        oid = lambda kind, n: mujoco.mj_name2id(self.m, kind, n)
        jn = [oid(mujoco.mjtObj.mjOBJ_JOINT, f"{l}_{j}_joint") for l in LEGS for j in ("hip", "thigh", "calf")]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jn])
        self.vadr = np.array([self.m.jnt_dofadr[j] for j in jn])
        self.aadr = np.array([oid(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{l}_{j}") for l in LEGS for j in ("hip", "thigh", "calf")])
        self.foot_gids = [oid(mujoco.mjtObj.mjOBJ_GEOM, l) for l in LEGS]
        self.trunk_bid = oid(mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.floor_gid = oid(mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.tau_lo = self.m.actuator_ctrlrange[self.aadr, 0].copy()
        self.tau_hi = self.m.actuator_ctrlrange[self.aadr, 1].copy()
        self.mass = float(np.sum(self.m.body_mass))
        self.q_home = self.m.key_qpos[0][self.qadr].copy()
        self._cf6 = np.zeros(6)
        self._viewer, self._wall0, self._nstep = None, None, 0
        self._speed = float(os.environ.get("VIEW_SPEED", "1.0"))
        if view is None:
            view = os.environ.get("VIEW", "0") != "0" or "--view" in sys.argv
        if view:
            if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
                print("[viewer] no DISPLAY found; running headless (unset VIEW to silence).")
            else:
                try:
                    import atexit, mujoco.viewer as mjv
                    mujoco.mj_forward(self.m, self.d)
                    self._viewer = mjv.launch_passive(self.m, self.d, key_callback=key_callback)
                    atexit.register(self.close)
                except BaseException as e:
                    print(f"[viewer] could not open a window ({e}); running headless.")
                    self._viewer = None

    def reset(self):
        self.d.qpos[:] = self.m.key_qpos[0]
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def leg_state(self):
        """q, qd (12,), ordered FR, FL, RR, RL x (hip, thigh, calf)."""
        return self.d.qpos[self.qadr].copy(), self.d.qvel[self.vadr].copy()

    def trunk_pose(self):
        """(position, quaternion) (7,) and (linear, angular velocity) (6,)."""
        return self.d.qpos[:7].copy(), self.d.qvel[:6].copy()

    def trunk_z(self):
        return float(self.d.qpos[2])

    def trunk_rpy(self):
        w, x, y, z = self.d.qpos[3:7]
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([roll, pitch, yaw])

    def bias_torque(self):
        """Gravity + Coriolis torque at the joints (mj_forward must be current)."""
        return self.d.qfrc_bias[self.vadr].copy()

    def foot_pos_world(self):
        return np.array([self.d.geom_xpos[g].copy() for g in self.foot_gids])

    def foot_grf(self):
        """Ground-reaction force on each foot, world frame (4, 3)."""
        F = np.zeros((4, 3))
        for c in range(self.d.ncon):
            con = self.d.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if self.floor_gid == g1 and g2 in self.foot_gids:
                idx, s = self.foot_gids.index(g2), +1.0
            elif self.floor_gid == g2 and g1 in self.foot_gids:
                idx, s = self.foot_gids.index(g1), -1.0
            else:
                continue
            mujoco.mj_contactForce(self.m, self.d, c, self._cf6)
            F[idx] += s * (con.frame.reshape(3, 3).T @ self._cf6[:3])
        return F

    def contacts(self):
        return (np.linalg.norm(self.foot_grf(), axis=1) > 1.0).astype(np.float32)

    def step(self, tau):
        """Apply joint torques (clipped to the limits) for one control tick."""
        tau = np.clip(np.asarray(tau, dtype=float), self.tau_lo, self.tau_hi)
        self.d.ctrl[self.aadr] = tau
        for _ in range(self.substeps):
            mujoco.mj_step(self.m, self.d)
        if self._viewer is not None:
            self._sync_view()
        return tau

    def _sync_view(self):
        try:
            if not self._viewer.is_running():
                self.close(); return
            self._viewer.sync()
        except Exception:
            return
        if self._wall0 is None:
            self._wall0 = time.time(); self._nstep = 0
        self._nstep += 1
        slack = self._wall0 + self._nstep * self.control_dt / max(1e-6, self._speed) - time.time()
        if slack > 0:
            time.sleep(slack)

    def hold(self):
        """Keep the viewer open until the user closes the window."""
        if self._viewer is None:
            return
        print("viewer open -- close the window to exit")
        try:
            while self._viewer.is_running():
                self._viewer.sync(); time.sleep(0.05)
        except Exception:
            pass
        self.close()

    def close(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
