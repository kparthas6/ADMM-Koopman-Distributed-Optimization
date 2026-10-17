import os, sys
root, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(root, "common")); sys.path.insert(0, os.path.join(root, "controller"))
os.chdir(os.path.join(root, "controller"))
import numpy as np
import koopman_admm as KA, jumps as JP
from torque_sim import TorqueSim

legs, res = KA.learn_operators(), {}
for name, kw in (("trot", dict(mode="trot", vx=0.3, seconds=2.0)),
                 ("long_stride_turn", dict(mode="trot", vx=0.4, gait_T=0.30, wz=0.2, seconds=2.0)),
                 ("trot_dist", dict(mode="trot", vx=0.3, seconds=1.0, body="dist")),
                 ("stand_push", dict(mode="stand", seconds=1.5, push=40.0))):
    res[name] = KA.run(TorqueSim(control_dt=0.005), legs, verbose=False, **kw)[0]["qpos"]
res["jump"] = JP.run_jumps(TorqueSim(control_dt=0.005), legs, [JP.Jump(h_apex=0.10, vx_to=0.2)], verbose=False)[0]["qpos"]
np.savez(out, **res)
