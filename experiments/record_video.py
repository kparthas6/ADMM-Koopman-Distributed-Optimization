"""Render a scenario run to mp4 (real time, 40 fps), with the body spheres and the planned trunk path drawn in.

    MUJOCO_GL=egl python record_video.py headon rows+steer          # -> ../results/videos/headon_rows_steer.mp4
    xvfb-run -a env MUJOCO_GL=glfw python record_video.py wall rows   # headless without EGL

Needs ffmpeg on the PATH, an offscreen OpenGL context (MUJOCO_GL=egl or glfw under xvfb) and Pillow."""
import os, sys, subprocess, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mujoco
from PIL import Image, ImageDraw
from scenarios import SCENARIOS, RESULTS, run_scenario      # first: puts controller/ and common/ on the path
import koopman_admm as KA
from torque_sim import TorqueSim

LABEL = {"none": "no keep-out", "rows": "keep-out rows only", "rows+steer": "keep-out rows + steering"}
W, H, EVERY, FPS = 640, 400, 5, 40                          # one frame per 5 ticks of 5 ms: real time


def add_sphere(scn, pos, r, rgba):
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), np.asarray(pos, float),
                            np.eye(3).ravel(), np.asarray(rgba, np.float32))
        scn.ngeom += 1


def record(name, variant, path, camera=None):
    sc, bod = SCENARIOS[name], np.asarray(KA.KEEPOUT_CFG["bodies"], float)
    sim = TorqueSim(control_dt=0.005, obstacles=sc["obst"])
    sim.m.vis.global_.offwidth, sim.m.vis.global_.offheight = W, H
    ren = mujoco.Renderer(sim.m, H, W)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cfg = {**dict(lookat=[1.2, 0.0, 0.0], distance=3.3, azimuth=90.0, elevation=-55.0, track=False), **(camera or {})}
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = cfg["lookat"], cfg["distance"], cfg["azimuth"], cfg["elevation"]
    plan = {"xp": None}
    solve = KA.DH.distributed_horizon
    def grab(*a, **k):                                       # keep the latest predicted trunk path for the overlay
        r = solve(*a, **k); plan["xp"] = k["state"]["h"].get("xp"); return r
    KA.DH.distributed_horizon = grab
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                           "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    n, step = [0], sim.step
    def stepped(tau):
        out = step(tau); n[0] += 1
        if n[0] > 200 and n[0] % EVERY == 0:                  # skip run()'s 1 s settle
            q = sim.d.qpos; pts = q[None, :3] + bod[:, :3] @ KA.quat2R(q[3:7]).T
            if cfg["track"]:
                cam.lookat[:] = [q[0], q[1], 0.15]
            ren.update_scene(sim.d, camera=cam); scn = ren.scene
            scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0; scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            if variant != "none":
                for p, b in zip(pts, bod):
                    add_sphere(scn, p, b[3], [0.23, 0.53, 0.90, 0.22])
                for xk in (plan["xp"] if plan["xp"] is not None else []):
                    add_sphere(scn, xk[:3], 0.012, [0.10, 0.36, 0.75, 0.9])
            img = Image.fromarray(ren.render()); dr = ImageDraw.Draw(img)
            dr.rectangle([0, 0, W, 26], fill=(252, 252, 251))
            dr.text((8, 7), f"{LABEL[variant]}   t {(n[0] - 200)*0.005:4.2f} s", fill=(11, 11, 11))
            ff.stdin.write(np.asarray(img, np.uint8).tobytes())
        return out
    sim.step = stepped
    try:
        summary, _ = run_scenario(name, variant, sim=sim)
    finally:
        KA.DH.distributed_horizon = solve
        ff.stdin.close(); ff.wait()
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=list(SCENARIOS))
    ap.add_argument("variant", choices=["none", "rows", "rows+steer"])
    ap.add_argument("--track", action="store_true", help="follow the robot instead of a fixed view of the course")
    a = ap.parse_args()
    out = os.path.join(RESULTS, "videos", f"{a.scenario}_{a.variant.replace('+', '_')}.mp4")
    s = record(a.scenario, a.variant, out, camera=dict(track=a.track, distance=2.8) if a.track else None)
    print(out, "|", {k: s[k] for k in ("upright", "min_clearance", "contact_ticks", "max_tilt")})
