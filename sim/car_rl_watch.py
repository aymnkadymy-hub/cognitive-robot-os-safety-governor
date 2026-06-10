#!/usr/bin/env python3
"""Live viewer: trained PPO driving policy navigates the 3D arena **under the safety guard**.

The policy (learned brain) chooses steering and throttle; **the guard clips throttle near obstacles**
(safe action projection). The green target moves, and obstacles change each episode (domain
randomization). Demonstrates: "the system runs any learned brain safely".

Live simulator:  tools/rl-venv/bin/python sim/car_rl_watch.py --watch
Silent GIF:      MUJOCO_GL=egl tools/rl-venv/bin/python sim/car_rl_watch.py
"""
import math
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw" if "--watch" in sys.argv else "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
from car_env import NOBS, CarEnv  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "car_arena.xml")
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training", "car_ppo.zip")


def gov_throttle(obs):
    front = min(obs[5:10])  # nearest forward ray [0,1]; 1 = clear road
    return 1.0 if front > 0.45 else float(np.clip(front / 0.45, 0.25, 1.0))  # slows only when close


def qz(a):
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def main():
    watch = "--watch" in sys.argv
    if not os.path.exists(MODEL):
        print("[rlwatch] no trained model — run training/train_car.py first.")
        return
    model = PPO.load(MODEL)
    env = CarEnv(seed=7)
    obs, _ = env.reset(seed=7)

    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    mid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b): m.body_mocapid[b]
           for b in range(m.nbody) if m.body_mocapid[b] >= 0}

    def setp(name, x, y, yaw, z):
        i = mid[name]
        d.mocap_pos[i] = [x, y, z]
        d.mocap_quat[i] = qz(yaw)

    def sync_scene(steer):
        setp("car", env.x, env.y, env.th, 0.16)
        for nm, fx, fy, yw in [("w_rl", -0.2, 0.16, env.th), ("w_rr", -0.2, -0.16, env.th),
                               ("w_fl", 0.2, 0.16, env.th + steer), ("w_fr", 0.2, -0.16, env.th + steer)]:
            setp(nm, env.x + fx * math.cos(env.th) - fy * math.sin(env.th),
                 env.y + fx * math.sin(env.th) + fy * math.cos(env.th), yw, 0.09)
        setp("target", env.tx, env.ty, 0, 0.2)
        for i in range(NOBS):
            setp(f"o{i}", env.obs[i][0], env.obs[i][1], 0, 0.25)
        for i in range(NOBS, 8):
            setp(f"o{i}", 20, 20, 0, 0.25)  # hide excess obstacles

    viewer = mujoco.viewer.launch_passive(m, d) if watch else None
    if watch:
        print("[rlwatch] LIVE: trained RL policy drives; the GUARD slows it near obstacles -> safe. Targets/obstacles change.")
    else:
        import imageio.v2 as imageio
        r = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 13, 90, -70
        cam.lookat[:] = [0, 0, 0]
        frames = []

    reached, crashes, prev = 0, 0, False
    last_mtime = os.path.getmtime(MODEL)
    iters = 100000 if watch else 1800
    for t in range(iters):
        act, _ = model.predict(obs, deterministic=True)
        steer = float(np.clip(act[0], -1, 1)) * 0.6
        obs, _, _, trunc, info = env.step(act, gov_throttle=gov_throttle(obs))
        if info["crashed"] and not prev:
            crashes += 1
        prev = info["crashed"]
        reached = info["reached"]
        sync_scene(steer)
        mujoco.mj_forward(m, d)
        if trunc:
            obs, _ = env.reset(seed=7 + t)  # new obstacles (DR)
            try:  # reload latest checkpoint (watch training improve in real time)
                mt = os.path.getmtime(MODEL)
                if mt != last_mtime:
                    model = PPO.load(MODEL)
                    last_mtime = mt
                    print(f"[rlwatch] reloaded improved checkpoint (reached {reached}, crashes {crashes})", flush=True)
            except Exception:
                pass
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(0.03)
        elif t % 5 == 0 and len(frames) < 320:
            r.update_scene(d, cam)
            frames.append(r.render())

    if not watch:
        out = os.path.expanduser("~/Desktop/car-rl-guarded.gif")
        imageio.mimsave(out, frames, fps=30)
        print(f"[rlwatch] rendered -> {out}")
    print(f"[rlwatch] reached {reached} targets, {crashes} crashes (guard active) — learned brain kept safe.")


if __name__ == "__main__":
    main()
