#!/usr/bin/env python3
"""Interactive live window for Steve — watch him and assign tasks via keyboard (run on your display).

Follows training in real time: pass a model directory and it reloads the latest checkpoint every few seconds.

Assign heading tasks live:
  Up / W = forward      Down / S = backward      Left / A = left      Right / D = right      R = reset

Usage:
  # Follow training live (reloads latest checkpoint):
  tools/rl-venv/bin/python sim/watch_steve_live.py tools/models
  # Specific model:
  tools/rl-venv/bin/python sim/watch_steve_live.py tools/models/steve.zip
  # No model (just view the character):
  tools/rl-venv/bin/python sim/watch_steve_live.py
"""
import glob
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
from minecraft_env import WORLD, SteveEnv  # noqa: E402

HEADINGS = {
    265: 0.0, 87: 0.0, 264: np.pi, 83: np.pi,
    263: np.pi / 2, 65: np.pi / 2, 262: -np.pi / 2, 68: -np.pi / 2,
}


def latest_ckpt(path):
    """Latest training checkpoint in a directory (or the path itself if it is a file)."""
    if path and os.path.isdir(path):
        files = glob.glob(os.path.join(path, "steve_ckpt_*.zip")) + glob.glob(os.path.join(path, "steve.zip"))
        return max(files, key=os.path.getmtime) if files else None
    return path if path and os.path.exists(path) else None


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    env = SteveEnv(xml_path=WORLD, render_mode="rgb_array")
    env.reset()
    env.set_task(0.0)

    from stable_baselines3 import PPO
    policy, loaded_path = None, None
    cur = latest_ckpt(arg)
    if cur:
        policy = PPO.load(cur)
        loaded_path = cur
        print(f"loaded: {os.path.basename(cur)}")
    else:
        print("no model yet -> Steve stands; will auto-load once training saves a checkpoint")

    st = {"obs": env._obs(), "reset": False}

    def key_cb(keycode):
        if keycode in HEADINGS:
            env.set_task(HEADINGS[keycode])
            print(f"TASK: walk heading={np.degrees(np.arctan2(*env.desired[::-1])):.0f}deg")
        elif keycode in (82, 32):
            st["reset"] = True

    print("live window open — arrows/WASD give tasks, R=reset, mouse rotates, Ctrl+C quits")
    i = 0
    with mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data,
                                      key_callback=key_cb) as viewer:
        while viewer.is_running():
            i += 1
            if i % 200 == 0 and arg and os.path.isdir(arg):  # follow training: load latest
                newest = latest_ckpt(arg)
                if newest and newest != loaded_path:
                    try:
                        policy = PPO.load(newest)
                        loaded_path = newest
                        print(f"reloaded improved model: {os.path.basename(newest)}")
                    except Exception:  # noqa: BLE001
                        pass
            if st["reset"]:
                env.reset()
                env.set_task(0.0)
                st["obs"] = env._obs()
                st["reset"] = False
            if policy is not None:
                act, _ = policy.predict(st["obs"], deterministic=True)
            else:
                act = np.zeros(env.action_space.shape, np.float32)
            st["obs"], _, term, _, _ = env.step(act)
            viewer.sync()
            time.sleep(env.dt)
            if term:
                st["reset"] = True


if __name__ == "__main__":
    main()
