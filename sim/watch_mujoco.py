#!/usr/bin/env python3
"""Live 3D window for watching the robot (run on your desktop — requires a display).

Opens an interactive MuJoCo window and runs the trained policy in real time
(you can rotate and zoom with the mouse).

Examples:
  # Watch the trained walker (after saving the model from train_walker.py):
  tools/rl-venv/bin/python sim/watch_mujoco.py Walker2d-v5 tools/models/walker.zip
  # Watch the humanoid:
  tools/rl-venv/bin/python sim/watch_mujoco.py Humanoid-v5 tools/models/humanoid.zip
  # Watch the body with random actions (no model — just to see the simulator itself):
  tools/rl-venv/bin/python sim/watch_mujoco.py Walker2d-v5
"""
import sys
import time

import gymnasium as gym
import mujoco.viewer


def main():
    env_id = sys.argv[1] if len(sys.argv) > 1 else "Walker2d-v5"
    model_path = sys.argv[2] if len(sys.argv) > 2 else None

    policy = None
    if model_path:
        from stable_baselines3 import PPO
        policy = PPO.load(model_path)
        print(f"loaded policy: {model_path}")
    else:
        print("no model -> random actions (watching the simulator body)")

    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset()
    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data

    print(f"opening live viewer for {env_id} — rotate/zoom with the mouse; Ctrl+C to quit")
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            if policy is not None:
                act, _ = policy.predict(obs, deterministic=True)
            else:
                act = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(act)
            viewer.sync()
            time.sleep(env.unwrapped.dt)
            if term or trunc:
                obs, _ = env.reset()


if __name__ == "__main__":
    main()
