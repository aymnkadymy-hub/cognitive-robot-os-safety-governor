#!/usr/bin/env python3
"""Train the blocky Steve character to walk (PPO, heading-conditioned) in MuJoCo.

Honest note: this is a custom model from scratch — training is harder than standard
benchmark environments and may require many steps to walk stably on CPU.
Saves the model for live viewing and live task assignment.
Usage: train_steve.py [steps]
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from minecraft_env import SteveEnv  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1_500_000


def main():
    venv = make_vec_env(SteveEnv, n_envs=8)
    model = PPO(
        "MlpPolicy", venv, verbose=1, n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    mdir = os.path.join(os.path.dirname(__file__), "..", "tools", "models")
    os.makedirs(mdir, exist_ok=True)
    # Save periodic checkpoints to observe improvement live in the viewer (every ~120k steps).
    ckpt = CheckpointCallback(save_freq=15000, save_path=mdir, name_prefix="steve_ckpt")
    print(f"training Minecraft Steve to walk for {STEPS} steps...")
    model.learn(total_timesteps=STEPS, progress_bar=False, callback=ckpt)
    model.save(os.path.join(mdir, "steve"))
    print("saved model -> tools/models/steve.zip")

    ev = SteveEnv()
    rewards = []
    for ep in range(5):
        obs, _ = ev.reset(seed=ep)
        ev.set_task(0.0)  # walk forward
        obs = ev._obs()
        tot, done = 0.0, False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = ev.step(a)
            tot += r
            done = term or trunc
        rewards.append(tot)
    print(f"EVAL mean_reward={np.mean(rewards):.0f} +/- {np.std(rewards):.0f}")
    render_gif(model)


def render_gif(model):
    try:
        import imageio.v2 as imageio
        env = SteveEnv(render_mode="rgb_array")
        obs, _ = env.reset(seed=0)
        env.set_task(0.0)
        obs = env._obs()
        frames = []
        for _ in range(400):
            frames.append(env.render())
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                obs, _ = env.reset()
                env.set_task(0.0)
                obs = env._obs()
        out = os.path.expanduser("~/Desktop/minecraft-steve-walk.gif")
        imageio.mimsave(out, frames, fps=30)
        print(f"rendered walking Steve: {out}")
    except Exception as e:  # noqa: BLE001
        print(f"(render skipped: {e})", file=sys.stderr)


if __name__ == "__main__":
    main()
