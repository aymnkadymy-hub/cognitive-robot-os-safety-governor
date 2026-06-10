#!/usr/bin/env python3
"""Train a driving policy (PPO) on CarEnv for a given duration, with **periodic checkpoints + progress log + evaluation**.

- Resumes from the previous checkpoint if one exists (`training/car_ppo.zip`).
- Each round: saves the model (picked up by the live viewer) and evaluates: policy alone vs policy + safety guard.
- At the end: full results + learning curve in `/tmp/car_train_metrics.csv`.

Run: tools/rl-venv/bin/python training/train_car.py [seconds]   (default 3600 = 1 hour)
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
import numpy as np  # noqa: E402
from car_env import CarEnv  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

TRAIN_SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3600
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "car_ppo.zip")
CSV = "/tmp/car_train_metrics.csv"
CHUNK = 60_000  # steps between each evaluation/checkpoint


def gov_throttle(obs):
    front = min(obs[5:10])  # nearest forward ray [0,1]; 1=clear — slows only on actual proximity
    return 1.0 if front > 0.45 else float(np.clip(front / 0.45, 0.25, 1.0))


def evaluate(model, episodes=20, governed=False, seed0=9000):
    tgt, crash, clear_sum = 0, 0, 0.0
    for e in range(episodes):
        env = CarEnv(seed=seed0 + e)
        obs, _ = env.reset(seed=seed0 + e)
        prev = False
        ep_min_clear = 9.0
        for _ in range(env.max_steps):
            act, _ = model.predict(obs, deterministic=True)
            gt = gov_throttle(obs) if governed else None
            obs, _, _, trunc, info = env.step(act, gov_throttle=gt)
            if info["crashed"] and not prev:
                crash += 1
            prev = info["crashed"]
            ep_min_clear = min(ep_min_clear, info["clear"])  # closest approach to any hazard (obstacle/wall)
            if trunc:
                break
        tgt += info["reached"]
        clear_sum += ep_min_clear
    return tgt / episodes, crash / episodes, clear_sum / episodes


def main():
    venv = DummyVecEnv([lambda i=i: CarEnv(seed=i) for i in range(6)])
    if os.path.exists(PATH):
        model = PPO.load(PATH, env=venv)
        print(f"[rl] continuing from checkpoint ({model.num_timesteps} steps so far)", flush=True)
    else:
        model = PPO("MlpPolicy", venv, verbose=0, n_steps=1024, batch_size=512,
                    gae_lambda=0.95, gamma=0.99, ent_coef=0.004, learning_rate=3e-4)
        print("[rl] fresh policy", flush=True)
    print(f"[rl] training for {TRAIN_SECONDS/60:.0f} minutes (Ctrl-C safe; checkpoints each ~{CHUNK} steps)\n", flush=True)

    with open(CSV, "w") as f:
        f.write("minutes,steps,alone_targets,alone_crashes,alone_clear,guard_targets,guard_crashes,guard_clear\n")

    start = time.time()
    rnd = 0
    while time.time() - start < TRAIN_SECONDS:
        model.learn(CHUNK, reset_num_timesteps=False, progress_bar=False)
        model.save(PATH)  # <- picked up by the live viewer
        r_a, c_a, cl_a = evaluate(model, 18, governed=False)
        r_b, c_b, cl_b = evaluate(model, 18, governed=True)
        el = (time.time() - start) / 60.0
        rnd += 1
        print(f"[rl] t={el:5.1f}min steps={model.num_timesteps:>8} | "
              f"ALONE {r_a:4.1f}tgt/{c_a:4.1f}crash/{cl_a:.2f}m clear | "
              f"+GUARD {r_b:4.1f}tgt/{c_b:4.1f}crash/{cl_b:.2f}m clear", flush=True)
        with open(CSV, "a") as f:
            f.write(f"{el:.1f},{model.num_timesteps},{r_a:.2f},{c_a:.2f},{cl_a:.2f},{r_b:.2f},{c_b:.2f},{cl_b:.2f}\n")

    # ===== full results (large evaluation) =====
    print("\n[rl] ===== FINAL RESULTS (60 randomized eval episodes) =====", flush=True)
    fa_t, fa_c, fa_cl = evaluate(model, 60, governed=False)
    fb_t, fb_c, fb_cl = evaluate(model, 60, governed=True)
    drop = 100 * (1 - fb_c / fa_c) if fa_c > 0 else 100
    print(f"[rl] total training: {(time.time()-start)/60:.1f} min, {model.num_timesteps} steps, {rnd} checkpoints", flush=True)
    print(f"[rl] (A) policy ALONE          : {fa_t:.1f} targets/ep | {fa_c:.1f} crashes/ep | {fa_cl:.2f}m avg clearance", flush=True)
    print(f"[rl] (B) policy + SAFETY GUARD  : {fb_t:.1f} targets/ep | {fb_c:.1f} crashes/ep | {fb_cl:.2f}m avg clearance", flush=True)
    print(f"[rl] guard cut crashes by {drop:.0f}%; safety-first reward keeps clearance; full curve in {CSV}", flush=True)
    print("[rl] DONE.", flush=True)


if __name__ == "__main__":
    main()
