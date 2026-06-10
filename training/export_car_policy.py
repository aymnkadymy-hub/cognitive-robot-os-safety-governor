#!/usr/bin/env python3
"""Export the driving policy (PPO MLP) to Rust `no_std` weights (Bridge B: deploying a learned brain on seL4-ready).

Policy: obs(11) → Linear(11,64)→tanh → Linear(64,64)→tanh → action_net(64,2) = [steer, throttle].
Writes `components/nav-policy/src/weights.rs` + verification samples (SIL-match).
"""
import os

import numpy as np
from stable_baselines3 import PPO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = PPO.load(os.path.join(ROOT, "training", "car_ppo.zip"))
sd = {n: p.detach().numpy() for n, p in m.policy.named_parameters()}

w0, b0 = sd["mlp_extractor.policy_net.0.weight"], sd["mlp_extractor.policy_net.0.bias"]
w2, b2 = sd["mlp_extractor.policy_net.2.weight"], sd["mlp_extractor.policy_net.2.bias"]
wa, ba = sd["action_net.weight"], sd["action_net.bias"]


def mat(name, a, rows, cols):
    rowsrc = ",\n    ".join("[" + ", ".join(f"{v:.6e}" for v in a[r]) + "]" for r in range(rows))
    return f"pub const {name}: [[f32; {cols}]; {rows}] = [\n    {rowsrc},\n];\n"


def vec(name, a):
    return f"pub const {name}: [f32; {len(a)}] = [{', '.join(f'{v:.6e}' for v in a)}];\n"


# Verification samples: Python outputs for fixed random inputs.
rng = np.random.default_rng(0)
samples = []
for _ in range(6):
    obs = rng.uniform(-1, 1, 11).astype(np.float32)
    act, _ = m.predict(obs, deterministic=True)
    samples.append((obs, np.asarray(act, np.float32)))

out = os.path.join(ROOT, "components", "nav-policy", "src", "weights.rs")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    f.write("//! Driving policy weights exported from PPO (auto-generated — do not edit manually).\n")
    f.write("#![allow(clippy::excessive_precision, clippy::large_const_arrays, dead_code)]\n\n")
    f.write(mat("W0", w0, 64, 11))
    f.write(vec("B0", b0))
    f.write(mat("W2", w2, 64, 64))
    f.write(vec("B2", b2))
    f.write(mat("WA", wa, 2, 64))
    f.write(vec("BA", ba))
    f.write(f"\npub const SAMPLES: [([f32; 11], [f32; 2]); {len(samples)}] = [\n")
    for obs, act in samples:
        o = ", ".join(f"{v:.6e}" for v in obs)
        a = ", ".join(f"{v:.6e}" for v in act)
        f.write(f"    ([{o}], [{a}]),\n")
    f.write("];\n")
print(f"wrote {out} (policy 11->64->64->2, {len(samples)} match samples)")
