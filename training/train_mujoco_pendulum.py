#!/usr/bin/env python3
"""Train a balancing policy on real MuJoCo physics (InvertedPendulum) and export it no_std for seL4.

Real MuJoCo physics (torque/friction/mass) instead of the hand-coded cartpole. Uses **DAgger**: imitate
a balanced controller, then run the MLP, collect the states it actually visits, label them with the
expert and retrain — so it learns to correct itself and overcomes the distribution-shift failure of naive
imitation on this unstable system. Result: MLP (4→32→1) exported no_std (seL4).

Output: components/mujoco-pendulum-policy/src/weights.rs  +  ~/Desktop/mujoco-pendulum.gif
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

np.random.seed(3)
IN, HID, OUT = 4, 32, 1
ENV = "InvertedPendulum-v5"
AMAX = 3.0


def controller(obs, k):
    x, th, xd, thd = obs
    a = k[0] * th + k[1] * thd + k[2] * x + k[3] * xd
    return np.clip([a], -AMAX, AMAX).astype(np.float32)


def relu(z):
    return np.maximum(z, 0.0)


def train_mlp(x_data, y_data, epochs=4000, lr=0.02, warm=None):
    rng = np.random.default_rng(0)
    if warm is None:
        w1 = (rng.standard_normal((HID, IN)) * 0.3).astype(np.float32)
        b1 = np.zeros(HID, np.float32)
        w2 = (rng.standard_normal((OUT, HID)) * 0.3).astype(np.float32)
        b2 = np.zeros(OUT, np.float32)
    else:
        w1, b1, w2, b2 = (v.copy() for v in warm)
    n = len(x_data)
    for _ in range(epochs):
        z1 = x_data @ w1.T + b1
        h = relu(z1)
        err = (h @ w2.T + b2) - y_data
        g = (2.0 / n) * err
        gw2, gb2 = g.T @ h, g.sum(0)
        gh = g @ w2
        gh[z1 <= 0] = 0.0
        gw1, gb1 = gh.T @ x_data, gh.sum(0)
        w2 -= lr * gw2
        b2 -= lr * gb2
        w1 -= lr * gw1
        b1 -= lr * gb1
    return w1, b1, w2, b2


def mlp_action(w, obs):
    w1, b1, w2, b2 = w
    h = relu(np.asarray(obs, np.float32) @ w1.T + b1)
    a = (h @ w2.T + b2)[0] * AMAX
    return np.clip([a], -AMAX, AMAX).astype(np.float32)


def episode_len(env, act_fn, seed):
    obs, _ = env.reset(seed=seed)
    for t in range(1000):
        obs, _, term, trunc, _ = env.step(act_fn(obs))
        if term or trunc:
            return t + 1
    return 1000


def main():
    env = gym.make(ENV)

    # 1) Tune an expert controller that balances in MuJoCo.
    base = np.array([20.0, 3.0, 1.0, 1.5])
    best_k, best_len = base, 0
    for scale in [0.8, 1.0, 1.25]:
        k = base * scale
        ln = np.mean([episode_len(env, lambda o, kk=k: controller(o, kk), s) for s in range(3)])
        if ln > best_len:
            best_len, best_k = ln, k
    print(f"expert tuned: k={best_k} mean_len={best_len:.0f}/1000")

    def expert(o):
        return controller(o, best_k)

    # 2) Initial data from the expert (with exploration noise).
    xs, ys = [], []
    for ep in range(20):
        obs, _ = env.reset(seed=100 + ep)
        for _ in range(1000):
            a = expert(obs)
            xs.append(np.asarray(obs, np.float32))
            ys.append(a / AMAX)
            noisy = np.clip(a + np.random.randn(1).astype(np.float32) * 0.6, -AMAX, AMAX)
            obs, _, term, trunc, _ = env.step(noisy)
            if term or trunc:
                break
    x_data = np.array(xs, np.float32)
    y_data = np.array(ys, np.float32)
    w = train_mlp(x_data, y_data)
    print(f"initial MLP: mean_len={np.mean([episode_len(env, lambda o: mlp_action(w, o), s) for s in range(3)]):.0f}")

    # 3) DAgger: run MLP, collect its states, label with expert, retrain.
    for it in range(6):
        nx = []
        for ep in range(6):
            obs, _ = env.reset(seed=500 + it * 10 + ep)
            for _ in range(1000):
                nx.append(np.asarray(obs, np.float32))
                obs, _, term, trunc, _ = env.step(mlp_action(w, obs))
                if term or trunc:
                    break
        nx = np.array(nx, np.float32)
        ny = np.array([expert(o) / AMAX for o in nx], np.float32)
        x_data = np.concatenate([x_data, nx])
        y_data = np.concatenate([y_data, ny])
        w = train_mlp(x_data, y_data, epochs=2500, warm=w)
        ml = np.mean([episode_len(env, lambda o: mlp_action(w, o), s) for s in range(5)])
        print(f"DAgger it{it + 1}: data={len(x_data)} MLP mean_len={ml:.0f}/1000")
        if ml >= 990:
            break

    final = np.mean([episode_len(env, lambda o: mlp_action(w, o), s) for s in range(10)])
    print(f"FINAL MLP balance in MuJoCo: mean_len={final:.0f}/1000")

    render_gif(lambda o: mlp_action(w, o))
    w1, b1, w2, b2 = w
    refs_in = x_data[:3]
    refs_out = (relu(refs_in @ w1.T + b1) @ w2.T + b2).astype(np.float32)
    emit(w1, b1, w2, b2, refs_in, refs_out)


def render_gif(act_fn):
    try:
        import imageio.v2 as imageio
        env = gym.make(ENV, render_mode="rgb_array")
        obs, _ = env.reset(seed=7)
        frames = []
        for _ in range(250):
            frames.append(env.render())
            obs, _, term, trunc, _ = env.step(act_fn(obs))
            if term or trunc:
                obs, _ = env.reset()
        out = os.path.expanduser("~/Desktop/mujoco-pendulum.gif")
        imageio.mimsave(out, frames, fps=30)
        print(f"rendered 3D robot: {out}")
    except Exception as e:  # noqa: BLE001
        print(f"(render skipped: {e})", file=sys.stderr)


def fa(v):
    return f"{np.float32(v):.8e}"


def mat(name, m, rows, cols):
    body = "\n".join("    [" + ", ".join(fa(v) for v in r) + "]," for r in m)
    return f"pub const {name}: [[f32; {cols}]; {rows}] = [\n{body}\n];\n"


def vec(name, v):
    return f"pub const {name}: [f32; {len(v)}] = [" + ", ".join(fa(x) for x in v) + "];\n"


def emit(w1, b1, w2, b2, refs_in, refs_out):
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "components", "mujoco-pendulum-policy", "src", "weights.rs"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("#![allow(clippy::excessive_precision)]\n\n")
        f.write("// Auto-generated by training/train_mujoco_pendulum.py — do not edit manually.\n")
        f.write("// Balancing policy trained on real MuJoCo physics (DAgger), runs no_std on seL4.\n\n")
        f.write(f"pub const IN: usize = {IN};\n")
        f.write(f"pub const HID: usize = {HID};\n")
        f.write(f"pub const OUT: usize = {OUT};\n\n")
        f.write(mat("W1", w1, HID, IN))
        f.write(vec("B1", b1))
        f.write("\n")
        f.write(mat("W2", w2, OUT, HID))
        f.write(vec("B2", b2))
        f.write("\n// Reference states (SIL).\n")
        f.write(mat("REF_IN", refs_in, 3, IN))
        f.write(mat("REF_OUT", refs_out, 3, OUT))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
