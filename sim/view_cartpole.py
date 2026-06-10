#!/usr/bin/env python3
"""Graphical viewer for the cart-pole — animates the system trajectory (controlled by the same seL4 crates).

Reads a CSV from sil-cartpole and produces an animated GIF: the cart moves, the pole tilts,
and turns red when the safety layer intervenes (action clamped). The target is marked with a line.

Usage:  python3 sim/view_cartpole.py [trajectory.csv] [out.gif]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless — save GIF
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

POLE_LEN = 1.0
CART_W, CART_H = 0.3, 0.18


def load(path):
    target_x = 0.0
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# target_x="):
                target_x = float(line.split("=")[1])
            elif line and not line.startswith(("x,", "#")):
                x, th, m, c = line.split(",")
                rows.append((float(x), float(th), float(m), int(c)))
    return target_x, rows


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "sil-cartpole", "cartpole_trajectory.csv")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.expanduser("~/Desktop"), "cartpole.gif")
    target_x, rows = load(csv)
    xs = np.array([r[0] for r in rows])
    ths = np.array([r[1] for r in rows])
    clamped = np.array([r[3] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 3.2))
    lo, hi = min(-2.5, xs.min() - 0.5), max(2.5, xs.max() + 0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.4, 1.4)
    ax.set_aspect("equal")
    ax.axhline(0, color="#999", lw=1)
    ax.axvline(target_x, color="#2a8", ls="--", lw=1, label=f"target x={target_x:g}")
    ax.set_title("Cognitive Robot OS — cartpole (same brain+safety as seL4)")
    ax.legend(loc="upper right", fontsize=8)

    cart = plt.Rectangle((0, 0), CART_W, CART_H, fc="#357")
    ax.add_patch(cart)
    (pole,) = ax.plot([], [], lw=4, color="#2a8")
    txt = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=9)

    def frame(i):
        x, th = xs[i], ths[i]
        cart.set_xy((x - CART_W / 2, 0))
        px, py = x + POLE_LEN * np.sin(th), CART_H + POLE_LEN * np.cos(th)
        pole.set_data([x, px], [CART_H, py])
        # red when safety intervention active.
        pole.set_color("#d33" if clamped[i] else "#2a8")
        txt.set_text(f"step {i}  x={x:+.2f}  theta={th:+.2f}"
                     + ("   [SAFETY CLAMP]" if clamped[i] else ""))
        return cart, pole, txt

    step = max(1, len(rows) // 200)  # limit frame count
    frames = range(0, len(rows), step)
    anim = animation.FuncAnimation(fig, frame, frames=frames, interval=40, blit=True)
    anim.save(out, writer=animation.PillowWriter(fps=25))
    print(f"saved animation: {out}  ({len(list(frames))} frames)")


if __name__ == "__main__":
    main()
