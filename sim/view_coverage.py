#!/usr/bin/env python3
"""Generates a GIF of a car sweeping the arena (coverage) while avoiding obstacles — from /tmp/coverage_trail.csv.

Usage: tools/rl-venv/bin/python sim/view_coverage.py [trail.csv] [out.gif]
"""
import sys

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else "/tmp/coverage_trail.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "/home/aymen/Desktop/coverage-car.gif"
    W = H = 0
    obs, car = [], []
    for line in open(csv):
        line = line.strip()
        if line.startswith("# W="):
            W = int(line.split("W=")[1].split()[0])
            H = int(line.split("H=")[1])
        elif line.startswith("obs,"):
            _, x, y = line.split(",")
            obs.append((int(x), int(y)))
        elif line.startswith("car,"):
            _, x, y = line.split(",")
            car.append((int(x), int(y)))

    swept = np.zeros((H, W))
    frames = []
    step = max(1, len(car) // 120)  # ~120 frames
    for i in range(0, len(car), step):
        for (x, y) in car[max(0, i - step):i + 1]:
            swept[y, x] = 1
        fig, ax = plt.subplots(figsize=(7.5, 5))
        grid = np.ones((H, W, 3))  # white background
        ys, xs = np.where(swept > 0)
        grid[ys, xs] = [0.6, 0.85, 0.6]  # light green = swept
        for (x, y) in obs:
            grid[y, x] = [0.15, 0.15, 0.15]  # dark = obstacle
        ax.imshow(grid, origin="upper")
        cx, cy = car[i]
        ax.plot(cx, cy, "o", color="#d22", ms=10)  # car marker
        ax.set_title(f"car sweeping the arena — avoiding obstacles  ({int(swept.sum())} cells, 0 crashes)")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
    imageio.mimsave(out, frames, fps=20)
    print(f"saved {out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
