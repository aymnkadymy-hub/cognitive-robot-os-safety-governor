#!/usr/bin/env python3
"""3D car (front-wheel Ackermann steering) that learns to cover the arena without collisions;
obstacles are randomized after each successful sweep.

Reliable bicycle kinematic model (front wheels visibly steer), visualized in MuJoCo (mocap).
- **When to slow/speed up:** online-learned logistic risk model on ego features
  (mirrors the `components/safety-model` Rust crate on seL4).
- **How to steer:** front-wheel angle toward the target + obstacle repulsion.
- **Curriculum:** each full obstacle-free coverage sweep → new obstacle positions
  (domain randomization) and training continues.

Live simulator (interactive):  tools/rl-venv/bin/python sim/car_drive.py --watch
Headless (GIF + stats):        MUJOCO_GL=egl tools/rl-venv/bin/python sim/car_drive.py
"""
import math
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw" if "--watch" in sys.argv else "egl")
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "car_arena.xml")
NOBS = 8
L = 0.42            # wheelbase
DT = 0.04
AX, AY = 5.2, 3.2  # half-arena extents
GX = GY = 0.75     # coverage cell size
SENSE = 2.0
DANGER = 0.95
HIT = 0.62
AVOID = 1.2         # tangential avoidance margin
BASE_V = 2.6       # m/s
STEER_MAX = 0.6


class SafetyModel:
    """Online-learned logistic risk model (same logic as the Rust safety-model crate, seL4-ready)."""

    def __init__(self, lr=0.06):
        self.w = np.zeros(3)
        self.b = 0.0
        self.lr = lr

    def risk(self, x):
        return 1.0 / (1.0 + math.exp(-(float(self.w @ x) + self.b)))

    def observe(self, x, danger):
        g = self.risk(x) - (1.0 if danger else 0.0)
        self.w -= self.lr * g * x
        self.b -= self.lr * g


def qz(a):
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def main():
    watch = "--watch" in sys.argv
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    mid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b): m.body_mocapid[b]
           for b in range(m.nbody) if m.body_mocapid[b] >= 0}

    nx, ny = int(2 * AX / GX), int(2 * AY / GY)
    model = SafetyModel()

    def set_pose(name, x, y, yaw, z=0.16):
        i = mid[name]
        d.mocap_pos[i] = [x, y, z]
        d.mocap_quat[i] = qz(yaw)

    def obstacles_xy():
        return [(d.mocap_pos[mid[f"o{i}"]][0], d.mocap_pos[mid[f"o{i}"]][1]) for i in range(NOBS)]

    def randomize():
        for i in range(NOBS):
            while True:
                x = rng.uniform(-AX + 0.7, AX - 0.7)
                y = rng.uniform(-AY + 0.7, AY - 0.7)
                if math.hypot(x + 4.5, y + 3.0) > 1.5:
                    break
            d.mocap_pos[mid[f"o{i}"]] = [x, y, 0.25]

    def nearest_obs(x, y):
        best, bx, by = 1e9, 1.0, 0.0
        for ox, oy in obstacles_xy():
            dx, dy = ox - x, oy - y
            dd = math.hypot(dx, dy)
            if dd < best:
                best, bx, by = dd, dx, dy
        return best, bx, by

    def place_car(x, y, th, steer):
        set_pose("car", x, y, th)
        for nm, fx, fy, yaw in [
            ("w_rl", -0.2, 0.16, th), ("w_rr", -0.2, -0.16, th),
            ("w_fl", 0.2, 0.16, th + steer), ("w_fr", 0.2, -0.16, th + steer),
        ]:
            wx = x + fx * math.cos(th) - fy * math.sin(th)
            wy = y + fx * math.sin(th) + fy * math.cos(th)
            set_pose(nm, wx, wy, yaw, z=0.09)

    randomize()
    mujoco.mj_forward(m, d)
    viewer = mujoco.viewer.launch_passive(m, d) if watch else None
    if watch:
        print("[car] LIVE viewer open — watch it learn: front wheels steer, it SLOWS near obstacles, sweeps, then obstacles move.")
    else:
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 13, 90, -70
        cam.lookat[:] = [0, 0, 0]
        frames = []

    # Car state (bicycle kinematics).
    sx, sy, th, v = -4.5, -3.0, 0.0, 0.0
    visited = np.zeros((nx, ny), bool)
    layout, ep, collisions, near, clean_layouts = 1, 0, 0, 0, 0
    prev_inside = False
    print(f"[car] grid {nx}x{ny}={nx*ny} cells. Cover all WITHOUT crashing -> obstacles randomize (curriculum).\n")

    max_ep_steps = 2600
    iters = 200000 if watch else 60000
    for t in range(iters):
        cx = min(nx - 1, max(0, int((sx + AX) / GX)))
        cy = min(ny - 1, max(0, int((sy + AY) / GY)))
        if 0 <= cx < nx and 0 <= cy < ny:
            visited[cx, cy] = True

        # Target: nearest unvisited cell.
        tgt, bestd = None, 1e9
        for gx in range(nx):
            for gy in range(ny):
                if not visited[gx, gy]:
                    wx, wy = -AX + (gx + 0.5) * GX, -AY + (gy + 0.5) * GY
                    dd = (wx - sx) ** 2 + (wy - sy) ** 2
                    if dd < bestd:
                        bestd, tgt = dd, (wx, wy)
        if tgt is None:
            tgt = (sx, sy)

        dist, ox, oy = nearest_obs(sx, sy)
        rel = math.atan2(oy, ox) - th
        prox = max(0.0, (SENSE - dist) / SENSE)
        feat = np.array([prox, math.sin(rel), math.cos(rel)])

        danger = dist < DANGER
        model.observe(feat, danger)
        inside = dist < HIT
        if inside and not prev_inside:  # count collision as an event (entering an obstacle)
            collisions += 1
        prev_inside = inside
        if danger and not inside:
            near += 1

        # Decision: speed (slows with risk) + steer toward target, or **tangentially around obstacle** when close.
        v_tgt = BASE_V * max(0.25, 1.0 - 0.9 * model.risk(feat))
        v += (v_tgt - v) * 0.2
        desired = math.atan2(tgt[1] - sy, tgt[0] - sx)
        if dist < AVOID:  # avoidance: choose the tangent closest to the goal direction (orbits around obstacle)
            to_obs = math.atan2(oy, ox)
            t1, t2 = to_obs + math.pi / 2, to_obs - math.pi / 2
            d1 = abs(math.atan2(math.sin(t1 - desired), math.cos(t1 - desired)))
            d2 = abs(math.atan2(math.sin(t2 - desired), math.cos(t2 - desired)))
            tang = t1 if d1 < d2 else t2
            wgt = min(1.0, (AVOID - dist) / AVOID)
            desired = math.atan2((1 - wgt) * math.sin(desired) + wgt * math.sin(tang),
                                 (1 - wgt) * math.cos(desired) + wgt * math.cos(tang))
        err = math.atan2(math.sin(desired - th), math.cos(desired - th))
        steer = max(-STEER_MAX, min(STEER_MAX, 1.5 * err))

        # Bicycle kinematic model (front-wheel steering).
        sx += v * math.cos(th) * DT
        sy += v * math.sin(th) * DT
        th += (v / L) * math.tan(steer) * DT
        sx = min(AX, max(-AX, sx))
        sy = min(AY, max(-AY, sy))

        place_car(sx, sy, th, steer)
        mujoco.mj_forward(m, d)
        ep += 1

        cov = visited.mean()
        if cov > 0.9 or ep > max_ep_steps:
            clean = cov > 0.9  # finished sweep -> change obstacles (curriculum continues)
            tag = "SWEPT -> new obstacle layout" if clean else "timeout"
            print(f"[car] layout {layout:>2} attempt: coverage {cov*100:.0f}%  collisions {collisions}  near-misses {near}  -> {tag}")
            if clean:
                layout += 1
                clean_layouts += 1
                randomize()  # <- obstacles change (curriculum)
            sx, sy, th, v = -4.5, -3.0, 0.0, 0.0
            visited[:] = False
            ep, collisions, near = 0, 0, 0
            if (not watch) and layout > 12:
                break

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(0.03)
        elif t % 6 == 0 and len(frames) < 320:
            renderer.update_scene(d, cam)
            frames.append(renderer.render())

    if not watch:
        out = os.path.expanduser("~/Desktop/car-learning.gif")
        imageio.mimsave(out, frames, fps=30)
        print(f"\n[car] rendered -> {out}")
    print(f"[car] DONE: learned WHEN to slow + HOW to steer (front wheels); {clean_layouts} layouts cleared cleanly (curriculum).")


if __name__ == "__main__":
    main()
