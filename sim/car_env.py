#!/usr/bin/env python3
"""Gymnasium environment: front-wheel-steered car that learns to reach goals
and avoid **solid** obstacles + **arena walls**.

Key improvements:
- **Lidar-style ray sensing (simplified):** 7 rays detect obstacle and **wall** distances
  in a forward arc → reveals gaps so the car can go around rather than oscillate
  (addresses "doesn't know which way to go around an obstacle").
- **Walls as obstacles:** safety margin from walls + hard stop on contact (no edge scratching).
- Hard stop + reverse + action smoothing (no jitter) + safety-first (quadratic penalty).

Observation (11): [goal distance, sin/cos of goal bearing, speed] + 7 rays.
Action: [steer∈[-1,1], throttle∈[-1,1] (negative = reverse)].
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

AX, AY = 5.0, 3.2
NOBS = 5
L = 0.42
DT = 0.06
BLOCK = 0.62       # obstacle collision radius
REACH = 0.5
RAY_MAX = 4.0      # ray range (for normalisation)
RAY_ANGLES = [-90, -55, -25, 0, 25, 55, 90]  # degrees relative to car heading
SAFE_MARGIN = 1.3  # safety penalty zone (quadratic penalty)
WALL_OFF = 0.62    # offset that puts the wall on the same clearance scale as obstacles


class CarEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=0):
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, (4 + len(RAY_ANGLES),), np.float32)
        # [steer∈[-1,1], throttle∈[-1,1]] — negative throttle = reverse
        self.action_space = spaces.Box(np.array([-1, -1], np.float32), np.array([1, 1], np.float32))
        self.rng = np.random.default_rng(seed)
        self.max_steps = 600

    def _obstacles(self):
        obs = []
        for _ in range(NOBS):
            while True:
                p = (self.rng.uniform(-AX + 0.9, AX - 0.9), self.rng.uniform(-AY + 0.9, AY - 0.9))
                if (math.hypot(p[0] - self.x, p[1] - self.y) > 1.2
                        and math.hypot(p[0] - self.tx, p[1] - self.ty) > 1.1
                        and all(math.hypot(p[0] - q[0], p[1] - q[1]) > 2.0 for q in obs)):
                    break
            obs.append(p)
        return obs

    def _new_target(self):
        for _ in range(60):
            self.tx = self.rng.uniform(-AX + 0.8, AX - 0.8)  # keep away from walls too
            self.ty = self.rng.uniform(-AY + 0.8, AY - 0.8)
            if not hasattr(self, "obs") or all(math.hypot(self.tx - o[0], self.ty - o[1]) > 1.1 for o in self.obs):
                return

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x, self.y, self.th, self.v = -4.2, -2.6, 0.0, 0.0
        self.tx, self.ty = 3.0, 2.0
        self.obs = self._obstacles()
        self._new_target()
        self.steps = 0
        self.reached = 0
        self.prev_throttle = 0.0
        self.prev_steer = 0.0
        self.prev_d = math.hypot(self.tx - self.x, self.ty - self.y)
        return self._obs(), {}

    def _wall_dist(self, px, py):
        return min(AX - px, AX + px, AY - py, AY + py)  # distance to nearest wall

    def _clearance_at(self, px, py):
        """Clearance to nearest hazard = min(distance-to-nearest-obstacle, wall-distance+offset)
        — treats walls as obstacles."""
        do = min((math.hypot(ox - px, oy - py) for ox, oy in self.obs), default=1e9)
        return min(do, self._wall_dist(px, py) + WALL_OFF)

    def _raycast(self):
        """7 ray distances (normalised) to the nearest obstacle or wall — vision for going around obstacles."""
        out = []
        for adeg in RAY_ANGLES:
            ang = self.th + math.radians(adeg)
            dx, dy = math.cos(ang), math.sin(ang)
            best = RAY_MAX
            for ox, oy in self.obs:  # ray-circle intersection
                fx, fy = self.x - ox, self.y - oy
                b = 2 * (fx * dx + fy * dy)
                c = fx * fx + fy * fy - BLOCK * BLOCK
                disc = b * b - 4 * c
                if disc >= 0:
                    t = (-b - math.sqrt(disc)) * 0.5
                    if 0 < t < best:
                        best = t
            for t in (  # ray-wall intersection
                (AX - self.x) / dx if dx > 1e-6 else (-AX - self.x) / dx if dx < -1e-6 else 1e9,
                (AY - self.y) / dy if dy > 1e-6 else (-AY - self.y) / dy if dy < -1e-6 else 1e9,
            ):
                if 0 < t < best:
                    best = t
            out.append(min(best, RAY_MAX) / RAY_MAX)
        return out

    def _obs(self):
        dt = math.hypot(self.tx - self.x, self.ty - self.y)
        at = math.atan2(self.ty - self.y, self.tx - self.x) - self.th
        return np.array(
            [min(1.0, dt / 8.0), math.sin(at), math.cos(at), self.v / 3.0] + self._raycast(),
            np.float32,
        )

    def step(self, action, gov_throttle=None):
        steer = float(np.clip(action[0], -1, 1)) * 0.6
        throttle = float(np.clip(action[1], -1, 1))  # negative = reverse
        if gov_throttle is not None:  # guard clips forward throttle only
            throttle = min(throttle, gov_throttle)
        smooth_pen = 0.07 * abs(throttle - self.prev_throttle) + 0.03 * abs(steer - self.prev_steer)
        self.prev_throttle, self.prev_steer = throttle, steer
        self.v += (throttle * 3.0 - self.v) * 0.25

        nx_raw = self.x + self.v * math.cos(self.th) * DT
        ny_raw = self.y + self.v * math.sin(self.th) * DT
        nx = min(AX, max(-AX, nx_raw))
        ny = min(AY, max(-AY, ny_raw))
        self.th += (self.v / L) * math.tan(steer) * DT

        blocked = False
        if nx != nx_raw or ny != ny_raw:  # wall hard stop (same as obstacle)
            self.v = 0.0
            blocked = True
        do, ox, oy = 1e9, 0.0, 0.0  # solid obstacle hard stop
        for cx, cy in self.obs:
            dd = math.hypot(cx - nx, cy - ny)
            if dd < do:
                do, ox, oy = dd, cx, cy
        if do < BLOCK:
            d = max(do, 1e-6)
            ux, uy = (nx - ox) / d, (ny - oy) / d
            nx, ny = ox + ux * BLOCK, oy + uy * BLOCK
            self.v = 0.0
            blocked = True
        self.x, self.y = nx, ny
        self.steps += 1

        d = math.hypot(self.tx - self.x, self.ty - self.y)
        clear = self._clearance_at(self.x, self.y)  # includes walls
        reward = 1.4 * (self.prev_d - d)             # progress toward goal (stronger incentive with new vision)
        if clear < SAFE_MARGIN:                       # safety first (obstacles + walls)
            reward -= 6.0 * (SAFE_MARGIN - clear) ** 2
        reward -= smooth_pen                          # smoothness (no jitter)
        reward -= 0.02                                # time cost
        self.prev_d = d
        if blocked:
            reward -= 1.0
        if d < REACH:
            reward += 6.0
            self.reached += 1
            self._new_target()
            self.prev_d = math.hypot(self.tx - self.x, self.ty - self.y)
        trunc = self.steps >= self.max_steps
        return self._obs(), reward, False, trunc, {"crashed": blocked, "reached": self.reached, "clear": clear}
