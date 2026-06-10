#!/usr/bin/env python3
"""Gymnasium environment for the blocky Steve character (Minecraft) in MuJoCo — conditioned on heading.

The task can be changed live: `desired` = the requested walking heading. Reward = progress in that
heading + staying upright - control cost. This allows giving the robot a "walk there" task on the fly.
"""
import os

import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "minecraft_steve.xml")
WORLD = os.path.join(HERE, "minecraft_world.xml")  # same Steve + 3D world objects


class SteveEnv(MujocoEnv):
    metadata = {"render_modes": ["rgb_array", "human", "depth_array"], "render_fps": 40}

    def __init__(self, xml_path=XML, **kwargs):
        # obs = qpos[2:] + qvel + desired heading (2); computed automatically from the model
        # (adapts to changes in joint count: elbows/waist/etc.).
        import mujoco
        _m = mujoco.MjModel.from_xml_path(xml_path)
        obs_dim = (_m.nq - 2) + _m.nv + 2
        obs_space = Box(-np.inf, np.inf, (obs_dim,), np.float64)
        super().__init__(xml_path, frame_skip=5, observation_space=obs_space, **kwargs)
        self.desired = np.array([1.0, 0.0])  # task heading (can be changed live)

    def set_task(self, heading_rad):
        """Assign a task: walk in direction heading_rad (0=forward, +pi/2=left, ...)."""
        self.desired = np.array([np.cos(heading_rad), np.sin(heading_rad)])

    def _obs(self):
        return np.concatenate([self.data.qpos[2:], self.data.qvel, self.desired])

    def step(self, action):
        xy0 = self.data.qpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy1 = self.data.qpos[:2].copy()
        vel = (xy1 - xy0) / self.dt
        forward = float(vel @ self.desired)  # progress in the desired heading
        z = float(self.data.qpos[2])
        healthy = 0.5 < z < 1.2  # upright (not fallen, not jumping)
        ctrl_cost = 0.02 * float(np.square(action).sum())
        # forward velocity (primary) + upright reward - effort cost.
        reward = 1.25 * forward + 1.0 * float(healthy) - ctrl_cost
        terminated = not healthy
        return self._obs(), reward, terminated, False, {"x": float(xy1[0]), "vx": forward}

    def reset_model(self):
        # Default training task: walk forward (+x). Static world objects do not change.
        self.desired = np.array([1.0, 0.0])
        qpos = self.init_qpos + self.np_random.uniform(-0.01, 0.01, self.model.nq)
        qvel = self.np_random.uniform(-0.01, 0.01, self.model.nv)
        self.set_state(qpos, qvel)
        return self._obs()
