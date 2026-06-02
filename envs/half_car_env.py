from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None

from models.half_car import HalfCarModel, HalfCarParams
from roads.road_profiles import RoadProfileFactory


class HalfCarEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any], scenario: dict[str, Any] | None = None, use_preview: bool = True):
        self.config = config
        self.dt = float(config["dt"])
        self.control_dt = float(config["control_dt"])
        self.substeps = max(1, int(round(self.control_dt / self.dt)))
        self.episode_seconds = float(config["episode_seconds"])
        self.max_steps = int(round(self.episode_seconds / self.control_dt))
        self.settle_steps = int(round(float(config.get("settle_seconds", 0.0)) / self.control_dt))
        self.wheelbase = float(config["wheelbase"])
        self.force_limit = float(config["force_limit"])
        self.preview_steps = int(config["preview"]["steps"])
        self.preview_dt = float(config["preview"].get("dt", self.control_dt))
        self.road_scale = float(config.get("road_scale", 0.1))
        self.use_preview = use_preview
        self.params = HalfCarParams.from_config(config)
        self.model = HalfCarModel(self.params)
        self.scenario = scenario or config["scenarios"][0]
        self.speed = float(self.scenario.get("speed", config.get("speed", 20.0)))
        self.road = RoadProfileFactory.create(self.scenario, self.episode_seconds, self.dt)
        self.state = np.zeros(8, dtype=np.float64)
        self.t = 0.0
        self.step_count = 0
        self.last_action = np.zeros(2, dtype=np.float64)
        self.last_derived = self.model.derived(self.state, np.zeros(4), self.last_action)

        self.base_obs_dim = 14
        self.obs_dim = self.base_obs_dim + (self.preview_steps * 2 if use_preview else 0)
        if spaces is not None:
            self.action_space = spaces.Box(-self.force_limit, self.force_limit, shape=(2,), dtype=np.float32)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            np.random.seed(seed)
        if options and "scenario" in options:
            self.scenario = options["scenario"]
            self.speed = float(self.scenario.get("speed", self.config.get("speed", 20.0)))
            self.road = RoadProfileFactory.create(self.scenario, self.episode_seconds, self.dt)
        self.state = np.zeros(8, dtype=np.float64)
        zf, dzf, zr, dzr = self.road.pair(0.0, self.speed, self.wheelbase)
        self.state[4] = zf
        self.state[6] = zr
        self.t = 0.0
        self.step_count = 0
        self.last_action = np.zeros(2, dtype=np.float64)
        self.last_derived = self.model.derived(self.state, np.asarray([zf, dzf, zr, dzr]), self.last_action)
        return self._obs(), self._info(0.0)

    def step(self, action):
        action = self.model.clip_action(action)
        total_reward = 0.0
        for _ in range(self.substeps):
            road = np.asarray(self.road.pair(self.t, self.speed, self.wheelbase), dtype=np.float64)
            self.state = self.model.rk4_step(self.state, action, road, self.dt)
            self.t += self.dt
        road = np.asarray(self.road.pair(self.t, self.speed, self.wheelbase), dtype=np.float64)
        self.last_action = action
        self.last_derived = self.model.derived(self.state, road, action)
        reward = self._reward(action)
        total_reward += reward
        self.step_count += 1
        finite = np.all(np.isfinite(self.state)) and np.all(np.isfinite(action))
        unsafe = not finite or abs(self.last_derived["delta_yf"]) > 0.25 or abs(self.last_derived["delta_yr"]) > 0.25
        terminated = not finite
        truncated = self.step_count >= self.max_steps
        if unsafe and self.step_count > self.settle_steps:
            total_reward -= 10.0
        return self._obs(), float(total_reward), terminated, truncated, self._info(total_reward, unsafe=unsafe)

    def _base_obs(self) -> np.ndarray:
        d = self.last_derived
        return np.asarray(
            [
                self.state[0],
                self.state[1],
                self.state[2],
                self.state[3],
                self.state[4],
                self.state[5],
                self.state[6],
                self.state[7],
                d["delta_yf"],
                d["delta_yr"],
                d["delta_dyf"],
                d["delta_dyr"],
                d["ddzb"],
                d["ddtheta"],
            ],
            dtype=np.float64,
        )

    def _obs(self) -> np.ndarray:
        base = self._base_obs()
        if not self.use_preview:
            return base.astype(np.float32)
        preview = self.road.preview(self.t, self.speed, self.wheelbase, self.preview_steps, self.preview_dt)
        return np.concatenate([base, preview.reshape(-1) / self.road_scale]).astype(np.float32)

    def _reward(self, action: np.ndarray) -> float:
        w = self.config["reward"]
        d = self.last_derived
        susp = d["delta_yf"] ** 2 + d["delta_yr"] ** 2
        tire = d["Fpf"] ** 2 + d["Fpr"] ** 2
        control = float(np.sum(np.square(action / self.force_limit)))
        return -float(
            w["body_acc"] * d["ddzb"] ** 2
            + w["pitch_acc"] * d["ddtheta"] ** 2
            + w["suspension"] * susp
            + w["tire_load"] * tire
            + w["control"] * control
        )

    def _info(self, reward: float, unsafe: bool = False) -> dict[str, Any]:
        road = np.asarray(self.road.pair(self.t, self.speed, self.wheelbase), dtype=np.float64)
        preview = self.road.preview(self.t, self.speed, self.wheelbase, self.preview_steps, self.preview_dt)
        return {
            "time": self.t,
            "state": self.state.copy(),
            "base_obs": self._base_obs(),
            "road": road,
            "road_preview": preview,
            "action": self.last_action.copy(),
            "derived": dict(self.last_derived),
            "reward": float(reward),
            "unsafe": bool(unsafe),
            "scenario": self.scenario.get("name", ""),
        }
