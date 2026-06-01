from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None

from models.half_car import HalfCarParams
from roads.road_profiles import BaseRoadProfile, RoadProfileFactory


class _FallbackBox:
    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        return np.random.uniform(self.low, self.high).astype(self.dtype)


class MuJoCoFullCarEnv(gym.Env if gym is not None else object):
    """MuJoCo full-car vertical dynamics environment.

    The model keeps the 14-value base observation interface while adding roll
    dynamics and four unsprung masses. It supports axle-level or four-corner
    actions, axle or corner road preview tokens, and optional preview error
    injection for already perceived road-height inputs.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: dict[str, Any],
        scenario: dict[str, Any] | None = None,
        use_preview: bool = True,
        render_mode: str | None = "rgb_array",
        width: int = 1280,
        height: int = 720,
    ):
        if mujoco is None:
            raise ImportError("MuJoCo is not installed. Install it with: python -m pip install mujoco glfw")
        self.config = config
        self._base_config = config
        self.rng = np.random.default_rng(int(config.get("seed", 42)))
        self.dt = float(config["dt"])
        self.control_dt = float(config["control_dt"])
        self.substeps = max(1, int(round(self.control_dt / self.dt)))
        self.episode_seconds = float(config["episode_seconds"])
        self.max_steps = int(round(self.episode_seconds / self.control_dt))
        self.settle_steps = int(round(float(config.get("settle_seconds", 0.0)) / self.control_dt))
        self.force_limit = float(config["force_limit"])
        self.preview_steps = int(config["preview"]["steps"])
        self.preview_dt = float(config["preview"].get("dt", self.control_dt))
        self.preview_error_cfg = dict(config.get("preview_error", {}))
        self.actuator_cfg = dict(config.get("actuator", {}))
        self.actuator_enabled = bool(self.actuator_cfg.get("enabled", False))
        self.actuator_tau = max(0.0, float(self.actuator_cfg.get("time_constant", 0.0)))
        self.actuator_rate_limit = float(self.actuator_cfg.get("rate_limit", np.inf))
        obs_cfg = dict(config.get("observation", {}))
        history_cfg = dict(obs_cfg.get("history", {}))
        self.observation_mode = str(obs_cfg.get("mode", "privileged")).lower()
        if self.observation_mode not in {"privileged", "estimated", "noisy"}:
            raise ValueError("observation.mode must be privileged, estimated, or noisy")
        self.state_noise_std = float(obs_cfg.get("state_noise_std", 0.0))
        self.estimator_alpha = float(obs_cfg.get("estimator_alpha", 0.35))
        self.history_steps = int(history_cfg.get("steps", 0)) if history_cfg.get("enabled", False) else 0
        safety_cfg = dict(config.get("safety_limits", {}))
        self.safety_limits = {
            "max_suspension_travel": float(safety_cfg.get("max_suspension_travel", 0.25)),
            "max_pitch": float(safety_cfg.get("max_pitch", 0.45)),
            "max_roll": float(safety_cfg.get("max_roll", 0.45)),
            "max_wheel_displacement": float(safety_cfg.get("max_wheel_displacement", 1.0)),
            "terminate_on_unsafe": bool(safety_cfg.get("terminate_on_unsafe", False)),
            "unsafe_penalty": float(safety_cfg.get("unsafe_penalty", 10.0)),
        }
        self.road_scale = float(config.get("road_scale", 0.1))
        self.use_preview = use_preview
        self.render_mode = render_mode
        self.width = int(width)
        self.height = int(height)

        self.params = HalfCarParams.from_seed(int(config.get("seed", 42)))
        full_cfg = config.get("full_car", {})
        self.track_width = float(full_cfg.get("track_width", 1.62))
        self.roll_inertia = float(full_cfg.get("roll_inertia", 520.0))
        self.lateral_roughness_scale = float(full_cfg.get("lateral_roughness_scale", 0.35))
        self.lateral_mode = str(full_cfg.get("lateral_mode", "correlated")).lower()
        self.action_mode = str(full_cfg.get("action_mode", "axle")).lower()
        self.preview_mode = str(full_cfg.get("preview_mode", "axle")).lower()
        if self.action_mode not in {"axle", "corner", "four_corner"}:
            raise ValueError("full_car.action_mode must be 'axle' or 'corner'")
        if self.preview_mode not in {"axle", "corner", "four_corner"}:
            raise ValueError("full_car.preview_mode must be 'axle' or 'corner'")
        self.act_dim = 4 if self.action_mode in {"corner", "four_corner"} else 2
        self.preview_token_dim = 4 if self.preview_mode in {"corner", "four_corner"} else 2
        self.scenario = scenario or config["scenarios"][0]
        self.speed = float(self.scenario.get("speed", config.get("speed", 20.0)))
        self.road_scale_randomization = 1.0
        self.road_left = RoadProfileFactory.create(self._randomized_scenario(self.scenario), self.episode_seconds, self.dt)
        self.road_right = self._make_right_road(self.scenario)

        self.mj_model = mujoco.MjModel.from_xml_string(self._xml())
        self.mj_model.opt.timestep = self.dt
        self.mj_data = mujoco.MjData(self.mj_model)
        self._renderer = None
        self._ids = self._lookup_ids()

        self.t = 0.0
        self.step_count = 0
        self.last_command = np.zeros(self.act_dim, dtype=np.float64)
        self.last_action = np.zeros(self.act_dim, dtype=np.float64)
        self.last_prior_action = np.zeros(self.act_dim, dtype=np.float64)
        self.prev_command = np.zeros(self.act_dim, dtype=np.float64)
        self.prev_action = np.zeros(self.act_dim, dtype=np.float64)
        self.last_corner_command = np.zeros(4, dtype=np.float64)
        self.last_corner_action = np.zeros(4, dtype=np.float64)
        self.last_road = np.zeros(8, dtype=np.float64)
        self.last_preview_clean = np.zeros((self.preview_steps, self.preview_token_dim), dtype=np.float64)
        self.last_preview = np.zeros((self.preview_steps, self.preview_token_dim), dtype=np.float64)
        self._last_preview_time: float | None = None
        self.last_derived: dict[str, float] = {}
        self.last_safety: dict[str, Any] = {}
        self.last_reward_components: dict[str, float] = {}
        self.base_obs_dim = 14
        self.estimated_state = np.zeros(14, dtype=np.float64)
        self.history_dim = self.base_obs_dim + 2 * self.act_dim
        self.history_buffer = np.zeros((self.history_steps, self.history_dim), dtype=np.float64)
        self._eq_state = np.zeros(14, dtype=np.float64)
        self._eq_susp = np.zeros(4, dtype=np.float64)
        self._eq_tire = np.zeros(4, dtype=np.float64)

        self.history_obs_dim = self.history_steps * self.history_dim
        self.obs_base_dim = self.base_obs_dim + self.history_obs_dim
        self.obs_dim = self.obs_base_dim + (self.preview_steps * self.preview_token_dim if use_preview else 0)
        if spaces is not None:
            self.action_space = spaces.Box(-self.force_limit, self.force_limit, shape=(self.act_dim,), dtype=np.float32)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        else:
            self.action_space = _FallbackBox(-self.force_limit, self.force_limit, shape=(self.act_dim,), dtype=np.float32)
            self.observation_space = _FallbackBox(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            np.random.seed(seed)
            self.rng = np.random.default_rng(seed)
        if options and "scenario" in options:
            self.scenario = options["scenario"]
            self.speed = float(self.scenario.get("speed", self.config.get("speed", 20.0)))
        self._apply_domain_randomization()
        self.road_left = RoadProfileFactory.create(self._randomized_scenario(self.scenario), self.episode_seconds, self.dt)
        self.road_right = self._make_right_road(self.scenario)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.mj_model = mujoco.MjModel.from_xml_string(self._xml())
        self.mj_model.opt.timestep = self.dt
        self.mj_data = mujoco.MjData(self.mj_model)
        self._ids = self._lookup_ids()
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.t = 0.0
        self.step_count = 0
        self.last_command = np.zeros(self.act_dim, dtype=np.float64)
        self.last_action = np.zeros(self.act_dim, dtype=np.float64)
        self.last_prior_action = np.zeros(self.act_dim, dtype=np.float64)
        self.prev_command = np.zeros(self.act_dim, dtype=np.float64)
        self.prev_action = np.zeros(self.act_dim, dtype=np.float64)
        self.last_corner_command = np.zeros(4, dtype=np.float64)
        self.last_corner_action = np.zeros(4, dtype=np.float64)
        self.last_safety = {}
        self.last_reward_components = {}
        self.estimated_state = np.zeros(self.base_obs_dim, dtype=np.float64)
        self.history_buffer = np.zeros((self.history_steps, self.history_dim), dtype=np.float64)
        self._last_preview_time = None
        self._set_initial_pose()
        self._update_road(0.0)
        self.mj_data.ctrl[:] = 0.0
        self._settle_static_equilibrium()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._capture_equilibrium()
        self.last_derived = self._derived()
        self.last_safety = self._safety(True)
        self._reward(np.zeros(self.act_dim, dtype=np.float64))
        self.estimated_state = self._state()
        self._push_history()
        return self._obs(), self._info(0.0)

    def step(self, action):
        prior_action = None
        if isinstance(action, dict):
            prior_action = action.get("prior_action")
            action = action.get("action")
        action = np.clip(np.asarray(action, dtype=np.float64), -self.force_limit, self.force_limit)
        if action.shape != (self.act_dim,):
            raise ValueError(f"Expected action shape {(self.act_dim,)}, got {action.shape}")
        if prior_action is None:
            self.last_prior_action = np.zeros(self.act_dim, dtype=np.float64)
        else:
            prior = np.clip(np.asarray(prior_action, dtype=np.float64), -self.force_limit, self.force_limit)
            if prior.shape != (self.act_dim,):
                raise ValueError(f"Expected prior_action shape {(self.act_dim,)}, got {prior.shape}")
            self.last_prior_action = prior
        self.prev_command = self.last_command.copy()
        self.prev_action = self.last_action.copy()
        self.last_command = action
        self.last_corner_command = self._corner_action(action)
        for _ in range(self.substeps):
            self._update_actuators(self.last_corner_command)
            self._update_road(self.t)
            mujoco.mj_step(self.mj_model, self.mj_data)
            self.t += self.dt
        self._update_road(self.t)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.last_derived = self._derived()
        reward = self._reward(action)
        self.estimated_state = self._estimate_state()
        self._push_history()
        self.step_count += 1
        finite = np.all(np.isfinite(self.mj_data.qpos)) and np.all(np.isfinite(self.mj_data.qvel))
        self.last_safety = self._safety(finite)
        unsafe = bool(self.last_safety["unsafe"])
        terminated = (not finite) or (unsafe and self.safety_limits["terminate_on_unsafe"] and self.step_count > self.settle_steps)
        truncated = self.step_count >= self.max_steps
        if unsafe and self.step_count > self.settle_steps:
            reward -= self.safety_limits["unsafe_penalty"]
        return self._obs(), float(reward), terminated, truncated, self._info(reward, unsafe=unsafe)

    def render(self):
        if self.render_mode != "rgb_array":
            raise NotImplementedError("MuJoCoFullCarEnv currently supports render_mode='rgb_array'.")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.mj_model, self.height, self.width)
        self._renderer.update_scene(self.mj_data, camera="rear_side")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def environment_metadata(self) -> dict[str, Any]:
        return {
            "engine": "mujoco_full_car",
            "dt": float(self.dt),
            "control_dt": float(self.control_dt),
            "substeps": int(self.substeps),
            "episode_seconds": float(self.episode_seconds),
            "force_limit": float(self.force_limit),
            "base_obs_dim": int(self.base_obs_dim),
            "obs_dim": int(self.obs_dim),
            "act_dim": int(self.act_dim),
            "preview_steps": int(self.preview_steps),
            "preview_dt": float(self.preview_dt),
            "preview_token_dim": int(self.preview_token_dim),
            "use_preview": bool(self.use_preview),
            "observation": {
                "mode": self.observation_mode,
                "obs_base_dim": int(self.obs_base_dim),
                "history_steps": int(self.history_steps),
                "history_dim": int(self.history_dim),
                "state_noise_std": float(self.state_noise_std),
                "estimator_alpha": float(self.estimator_alpha),
            },
            "full_car": {
                "track_width": float(self.track_width),
                "roll_inertia": float(self.roll_inertia),
                "lateral_roughness_scale": float(self.lateral_roughness_scale),
                "lateral_mode": self.lateral_mode,
                "action_mode": self.action_mode,
                "preview_mode": self.preview_mode,
            },
            "vehicle_params": {key: float(value) for key, value in asdict(self.params).items()},
            "actuator": {
                "enabled": bool(self.actuator_enabled),
                "time_constant": float(self.actuator_tau),
                "rate_limit": float(self.actuator_rate_limit),
            },
            "preview_error": {
                "enabled": bool(self.preview_error_cfg.get("enabled", False)),
                "delay_steps": int(self.preview_error_cfg.get("delay_steps", 0)),
                "height_noise_std": float(self.preview_error_cfg.get("height_noise_std", 0.0)),
                "bias_std": float(self.preview_error_cfg.get("bias_std", 0.0)),
                "dropout_prob": float(self.preview_error_cfg.get("dropout_prob", 0.0)),
                "scale_error_std": float(self.preview_error_cfg.get("scale_error_std", 0.0)),
            },
            "domain_randomization": {
                "enabled": bool(self.config.get("domain_randomization", {}).get("enabled", False)),
                "speed": float(self.speed),
                "road_scale": float(self.road_scale_randomization),
            },
            "safety_limits": dict(self.safety_limits),
        }

    def _make_right_road(self, scenario: dict) -> BaseRoadProfile:
        if self.lateral_mode == "independent":
            right_scenario = self._randomized_scenario(scenario)
            right_scenario["seed"] = int(scenario.get("seed", 42)) + 1009
            return RoadProfileFactory.create(right_scenario, self.episode_seconds, self.dt)
        return self.road_left

    def _apply_domain_randomization(self):
        dr = self.config.get("domain_randomization", {})
        self.params = HalfCarParams.from_seed(int(self.config.get("seed", 42)))
        if not dr.get("enabled", False):
            self.speed = float(self.scenario.get("speed", self.config.get("speed", 20.0)))
            self.road_scale_randomization = 1.0
            return

        def scale(value: float, key: str) -> float:
            span = float(dr.get(key, 0.0))
            if span <= 0.0:
                return value
            return float(value * self.rng.uniform(1.0 - span, 1.0 + span))

        self.params.mb_real = scale(self.params.mb_real, "mass_scale")
        self.params.Ip_real = scale(self.params.Ip_real, "inertia_scale")
        self.roll_inertia = scale(float(self.config.get("full_car", {}).get("roll_inertia", 520.0)), "inertia_scale")
        self.params.kf1_real = scale(self.params.kf1_real, "suspension_stiffness_scale")
        self.params.kr1_real = scale(self.params.kr1_real, "suspension_stiffness_scale")
        self.params.be_real = scale(self.params.be_real, "suspension_damping_scale")
        self.params.bc_real = scale(self.params.bc_real, "suspension_damping_scale")
        self.params.kf2 = scale(self.params.kf2, "tire_stiffness_scale")
        self.params.kr2 = scale(self.params.kr2, "tire_stiffness_scale")
        self.params.bf2 = scale(self.params.bf2, "tire_damping_scale")
        self.params.br2 = scale(self.params.br2, "tire_damping_scale")
        self.speed = scale(float(self.scenario.get("speed", self.config.get("speed", 20.0))), "speed_scale")
        road_span = float(dr.get("road_amplitude_scale", 0.0))
        self.road_scale_randomization = float(self.rng.uniform(1.0 - road_span, 1.0 + road_span)) if road_span > 0 else 1.0

    def _randomized_scenario(self, scenario: dict) -> dict:
        scaled = dict(scenario)
        scaled["speed"] = self.speed
        if self.road_scale_randomization == 1.0:
            return scaled
        for key in ("height", "bump_height", "depth", "amplitude", "scale"):
            if key in scaled:
                scaled[key] = float(scaled[key]) * self.road_scale_randomization
        if scaled.get("type", "").lower() == "composite":
            components = []
            for part in scaled.get("components", []):
                part_scaled = dict(part)
                for key in ("height", "bump_height", "depth", "amplitude", "scale"):
                    if key in part_scaled:
                        part_scaled[key] = float(part_scaled[key]) * self.road_scale_randomization
                components.append(part_scaled)
            scaled["components"] = components
        return scaled

    def _corner_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_mode in {"corner", "four_corner"}:
            return np.asarray(action, dtype=np.float64)
        return np.asarray([0.5 * action[0], 0.5 * action[0], 0.5 * action[1], 0.5 * action[1]], dtype=np.float64)

    def _action_from_corner(self, corner_action: np.ndarray) -> np.ndarray:
        if self.action_mode in {"corner", "four_corner"}:
            return np.asarray(corner_action, dtype=np.float64)
        return np.asarray([corner_action[0] + corner_action[1], corner_action[2] + corner_action[3]], dtype=np.float64)

    def _update_actuators(self, corner_command: np.ndarray):
        target = np.clip(np.asarray(corner_command, dtype=np.float64), -self.force_limit, self.force_limit)
        if not self.actuator_enabled:
            next_corner = target
        else:
            next_corner = self.last_corner_action.copy()
            if self.actuator_tau > 0.0:
                alpha = min(1.0, self.dt / self.actuator_tau)
                next_corner = next_corner + alpha * (target - next_corner)
            else:
                next_corner = target
            if np.isfinite(self.actuator_rate_limit) and self.actuator_rate_limit > 0.0:
                max_delta = self.actuator_rate_limit * self.dt
                next_corner = self.last_corner_action + np.clip(next_corner - self.last_corner_action, -max_delta, max_delta)
            next_corner = np.clip(next_corner, -self.force_limit, self.force_limit)
        self.last_corner_action = next_corner
        self.last_action = self._action_from_corner(next_corner)
        for name, value in zip(("fl_active", "fr_active", "rl_active", "rr_active"), next_corner):
            self.mj_data.ctrl[self._ids[f"{name}_act"]] = value

    def _lookup_ids(self) -> dict[str, int]:
        obj = mujoco.mjtObj
        ids = {
            "chassis_z": mujoco.mj_name2id(self.mj_model, obj.mjOBJ_JOINT, "chassis_z"),
            "chassis_pitch": mujoco.mj_name2id(self.mj_model, obj.mjOBJ_JOINT, "chassis_pitch"),
            "chassis_roll": mujoco.mj_name2id(self.mj_model, obj.mjOBJ_JOINT, "chassis_roll"),
        }
        for c in ("fl", "fr", "rl", "rr"):
            ids[f"{c}_wheel_z"] = mujoco.mj_name2id(self.mj_model, obj.mjOBJ_JOINT, f"{c}_wheel_z")
            ids[f"{c}_road"] = mujoco.mj_name2id(self.mj_model, obj.mjOBJ_BODY, f"{c}_road")
            ids[f"{c}_active_act"] = mujoco.mj_name2id(self.mj_model, obj.mjOBJ_ACTUATOR, f"{c}_active")
        return ids

    def _joint_addr(self, joint_name: str) -> tuple[int, int]:
        jid = self._ids[joint_name]
        return int(self.mj_model.jnt_qposadr[jid]), int(self.mj_model.jnt_dofadr[jid])

    def _joint_state(self, joint_name: str) -> tuple[float, float]:
        qadr, dadr = self._joint_addr(joint_name)
        return float(self.mj_data.qpos[qadr]), float(self.mj_data.qvel[dadr])

    def _set_joint(self, joint_name: str, qpos: float, qvel: float = 0.0):
        qadr, dadr = self._joint_addr(joint_name)
        self.mj_data.qpos[qadr] = qpos
        self.mj_data.qvel[dadr] = qvel

    def _road_corner_values(self, t: float) -> np.ndarray:
        p = self.params
        delay = (p.a_real + p.b_real) / max(self.speed, 1e-6)
        tr = max(0.0, t - delay)
        zfl = self.road_left.value(t)
        dzfl = self.road_left.derivative(t)
        zrl = self.road_left.value(tr)
        dzrl = self.road_left.derivative(tr)
        if self.lateral_mode == "correlated":
            zfr = zfl + self.lateral_roughness_scale * (self.road_right.value(t + 0.017) - zfl)
            dzfr = dzfl + self.lateral_roughness_scale * (self.road_right.derivative(t + 0.017) - dzfl)
            zrr = zrl + self.lateral_roughness_scale * (self.road_right.value(tr + 0.017) - zrl)
            dzrr = dzrl + self.lateral_roughness_scale * (self.road_right.derivative(tr + 0.017) - dzrl)
        else:
            zfr = self.road_right.value(t)
            dzfr = self.road_right.derivative(t)
            zrr = self.road_right.value(tr)
            dzrr = self.road_right.derivative(tr)
        return np.asarray([zfl, dzfl, zfr, dzfr, zrl, dzrl, zrr, dzrr], dtype=np.float64)

    def _update_road(self, t: float):
        self.last_road = self._road_corner_values(t)
        p = self.params
        half_track = 0.5 * self.track_width
        positions = {
            "fl": (p.a_real, half_track, self.last_road[0]),
            "fr": (p.a_real, -half_track, self.last_road[2]),
            "rl": (-p.b_real, half_track, self.last_road[4]),
            "rr": (-p.b_real, -half_track, self.last_road[6]),
        }
        for c, pos in positions.items():
            mid = self.mj_model.body_mocapid[self._ids[f"{c}_road"]]
            self.mj_data.mocap_pos[mid] = np.asarray(pos, dtype=np.float64)
            self.mj_data.mocap_quat[mid] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _set_initial_pose(self):
        p = self.params
        road = self._road_corner_values(0.0)
        tire_rest = self._tire_rest_length()
        susp_rest = self._susp_rest_length()
        wheel_z = np.asarray([road[0], road[2], road[4], road[6]], dtype=np.float64) + tire_rest
        front_avg = float(np.mean(wheel_z[:2]))
        rear_avg = float(np.mean(wheel_z[2:]))
        left_avg = float(np.mean(wheel_z[[0, 2]]))
        right_avg = float(np.mean(wheel_z[[1, 3]]))
        z = float(np.mean(wheel_z) + susp_rest)
        pitch = (front_avg - rear_avg) / max(p.a_real + p.b_real, 1e-6)
        roll = (left_avg - right_avg) / max(self.track_width, 1e-6)
        self._set_joint("chassis_z", z, 0.0)
        self._set_joint("chassis_pitch", pitch, 0.0)
        self._set_joint("chassis_roll", roll, 0.0)
        for c, value in zip(("fl", "fr", "rl", "rr"), wheel_z):
            self._set_joint(f"{c}_wheel_z", float(value), 0.0)

    def _physical_state(self) -> np.ndarray:
        z, dz = self._joint_state("chassis_z")
        pitch, dpitch = self._joint_state("chassis_pitch")
        roll, droll = self._joint_state("chassis_roll")
        wheels = []
        for c in ("fl", "fr", "rl", "rr"):
            wheels.extend(self._joint_state(f"{c}_wheel_z"))
        return np.asarray([z, dz, pitch, dpitch, roll, droll, *wheels], dtype=np.float64)

    def _state(self) -> np.ndarray:
        state = self._physical_state()
        rel = state.copy()
        rel[0] -= self._eq_state[0]
        rel[2] -= self._eq_state[2]
        rel[4] -= self._eq_state[4]
        for idx in (6, 8, 10, 12):
            rel[idx] -= self._eq_state[idx]
        return rel

    def _corner_body_height(self, z: float, pitch: float, roll: float, corner: str) -> float:
        p = self.params
        half_track = 0.5 * self.track_width
        x = p.a_real if corner[0] == "f" else -p.b_real
        y = half_track if corner[1] == "l" else -half_track
        return float(z + x * np.sin(pitch) + y * np.sin(roll))

    def _derived(self) -> dict[str, float]:
        p = self.params
        state = self._physical_state()
        z, dz, pitch, dpitch, roll, droll = state[:6]
        wheel_state = {"fl": state[6:8], "fr": state[8:10], "rl": state[10:12], "rr": state[12:14]}
        road = {
            "fl": self.last_road[0:2],
            "fr": self.last_road[2:4],
            "rl": self.last_road[4:6],
            "rr": self.last_road[6:8],
        }
        qacc = self.mj_data.qacc
        _, z_dof = self._joint_addr("chassis_z")
        _, pitch_dof = self._joint_addr("chassis_pitch")
        _, roll_dof = self._joint_addr("chassis_roll")
        axle_action = self.last_action if self.action_mode == "axle" else np.asarray(
            [
                self.last_action[0] + self.last_action[1],
                self.last_action[2] + self.last_action[3],
            ],
            dtype=np.float64,
        )
        out = {
            "ddzb": float(qacc[z_dof]),
            "ddtheta": float(qacc[pitch_dof]),
            "ddroll": float(qacc[roll_dof]),
            "Uaf": float(axle_action[0]),
            "Uar": float(axle_action[1]),
        }
        for i, c in enumerate(("fl", "fr", "rl", "rr")):
            wz, dwz = wheel_state[c]
            rz, drz = road[c]
            body_z = self._corner_body_height(z, pitch, roll, c)
            body_dz = dz + (p.a_real if c[0] == "f" else -p.b_real) * dpitch * np.cos(pitch)
            body_dz += (0.5 * self.track_width if c[1] == "l" else -0.5 * self.track_width) * droll * np.cos(roll)
            susp = body_z - wz - self._eq_susp[i]
            dsusp = body_dz - dwz
            tire = wz - rz - self._eq_tire[i]
            dtire = dwz - drz
            kt = p.kf2 if c[0] == "f" else p.kr2
            bt = p.bf2 if c[0] == "f" else p.br2
            _, wdof = self._joint_addr(f"{c}_wheel_z")
            out[f"delta_y_{c}"] = float(susp)
            out[f"delta_dy_{c}"] = float(dsusp)
            out[f"F_tire_{c}"] = float(kt * tire + bt * dtire)
            out[f"ddzw_{c}"] = float(qacc[wdof])
        out["delta_yf"] = float(0.5 * (out["delta_y_fl"] + out["delta_y_fr"]))
        out["delta_yr"] = float(0.5 * (out["delta_y_rl"] + out["delta_y_rr"]))
        out["delta_dyf"] = float(0.5 * (out["delta_dy_fl"] + out["delta_dy_fr"]))
        out["delta_dyr"] = float(0.5 * (out["delta_dy_rl"] + out["delta_dy_rr"]))
        out["Fpf"] = float(out["F_tire_fl"] + out["F_tire_fr"])
        out["Fpr"] = float(out["F_tire_rl"] + out["F_tire_rr"])
        return out

    def _base_obs(self) -> np.ndarray:
        if self.observation_mode == "privileged":
            state = self._state()
        elif self.observation_mode == "estimated":
            state = self.estimated_state
        else:
            state = self._state()
            if self.state_noise_std > 0.0:
                state = state + self.rng.normal(0.0, self.state_noise_std, size=state.shape)
        if self.history_steps <= 0:
            return state.astype(np.float64)
        return np.concatenate([state, self.history_buffer.reshape(-1)]).astype(np.float64)

    def _estimate_state(self) -> np.ndarray:
        measured = self._state()
        if self.observation_mode == "estimated":
            alpha = float(np.clip(self.estimator_alpha, 0.0, 1.0))
            return (1.0 - alpha) * self.estimated_state + alpha * measured
        if self.observation_mode == "noisy" and self.state_noise_std > 0.0:
            return measured + self.rng.normal(0.0, self.state_noise_std, size=measured.shape)
        return measured

    def _push_history(self):
        if self.history_steps <= 0:
            return
        row = np.concatenate([self._state(), self.last_command, self.last_action]).astype(np.float64)
        self.history_buffer = np.roll(self.history_buffer, -1, axis=0)
        self.history_buffer[-1] = row

    def _preview_clean(self, t: float | None = None) -> np.ndarray:
        if t is None:
            t = self.t
        rows = []
        for k in range(1, self.preview_steps + 1):
            road = self._road_corner_values(t + k * self.preview_dt)
            if self.preview_mode in {"corner", "four_corner"}:
                rows.append([road[0], road[2], road[4], road[6]])
            else:
                front = 0.5 * (road[0] + road[2])
                rear = 0.5 * (road[4] + road[6])
                rows.append([front, rear])
        return np.asarray(rows, dtype=np.float64)

    def _preview(self) -> np.ndarray:
        if self._last_preview_time is not None and np.isclose(self._last_preview_time, self.t):
            return self.last_preview.copy()
        reference_clean = self._preview_clean()
        cfg = self.preview_error_cfg
        if not cfg.get("enabled", False):
            self.last_preview_clean = reference_clean
            self.last_preview = reference_clean
            self._last_preview_time = self.t
            return reference_clean.copy()

        delay_steps = max(0, int(cfg.get("delay_steps", 0)))
        delayed_t = max(0.0, self.t - delay_steps * self.preview_dt)
        delayed_clean = self._preview_clean(delayed_t)
        preview = delayed_clean.copy()

        scale_std = float(cfg.get("scale_error_std", 0.0))
        if scale_std > 0.0:
            preview *= self.rng.normal(1.0, scale_std, size=(1, preview.shape[1]))

        bias_std = float(cfg.get("bias_std", 0.0))
        if bias_std > 0.0:
            preview += self.rng.normal(0.0, bias_std, size=(1, preview.shape[1]))

        noise_std = float(cfg.get("height_noise_std", 0.0))
        if noise_std > 0.0:
            preview += self.rng.normal(0.0, noise_std, size=preview.shape)

        dropout_prob = float(cfg.get("dropout_prob", 0.0))
        if dropout_prob > 0.0:
            dropout = self.rng.random(preview.shape) < np.clip(dropout_prob, 0.0, 1.0)
            preview[dropout] = float(cfg.get("dropout_value", 0.0))

        self.last_preview_clean = reference_clean
        self.last_preview = preview
        self._last_preview_time = self.t
        return preview.copy()

    def _obs(self) -> np.ndarray:
        base = self._base_obs()
        if not self.use_preview:
            return base.astype(np.float32)
        return np.concatenate([base, self._preview().reshape(-1) / self.road_scale]).astype(np.float32)

    def _reward(self, action: np.ndarray) -> float:
        w = self.config["reward"]
        d = self.last_derived
        susp = sum(d[f"delta_y_{c}"] ** 2 for c in ("fl", "fr", "rl", "rr"))
        tire = sum(d[f"F_tire_{c}"] ** 2 for c in ("fl", "fr", "rl", "rr"))
        roll_w = float(w.get("roll_acc", 0.25))
        control = float(np.sum(np.square(action / self.force_limit)))
        action_delta = float(np.sum(np.square((self.last_action - self.prev_action) / self.force_limit)))
        command_delta = float(np.sum(np.square((self.last_command - self.prev_command) / self.force_limit)))
        actuator_tracking = float(np.sum(np.square((self.last_command - self.last_action) / self.force_limit)))
        saturation = float(np.mean(np.isclose(np.abs(self.last_action), self.force_limit, rtol=0.0, atol=1e-6)))
        deviation = float(np.sum(np.square((self.last_command - self.last_prior_action) / self.force_limit)))
        power = 0.0
        for value, c in zip(self.last_corner_action, ("fl", "fr", "rl", "rr")):
            power += abs(float(value) * d[f"delta_dy_{c}"])
        components = {
            "body_acc": float(w["body_acc"] * d["ddzb"] ** 2),
            "pitch_acc": float(w["pitch_acc"] * d["ddtheta"] ** 2),
            "roll_acc": float(roll_w * d["ddroll"] ** 2),
            "suspension": float(w["suspension"] * susp),
            "tire_load": float(w["tire_load"] * tire),
            "control": float(w["control"] * control),
            "action_delta": float(w.get("action_delta", 0.0) * action_delta),
            "command_delta": float(w.get("command_delta", 0.0) * command_delta),
            "actuator_tracking": float(w.get("actuator_tracking", 0.0) * actuator_tracking),
            "saturation": float(w.get("saturation", 0.0) * saturation),
            "energy": float(w.get("energy", 0.0) * power),
            "deviation": float(w.get("deviation", 0.0) * deviation),
        }
        self.last_reward_components = components
        return -float(sum(components.values()))

    def _safety(self, finite: bool) -> dict[str, Any]:
        state = self._state()
        susp = np.asarray([self.last_derived[f"delta_y_{c}"] for c in ("fl", "fr", "rl", "rr")], dtype=np.float64)
        wheel = np.asarray([state[i] for i in (6, 8, 10, 12)], dtype=np.float64)
        checks = {
            "finite": bool(finite),
            "suspension_travel": bool(np.max(np.abs(susp)) <= self.safety_limits["max_suspension_travel"]),
            "pitch": bool(abs(state[2]) <= self.safety_limits["max_pitch"]),
            "roll": bool(abs(state[4]) <= self.safety_limits["max_roll"]),
            "wheel_displacement": bool(np.max(np.abs(wheel)) <= self.safety_limits["max_wheel_displacement"]),
        }
        violations = [name for name, ok in checks.items() if not ok]
        return {
            "unsafe": bool(violations),
            "checks": checks,
            "violations": violations,
            "max_abs_suspension_travel": float(np.max(np.abs(susp))),
            "max_abs_wheel_displacement": float(np.max(np.abs(wheel))),
            "abs_pitch": float(abs(state[2])),
            "abs_roll": float(abs(state[4])),
            "limits": dict(self.safety_limits),
        }

    def _info(self, reward: float, unsafe: bool = False) -> dict[str, Any]:
        avg_road = np.asarray(
            [
                0.5 * (self.last_road[0] + self.last_road[2]),
                0.5 * (self.last_road[1] + self.last_road[3]),
                0.5 * (self.last_road[4] + self.last_road[6]),
                0.5 * (self.last_road[5] + self.last_road[7]),
            ],
            dtype=np.float64,
        )
        preview = self._preview()
        action_delta = self.last_action - self.prev_action
        command_delta = self.last_command - self.prev_command
        tracking_error = self.last_command - self.last_action
        saturation_mask = np.isclose(np.abs(self.last_action), self.force_limit, rtol=0.0, atol=1e-6)
        return {
            "time": self.t,
            "state": self._state(),
            "base_obs": self._base_obs(),
            "road": avg_road,
            "road_corners": self.last_road.copy(),
            "road_preview": preview,
            "road_preview_clean": self.last_preview_clean.copy(),
            "preview_error": {
                "enabled": bool(self.preview_error_cfg.get("enabled", False)),
                "delay_steps": int(self.preview_error_cfg.get("delay_steps", 0)),
                "height_noise_std": float(self.preview_error_cfg.get("height_noise_std", 0.0)),
                "bias_std": float(self.preview_error_cfg.get("bias_std", 0.0)),
                "dropout_prob": float(self.preview_error_cfg.get("dropout_prob", 0.0)),
                "scale_error_std": float(self.preview_error_cfg.get("scale_error_std", 0.0)),
            },
            "command_action": self.last_command.copy(),
            "command_corner_action": self.last_corner_command.copy(),
            "action": self.last_action.copy(),
            "corner_action": self.last_corner_action.copy(),
            "prior_action": self.last_prior_action.copy(),
            "action_metrics": {
                "action_delta": action_delta.copy(),
                "command_delta": command_delta.copy(),
                "actuator_tracking_error": tracking_error.copy(),
                "action_delta_rms": float(np.sqrt(np.mean(np.square(action_delta)))) if action_delta.size else 0.0,
                "command_delta_rms": float(np.sqrt(np.mean(np.square(command_delta)))) if command_delta.size else 0.0,
                "actuator_tracking_rms": float(np.sqrt(np.mean(np.square(tracking_error)))) if tracking_error.size else 0.0,
                "saturation_ratio": float(np.mean(saturation_mask)) if saturation_mask.size else 0.0,
                "deviation_rms": float(np.sqrt(np.mean(np.square(self.last_command - self.last_prior_action)))),
                "force_limit": float(self.force_limit),
            },
            "actuator": {
                "enabled": bool(self.actuator_enabled),
                "time_constant": float(self.actuator_tau),
                "rate_limit": float(self.actuator_rate_limit),
            },
            "derived": dict(self.last_derived),
            "safety": dict(self.last_safety),
            "reward_components": dict(self.last_reward_components),
            "reward": float(reward),
            "unsafe": bool(unsafe),
            "scenario": self.scenario.get("name", ""),
            "engine": "mujoco_full_car",
            "domain_randomization": {
                "enabled": bool(self.config.get("domain_randomization", {}).get("enabled", False)),
                "speed": float(self.speed),
                "road_scale": float(self.road_scale_randomization),
                "mb": float(self.params.mb_real),
                "Ip": float(self.params.Ip_real),
                "roll_inertia": float(self.roll_inertia),
            },
        }

    def _susp_rest_length(self) -> float:
        return 0.45

    def _tire_rest_length(self) -> float:
        return 0.30

    def _settle_static_equilibrium(self):
        settle_seconds = float(self.config.get("mujoco", {}).get("settle_seconds", 1.5))
        for _ in range(max(1, int(round(settle_seconds / self.dt)))):
            self._update_road(0.0)
            self.mj_data.ctrl[:] = 0.0
            mujoco.mj_step(self.mj_model, self.mj_data)
        self.mj_data.qvel[:] = 0.0
        self.mj_data.ctrl[:] = 0.0

    def _capture_equilibrium(self):
        state = self._physical_state()
        self._eq_state = state.copy()
        z, _, pitch, _, roll, _ = state[:6]
        for i, c in enumerate(("fl", "fr", "rl", "rr")):
            wheel_z = state[6 + 2 * i]
            self._eq_susp[i] = self._corner_body_height(z, pitch, roll, c) - wheel_z
            self._eq_tire[i] = wheel_z

    def _xml(self) -> str:
        p = self.params
        half_track = 0.5 * self.track_width
        body_len = 0.5 * (p.a_real + p.b_real)
        body_center = 0.5 * (p.a_real - p.b_real)
        susp_rest = self._susp_rest_length()
        tire_rest = self._tire_rest_length()

        def corner_body(c: str, x: float, y: float, mass: float, tire_k: float, tire_b: float) -> str:
            return f"""
    <body name="{c}_wheel" pos="{x:.6f} {y:.6f} .30">
      <joint name="{c}_wheel_z" type="slide" axis="0 0 1" limited="true" range="-1.0 2.0"/>
      <inertial pos="0 0 0" mass="{mass:.9f}" diaginertia="2 2 2"/>
      <geom name="{c}_wheel_geom" type="cylinder" size=".24 .07" euler="1.57079632679 0 0" material="wheel"/>
      <site name="{c}_wheel_site" pos="0 0 0" size=".022"/>
    </body>
    <body name="{c}_road" mocap="true" pos="{x:.6f} {y:.6f} 0">
      <geom name="{c}_road_patch" type="box" size=".34 .34 .025" pos="0 0 -.025" material="road"/>
      <site name="{c}_road_site" pos="0 0 0" size=".018"/>
    </body>"""

        def tendons_and_actuators() -> str:
            rows = []
            for c in ("fl", "fr", "rl", "rr"):
                is_front = c[0] == "f"
                k_s = p.kf1_real if is_front else p.kr1_real
                k_t = p.kf2 if is_front else p.kr2
                b_t = p.bf2 if is_front else p.br2
                rows.append(
                    f"""
    <spatial name="{c}_susp" stiffness="{k_s:.9f}" damping="{p.be_real:.9f}" springlength="{susp_rest:.9f}">
      <site site="{c}_body_site"/>
      <site site="{c}_wheel_site"/>
    </spatial>
    <spatial name="{c}_tire" stiffness="{k_t:.9f}" damping="{b_t:.9f}" springlength="{tire_rest:.9f}">
      <site site="{c}_wheel_site"/>
      <site site="{c}_road_site"/>
    </spatial>"""
                )
            tendon = "<tendon>" + "".join(rows) + "\n  </tendon>"
            acts = ["  <actuator>"]
            for c in ("fl", "fr", "rl", "rr"):
                acts.append(
                    f'    <motor name="{c}_active" tendon="{c}_susp" gear="1" ctrllimited="true" ctrlrange="{-p.fmax:.9f} {p.fmax:.9f}"/>'
                )
            acts.append("  </actuator>")
            return tendon + "\n" + "\n".join(acts)

        return f"""
<mujoco model="industrial_full_car_vertical">
  <compiler angle="radian"/>
  <option timestep="{self.dt}" gravity="0 0 -9.81" integrator="RK4"/>
  <default>
    <joint damping="0" armature="0.01"/>
    <geom contype="0" conaffinity="0"/>
  </default>
  <visual>
    <global offwidth="{self.width}" offheight="{self.height}"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".18 .20 .22" rgb2=".28 .30 .32" width="256" height="256"/>
    <material name="ground" texture="grid" texrepeat="12 4" reflectance="0.04"/>
    <material name="body" rgba="0.04 0.10 0.18 1"/>
    <material name="wheel" rgba="0.94 0.42 0.10 1"/>
    <material name="road" rgba="0.18 0.18 0.18 1"/>
  </asset>
  <worldbody>
    <light pos="0 -4 5" dir="0 1 -1"/>
    <geom name="ground" type="plane" size="6 3 .05" material="ground"/>
    <body name="chassis" pos="0 0 .75">
      <joint name="chassis_z" type="slide" axis="0 0 1" limited="true" range="-1.0 2.0"/>
      <joint name="chassis_pitch" type="hinge" axis="0 1 0" limited="true" range="-0.5 0.5"/>
      <joint name="chassis_roll" type="hinge" axis="1 0 0" limited="true" range="-0.5 0.5"/>
      <inertial pos="0 0 0" mass="{p.mb_real:.9f}" diaginertia="{self.roll_inertia:.9f} {p.Ip_real:.9f} {p.Ip_real:.9f}"/>
      <geom name="body_box" type="box" pos="{body_center:.6f} 0 0" size="{body_len:.6f} {0.5 * self.track_width:.6f} .11" material="body"/>
      <site name="fl_body_site" pos="{p.a_real:.6f} {half_track:.6f} -.08" size=".022"/>
      <site name="fr_body_site" pos="{p.a_real:.6f} {-half_track:.6f} -.08" size=".022"/>
      <site name="rl_body_site" pos="{-p.b_real:.6f} {half_track:.6f} -.08" size=".022"/>
      <site name="rr_body_site" pos="{-p.b_real:.6f} {-half_track:.6f} -.08" size=".022"/>
    </body>
{corner_body("fl", p.a_real, half_track, 0.5 * p.mwf, p.kf2, p.bf2)}
{corner_body("fr", p.a_real, -half_track, 0.5 * p.mwf, p.kf2, p.bf2)}
{corner_body("rl", -p.b_real, half_track, 0.5 * p.mwr, p.kr2, p.br2)}
{corner_body("rr", -p.b_real, -half_track, 0.5 * p.mwr, p.kr2, p.br2)}
    <camera name="rear_side" pos="-3.5 -5 1.7" xyaxes="0.82 -0.57 0 0.17 0.24 0.96"/>
  </worldbody>
{tendons_and_actuators()}
</mujoco>
"""
