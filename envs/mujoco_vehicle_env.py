from __future__ import annotations

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
from roads.road_profiles import RoadProfileFactory


class _FallbackBox:
    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        return np.random.uniform(self.low, self.high).astype(self.dtype)


class MuJoCoVehicleEnv(gym.Env if gym is not None else object):
    """MuJoCo-based half-car active suspension environment.

    Unlike MuJoCoHalfCarEnv, this environment lets MuJoCo integrate the vehicle
    dynamics. The vehicle is represented as a planar half-car: chassis heave,
    chassis pitch, and front/rear unsprung vertical motions. Passive suspension
    and tire forces are modeled with MuJoCo spatial tendons; active suspension
    commands are motor actuators on the suspension tendons.
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
        vertical_scale: float = 1.0,
    ):
        if mujoco is None:
            raise ImportError("MuJoCo is not installed. Install it with: python -m pip install mujoco glfw")
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
        self.render_mode = render_mode
        self.width = int(width)
        self.height = int(height)
        self.vertical_scale = float(vertical_scale)

        self.params = HalfCarParams.from_config(config)
        self.scenario = scenario or config["scenarios"][0]
        self.speed = float(self.scenario.get("speed", config.get("speed", 20.0)))
        self.road = RoadProfileFactory.create(self.scenario, self.episode_seconds, self.dt)

        self.mj_model = mujoco.MjModel.from_xml_string(self._xml())
        self.mj_model.opt.timestep = self.dt
        self.mj_data = mujoco.MjData(self.mj_model)
        self._renderer = None
        self._ids = self._lookup_ids()

        self.t = 0.0
        self.step_count = 0
        self.last_action = np.zeros(2, dtype=np.float64)
        self.last_road = np.zeros(4, dtype=np.float64)
        self.last_derived: dict[str, float] = {}
        self._eq_state = np.zeros(8, dtype=np.float64)
        self._eq_front_susp = 0.0
        self._eq_rear_susp = 0.0
        self._eq_front_tire = 0.0
        self._eq_rear_tire = 0.0

        self.base_obs_dim = 14
        self.obs_dim = self.base_obs_dim + (self.preview_steps * 2 if use_preview else 0)
        if spaces is not None:
            self.action_space = spaces.Box(-self.force_limit, self.force_limit, shape=(2,), dtype=np.float32)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        else:
            self.action_space = _FallbackBox(-self.force_limit, self.force_limit, shape=(2,), dtype=np.float32)
            self.observation_space = _FallbackBox(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            np.random.seed(seed)
        if options and "scenario" in options:
            self.scenario = options["scenario"]
            self.speed = float(self.scenario.get("speed", self.config.get("speed", 20.0)))
            self.road = RoadProfileFactory.create(self.scenario, self.episode_seconds, self.dt)

        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.t = 0.0
        self.step_count = 0
        self.last_action = np.zeros(2, dtype=np.float64)
        self._set_initial_pose()
        self._update_road(self.t)
        self.mj_data.ctrl[:] = 0.0
        self._settle_static_equilibrium()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._capture_equilibrium()
        self.last_derived = self._derived()
        return self._obs(), self._info(0.0)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -self.force_limit, self.force_limit)
        self.last_action = action
        self.mj_data.ctrl[self._ids["front_active_act"]] = action[0]
        self.mj_data.ctrl[self._ids["rear_active_act"]] = action[1]
        for _ in range(self.substeps):
            self._update_road(self.t)
            mujoco.mj_step(self.mj_model, self.mj_data)
            self.t += self.dt
        self._update_road(self.t)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.last_derived = self._derived()
        reward = self._reward(action)
        self.step_count += 1
        finite = np.all(np.isfinite(self.mj_data.qpos)) and np.all(np.isfinite(self.mj_data.qvel))
        unsafe = not finite or abs(self.last_derived["delta_yf"]) > 0.25 or abs(self.last_derived["delta_yr"]) > 0.25
        terminated = not finite
        truncated = self.step_count >= self.max_steps
        if unsafe and self.step_count > self.settle_steps:
            reward -= 10.0
        return self._obs(), float(reward), terminated, truncated, self._info(reward, unsafe=unsafe)

    def render(self):
        if self.render_mode != "rgb_array":
            raise NotImplementedError("MuJoCoVehicleEnv currently supports render_mode='rgb_array'.")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.mj_model, self.height, self.width)
        self._renderer.update_scene(self.mj_data, camera="side")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _lookup_ids(self) -> dict[str, int]:
        obj = mujoco.mjtObj
        return {
            "chassis_z": mujoco.mj_name2id(self.mj_model, obj.mjOBJ_JOINT, "chassis_z"),
            "chassis_pitch": mujoco.mj_name2id(self.mj_model, obj.mjOBJ_JOINT, "chassis_pitch"),
            "front_wheel_z": mujoco.mj_name2id(self.mj_model, obj.mjOBJ_JOINT, "front_wheel_z"),
            "rear_wheel_z": mujoco.mj_name2id(self.mj_model, obj.mjOBJ_JOINT, "rear_wheel_z"),
            "front_road": mujoco.mj_name2id(self.mj_model, obj.mjOBJ_BODY, "front_road"),
            "rear_road": mujoco.mj_name2id(self.mj_model, obj.mjOBJ_BODY, "rear_road"),
            "front_active_act": mujoco.mj_name2id(self.mj_model, obj.mjOBJ_ACTUATOR, "front_active"),
            "rear_active_act": mujoco.mj_name2id(self.mj_model, obj.mjOBJ_ACTUATOR, "rear_active"),
        }

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

    def _set_initial_pose(self):
        p = self.params
        zf, _, zr, _ = self.road.pair(0.0, self.speed, self.wheelbase)
        tire_rest = self._tire_rest_length()
        susp_rest = self._susp_rest_length()
        front_wheel_z = zf + tire_rest
        rear_wheel_z = zr + tire_rest
        chassis_z = 0.5 * (front_wheel_z + rear_wheel_z) + susp_rest
        theta = (front_wheel_z - rear_wheel_z) / max(p.a_real + p.b_real, 1e-6)
        self._set_joint("chassis_z", chassis_z, 0.0)
        self._set_joint("chassis_pitch", theta, 0.0)
        self._set_joint("front_wheel_z", front_wheel_z, 0.0)
        self._set_joint("rear_wheel_z", rear_wheel_z, 0.0)

    def _update_road(self, t: float):
        road = np.asarray(self.road.pair(t, self.speed, self.wheelbase), dtype=np.float64)
        self.last_road = road
        zdf, _, zdr, _ = road
        p = self.params
        front_mid = self.mj_model.body_mocapid[self._ids["front_road"]]
        rear_mid = self.mj_model.body_mocapid[self._ids["rear_road"]]
        self.mj_data.mocap_pos[front_mid] = np.asarray([p.a_real, 0.0, zdf * self.vertical_scale], dtype=np.float64)
        self.mj_data.mocap_pos[rear_mid] = np.asarray([-p.b_real, 0.0, zdr * self.vertical_scale], dtype=np.float64)
        self.mj_data.mocap_quat[front_mid] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.mj_data.mocap_quat[rear_mid] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _physical_state(self) -> np.ndarray:
        zb, dzb = self._joint_state("chassis_z")
        theta, dtheta = self._joint_state("chassis_pitch")
        zwf, dzwf = self._joint_state("front_wheel_z")
        zwr, dzwr = self._joint_state("rear_wheel_z")
        return np.asarray([zb, dzb, theta, dtheta, zwf, dzwf, zwr, dzwr], dtype=np.float64)

    def _state(self) -> np.ndarray:
        state = self._physical_state()
        return np.asarray(
            [
                state[0] - self._eq_state[0],
                state[1],
                state[2] - self._eq_state[2],
                state[3],
                state[4] - self._eq_state[4],
                state[5],
                state[6] - self._eq_state[6],
                state[7],
            ],
            dtype=np.float64,
        )

    def _derived(self) -> dict[str, float]:
        p = self.params
        state = self._physical_state()
        zb, dzb, theta, dtheta, zwf, dzwf, zwr, dzwr = state
        zdf, dzdf, zdr, dzdr = self.last_road
        uaf, uar = self.last_action
        front_susp = zb + p.a_real * np.sin(theta) - zwf
        rear_susp = zb - p.b_real * np.sin(theta) - zwr
        front_tire = zwf - zdf
        rear_tire = zwr - zdr
        delta_yf = front_susp - self._eq_front_susp
        delta_yr = rear_susp - self._eq_rear_susp
        delta_dyf = dzb + p.a_real * dtheta * np.cos(theta) - dzwf
        delta_dyr = dzb - p.b_real * dtheta * np.cos(theta) - dzwr
        tire_yf = front_tire - self._eq_front_tire
        tire_yr = rear_tire - self._eq_rear_tire
        tire_dyf = dzwf - dzdf
        tire_dyr = dzwr - dzdr
        fpf = p.kf2 * tire_yf + p.bf2 * tire_dyf
        fpr = p.kr2 * tire_yr + p.br2 * tire_dyr
        qacc = self.mj_data.qacc
        _, chassis_z_dof = self._joint_addr("chassis_z")
        _, pitch_dof = self._joint_addr("chassis_pitch")
        _, front_wheel_dof = self._joint_addr("front_wheel_z")
        _, rear_wheel_dof = self._joint_addr("rear_wheel_z")
        return {
            "delta_yf": float(delta_yf),
            "delta_dyf": float(delta_dyf),
            "delta_yr": float(delta_yr),
            "delta_dyr": float(delta_dyr),
            "Fpf": float(fpf),
            "Fpr": float(fpr),
            "ddzb": float(qacc[chassis_z_dof]),
            "ddtheta": float(qacc[pitch_dof]),
            "ddzwf": float(qacc[front_wheel_dof]),
            "ddzwr": float(qacc[rear_wheel_dof]),
            "Uaf": float(uaf),
            "Uar": float(uar),
        }

    def _base_obs(self) -> np.ndarray:
        d = self.last_derived
        return np.asarray(
            [
                *self._state(),
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
        preview = self.road.preview(self.t, self.speed, self.wheelbase, self.preview_steps, self.preview_dt)
        return {
            "time": self.t,
            "state": self._state(),
            "base_obs": self._base_obs(),
            "road": self.last_road.copy(),
            "road_preview": preview,
            "action": self.last_action.copy(),
            "derived": dict(self.last_derived),
            "reward": float(reward),
            "unsafe": bool(unsafe),
            "scenario": self.scenario.get("name", ""),
            "engine": "mujoco",
        }

    def _susp_rest_length(self) -> float:
        return 0.45

    def _tire_rest_length(self) -> float:
        return 0.30

    def _settle_static_equilibrium(self):
        settle_seconds = float(self.config.get("mujoco", {}).get("settle_seconds", 1.5))
        settle_steps = max(1, int(round(settle_seconds / self.dt)))
        for _ in range(settle_steps):
            self._update_road(0.0)
            self.mj_data.ctrl[:] = 0.0
            mujoco.mj_step(self.mj_model, self.mj_data)
        self.mj_data.qvel[:] = 0.0
        self.mj_data.ctrl[:] = 0.0

    def _capture_equilibrium(self):
        p = self.params
        self._eq_state = self._physical_state()
        zb, _, theta, _, zwf, _, zwr, _ = self._eq_state
        self._eq_front_susp = float(zb + p.a_real * np.sin(theta) - zwf)
        self._eq_rear_susp = float(zb - p.b_real * np.sin(theta) - zwr)
        self._eq_front_tire = float(zwf)
        self._eq_rear_tire = float(zwr)

    def _xml(self) -> str:
        p = self.params
        half_length = 0.5 * (p.a_real + p.b_real)
        chassis_center = 0.5 * (p.a_real - p.b_real)
        susp_rest = self._susp_rest_length()
        tire_rest = self._tire_rest_length()
        return f"""
<mujoco model="industrial_half_car">
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
    <material name="ground" texture="grid" texrepeat="12 2" reflectance="0.04"/>
    <material name="body" rgba="0.04 0.10 0.18 1"/>
    <material name="wheel" rgba="0.94 0.42 0.10 1"/>
    <material name="road" rgba="0.18 0.18 0.18 1"/>
    <material name="link" rgba="0.10 0.35 0.80 1"/>
  </asset>
  <worldbody>
    <light pos="0 -4 5" dir="0 1 -1"/>
    <geom name="ground" type="plane" size="6 2 .05" material="ground"/>
    <body name="chassis" pos="0 0 .75">
      <joint name="chassis_z" type="slide" axis="0 0 1" limited="true" range="-1.0 2.0"/>
      <joint name="chassis_pitch" type="hinge" axis="0 1 0" limited="true" range="-0.5 0.5"/>
      <inertial pos="0 0 0" mass="{p.mb_real:.9f}" diaginertia="{p.Ip_real:.9f} {p.Ip_real:.9f} {p.Ip_real:.9f}"/>
      <geom name="body_box" type="box" pos="{chassis_center:.6f} 0 0" size="{half_length:.6f} .32 .11" material="body"/>
      <site name="front_body_site" pos="{p.a_real:.6f} 0 -.08" size=".025"/>
      <site name="rear_body_site" pos="{-p.b_real:.6f} 0 -.08" size=".025"/>
    </body>
    <body name="front_wheel" pos="{p.a_real:.6f} 0 .30">
      <joint name="front_wheel_z" type="slide" axis="0 0 1" limited="true" range="-1.0 2.0"/>
      <inertial pos="0 0 0" mass="{p.mwf:.9f}" diaginertia="2 2 2"/>
      <geom name="front_wheel_geom" type="cylinder" size=".26 .08" euler="1.57079632679 0 0" material="wheel"/>
      <site name="front_wheel_site" pos="0 0 0" size=".025"/>
    </body>
    <body name="rear_wheel" pos="{-p.b_real:.6f} 0 .30">
      <joint name="rear_wheel_z" type="slide" axis="0 0 1" limited="true" range="-1.0 2.0"/>
      <inertial pos="0 0 0" mass="{p.mwr:.9f}" diaginertia="2 2 2"/>
      <geom name="rear_wheel_geom" type="cylinder" size=".26 .08" euler="1.57079632679 0 0" material="wheel"/>
      <site name="rear_wheel_site" pos="0 0 0" size=".025"/>
    </body>
    <body name="front_road" mocap="true" pos="{p.a_real:.6f} 0 0">
      <geom name="front_road_patch" type="box" size=".35 .42 .025" pos="0 0 -.025" material="road"/>
      <site name="front_road_site" pos="0 0 0" size=".02"/>
    </body>
    <body name="rear_road" mocap="true" pos="{-p.b_real:.6f} 0 0">
      <geom name="rear_road_patch" type="box" size=".35 .42 .025" pos="0 0 -.025" material="road"/>
      <site name="rear_road_site" pos="0 0 0" size=".02"/>
    </body>
    <camera name="side" pos="0 -5 1.35" xyaxes="1 0 0 0 0 1"/>
  </worldbody>
  <tendon>
    <spatial name="front_susp" stiffness="{p.kf1_real:.9f}" damping="{p.be_real:.9f}" springlength="{susp_rest:.9f}">
      <site site="front_body_site"/>
      <site site="front_wheel_site"/>
    </spatial>
    <spatial name="rear_susp" stiffness="{p.kr1_real:.9f}" damping="{p.be_real:.9f}" springlength="{susp_rest:.9f}">
      <site site="rear_body_site"/>
      <site site="rear_wheel_site"/>
    </spatial>
    <spatial name="front_tire" stiffness="{p.kf2:.9f}" damping="{p.bf2:.9f}" springlength="{tire_rest:.9f}">
      <site site="front_wheel_site"/>
      <site site="front_road_site"/>
    </spatial>
    <spatial name="rear_tire" stiffness="{p.kr2:.9f}" damping="{p.br2:.9f}" springlength="{tire_rest:.9f}">
      <site site="rear_wheel_site"/>
      <site site="rear_road_site"/>
    </spatial>
  </tendon>
  <actuator>
    <motor name="front_active" tendon="front_susp" gear="1" ctrllimited="true" ctrlrange="{-p.fmax:.9f} {p.fmax:.9f}"/>
    <motor name="rear_active" tendon="rear_susp" gear="1" ctrllimited="true" ctrlrange="{-p.fmax:.9f} {p.fmax:.9f}"/>
  </actuator>
</mujoco>
"""
