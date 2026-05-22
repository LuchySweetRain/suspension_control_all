from __future__ import annotations

from typing import Any

import numpy as np

from .half_car_env import HalfCarEnv


try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None


def _quat_from_pitch(theta: float) -> np.ndarray:
    half = 0.5 * float(theta)
    return np.asarray([np.cos(half), 0.0, np.sin(half), 0.0], dtype=np.float64)


class MuJoCoHalfCarEnv(HalfCarEnv):
    """Half-car environment with a MuJoCo visual scene.

    The suspension dynamics, observations, rewards, and termination logic come
    from HalfCarEnv. MuJoCo is used as a rendering backend so controller
    comparisons stay numerically consistent with the existing RK4 model.
    """

    def __init__(
        self,
        config: dict[str, Any],
        scenario: dict[str, Any] | None = None,
        use_preview: bool = True,
        render_mode: str | None = "rgb_array",
        width: int = 1280,
        height: int = 720,
        vertical_scale: float = 12.0,
    ):
        if mujoco is None:
            raise ImportError("MuJoCo is not installed. Install it with: python -m pip install mujoco glfw")
        super().__init__(config, scenario=scenario, use_preview=use_preview)
        self.render_mode = render_mode
        self.width = int(width)
        self.height = int(height)
        self.vertical_scale = float(vertical_scale)
        self.mj_model = mujoco.MjModel.from_xml_string(self._xml())
        self.mj_data = mujoco.MjData(self.mj_model)
        self._renderer = None
        self._mocap_ids = {
            name: mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("chassis", "front_wheel", "rear_wheel", "front_road", "rear_road")
        }
        self._sync_scene()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._sync_scene()
        return obs, info

    def step(self, action):
        result = super().step(action)
        self._sync_scene()
        return result

    def render(self):
        self._sync_scene()
        if self.render_mode == "human":
            raise NotImplementedError(
                "Use scripts/view_mujoco_env.py for an interactive viewer, or render_mode='rgb_array'."
            )
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.mj_model, self.height, self.width)
        self._renderer.update_scene(self.mj_data, camera="side")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _sync_scene(self):
        road = np.asarray(self.road.pair(self.t, self.speed, self.wheelbase), dtype=np.float64)
        zdf, _, zdr, _ = road
        zb, _, theta, _, zwf, _, zwr, _ = self.state
        p = self.params

        chassis_z = 0.95 + zb * self.vertical_scale
        front_x = p.a_real
        rear_x = -p.b_real
        front_wheel_z = 0.35 + zwf * self.vertical_scale
        rear_wheel_z = 0.35 + zwr * self.vertical_scale
        front_road_z = zdf * self.vertical_scale
        rear_road_z = zdr * self.vertical_scale

        self._set_mocap("chassis", [0.0, 0.0, chassis_z], _quat_from_pitch(theta))
        self._set_mocap("front_wheel", [front_x, 0.0, front_wheel_z], [1.0, 0.0, 0.0, 0.0])
        self._set_mocap("rear_wheel", [rear_x, 0.0, rear_wheel_z], [1.0, 0.0, 0.0, 0.0])
        self._set_mocap("front_road", [front_x, 0.0, front_road_z - 0.03], [1.0, 0.0, 0.0, 0.0])
        self._set_mocap("rear_road", [rear_x, 0.0, rear_road_z - 0.03], [1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _set_mocap(self, body_name: str, pos, quat):
        body_id = self._mocap_ids[body_name]
        mocap_id = self.mj_model.body_mocapid[body_id]
        self.mj_data.mocap_pos[mocap_id] = np.asarray(pos, dtype=np.float64)
        self.mj_data.mocap_quat[mocap_id] = np.asarray(quat, dtype=np.float64)

    def _xml(self) -> str:
        p = self.params
        half_length = 0.5 * (p.a_real + p.b_real)
        chassis_center = 0.5 * (p.a_real - p.b_real)
        return f"""
<mujoco model="half_car_visual">
  <compiler angle="radian"/>
  <option timestep="{self.dt}" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="{self.width}" offheight="{self.height}"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2"/>
    <rgba haze="0.95 0.97 1 1"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .25 .28" rgb2=".3 .35 .38" width="256" height="256"/>
    <material name="ground" texture="grid" texrepeat="8 2" reflectance="0.05"/>
    <material name="body" rgba="0.05 0.12 0.20 1"/>
    <material name="wheel" rgba="0.95 0.45 0.12 1"/>
    <material name="road" rgba="0.25 0.25 0.25 1"/>
    <material name="spring" rgba="0.1 0.35 0.8 1"/>
  </asset>
  <worldbody>
    <light pos="0 -3 4" dir="0 1 -1"/>
    <geom name="ground" type="plane" size="5 2 .05" material="ground"/>
    <body name="chassis" mocap="true" pos="0 0 .95">
      <geom name="body_box" type="box" pos="{chassis_center:.6f} 0 0" size="{half_length:.6f} .32 .12" material="body"/>
      <geom name="front_strut" type="capsule" fromto="{p.a_real:.6f} 0 -.08 {p.a_real:.6f} 0 -.58" size=".025" material="spring"/>
      <geom name="rear_strut" type="capsule" fromto="{-p.b_real:.6f} 0 -.08 {-p.b_real:.6f} 0 -.58" size=".025" material="spring"/>
    </body>
    <body name="front_wheel" mocap="true" pos="{p.a_real:.6f} 0 .35">
      <geom name="front_wheel_geom" type="cylinder" size=".26 .08" euler="1.57079632679 0 0" material="wheel"/>
    </body>
    <body name="rear_wheel" mocap="true" pos="{-p.b_real:.6f} 0 .35">
      <geom name="rear_wheel_geom" type="cylinder" size=".26 .08" euler="1.57079632679 0 0" material="wheel"/>
    </body>
    <body name="front_road" mocap="true" pos="{p.a_real:.6f} 0 -.03">
      <geom name="front_road_patch" type="box" size=".34 .38 .03" material="road"/>
    </body>
    <body name="rear_road" mocap="true" pos="{-p.b_real:.6f} 0 -.03">
      <geom name="rear_road_patch" type="box" size=".34 .38 .03" material="road"/>
    </body>
    <camera name="side" pos="0 -5 1.45" xyaxes="1 0 0 0 0 1"/>
  </worldbody>
</mujoco>
"""
