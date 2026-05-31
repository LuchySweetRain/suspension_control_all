from __future__ import annotations

import numpy as np

from .base import Controller


class ReducedFullCarPreviewController(Controller):
    """Fast reduced full-car preview baseline.

    This is an MPC-lite/LPV-style baseline rather than a constrained optimizer:
    it uses a linear heave-pitch-roll feedback law plus preview feedforward.
    The controller is intentionally cheap enough for benchmark sweeps and
    residual-RL priors.
    """

    name = "FULL_CAR_MPC_LITE"

    def __init__(self, env, config: dict):
        self.force_limit = float(config["force_limit"])
        cfg = dict(config.get("reduced_full_car_mpc", {}))
        self.k_z = float(cfg.get("k_z", 1200.0))
        self.k_dz = float(cfg.get("k_dz", 1800.0))
        self.k_pitch = float(cfg.get("k_pitch", 950.0))
        self.k_dpitch = float(cfg.get("k_dpitch", 700.0))
        self.k_roll = float(cfg.get("k_roll", 850.0))
        self.k_droll = float(cfg.get("k_droll", 650.0))
        self.k_preview = float(cfg.get("k_preview", 18000.0))
        self.track_width = float(getattr(env, "track_width", config.get("full_car", {}).get("track_width", 1.62)))
        self.a = float(getattr(env.params, "a_real", getattr(env.params, "a", 1.2)))
        self.b = float(getattr(env.params, "b_real", getattr(env.params, "b", 1.5)))

    def reset(self):
        pass

    def compute_action(self, obs: np.ndarray, info: dict) -> np.ndarray:
        state = np.asarray(info["state"], dtype=np.float64)
        z, dz, pitch, dpitch, roll, droll = state[:6]
        preview = np.asarray(info.get("road_preview", np.zeros((1, 4))), dtype=np.float64)
        if preview.ndim == 2 and preview.shape[1] == 4:
            road = preview[0]
        elif preview.ndim == 2 and preview.shape[1] == 2:
            road = np.asarray([preview[0, 0], preview[0, 0], preview[0, 1], preview[0, 1]], dtype=np.float64)
        else:
            road = np.zeros(4, dtype=np.float64)

        half_track = 0.5 * self.track_width
        corner_body = np.asarray(
            [
                z + self.a * pitch + half_track * roll,
                z + self.a * pitch - half_track * roll,
                z - self.b * pitch + half_track * roll,
                z - self.b * pitch - half_track * roll,
            ],
            dtype=np.float64,
        )
        corner_vel = np.asarray(
            [
                dz + self.a * dpitch + half_track * droll,
                dz + self.a * dpitch - half_track * droll,
                dz - self.b * dpitch + half_track * droll,
                dz - self.b * dpitch - half_track * droll,
            ],
            dtype=np.float64,
        )
        heave_term = -self.k_z * corner_body - self.k_dz * corner_vel
        pitch_term = np.asarray([-self.k_pitch * pitch - self.k_dpitch * dpitch] * 2 + [self.k_pitch * pitch + self.k_dpitch * dpitch] * 2)
        roll_term = np.asarray(
            [
                -self.k_roll * roll - self.k_droll * droll,
                self.k_roll * roll + self.k_droll * droll,
                -self.k_roll * roll - self.k_droll * droll,
                self.k_roll * roll + self.k_droll * droll,
            ],
            dtype=np.float64,
        )
        preview_term = self.k_preview * (road - corner_body)
        action = heave_term + pitch_term + roll_term + preview_term
        return np.clip(action, -self.force_limit, self.force_limit).astype(np.float32)

