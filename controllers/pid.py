from __future__ import annotations

import numpy as np

from .base import Controller, split_generalized_force
from models.half_car import HalfCarParams


class PIDController(Controller):
    name = "PID"

    def __init__(self, params: HalfCarParams, dt: float, fmax: float):
        self.params = params
        self.dt = dt
        self.fmax = fmax
        # Match active_suspension_sim.slx:
        # error = -state, then Kp=50000, Ki=5000, Kd=500000 for both channels.
        self.kp_z = 50000.0
        self.ki_z = 5000.0
        self.kd_z = 500000.0
        self.kp_theta = 50000.0
        self.ki_theta = 5000.0
        self.kd_theta = 500000.0
        self.reset()

    def reset(self):
        self.int_z = 0.0
        self.int_theta = 0.0
        self.prev_z = 0.0
        self.prev_theta = 0.0

    def compute_action(self, obs: np.ndarray, info: dict) -> np.ndarray:
        state = info["state"]
        z = float(state[0])
        theta = float(state[2])
        error_z = -z
        error_theta = -theta
        derivative_z = (error_z - self.prev_z) / self.dt
        derivative_theta = (error_theta - self.prev_theta) / self.dt
        self.int_z += error_z * self.dt
        self.int_theta += error_theta * self.dt
        self.prev_z = error_z
        self.prev_theta = error_theta
        ub = self.kp_z * error_z + self.ki_z * self.int_z + self.kd_z * derivative_z
        utheta = self.kp_theta * error_theta + self.ki_theta * self.int_theta + self.kd_theta * derivative_theta
        return split_generalized_force(ub, utheta, self.params.a, self.params.b, self.fmax)
