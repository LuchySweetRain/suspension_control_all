from __future__ import annotations

import numpy as np

from .base import Controller
from models.half_car import HalfCarModel

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    minimize = None


class MPCController(Controller):
    name = "MPC"

    def __init__(self, model: HalfCarModel, config: dict):
        self.model = model
        self.config = config
        self.horizon = int(config["mpc"]["horizon"])
        self.dt = float(config["control_dt"])
        self.force_limit = float(config["force_limit"])
        self.max_iter = int(config["mpc"].get("max_iter", 40))
        self.replan_interval = int(config["mpc"].get("replan_interval", 1))
        self.control_weight = float(config["mpc"].get("control_weight", 1e-7))
        self.delta_weight = float(config["mpc"].get("delta_weight", 1e-8))
        self.prev_solution = np.zeros((self.horizon, 2), dtype=np.float64)
        self.step_count = 0

    def reset(self):
        self.prev_solution[:] = 0.0
        self.step_count = 0

    def compute_action(self, obs: np.ndarray, info: dict) -> np.ndarray:
        if minimize is None:
            raise RuntimeError("scipy is required for MPCController")
        if self.step_count % self.replan_interval != 0:
            action = self.prev_solution[0].copy()
            self.prev_solution = np.roll(self.prev_solution, -1, axis=0)
            self.prev_solution[-1] = self.prev_solution[-2]
            self.step_count += 1
            return np.clip(action, -self.force_limit, self.force_limit)

        preview = np.asarray(info["road_preview"], dtype=np.float64)
        x0 = np.asarray(info["state"], dtype=np.float64)
        current_road = np.asarray(info["road"], dtype=np.float64)
        z_current = current_road[[0, 2]]
        z_seq = np.vstack([z_current, preview[: self.horizon]])

        def cost(flat_u):
            u_seq = flat_u.reshape(self.horizon, 2)
            x = x0.copy()
            total = 0.0
            last_u = np.zeros(2)
            for k in range(self.horizon):
                dzf = (z_seq[k + 1, 0] - z_seq[k, 0]) / self.dt
                dzr = (z_seq[k + 1, 1] - z_seq[k, 1]) / self.dt
                road = np.asarray([z_seq[k, 0], dzf, z_seq[k, 1], dzr])
                u = np.clip(u_seq[k], -self.force_limit, self.force_limit)
                d = self.model.derived(x, road, u)
                total += (
                    d["ddzb"] ** 2
                    + 0.4 * d["ddtheta"] ** 2
                    + 30.0 * (d["delta_yf"] ** 2 + d["delta_yr"] ** 2)
                    + 1e-8 * (d["Fpf"] ** 2 + d["Fpr"] ** 2)
                    + self.control_weight * np.sum(u**2)
                    + self.delta_weight * np.sum((u - last_u) ** 2)
                )
                x = self.model.rk4_step(x, u, road, self.dt)
                last_u = u
            return float(total)

        x_init = np.roll(self.prev_solution, -1, axis=0)
        x_init[-1] = x_init[-2]
        bounds = [(-self.force_limit, self.force_limit)] * (self.horizon * 2)
        result = minimize(
            cost,
            x_init.reshape(-1),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": 1e-5},
        )
        if result.success or np.isfinite(result.fun):
            self.prev_solution = result.x.reshape(self.horizon, 2)
        action = self.prev_solution[0].copy()
        self.prev_solution = np.roll(self.prev_solution, -1, axis=0)
        self.prev_solution[-1] = self.prev_solution[-2]
        self.step_count += 1
        return np.clip(action, -self.force_limit, self.force_limit)
