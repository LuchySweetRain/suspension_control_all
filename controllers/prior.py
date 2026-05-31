from __future__ import annotations

import numpy as np

from .mpc import MPCController
from .pid import PIDController
from .spdf import SPDFController


def adapt_action_to_env(action: np.ndarray, env) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    act_dim = int(env.action_space.shape[0]) if hasattr(env, "action_space") else len(action)
    force_limit = float(getattr(env, "force_limit", np.inf))
    if action.shape == (act_dim,):
        return np.clip(action, -force_limit, force_limit).astype(np.float32)
    if action.shape == (2,) and act_dim == 4:
        adapted = np.asarray([0.5 * action[0], 0.5 * action[0], 0.5 * action[1], 0.5 * action[1]], dtype=np.float64)
        return np.clip(adapted, -force_limit, force_limit).astype(np.float32)
    if action.shape == (4,) and act_dim == 2:
        adapted = np.asarray([action[0] + action[1], action[2] + action[3]], dtype=np.float64)
        return np.clip(adapted, -force_limit, force_limit).astype(np.float32)
    raise ValueError(f"Cannot adapt prior action shape {action.shape} to env action shape {(act_dim,)}")


def make_prior_controller(name: str, env, config: dict):
    key = name.lower()
    if key in {"passive", "zero", "none"}:
        return None
    if key == "pid":
        return PIDController(env.params, env.control_dt, env.force_limit)
    if key == "spdf":
        return SPDFController(env.params, env.control_dt)
    if key == "mpc":
        if not hasattr(env, "model"):
            raise ValueError("MPC prior requires the Python HalfCarEnv model.")
        return MPCController(env.model, config)
    if key in {"full_car_mpc_lite", "mpc_lite", "lqr", "lpv"}:
        from .reduced_full_car import ReducedFullCarPreviewController

        return ReducedFullCarPreviewController(env, config)
    raise ValueError(f"Unknown prior controller: {name}")


def compute_prior_action(controller, env, obs: np.ndarray, info: dict) -> np.ndarray:
    if controller is None:
        return np.zeros(env.action_space.shape, dtype=np.float32)
    action = controller.compute_action(obs, info)
    return adapt_action_to_env(action, env)

