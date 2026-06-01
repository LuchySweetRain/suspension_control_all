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


def safety_margin(info: dict) -> float:
    safety = dict(info.get("safety", {}))
    limits = dict(safety.get("limits", {}))
    margins = []
    for value_key, limit_key in (
        ("max_abs_suspension_travel", "max_suspension_travel"),
        ("abs_pitch", "max_pitch"),
        ("abs_roll", "max_roll"),
        ("max_abs_wheel_displacement", "max_wheel_displacement"),
    ):
        limit = float(limits.get(limit_key, 0.0))
        if limit > 0.0:
            value = float(safety.get(value_key, 0.0))
            margins.append(np.clip((limit - value) / limit, 0.0, 1.0))
    margin = float(min(margins)) if margins else 1.0
    if bool(safety.get("unsafe", False)):
        margin = 0.0
    return margin


def residual_gate(info: dict, cfg: dict) -> float:
    gate_cfg = dict(cfg.get("gate", {}))
    base = float(cfg.get("scale", 1.0))
    if not gate_cfg.get("enabled", False):
        return base
    preview_error = dict(info.get("preview_error", {}))
    delay = float(preview_error.get("delay_steps", 0.0))
    noise = float(preview_error.get("height_noise_std", 0.0))
    bias = float(preview_error.get("bias_std", 0.0))
    dropout = float(preview_error.get("dropout_prob", 0.0))
    scale_error = float(preview_error.get("scale_error_std", 0.0))
    preview_penalty = (
        float(gate_cfg.get("delay_weight", 0.05)) * delay
        + float(gate_cfg.get("noise_weight", 80.0)) * noise
        + float(gate_cfg.get("bias_weight", 50.0)) * bias
        + float(gate_cfg.get("dropout_weight", 1.0)) * dropout
        + float(gate_cfg.get("scale_error_weight", 1.0)) * scale_error
    )
    preview_confidence = float(np.exp(-max(0.0, preview_penalty)))

    margin = safety_margin(info)
    deadband = float(gate_cfg.get("safety_deadband", 0.0))
    if deadband > 0.0:
        margin = 0.0 if margin <= deadband else (margin - deadband) / max(1.0 - deadband, 1e-6)
    margin = float(np.clip(margin, 0.0, 1.0)) ** float(gate_cfg.get("safety_power", 1.0))

    min_scale = float(gate_cfg.get("min_scale", 0.0))
    max_scale = float(gate_cfg.get("max_scale", base))
    gated = base * preview_confidence * margin
    return float(np.clip(gated, min_scale, max_scale))


def shield_residual_action(residual_action: np.ndarray, env, info: dict, cfg: dict) -> np.ndarray:
    shield_cfg = dict(cfg.get("shield", {}))
    residual = adapt_action_to_env(residual_action, env).astype(np.float64)
    force_limit = float(getattr(env, "force_limit", np.inf))
    max_fraction = float(shield_cfg.get("max_residual_fraction", 1.0))
    if np.isfinite(force_limit) and max_fraction > 0.0:
        residual = np.clip(residual, -max_fraction * force_limit, max_fraction * force_limit)
    if not shield_cfg.get("enabled", False):
        return residual.astype(np.float32)

    margin = safety_margin(info)
    hard_margin = float(shield_cfg.get("hard_margin", 0.0))
    soft_margin = float(shield_cfg.get("soft_margin", hard_margin))
    if margin <= hard_margin:
        residual *= 0.0
    elif soft_margin > hard_margin and margin < soft_margin:
        residual *= (margin - hard_margin) / max(soft_margin - hard_margin, 1e-6)
    return residual.astype(np.float32)
