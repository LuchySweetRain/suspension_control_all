from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from config import load_config
from envs.half_car_env import HalfCarEnv
from envs.mujoco_full_car_env import MuJoCoFullCarEnv
from envs.mujoco_vehicle_env import MuJoCoVehicleEnv

ROOT = Path(__file__).resolve().parents[1]

GYM_IDS = {
    "half_car": "ActiveSuspensionHalfCar-v0",
    "mujoco_vehicle": "ActiveSuspensionMuJoCoVehicle-v0",
    "mujoco_full_car": "ActiveSuspensionMuJoCoFullCar-v0",
}


def _resolve_config(config: dict[str, Any] | None = None, config_path: str | Path | None = None) -> dict[str, Any]:
    if config is not None:
        return deepcopy(config)
    if config_path is None:
        config_path = ROOT / "configs" / "mujoco_full_car_corner.yaml"
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    return load_config(path)


def _resolve_scenario(
    config: dict[str, Any],
    scenario: dict[str, Any] | None = None,
    scenario_index: int = 0,
    scenario_name: str | None = None,
) -> dict[str, Any]:
    if scenario is not None:
        return scenario
    scenarios = list(config.get("scenarios", []))
    if not scenarios:
        raise ValueError("Config must define at least one scenario.")
    if scenario_name is not None:
        for item in scenarios:
            if item.get("name") == scenario_name:
                return item
        raise KeyError(f"Unknown scenario_name: {scenario_name}")
    return scenarios[int(scenario_index)]


def make_half_car_env(
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    scenario: dict[str, Any] | None = None,
    scenario_index: int = 0,
    scenario_name: str | None = None,
    use_preview: bool = True,
    **kwargs,
) -> HalfCarEnv:
    cfg = _resolve_config(config=config, config_path=config_path)
    return HalfCarEnv(
        cfg,
        scenario=_resolve_scenario(cfg, scenario=scenario, scenario_index=scenario_index, scenario_name=scenario_name),
        use_preview=use_preview,
    )


def make_mujoco_vehicle_env(
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    scenario: dict[str, Any] | None = None,
    scenario_index: int = 0,
    scenario_name: str | None = None,
    use_preview: bool = True,
    **kwargs,
) -> MuJoCoVehicleEnv:
    cfg = _resolve_config(config=config, config_path=config_path)
    return MuJoCoVehicleEnv(
        cfg,
        scenario=_resolve_scenario(cfg, scenario=scenario, scenario_index=scenario_index, scenario_name=scenario_name),
        use_preview=use_preview,
        **kwargs,
    )


def make_mujoco_full_car_env(
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    scenario: dict[str, Any] | None = None,
    scenario_index: int = 0,
    scenario_name: str | None = None,
    use_preview: bool = True,
    **kwargs,
) -> MuJoCoFullCarEnv:
    cfg = _resolve_config(config=config, config_path=config_path)
    return MuJoCoFullCarEnv(
        cfg,
        scenario=_resolve_scenario(cfg, scenario=scenario, scenario_index=scenario_index, scenario_name=scenario_name),
        use_preview=use_preview,
        **kwargs,
    )


def register_gymnasium_envs() -> bool:
    try:
        import gymnasium as gym
    except ImportError:
        return False

    specs = getattr(gym.envs.registration, "registry", {})
    registrations = [
        (GYM_IDS["half_car"], "envs.registration:make_half_car_env"),
        (GYM_IDS["mujoco_vehicle"], "envs.registration:make_mujoco_vehicle_env"),
        (GYM_IDS["mujoco_full_car"], "envs.registration:make_mujoco_full_car_env"),
    ]
    for env_id, entry_point in registrations:
        if env_id not in specs:
            gym.register(id=env_id, entry_point=entry_point)
    return True
