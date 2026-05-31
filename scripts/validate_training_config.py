from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from roads.road_profiles import RoadProfileFactory


def _error(errors: list[str], message: str):
    errors.append(message)


def _expected_dims(config: dict[str, Any]) -> tuple[int, int, int]:
    full_cfg = config.get("full_car", {})
    action_mode = str(full_cfg.get("action_mode", "axle")).lower()
    preview_mode = str(full_cfg.get("preview_mode", "axle")).lower()
    act_dim = 4 if action_mode in {"corner", "four_corner"} else 2
    preview_token_dim = 4 if preview_mode in {"corner", "four_corner"} else 2
    history_cfg = dict(config.get("observation", {}).get("history", {}))
    history_steps = int(history_cfg.get("steps", 0)) if history_cfg.get("enabled", False) else 0
    obs_dim_base = 14 + history_steps * (14 + 2 * act_dim)
    return act_dim, preview_token_dim, obs_dim_base


def validate_config(config_path: Path, check_roads: bool = True) -> dict:
    config = load_config(config_path)
    errors: list[str] = []
    warnings: list[str] = []
    engine = str(config.get("environment", {}).get("engine", "python")).lower()
    scenarios = list(config.get("scenarios", []))
    names = [str(s.get("name", "")) for s in scenarios]

    for key in ("dt", "control_dt", "episode_seconds", "force_limit", "preview", "rl", "scenarios"):
        if key not in config:
            _error(errors, f"missing required config key: {key}")
    if "dt" in config and "control_dt" in config:
        dt = float(config["dt"])
        control_dt = float(config["control_dt"])
        if dt <= 0.0 or control_dt <= 0.0:
            _error(errors, "dt and control_dt must be positive")
        elif abs(round(control_dt / dt) - control_dt / dt) > 1e-6:
            _error(errors, f"control_dt ({control_dt}) must be an integer multiple of dt ({dt})")
    if not scenarios:
        _error(errors, "scenarios must contain at least one scenario")
    if len(set(names)) != len(names):
        _error(errors, "scenario names must be unique")

    act_dim, preview_token_dim, obs_dim_base = _expected_dims(config)
    rl_cfg = config.get("rl", {})
    if int(rl_cfg.get("obs_dim_base", obs_dim_base)) != obs_dim_base:
        _error(errors, f"rl.obs_dim_base must be {obs_dim_base} for current observation history/action mode")
    if int(rl_cfg.get("act_dim", act_dim)) != act_dim:
        _error(errors, f"rl.act_dim must be {act_dim} for current action mode")
    if int(rl_cfg.get("preview_token_dim", preview_token_dim)) != preview_token_dim:
        _error(errors, f"rl.preview_token_dim must be {preview_token_dim} for current preview mode")
    preview_steps = int(config.get("preview", {}).get("steps", 0))
    if preview_steps <= 0:
        _error(errors, "preview.steps must be positive")
    obs_cfg = config.get("observation", {})
    obs_mode = str(obs_cfg.get("mode", "privileged")).lower()
    if obs_mode not in {"privileged", "estimated", "noisy"}:
        _error(errors, "observation.mode must be privileged, estimated, or noisy")
    history_cfg = dict(obs_cfg.get("history", {}))
    if history_cfg.get("enabled", False) and int(history_cfg.get("steps", 0)) <= 0:
        _error(errors, "observation.history.steps must be positive when history is enabled")

    if engine in {"mujoco_full", "mujoco_full_car", "full_car"}:
        safety = config.get("safety_limits", {})
        for key in ("max_suspension_travel", "max_pitch", "max_roll", "max_wheel_displacement"):
            if float(safety.get(key, 0.0)) <= 0.0:
                _error(errors, f"safety_limits.{key} must be positive for full-car MuJoCo training")
        actuator = config.get("actuator", {})
        if actuator.get("enabled", False):
            if float(actuator.get("time_constant", 0.0)) < 0.0:
                _error(errors, "actuator.time_constant must be non-negative")
            if float(actuator.get("rate_limit", 0.0)) <= 0.0:
                _error(errors, "actuator.rate_limit must be positive when actuator.enabled=true")

    sampling = config.get("scenario_sampling", {})
    sampling_mode = str(sampling.get("mode", "cycle")).lower()
    if sampling_mode not in {"cycle", "uniform", "weighted"}:
        _error(errors, "scenario_sampling.mode must be cycle, uniform, or weighted")
    weights = sampling.get("weights")
    if sampling_mode == "weighted":
        if isinstance(weights, dict):
            missing = [name for name in names if name not in weights]
            unknown = [name for name in weights if name not in names]
            if missing:
                _error(errors, f"scenario_sampling.weights missing scenarios: {missing}")
            if unknown:
                _error(errors, f"scenario_sampling.weights contains unknown scenarios: {unknown}")
            if sum(float(v) for v in weights.values()) <= 0.0:
                _error(errors, "scenario_sampling.weights must sum to a positive value")
        elif isinstance(weights, list):
            if len(weights) != len(scenarios):
                _error(errors, "scenario_sampling.weights list length must match scenario count")
        else:
            warnings.append("weighted scenario_sampling has no weights; uniform weights will be used")
    known_names = set(names)
    for phase in sampling.get("curriculum", []):
        for name in phase.get("scenarios", []):
            if str(name) not in known_names:
                _error(errors, f"curriculum references unknown scenario: {name}")

    road_results = []
    if check_roads:
        for scenario in scenarios:
            item = {"name": scenario.get("name", ""), "type": scenario.get("type", "iso"), "ok": True, "error": None}
            try:
                if str(scenario.get("type", "")).lower() in {"csv", "file"}:
                    path = Path(str(scenario.get("path", "")))
                    if not path.is_absolute():
                        path = config_path.parent / path
                    if not path.is_file():
                        raise FileNotFoundError(f"road file does not exist: {path}")
                RoadProfileFactory.create(scenario, duration=float(config.get("episode_seconds", 1.0)), dt=float(config.get("dt", 0.001)))
            except Exception as exc:  # noqa: BLE001 - config validator reports all errors as data.
                item["ok"] = False
                item["error"] = str(exc)
                _error(errors, f"scenario {scenario.get('name', '')} road profile invalid: {exc}")
            road_results.append(item)

    return {
        "config": str(config_path.resolve()),
        "engine": engine,
        "scenario_count": len(scenarios),
        "expected_act_dim": act_dim,
        "expected_preview_token_dim": preview_token_dim,
        "expected_obs_dim_base": obs_dim_base,
        "errors": errors,
        "warnings": warnings,
        "roads": road_results,
        "passed": not errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_full_car_corner.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-road-check", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    report = validate_config(config_path, check_roads=not args.skip_road_check)
    text = json.dumps(report, indent=2)
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
