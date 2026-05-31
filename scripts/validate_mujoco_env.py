from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from scripts.train_rl import make_env


def _sample_action(env, mode: str) -> np.ndarray:
    if mode == "random":
        return np.asarray(env.action_space.sample(), dtype=np.float32)
    return np.zeros(env.action_space.shape, dtype=np.float32)


def validate_environment(
    config_path: Path,
    out_path: Path | None = None,
    max_steps: int = 200,
    action_mode: str = "passive",
    render: bool = False,
    max_unsafe_fraction: float = 0.0,
    require_full_horizon: bool = False,
) -> dict:
    config = load_config(config_path)
    scenarios = list(config.get("scenarios", []))
    if not scenarios:
        raise ValueError("Config must define at least one scenario.")

    report = {
        "config": str(config_path.resolve()),
        "engine": str(config.get("environment", {}).get("engine", "python")),
        "action_mode": action_mode,
        "max_steps": int(max_steps),
        "max_unsafe_fraction": float(max_unsafe_fraction),
        "require_full_horizon": bool(require_full_horizon),
        "render": bool(render),
        "passed": True,
        "scenarios": [],
    }
    total_steps = 0
    start_all = time.perf_counter()
    for scenario in scenarios:
        env = make_env(config, scenario=scenario, use_preview=True)
        scenario_report = {
            "name": scenario.get("name", ""),
            "environment_metadata": None,
            "steps": 0,
            "terminated": False,
            "truncated": False,
            "finite": True,
            "unsafe_steps": 0,
            "safety_violations": {},
            "return": 0.0,
            "max_abs_observation": 0.0,
            "max_abs_action": 0.0,
            "max_abs_command": 0.0,
            "render_shape": None,
            "errors": [],
        }
        try:
            obs, info = env.reset(seed=int(config.get("seed", 42)))
            if hasattr(env, "environment_metadata"):
                scenario_report["environment_metadata"] = env.environment_metadata()
            if not np.all(np.isfinite(obs)):
                scenario_report["finite"] = False
                scenario_report["errors"].append("reset observation contains non-finite values")
            if "road_preview" in info and not np.all(np.isfinite(info["road_preview"])):
                scenario_report["finite"] = False
                scenario_report["errors"].append("reset road_preview contains non-finite values")
            for _ in range(max_steps):
                action = _sample_action(env, action_mode)
                obs, reward, terminated, truncated, info = env.step(action)
                scenario_report["steps"] += 1
                total_steps += 1
                scenario_report["return"] += float(reward)
                scenario_report["terminated"] = bool(terminated)
                scenario_report["truncated"] = bool(truncated)
                scenario_report["unsafe_steps"] += int(info.get("unsafe", False))
                for violation in info.get("safety", {}).get("violations", []):
                    scenario_report["safety_violations"][violation] = (
                        scenario_report["safety_violations"].get(violation, 0) + 1
                    )
                scenario_report["max_abs_observation"] = max(
                    scenario_report["max_abs_observation"], float(np.max(np.abs(obs)))
                )
                actual = np.asarray(info.get("corner_action", info.get("action", action)), dtype=np.float64)
                command = np.asarray(info.get("command_corner_action", action), dtype=np.float64)
                scenario_report["max_abs_action"] = max(scenario_report["max_abs_action"], float(np.max(np.abs(actual))))
                scenario_report["max_abs_command"] = max(
                    scenario_report["max_abs_command"], float(np.max(np.abs(command)))
                )
                finite = np.all(np.isfinite(obs)) and np.isfinite(reward)
                if "road_preview" in info:
                    finite = finite and np.all(np.isfinite(info["road_preview"]))
                if not finite:
                    scenario_report["finite"] = False
                    scenario_report["errors"].append(f"non-finite value at step {scenario_report['steps']}")
                    break
                if render and scenario_report["render_shape"] is None:
                    frame = env.render()
                    scenario_report["render_shape"] = list(frame.shape)
                    if frame.ndim != 3 or frame.shape[2] != 3:
                        scenario_report["errors"].append("render frame is not HxWx3")
                if terminated or truncated:
                    break
        finally:
            env.close()
        unsafe_fraction = scenario_report["unsafe_steps"] / max(1, scenario_report["steps"])
        scenario_report["unsafe_fraction"] = float(unsafe_fraction)
        if unsafe_fraction > max_unsafe_fraction:
            scenario_report["errors"].append(
                f"unsafe fraction {unsafe_fraction:.3f} exceeds threshold {max_unsafe_fraction:.3f}"
            )
        if require_full_horizon and scenario_report["steps"] < max_steps:
            scenario_report["errors"].append(
                f"rollout ended at {scenario_report['steps']} steps before requested horizon {max_steps}"
            )
        scenario_report["passed"] = scenario_report["finite"] and not scenario_report["errors"]
        report["passed"] = report["passed"] and scenario_report["passed"]
        report["scenarios"].append(scenario_report)
    elapsed = time.perf_counter() - start_all
    report["total_steps"] = total_steps
    report["elapsed_seconds"] = elapsed
    report["steps_per_second"] = float(total_steps / elapsed) if elapsed > 0.0 else 0.0

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_full_car_corner.yaml")
    parser.add_argument("--out", default="results/mujoco_env_validation.json")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--action-mode", choices=["passive", "random"], default="passive")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-unsafe-fraction", type=float, default=0.0)
    parser.add_argument("--require-full-horizon", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    report = validate_environment(
        config_path=config_path,
        out_path=Path(args.out),
        max_steps=args.max_steps,
        action_mode=args.action_mode,
        render=args.render,
        max_unsafe_fraction=args.max_unsafe_fraction,
        require_full_horizon=args.require_full_horizon,
    )
    status = "PASSED" if report["passed"] else "FAILED"
    print(f"{status}: {len(report['scenarios'])} scenarios, {report['total_steps']} steps, {report['steps_per_second']:.1f} steps/s")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
