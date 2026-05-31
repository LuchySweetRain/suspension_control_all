from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from scripts.validate_mujoco_env import validate_environment


DEFAULT_CASES = [
    {
        "name": "nominal_passive",
        "action_mode": "passive",
        "overrides": {
            "domain_randomization": {"enabled": False},
            "preview_error": {"enabled": False},
            "actuator": {"enabled": False},
        },
    },
    {
        "name": "nominal_random",
        "action_mode": "random",
        "overrides": {
            "domain_randomization": {"enabled": False},
            "preview_error": {"enabled": False},
            "actuator": {"enabled": False},
        },
    },
    {
        "name": "sensor_actuator_random",
        "action_mode": "random",
        "overrides": {
            "domain_randomization": {"enabled": False},
            "preview_error": {
                "enabled": True,
                "delay_steps": 2,
                "height_noise_std": 0.003,
                "bias_std": 0.002,
                "dropout_prob": 0.05,
                "scale_error_std": 0.05,
            },
            "actuator": {"enabled": True, "time_constant": 0.03, "rate_limit": 120000.0},
        },
    },
    {
        "name": "domain_randomized_random",
        "action_mode": "random",
        "overrides": {
            "domain_randomization": {
                "enabled": True,
                "speed_scale": 0.20,
                "mass_scale": 0.12,
                "inertia_scale": 0.18,
                "suspension_stiffness_scale": 0.25,
                "suspension_damping_scale": 0.25,
                "tire_stiffness_scale": 0.18,
                "tire_damping_scale": 0.18,
                "road_amplitude_scale": 0.30,
            }
        },
    },
]


def _deep_update(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def run_robustness_matrix(
    config_path: Path,
    out_dir: Path,
    max_steps: int = 200,
    scenario_limit: int | None = None,
    max_unsafe_fraction: float = 0.0,
    require_full_horizon: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    case_dir = out_dir / "case_configs"
    case_dir.mkdir(exist_ok=True)
    report_dir = out_dir / "case_reports"
    report_dir.mkdir(exist_ok=True)

    base_config = load_config(config_path)
    if scenario_limit is not None:
        base_config["scenarios"] = list(base_config.get("scenarios", []))[:scenario_limit]

    matrix = {
        "base_config": str(config_path.resolve()),
        "max_steps": int(max_steps),
        "scenario_limit": scenario_limit,
        "max_unsafe_fraction": float(max_unsafe_fraction),
        "require_full_horizon": bool(require_full_horizon),
        "passed": True,
        "cases": [],
    }
    for case in DEFAULT_CASES:
        config = _deep_update(base_config, case["overrides"])
        case_config_path = case_dir / f"{case['name']}.yaml"
        with case_config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        case_report_path = report_dir / f"{case['name']}.json"
        report = validate_environment(
            config_path=case_config_path,
            out_path=case_report_path,
            max_steps=max_steps,
            action_mode=case["action_mode"],
            render=False,
            max_unsafe_fraction=max_unsafe_fraction,
            require_full_horizon=require_full_horizon,
        )
        summary = {
            "name": case["name"],
            "action_mode": case["action_mode"],
            "config": str(case_config_path),
            "report": str(case_report_path),
            "passed": bool(report["passed"]),
            "total_steps": int(report["total_steps"]),
            "steps_per_second": float(report["steps_per_second"]),
            "failed_scenarios": [
                item["name"] for item in report["scenarios"] if not item.get("passed", False)
            ],
        }
        matrix["passed"] = matrix["passed"] and summary["passed"]
        matrix["cases"].append(summary)

    matrix_path = out_dir / "robustness_matrix.json"
    with matrix_path.open("w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)
    matrix["matrix_report"] = str(matrix_path)
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_full_car_corner.yaml")
    parser.add_argument("--out", default="results/mujoco_robustness_matrix")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--scenario-limit", type=int, default=None)
    parser.add_argument("--max-unsafe-fraction", type=float, default=0.0)
    parser.add_argument("--require-full-horizon", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    matrix = run_robustness_matrix(
        config_path=config_path,
        out_dir=out_dir,
        max_steps=args.max_steps,
        scenario_limit=args.scenario_limit,
        max_unsafe_fraction=args.max_unsafe_fraction,
        require_full_horizon=args.require_full_horizon,
    )
    status = "PASSED" if matrix["passed"] else "FAILED"
    print(f"{status}: {len(matrix['cases'])} validation cases")
    print(f"Matrix report: {matrix['matrix_report']}")
    if not matrix["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
