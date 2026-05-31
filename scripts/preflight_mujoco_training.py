from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_vector_env import benchmark_vector_env
from scripts.collect_env_statistics import collect_env_statistics
from scripts.export_mujoco_env_spec import export_env_spec
from scripts.validate_mujoco_env import validate_environment
from scripts.validate_training_config import validate_config


def preflight_mujoco_training(
    config_path: Path,
    out_dir: Path,
    validation_steps: int = 50,
    vector_envs: int = 2,
    vector_steps: int = 20,
    statistics_episodes: int = 2,
    statistics_steps: int = 50,
    action_mode: str = "random",
    max_unsafe_fraction: float = 0.1,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    config_report_path = out_dir / "config_validation.json"
    spec_path = out_dir / "environment_spec.json"
    env_validation_path = out_dir / "rollout_validation.json"
    vector_path = out_dir / "vector_benchmark.json"
    stats_path = out_dir / "env_statistics.json"

    config_report = validate_config(config_path)
    config_report_path.write_text(json.dumps(config_report, indent=2) + "\n", encoding="utf-8")
    spec = export_env_spec(config_path, spec_path)
    env_report = validate_environment(
        config_path=config_path,
        out_path=env_validation_path,
        max_steps=validation_steps,
        action_mode=action_mode,
        max_unsafe_fraction=max_unsafe_fraction,
    )
    vector_report = benchmark_vector_env(
        config_path=config_path,
        out_path=vector_path,
        num_envs=vector_envs,
        steps=vector_steps,
        action_mode=action_mode,
    )
    stats_report = collect_env_statistics(
        config_path=config_path,
        out_path=stats_path,
        episodes=statistics_episodes,
        max_steps=statistics_steps,
        action_mode=action_mode,
    )

    passed = bool(
        config_report["passed"]
        and env_report["passed"]
        and vector_report["passed"]
        and stats_report["unsafe_fraction"] <= max_unsafe_fraction
    )
    manifest = {
        "config": str(config_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "passed": passed,
        "action_mode": action_mode,
        "max_unsafe_fraction": float(max_unsafe_fraction),
        "artifacts": {
            "config_validation": str(config_report_path),
            "environment_spec": str(spec_path),
            "rollout_validation": str(env_validation_path),
            "vector_benchmark": str(vector_path),
            "env_statistics": str(stats_path),
        },
        "summary": {
            "engine": spec["environment_metadata"].get("engine", ""),
            "obs_shape": spec["observation_shape"],
            "action_shape": spec["action_shape"],
            "scenario_count": config_report["scenario_count"],
            "validation_steps": env_report["total_steps"],
            "validation_steps_per_second": env_report["steps_per_second"],
            "vector_transitions_per_second": vector_report["transitions_per_second"],
            "statistics_steps": stats_report["total_steps"],
            "statistics_reward_mean": stats_report["reward"]["mean"],
            "statistics_unsafe_fraction": stats_report["unsafe_fraction"],
        },
    }
    manifest_path = out_dir / "preflight_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_full_car_corner.yaml")
    parser.add_argument("--out", default="results/mujoco_preflight")
    parser.add_argument("--validation-steps", type=int, default=50)
    parser.add_argument("--vector-envs", type=int, default=2)
    parser.add_argument("--vector-steps", type=int, default=20)
    parser.add_argument("--statistics-episodes", type=int, default=2)
    parser.add_argument("--statistics-steps", type=int, default=50)
    parser.add_argument("--action-mode", choices=["passive", "random"], default="random")
    parser.add_argument("--max-unsafe-fraction", type=float, default=0.1)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    manifest = preflight_mujoco_training(
        config_path=config_path,
        out_dir=out_dir,
        validation_steps=args.validation_steps,
        vector_envs=args.vector_envs,
        vector_steps=args.vector_steps,
        statistics_episodes=args.statistics_episodes,
        statistics_steps=args.statistics_steps,
        action_mode=args.action_mode,
        max_unsafe_fraction=args.max_unsafe_fraction,
    )
    status = "PASSED" if manifest["passed"] else "FAILED"
    print(f"{status}: preflight manifest {manifest['manifest']}")
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
