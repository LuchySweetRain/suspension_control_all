from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from scripts.train_rl import make_env


def export_env_spec(config_path: Path, out_path: Path, scenario_index: int = 0) -> dict:
    config = load_config(config_path)
    scenarios = list(config.get("scenarios", []))
    if not scenarios:
        raise ValueError("Config must define at least one scenario.")
    scenario = scenarios[scenario_index]
    env = make_env(config, scenario=scenario, use_preview=True)
    try:
        obs, info = env.reset(seed=int(config.get("seed", 42)))
        if hasattr(env, "environment_metadata"):
            metadata = env.environment_metadata()
        else:
            metadata = {
                "engine": str(config.get("environment", {}).get("engine", "python")),
                "obs_dim": int(env.obs_dim),
                "act_dim": int(env.action_space.shape[0]),
                "base_obs_dim": int(env.base_obs_dim),
            }
        spec = {
            "config": str(config_path.resolve()),
            "scenario": scenario,
            "observation_shape": list(obs.shape),
            "action_shape": list(env.action_space.shape),
            "reset_info_keys": sorted(info.keys()),
            "environment_metadata": metadata,
        }
    finally:
        env.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    return spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_full_car_corner.yaml")
    parser.add_argument("--out", default="results/mujoco_env_spec.json")
    parser.add_argument("--scenario-index", type=int, default=0)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    spec = export_env_spec(config_path, out_path, scenario_index=args.scenario_index)
    print(f"Exported {spec['environment_metadata']['engine']} spec to {out_path}")


if __name__ == "__main__":
    main()
