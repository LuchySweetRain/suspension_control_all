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
from scripts.train_rl import ScenarioSampler, make_env


def benchmark_vector_env(
    config_path: Path,
    out_path: Path | None = None,
    num_envs: int = 4,
    steps: int = 200,
    action_mode: str = "random",
) -> dict:
    config = load_config(config_path)
    sampler = ScenarioSampler(list(config["scenarios"]), config)
    envs = []
    report = {
        "config": str(config_path.resolve()),
        "num_envs": int(num_envs),
        "steps": int(steps),
        "action_mode": action_mode,
        "passed": True,
        "total_transitions": 0,
        "unsafe_steps": 0,
        "envs": [],
    }
    try:
        for idx in range(num_envs):
            scenario = sampler.select(idx)
            env = make_env(config, scenario=scenario, use_preview=True)
            obs, info = env.reset(seed=int(config.get("seed", 42)) + idx)
            envs.append(env)
            report["envs"].append(
                {
                    "index": idx,
                    "scenario": scenario.get("name", ""),
                    "obs_shape": list(obs.shape),
                    "action_shape": list(env.action_space.shape),
                    "engine": info.get("engine", ""),
                    "errors": [],
                }
            )
        start = time.perf_counter()
        returns = np.zeros(num_envs, dtype=np.float64)
        for _ in range(steps):
            for idx, env in enumerate(envs):
                if action_mode == "random":
                    action = env.action_space.sample()
                else:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                returns[idx] += float(reward)
                report["total_transitions"] += 1
                report["unsafe_steps"] += int(info.get("unsafe", False))
                finite = np.all(np.isfinite(obs)) and np.isfinite(reward)
                if not finite:
                    report["passed"] = False
                    report["envs"][idx]["errors"].append("non-finite observation or reward")
                if terminated or truncated:
                    obs, info = env.reset()
                    if not np.all(np.isfinite(obs)):
                        report["passed"] = False
                        report["envs"][idx]["errors"].append("non-finite observation after reset")
        elapsed = time.perf_counter() - start
        report["elapsed_seconds"] = elapsed
        report["transitions_per_second"] = float(report["total_transitions"] / elapsed) if elapsed > 0.0 else 0.0
        report["mean_return"] = float(np.mean(returns))
        report["returns"] = [float(value) for value in returns]
        report["unsafe_fraction"] = float(report["unsafe_steps"] / max(1, report["total_transitions"]))
    finally:
        for env in envs:
            env.close()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_full_car_corner.yaml")
    parser.add_argument("--out", default="results/mujoco_vector_env_benchmark.json")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--action-mode", choices=["passive", "random"], default="random")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    report = benchmark_vector_env(
        config_path=config_path,
        out_path=out_path,
        num_envs=args.num_envs,
        steps=args.steps,
        action_mode=args.action_mode,
    )
    status = "PASSED" if report["passed"] else "FAILED"
    print(
        f"{status}: {report['total_transitions']} transitions, "
        f"{report['transitions_per_second']:.1f} transitions/s, unsafe_fraction={report['unsafe_fraction']:.3f}"
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
