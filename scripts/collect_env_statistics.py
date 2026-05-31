from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from scripts.train_rl import ScenarioSampler, make_env


class RunningStats:
    def __init__(self, shape: tuple[int, ...]):
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)
        self.min = np.full(shape, np.inf, dtype=np.float64)
        self.max = np.full(shape, -np.inf, dtype=np.float64)

    def update(self, value: np.ndarray):
        value = np.asarray(value, dtype=np.float64)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.min = np.minimum(self.min, value)
        self.max = np.maximum(self.max, value)

    def to_dict(self) -> dict:
        variance = self.m2 / max(1, self.count - 1)
        return {
            "count": int(self.count),
            "mean": self.mean.tolist(),
            "std": np.sqrt(np.maximum(variance, 0.0)).tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }


def _action(env, mode: str) -> np.ndarray:
    if mode == "random":
        return np.asarray(env.action_space.sample(), dtype=np.float32)
    return np.zeros(env.action_space.shape, dtype=np.float32)


def collect_env_statistics(
    config_path: Path,
    out_path: Path | None = None,
    episodes: int = 4,
    max_steps: int = 200,
    action_mode: str = "random",
) -> dict:
    config = load_config(config_path)
    sampler = ScenarioSampler(list(config["scenarios"]), config)
    obs_stats: RunningStats | None = None
    action_stats: RunningStats | None = None
    reward_values: list[float] = []
    reward_components: dict[str, list[float]] = {}
    action_metrics: dict[str, list[float]] = {}
    episode_reports = []
    unsafe_steps = 0
    total_steps = 0

    for episode in range(episodes):
        scenario = sampler.select(episode)
        env = make_env(config, scenario=scenario, use_preview=True)
        try:
            obs, _ = env.reset(seed=int(config.get("seed", 42)) + episode)
            if obs_stats is None:
                obs_stats = RunningStats(obs.shape)
                action_stats = RunningStats(env.action_space.shape)
            done = False
            ep_return = 0.0
            ep_steps = 0
            while not done and ep_steps < max_steps:
                obs_stats.update(obs)
                action = _action(env, action_mode)
                action_stats.update(action)
                obs, reward, terminated, truncated, info = env.step(action)
                reward_values.append(float(reward))
                for key, value in info.get("reward_components", {}).items():
                    reward_components.setdefault(key, []).append(float(value))
                for key, value in info.get("action_metrics", {}).items():
                    if np.isscalar(value):
                        action_metrics.setdefault(key, []).append(float(value))
                ep_return += float(reward)
                ep_steps += 1
                total_steps += 1
                unsafe_steps += int(info.get("unsafe", False))
                done = terminated or truncated
            episode_reports.append(
                {
                    "episode": episode + 1,
                    "scenario": scenario.get("name", ""),
                    "steps": ep_steps,
                    "return": float(ep_return),
                }
            )
        finally:
            env.close()

    rewards = np.asarray(reward_values, dtype=np.float64)
    report = {
        "config": str(config_path.resolve()),
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "action_mode": action_mode,
        "total_steps": int(total_steps),
        "unsafe_steps": int(unsafe_steps),
        "unsafe_fraction": float(unsafe_steps / max(1, total_steps)),
        "episodes_detail": episode_reports,
        "observation": obs_stats.to_dict() if obs_stats is not None else {},
        "action": action_stats.to_dict() if action_stats is not None else {},
        "reward": {
            "count": int(len(rewards)),
            "mean": float(np.mean(rewards)) if len(rewards) else 0.0,
            "std": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
            "min": float(np.min(rewards)) if len(rewards) else 0.0,
            "max": float(np.max(rewards)) if len(rewards) else 0.0,
        },
        "reward_components": {
            key: {
                "count": int(len(values)),
                "mean": float(np.mean(values)) if values else 0.0,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)) if values else 0.0,
                "max": float(np.max(values)) if values else 0.0,
            }
            for key, values in sorted(reward_components.items())
        },
        "action_metrics": {
            key: {
                "count": int(len(values)),
                "mean": float(np.mean(values)) if values else 0.0,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)) if values else 0.0,
                "max": float(np.max(values)) if values else 0.0,
            }
            for key, values in sorted(action_metrics.items())
        },
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_full_car_corner.yaml")
    parser.add_argument("--out", default="results/mujoco_env_statistics.json")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--action-mode", choices=["passive", "random"], default="random")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    report = collect_env_statistics(
        config_path=config_path,
        out_path=out_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        action_mode=args.action_mode,
    )
    print(
        f"Collected {report['total_steps']} steps: "
        f"reward_mean={report['reward']['mean']:.4g}, unsafe_fraction={report['unsafe_fraction']:.3f}"
    )


if __name__ == "__main__":
    main()
