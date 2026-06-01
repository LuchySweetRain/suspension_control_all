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
from controllers.prior import adapt_action_to_env, compute_prior_action, make_prior_controller
from experiments.evaluation import make_controller
from scripts.train_rl import ScenarioSampler, make_env


def _flatten_numeric(prefix: str, data: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in data.items():
        if np.isscalar(value):
            out[f"{prefix}_{key}"] = float(value)
    return out


def collect_expert_dataset(
    config_path: Path,
    out_path: Path,
    expert: str,
    episodes: int,
    max_steps: int | None = None,
    residual_prior: str | None = None,
    seed: int | None = None,
    skip_unsafe: bool = False,
) -> dict:
    config = load_config(config_path)
    rng_seed = int(seed if seed is not None else config.get("seed", 42))
    sampler = ScenarioSampler(list(config["scenarios"]), config, rng=np.random.default_rng(rng_seed))
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    residual_actions: list[np.ndarray] = []
    prior_actions: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    scenario_names: list[str] = []
    step_metrics: list[dict[str, float]] = []
    raw_step_metrics: list[dict[str, float]] = []
    skipped_unsafe = 0

    for episode in range(int(episodes)):
        scenario = sampler.select(episode)
        env = make_env(config, scenario=scenario, use_preview=True)
        expert_controller = make_controller(expert, env, config)
        prior_controller = make_prior_controller(residual_prior, env, config) if residual_prior else None
        try:
            obs, info = env.reset(seed=rng_seed + episode)
            expert_controller.reset()
            if prior_controller is not None:
                prior_controller.reset()
            done = False
            steps = 0
            while not done and (max_steps is None or steps < max_steps):
                action = adapt_action_to_env(expert_controller.compute_action(obs, info), env)
                if prior_controller is not None:
                    prior_action = compute_prior_action(prior_controller, env, obs, info)
                else:
                    prior_action = np.zeros_like(action)
                residual_action = np.clip(action - prior_action, env.action_space.low, env.action_space.high)
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                done = bool(terminated or truncated)
                metrics = {
                    "episode": float(episode + 1),
                    "step": float(steps + 1),
                    "reward": float(reward),
                    "unsafe": float(next_info.get("unsafe", False)),
                }
                metrics.update(_flatten_numeric("action", next_info.get("action_metrics", {})))
                metrics.update(_flatten_numeric("reward", next_info.get("reward_components", {})))
                raw_step_metrics.append(metrics)
                if skip_unsafe and bool(next_info.get("unsafe", False)):
                    skipped_unsafe += 1
                else:
                    observations.append(np.asarray(obs, dtype=np.float32))
                    actions.append(np.asarray(action, dtype=np.float32))
                    residual_actions.append(np.asarray(residual_action, dtype=np.float32))
                    prior_actions.append(np.asarray(prior_action, dtype=np.float32))
                    next_observations.append(np.asarray(next_obs, dtype=np.float32))
                    rewards.append(float(reward))
                    dones.append(done)
                    scenario_names.append(str(scenario.get("name", "")))
                    step_metrics.append(metrics)
                obs, info = next_obs, next_info
                steps += 1
        finally:
            env.close()

    if not observations:
        raise RuntimeError("No expert transitions were collected.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = sorted({key for row in step_metrics for key in row})
    metric_array = np.zeros((len(step_metrics), len(metric_keys)), dtype=np.float32)
    for i, row in enumerate(step_metrics):
        for j, key in enumerate(metric_keys):
            metric_array[i, j] = float(row.get(key, 0.0))
    np.savez_compressed(
        out_path,
        obs=np.asarray(observations, dtype=np.float32),
        act=np.asarray(actions, dtype=np.float32),
        residual_act=np.asarray(residual_actions, dtype=np.float32),
        prior_act=np.asarray(prior_actions, dtype=np.float32),
        next_obs=np.asarray(next_observations, dtype=np.float32),
        rew=np.asarray(rewards, dtype=np.float32),
        done=np.asarray(dones, dtype=np.float32),
        scenario=np.asarray(scenario_names),
        metrics=metric_array,
        metric_keys=np.asarray(metric_keys),
    )
    manifest = {
        "config": str(config_path.resolve()),
        "out": str(out_path.resolve()),
        "expert": expert,
        "residual_prior": residual_prior,
        "episodes": int(episodes),
        "max_steps": max_steps,
        "skip_unsafe": bool(skip_unsafe),
        "raw_transitions": len(raw_step_metrics),
        "skipped_unsafe": int(skipped_unsafe),
        "transitions": len(observations),
        "obs_dim": int(observations[0].shape[0]),
        "act_dim": int(actions[0].shape[0]),
        "scenarios": sorted(set(scenario_names)),
        "return_sum": float(np.sum(rewards)),
        "raw_unsafe_fraction": float(np.mean([row.get("unsafe", 0.0) for row in raw_step_metrics])),
        "unsafe_fraction": float(np.mean([row.get("unsafe", 0.0) for row in step_metrics])),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_full_car_residual.yaml")
    parser.add_argument("--out", default="datasets/expert/full_car_mpc_lite.npz")
    parser.add_argument("--expert", default="FULL_CAR_MPC_LITE")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--residual-prior", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-unsafe", action="store_true", help="Drop expert transitions whose next state is marked unsafe.")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    manifest = collect_expert_dataset(
        config_path=config_path,
        out_path=out_path,
        expert=args.expert,
        episodes=args.episodes,
        max_steps=args.max_steps,
        residual_prior=args.residual_prior,
        seed=args.seed,
        skip_unsafe=args.skip_unsafe,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
