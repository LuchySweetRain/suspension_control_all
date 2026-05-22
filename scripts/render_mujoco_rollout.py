from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from envs import MuJoCoHalfCarEnv
from experiments.evaluation import make_controller


def resolve_checkpoint(value: str | None, algorithm: str) -> str | None:
    if not value:
        return None
    if value.lower() == "latest":
        candidates = sorted(
            (ROOT / "results").glob(f"*_{algorithm}/checkpoints/best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            candidates = sorted(
                (ROOT / "results").glob(f"*_{algorithm}/checkpoints/final.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found for {algorithm}")
        return str(candidates[0])
    checkpoint = Path(value)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    return str(checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_fast.yaml")
    parser.add_argument("--controller", default="sac", help="PID, SPDF, MPC, TD3, DDPG, SAC, or PPO")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--scenario", default="bump")
    parser.add_argument("--out", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    scenario = next((s for s in config["scenarios"] if s["name"] == args.scenario), config["scenarios"][0])
    controller_name = args.controller.upper()
    algorithm = controller_name.lower() if controller_name.lower() in {"td3", "ddpg", "sac", "ppo"} else None
    checkpoint = resolve_checkpoint(args.checkpoint, algorithm) if algorithm else None
    use_preview = bool(algorithm) or controller_name.lower() in set(config["preview"]["enabled_for"])

    env_config = dict(config)
    if controller_name.lower() in {"pid", "spdf"}:
        env_config["control_dt"] = env_config["dt"]
    env = MuJoCoHalfCarEnv(env_config, scenario=scenario, use_preview=use_preview, width=args.width, height=args.height)
    controller = make_controller(controller_name, env, env_config, checkpoint=checkpoint, algorithm=algorithm)
    obs, info = env.reset()
    controller.reset()

    frames = []
    done = False
    step = 0
    while not done:
        obs_for_controller = obs if use_preview else obs[: env.base_obs_dim]
        action = controller.compute_action(obs_for_controller, info)
        obs, _, terminated, truncated, info = env.step(action)
        if step % max(1, args.stride) == 0:
            frames.append(env.render())
        done = terminated or truncated
        step += 1
    env.close()

    out = Path(args.out) if args.out else ROOT / "results" / f"{controller_name}_{args.scenario}_mujoco.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps)
    print(f"Saved MuJoCo rollout to {out}")


if __name__ == "__main__":
    main()
