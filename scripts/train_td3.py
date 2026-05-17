from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from controllers import MPCController, PIDController, SPDFController
from envs import HalfCarEnv
from rl.replay_buffer import MixedReplaySampler, ReplayBuffer
from rl.td3 import TD3Agent, TD3Config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_expert_controller(name: str, env: HalfCarEnv, config: dict):
    name = name.lower()
    if name == "pid":
        return PIDController(env.params, env.control_dt, env.force_limit), False
    if name == "spdf":
        return SPDFController(env.params, env.control_dt), False
    if name == "mpc":
        return MPCController(env.model, config), True
    raise ValueError(f"Unknown expert controller: {name}")


def collect_expert_buffer(config: dict, rl_cfg: TD3Config, capacity: int) -> ReplayBuffer:
    expert_cfg = config.get("expert", {})
    expert_buffer = ReplayBuffer(rl_cfg.obs_dim, rl_cfg.act_dim, capacity)
    if not expert_cfg.get("enabled", False):
        return expert_buffer

    controllers = list(expert_cfg.get("controllers", ["pid", "spdf", "mpc"]))
    episodes_per_scenario = int(
        expert_cfg.get("episodes_per_scenario", expert_cfg.get("episodes_per_controller", 1))
    )
    scenarios = list(config["scenarios"])
    print(
        "Collecting expert experience from "
        f"{controllers}, {episodes_per_scenario} episode(s) per scenario..."
    )
    for controller_name in controllers:
        for scenario in scenarios:
            for _ in range(episodes_per_scenario):
                env_config = dict(config)
                if controller_name.lower() in {"pid", "spdf"}:
                    env_config = json.loads(json.dumps(config))
                    env_config["control_dt"] = env_config["dt"]
                env = HalfCarEnv(env_config, scenario=scenario, use_preview=True)
                controller, controller_uses_preview = build_expert_controller(controller_name, env, env_config)
                obs, info = env.reset()
                controller.reset()
                done = False
                ep_return = 0.0
                while not done:
                    controller_obs = obs if controller_uses_preview else obs[: env.base_obs_dim]
                    action = controller.compute_action(controller_obs, info)
                    next_obs, reward, terminated, truncated, next_info = env.step(action)
                    done = terminated or truncated
                    expert_buffer.store(obs, action, reward, next_obs, done)
                    obs = next_obs
                    info = next_info
                    ep_return += reward
                print(f"  expert={controller_name:<4} scenario={scenario['name']:<7} return={ep_return:.3f}")
    print(f"Expert buffer size: {len(expert_buffer)}")
    return expert_buffer


def linear_decay(initial: float, final: float, step: int, decay_steps: int) -> float:
    progress = min(1.0, max(0, step) / max(1, decay_steps))
    return initial + progress * (final - initial)


def evaluate_agent(agent: TD3Agent, config: dict) -> dict:
    scenario_returns = {}
    for scenario in config["scenarios"]:
        env = HalfCarEnv(config, scenario=scenario, use_preview=True)
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = agent.select_action(obs, noise=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        scenario_returns[scenario["name"]] = float(ep_return)
    mean_return = float(np.mean(list(scenario_returns.values())))
    return {"mean_return": mean_return, "scenario_returns": scenario_returns}


def save_best_if_needed(
    agent: TD3Agent,
    ckpt_dir: Path,
    eval_summary: dict,
    best_eval_return: float,
    episode: int,
) -> float:
    if eval_summary["mean_return"] > best_eval_return:
        agent.save(ckpt_dir / "best.pt")
        best_summary = {
            "episode": episode,
            "mean_return": eval_summary["mean_return"],
            "scenario_returns": eval_summary["scenario_returns"],
        }
        with (ckpt_dir / "best_summary.json").open("w", encoding="utf-8") as f:
            json.dump(best_summary, f, indent=2)
        return eval_summary["mean_return"]
    return best_eval_return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_fast.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    set_seed(int(config.get("seed", 42)))
    rl_cfg = TD3Config.from_project_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TD3Agent(rl_cfg, device)
    replay = ReplayBuffer(rl_cfg.obs_dim, rl_cfg.act_dim, int(config["rl"]["buffer_size"]))
    expert_capacity = int(config.get("expert", {}).get("capacity", config["rl"]["buffer_size"]))
    expert_replay = collect_expert_buffer(config, rl_cfg, expert_capacity)
    expert_cfg = config.get("expert", {})
    if expert_cfg.get("bc_enabled", True) and len(expert_replay) > 0:
        pretrain_epochs = int(expert_cfg.get("bc_pretrain_epochs", 0))
        if pretrain_epochs > 0:
            losses = agent.pretrain_actor_bc(
                expert_replay,
                epochs=pretrain_epochs,
                batch_size=int(expert_cfg.get("bc_batch_size", config["rl"]["batch_size"])),
                max_steps_per_epoch=expert_cfg.get("bc_pretrain_steps_per_epoch"),
            )
            if losses:
                print(
                    "Actor BC pretrain losses: "
                    + ", ".join(f"{loss:.3f}" for loss in losses)
                )
    mixed_sampler = MixedReplaySampler(
        online_buffer=replay,
        expert_buffer=expert_replay,
        initial_expert_ratio=float(expert_cfg.get("initial_ratio", 0.0)),
        final_expert_ratio=float(expert_cfg.get("final_ratio", 0.0)),
        decay_episodes=int(expert_cfg.get("decay_episodes", 1)),
    )
    episodes = int(args.episodes or config["rl"].get("episodes", 200))
    batch_size = int(config["rl"]["batch_size"])
    warmup_steps = int(config["rl"]["warmup_steps"])
    update_after = int(config["rl"]["update_after"])
    update_every = int(config["rl"].get("update_every", 1))
    sigma = float(config["rl"]["exploration_sigma"])
    random_warmup = bool(config["rl"].get("random_warmup", len(expert_replay) == 0))

    run_dir = ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_td3")
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    total_steps = 0
    history = []
    eval_history = []
    best_eval_return = -float("inf")
    eval_every = int(config["rl"].get("eval_every", 0))
    if len(expert_replay) > 0 and expert_cfg.get("bc_enabled", True):
        agent.save(ckpt_dir / "bc_pretrain.pt")
    if eval_every > 0:
        eval_summary = evaluate_agent(agent, config)
        eval_summary["episode"] = 0
        eval_history.append(eval_summary)
        best_eval_return = save_best_if_needed(agent, ckpt_dir, eval_summary, best_eval_return, episode=0)
        print(
            "Initial eval mean_return="
            f"{eval_summary['mean_return']:.3f}, best={best_eval_return:.3f}"
        )
    scenarios = list(config["scenarios"])
    pbar = trange(episodes, desc="TD3")
    for episode in pbar:
        scenario = scenarios[episode % len(scenarios)]
        env = HalfCarEnv(config, scenario=scenario, use_preview=True)
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        critic_loss = None
        actor_loss = None
        mixed_sampler.set_episode(episode)
        bc_weight = 0.0
        if expert_cfg.get("bc_enabled", True):
            bc_weight = linear_decay(
                float(expert_cfg.get("bc_initial_weight", 0.0)),
                float(expert_cfg.get("bc_final_weight", 0.0)),
                episode,
                int(expert_cfg.get("bc_decay_episodes", expert_cfg.get("decay_episodes", 1))),
            )
        while not done:
            if total_steps < warmup_steps and random_warmup:
                if hasattr(env, "action_space"):
                    action = env.action_space.sample()
                else:
                    action = np.random.uniform(
                        -float(config["force_limit"]),
                        float(config["force_limit"]),
                        size=2,
                    ).astype(np.float32)
            else:
                action = agent.select_action(obs, noise=sigma)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.store(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_return += reward
            total_steps += 1
            if total_steps >= update_after and total_steps % update_every == 0 and len(replay) >= batch_size:
                critic_loss, actor_loss = agent.train_step(
                    mixed_sampler,
                    batch_size,
                    expert_replay=expert_replay,
                    bc_weight=bc_weight,
                    bc_batch_size=int(expert_cfg.get("bc_batch_size", batch_size)),
                )
        record = {
            "episode": episode + 1,
            "scenario": scenario["name"],
            "return": ep_return,
            "steps": env.step_count,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "expert_ratio": mixed_sampler.expert_ratio,
            "bc_weight": bc_weight,
            "online_buffer_size": len(replay),
            "expert_buffer_size": len(expert_replay),
            "eval_mean_return": None,
            "best_eval_return": best_eval_return if np.isfinite(best_eval_return) else None,
        }
        if eval_every > 0 and (episode + 1) % eval_every == 0:
            eval_summary = evaluate_agent(agent, config)
            eval_summary["episode"] = episode + 1
            eval_history.append(eval_summary)
            best_eval_return = save_best_if_needed(
                agent,
                ckpt_dir,
                eval_summary,
                best_eval_return,
                episode=episode + 1,
            )
            record["eval_mean_return"] = eval_summary["mean_return"]
            record["best_eval_return"] = best_eval_return
        history.append(record)
        pbar.set_postfix({
            "return": f"{ep_return:.2f}",
            "buffer": len(replay),
            "expert": f"{mixed_sampler.expert_ratio:.2f}",
            "best": f"{best_eval_return:.2f}" if np.isfinite(best_eval_return) else "n/a",
        })
        if (episode + 1) % int(config["rl"].get("save_every", 50)) == 0:
            agent.save(ckpt_dir / f"episode_{episode+1}.pt")

    agent.save(ckpt_dir / "final.pt")
    if eval_every > 0:
        eval_summary = evaluate_agent(agent, config)
        eval_summary["episode"] = episodes
        eval_summary["label"] = "final"
        eval_history.append(eval_summary)
        best_eval_return = save_best_if_needed(agent, ckpt_dir, eval_summary, best_eval_return, episode=episodes)
    with (run_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (run_dir / "eval_history.json").open("w", encoding="utf-8") as f:
        json.dump(eval_history, f, indent=2)
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
