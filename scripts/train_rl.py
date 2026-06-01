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

from config import apply_algorithm_overrides, load_config
from controllers import MPCController, PIDController, SPDFController
from controllers.prior import adapt_action_to_env, compute_prior_action, make_prior_controller, residual_gate, shield_residual_action
from envs import HalfCarEnv, MuJoCoFullCarEnv, MuJoCoVehicleEnv
from rl.ddpg import DDPGAgent, DDPGConfig
from rl.ppo import PPOAgent, PPOConfig, finish_ppo_trajectory
from rl.replay_buffer import MixedReplaySampler, ReplayBuffer
from rl.sac import SACAgent, SACConfig
from rl.td3 import TD3Agent, TD3Config


AGENTS = {
    "td3": (TD3Config, TD3Agent),
    "ddpg": (DDPGConfig, DDPGAgent),
    "sac": (SACConfig, SACAgent),
    "ppo": (PPOConfig, PPOAgent),
}


class ScenarioSampler:
    def __init__(self, scenarios: list[dict], config: dict, rng: np.random.Generator | None = None):
        if not scenarios:
            raise ValueError("At least one scenario is required for training.")
        self.scenarios = list(scenarios)
        self.cfg = dict(config.get("scenario_sampling", {}))
        self.mode = str(self.cfg.get("mode", "cycle")).lower()
        self.rng = rng or np.random.default_rng(int(config.get("seed", 42)))
        self._name_to_scenario = {str(s.get("name", i)): s for i, s in enumerate(self.scenarios)}

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "scenario_names": [str(s.get("name", i)) for i, s in enumerate(self.scenarios)],
            "weights": self.cfg.get("weights"),
            "curriculum": self.cfg.get("curriculum", []),
        }

    def select(self, episode: int) -> dict:
        active_scenarios = self.scenarios
        active_weights = self.cfg.get("weights")
        active_mode = self.mode
        for phase in self.cfg.get("curriculum", []):
            until_episode = int(phase.get("until_episode", episode + 1))
            if episode + 1 <= until_episode:
                names = phase.get("scenarios")
                if names:
                    active_scenarios = [self._name_to_scenario[str(name)] for name in names]
                active_weights = phase.get("weights", active_weights)
                active_mode = str(phase.get("mode", active_mode)).lower()
                break
        if active_mode == "cycle":
            return active_scenarios[episode % len(active_scenarios)]
        if active_mode == "uniform":
            return active_scenarios[int(self.rng.integers(0, len(active_scenarios)))]
        if active_mode == "weighted":
            weights = self._weights_for(active_scenarios, active_weights)
            index = int(self.rng.choice(len(active_scenarios), p=weights))
            return active_scenarios[index]
        raise ValueError("scenario_sampling.mode must be one of: cycle, uniform, weighted")

    def _weights_for(self, scenarios: list[dict], weights_cfg) -> np.ndarray:
        if isinstance(weights_cfg, dict):
            weights = np.asarray([float(weights_cfg.get(str(s.get("name", "")), 0.0)) for s in scenarios], dtype=np.float64)
        elif isinstance(weights_cfg, list):
            weights = np.asarray(weights_cfg, dtype=np.float64)
            if len(weights) != len(scenarios):
                raise ValueError("scenario_sampling.weights list must match the active scenario count.")
        else:
            weights = np.ones(len(scenarios), dtype=np.float64)
        if np.any(weights < 0.0) or not np.any(weights > 0.0):
            raise ValueError("scenario_sampling.weights must contain at least one positive non-negative weight.")
        return weights / np.sum(weights)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(config: dict, scenario: dict, use_preview: bool = True):
    engine = str(config.get("environment", {}).get("engine", "python")).lower()
    if engine in {"mujoco_full", "mujoco_full_car", "full_car"}:
        return MuJoCoFullCarEnv(config, scenario=scenario, use_preview=use_preview)
    if engine in {"mujoco", "mujoco_vehicle"}:
        return MuJoCoVehicleEnv(config, scenario=scenario, use_preview=use_preview)
    return HalfCarEnv(config, scenario=scenario, use_preview=use_preview)


def build_expert_controller(name: str, env: HalfCarEnv, config: dict):
    name = name.lower()
    if name == "pid":
        return PIDController(env.params, env.control_dt, env.force_limit), False
    if name == "spdf":
        return SPDFController(env.params, env.control_dt), False
    if name == "mpc":
        if not hasattr(env, "model"):
            raise ValueError("MPC expert collection requires the Python HalfCarEnv model.")
        return MPCController(env.model, config), True
    raise ValueError(f"Unknown expert controller: {name}")


def collect_expert_buffer(config: dict, rl_cfg, capacity: int) -> ReplayBuffer:
    expert_cfg = config.get("expert", {})
    expert_buffer = ReplayBuffer(rl_cfg.obs_dim, rl_cfg.act_dim, capacity)
    if not expert_cfg.get("enabled", False):
        return expert_buffer
    if str(config.get("environment", {}).get("engine", "python")).lower() in {
        "mujoco",
        "mujoco_vehicle",
        "mujoco_full",
        "mujoco_full_car",
        "full_car",
    }:
        print("Skipping expert collection for MuJoCo dynamics envs; set expert.enabled=false for pure MuJoCo RL runs.")
        return expert_buffer
    controllers = list(expert_cfg.get("controllers", ["pid", "spdf", "mpc"]))
    episodes_per_scenario = int(expert_cfg.get("episodes_per_scenario", 1))
    print(f"Collecting expert experience from {controllers}...")
    for controller_name in controllers:
        for scenario in config["scenarios"]:
            for _ in range(episodes_per_scenario):
                env_config = json.loads(json.dumps(config))
                if controller_name.lower() in {"pid", "spdf"}:
                    env_config["control_dt"] = env_config["dt"]
                env = make_env(env_config, scenario=scenario, use_preview=True)
                controller, controller_uses_preview = build_expert_controller(controller_name, env, env_config)
                obs, info = env.reset()
                controller.reset()
                done = False
                while not done:
                    action = controller.compute_action(obs if controller_uses_preview else obs[: env.base_obs_dim], info)
                    next_obs, reward, terminated, truncated, next_info = env.step(action)
                    done = terminated or truncated
                    expert_buffer.store(obs, action, reward, next_obs, done)
                    obs, info = next_obs, next_info
    print(f"Expert buffer size: {len(expert_buffer)}")
    return expert_buffer


def linear_decay(initial: float, final: float, step: int, decay_steps: int) -> float:
    progress = min(1.0, max(0, step) / max(1, decay_steps))
    return initial + progress * (final - initial)


def evaluate_agent(agent, config: dict) -> dict:
    scenario_returns = {}
    residual_cfg = dict(config.get("residual_control", {}))
    residual_enabled = bool(residual_cfg.get("enabled", False))
    for scenario in config["scenarios"]:
        env = make_env(config, scenario=scenario, use_preview=True)
        obs, info = env.reset()
        prior = make_prior_controller(str(residual_cfg.get("prior", "spdf")), env, config) if residual_enabled else None
        if prior is not None:
            prior.reset()
        done = False
        ep_return = 0.0
        while not done:
            residual_action = agent.select_action(obs, noise=0.0)
            if residual_enabled:
                prior_action = compute_prior_action(prior, env, obs, info)
                residual_action = shield_residual_action(residual_action, env, info, residual_cfg)
                scale = residual_gate(info, residual_cfg)
                action = np.clip(
                    prior_action + scale * residual_action,
                    env.action_space.low,
                    env.action_space.high,
                )
            else:
                action = residual_action
                prior_action = None
            step_action = {"action": action, "prior_action": prior_action} if residual_enabled else action
            obs, reward, terminated, truncated, info = env.step(step_action)
            ep_return += reward
            done = terminated or truncated
        env.close()
        scenario_returns[scenario["name"]] = float(ep_return)
    return {"mean_return": float(np.mean(list(scenario_returns.values()))), "scenario_returns": scenario_returns}


def save_best_if_needed(agent, ckpt_dir: Path, eval_summary: dict, best_eval_return: float, episode: int) -> float:
    if eval_summary["mean_return"] > best_eval_return:
        agent.save(ckpt_dir / "best.pt")
        with (ckpt_dir / "best_summary.json").open("w", encoding="utf-8") as f:
            json.dump({"episode": episode, **eval_summary}, f, indent=2)
        return eval_summary["mean_return"]
    return best_eval_return


def pretrain_ppo_from_imitation(agent: PPOAgent, config: dict, run_dir: Path) -> list[float]:
    imitation_cfg = dict(config.get("imitation", {}))
    if not imitation_cfg.get("enabled", False):
        return []
    dataset_path = Path(str(imitation_cfg.get("dataset", "")))
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not dataset_path.is_file():
        raise FileNotFoundError(f"imitation.dataset does not exist: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    obs = np.asarray(data["obs"], dtype=np.float32)
    action_key = "residual_act" if imitation_cfg.get("residual_targets", False) else "act"
    if action_key not in data:
        raise KeyError(f"imitation dataset missing {action_key!r}")
    act = np.asarray(data[action_key], dtype=np.float32)
    if obs.shape[1] != agent.cfg.obs_dim:
        raise ValueError(f"imitation obs dim {obs.shape[1]} does not match PPO obs dim {agent.cfg.obs_dim}")
    if act.shape[1] != agent.cfg.act_dim:
        raise ValueError(f"imitation act dim {act.shape[1]} does not match PPO act dim {agent.cfg.act_dim}")
    losses = agent.pretrain_actor_bc(
        obs,
        act,
        epochs=int(imitation_cfg.get("epochs", 0)),
        batch_size=int(imitation_cfg.get("batch_size", config["rl"].get("batch_size", 128))),
        max_steps_per_epoch=imitation_cfg.get("max_steps_per_epoch"),
    )
    if imitation_cfg.get("anchor_enabled", False) and hasattr(agent, "set_bc_anchor"):
        max_anchor_samples = imitation_cfg.get("anchor_max_samples")
        if max_anchor_samples is not None and int(max_anchor_samples) > 0 and obs.shape[0] > int(max_anchor_samples):
            idx = np.linspace(0, obs.shape[0] - 1, int(max_anchor_samples), dtype=int)
            agent.set_bc_anchor(obs[idx], act[idx])
        else:
            agent.set_bc_anchor(obs, act)
    if losses:
        out = {
            "dataset": str(dataset_path.resolve()),
            "action_key": action_key,
            "samples": int(obs.shape[0]),
            "losses": losses,
        }
        with (run_dir / "imitation_pretrain.json").open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("PPO imitation BC losses: " + ", ".join(f"{loss:.6f}" for loss in losses))
    return losses


def train_off_policy(agent, rl_cfg, config: dict, algorithm: str, run_dir: Path, episodes: int):
    replay = ReplayBuffer(rl_cfg.obs_dim, rl_cfg.act_dim, int(config["rl"]["buffer_size"]))
    expert_cfg = config.get("expert", {})
    expert_replay = ReplayBuffer(rl_cfg.obs_dim, rl_cfg.act_dim, 1)
    if algorithm in {"td3", "ddpg"}:
        expert_capacity = int(expert_cfg.get("capacity", config["rl"]["buffer_size"]))
        expert_replay = collect_expert_buffer(config, rl_cfg, expert_capacity)
        if hasattr(agent, "pretrain_actor_bc") and expert_cfg.get("bc_enabled", True):
            losses = agent.pretrain_actor_bc(
                expert_replay,
                epochs=int(expert_cfg.get("bc_pretrain_epochs", 0)),
                batch_size=int(expert_cfg.get("bc_batch_size", config["rl"]["batch_size"])),
                max_steps_per_epoch=expert_cfg.get("bc_pretrain_steps_per_epoch"),
            )
            if losses:
                print("Actor BC pretrain losses: " + ", ".join(f"{loss:.3f}" for loss in losses))
    sampler = MixedReplaySampler(
        replay,
        expert_replay,
        float(expert_cfg.get("initial_ratio", 0.0)) if algorithm in {"td3", "ddpg"} else 0.0,
        float(expert_cfg.get("final_ratio", 0.0)) if algorithm in {"td3", "ddpg"} else 0.0,
        int(expert_cfg.get("decay_episodes", 1)),
    )
    batch_size = int(config["rl"]["batch_size"])
    warmup_steps = int(config["rl"]["warmup_steps"])
    update_after = int(config["rl"]["update_after"])
    update_every = int(config["rl"].get("update_every", 1))
    sigma = float(config["rl"]["exploration_sigma"])
    random_warmup = bool(config["rl"].get("random_warmup", len(expert_replay) == 0))
    residual_cfg = dict(config.get("residual_control", {}))
    residual_enabled = bool(residual_cfg.get("enabled", False))
    history, eval_history = [], []
    total_steps = 0
    best_eval_return = -float("inf")
    eval_every = int(config["rl"].get("eval_every", 0))
    ckpt_dir = run_dir / "checkpoints"
    scenario_sampler = ScenarioSampler(list(config["scenarios"]), config)
    pbar = trange(episodes, desc=algorithm.upper())
    for episode in pbar:
        scenario = scenario_sampler.select(episode)
        env = make_env(config, scenario=scenario, use_preview=True)
        obs, info = env.reset()
        prior = make_prior_controller(str(residual_cfg.get("prior", "spdf")), env, config) if residual_enabled else None
        if prior is not None:
            prior.reset()
        done = False
        ep_return = 0.0
        critic_loss = None
        actor_loss = None
        sampler.set_episode(episode)
        bc_weight = linear_decay(
            float(expert_cfg.get("bc_initial_weight", 0.0)),
            float(expert_cfg.get("bc_final_weight", 0.0)),
            episode,
            int(expert_cfg.get("bc_decay_episodes", expert_cfg.get("decay_episodes", 1))),
        )
        while not done:
            if total_steps < warmup_steps and random_warmup:
                residual_action = env.action_space.sample()
            else:
                residual_action = agent.select_action(obs, noise=sigma)
            if residual_enabled:
                prior_action = compute_prior_action(prior, env, obs, info)
                residual_action = shield_residual_action(residual_action, env, info, residual_cfg)
                scale = residual_gate(info, residual_cfg)
                action = np.clip(
                    prior_action + scale * residual_action,
                    env.action_space.low,
                    env.action_space.high,
                )
            else:
                action = residual_action
                prior_action = None
            step_action = {"action": action, "prior_action": prior_action} if residual_enabled else action
            next_obs, reward, terminated, truncated, next_info = env.step(step_action)
            done = terminated or truncated
            replay.store(obs, residual_action, reward, next_obs, done)
            obs = next_obs
            info = next_info
            ep_return += reward
            total_steps += 1
            if total_steps >= update_after and total_steps % update_every == 0 and len(replay) >= batch_size:
                critic_loss, actor_loss = agent.train_step(
                    sampler,
                    batch_size,
                    expert_replay=expert_replay,
                    bc_weight=bc_weight,
                    bc_batch_size=int(expert_cfg.get("bc_batch_size", batch_size)),
                )
        env.close()
        record = {
            "episode": episode + 1,
            "scenario": scenario["name"],
            "return": ep_return,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "online_buffer_size": len(replay),
            "expert_buffer_size": len(expert_replay),
            "eval_mean_return": None,
        }
        if eval_every > 0 and (episode + 1) % eval_every == 0:
            eval_summary = evaluate_agent(agent, config)
            eval_summary["episode"] = episode + 1
            eval_history.append(eval_summary)
            best_eval_return = save_best_if_needed(agent, ckpt_dir, eval_summary, best_eval_return, episode + 1)
            record["eval_mean_return"] = eval_summary["mean_return"]
        history.append(record)
        pbar.set_postfix({"return": f"{ep_return:.2f}", "buffer": len(replay)})
        if (episode + 1) % int(config["rl"].get("save_every", 50)) == 0:
            agent.save(ckpt_dir / f"episode_{episode+1}.pt")
    return history, eval_history


def train_ppo(agent: PPOAgent, config: dict, run_dir: Path, episodes: int):
    history, eval_history = [], []
    best_eval_return = -float("inf")
    eval_every = int(config["rl"].get("eval_every", 0))
    ckpt_dir = run_dir / "checkpoints"
    scenario_sampler = ScenarioSampler(list(config["scenarios"]), config)
    residual_cfg = dict(config.get("residual_control", {}))
    residual_enabled = bool(residual_cfg.get("enabled", False))
    imitation_cfg = dict(config.get("imitation", {}))
    pbar = trange(episodes, desc="PPO")
    for episode in pbar:
        scenario = scenario_sampler.select(episode)
        env = make_env(config, scenario=scenario, use_preview=True)
        obs, info = env.reset()
        prior = make_prior_controller(str(residual_cfg.get("prior", "spdf")), env, config) if residual_enabled else None
        if prior is not None:
            prior.reset()
        done = False
        trajectory = {"obs": [], "act": [], "logp": [], "rew": [], "value": [], "done": []}
        ep_return = 0.0
        while not done:
            residual_action, logp, value = agent.act_for_training(obs)
            if residual_enabled:
                prior_action = compute_prior_action(prior, env, obs, info)
                residual_action = shield_residual_action(residual_action, env, info, residual_cfg)
                scale = residual_gate(info, residual_cfg)
                action = np.clip(
                    prior_action + scale * residual_action,
                    env.action_space.low,
                    env.action_space.high,
                )
            else:
                action = residual_action
                prior_action = None
            step_action = {"action": action, "prior_action": prior_action} if residual_enabled else action
            next_obs, reward, terminated, truncated, next_info = env.step(step_action)
            done = terminated or truncated
            trajectory["obs"].append(obs)
            trajectory["act"].append(residual_action)
            trajectory["logp"].append(logp)
            trajectory["rew"].append(reward)
            trajectory["value"].append(value)
            trajectory["done"].append(done)
            obs = next_obs
            info = next_info
            ep_return += reward
        env.close()
        returns, adv = finish_ppo_trajectory(
            trajectory["rew"],
            trajectory["value"],
            trajectory["done"],
            agent.cfg.gamma,
            agent.cfg.lam,
        )
        trajectory["ret"] = returns
        trajectory["adv"] = adv
        bc_anchor_weight = linear_decay(
            float(imitation_cfg.get("anchor_initial_weight", 0.0)),
            float(imitation_cfg.get("anchor_final_weight", 0.0)),
            episode,
            int(imitation_cfg.get("anchor_decay_episodes", max(1, episodes))),
        )
        critic_loss, actor_loss = agent.train_trajectory(
            trajectory,
            bc_anchor_weight=bc_anchor_weight,
            bc_anchor_batch_size=imitation_cfg.get("anchor_batch_size"),
        )
        record = {
            "episode": episode + 1,
            "scenario": scenario["name"],
            "return": ep_return,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "bc_anchor_weight": bc_anchor_weight,
            "eval_mean_return": None,
        }
        if eval_every > 0 and (episode + 1) % eval_every == 0:
            eval_summary = evaluate_agent(agent, config)
            eval_summary["episode"] = episode + 1
            eval_history.append(eval_summary)
            best_eval_return = save_best_if_needed(agent, ckpt_dir, eval_summary, best_eval_return, episode + 1)
            record["eval_mean_return"] = eval_summary["mean_return"]
        history.append(record)
        pbar.set_postfix({"return": f"{ep_return:.2f}"})
        if (episode + 1) % int(config["rl"].get("save_every", 50)) == 0:
            agent.save(ckpt_dir / f"episode_{episode+1}.pt")
    return history, eval_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_fast.yaml")
    parser.add_argument("--algorithm", choices=sorted(AGENTS), default="td3")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    config = apply_algorithm_overrides(load_config(ROOT / args.config), args.algorithm)
    set_seed(int(config.get("seed", 42)))
    config_cls, agent_cls = AGENTS[args.algorithm]
    rl_cfg = config_cls.from_project_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = agent_cls(rl_cfg, device)
    episodes = int(args.episodes or config["rl"].get("episodes", 200))
    run_dir = ROOT / "results" / datetime.now().strftime(f"%Y%m%d_%H%M%S_{args.algorithm}")
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    scenario_sampler = ScenarioSampler(list(config["scenarios"]), config)
    with (run_dir / "training_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "algorithm": args.algorithm,
                "episodes": episodes,
                "device": str(device),
                "config": str((ROOT / args.config).resolve()),
                "scenario_sampling": scenario_sampler.summary(),
                "environment_engine": str(config.get("environment", {}).get("engine", "python")),
                "rl_obs_dim": int(rl_cfg.obs_dim),
                "rl_act_dim": int(rl_cfg.act_dim),
            },
            f,
            indent=2,
        )

    if args.algorithm == "ppo":
        pretrain_losses = pretrain_ppo_from_imitation(agent, config, run_dir)
        if pretrain_losses:
            agent.save(ckpt_dir / "imitation_pretrained.pt")
            if config.get("imitation", {}).get("eval_pretrained", True):
                eval_summary = evaluate_agent(agent, config)
                with (run_dir / "imitation_pretrained_eval.json").open("w", encoding="utf-8") as f:
                    json.dump(eval_summary, f, indent=2)
        history, eval_history = train_ppo(agent, config, run_dir, episodes)
    else:
        history, eval_history = train_off_policy(agent, rl_cfg, config, args.algorithm, run_dir, episodes)
    agent.save(ckpt_dir / "final.pt")
    with (run_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (run_dir / "eval_history.json").open("w", encoding="utf-8") as f:
        json.dump(eval_history, f, indent=2)
    print(f"Saved {args.algorithm.upper()} run to {run_dir}")


if __name__ == "__main__":
    main()
