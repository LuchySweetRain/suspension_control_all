from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import TransformerActor, TransformerCritic


@dataclass
class TD3Config:
    obs_dim_base: int
    preview_steps: int
    preview_token_dim: int
    act_dim: int
    act_limit: float
    hidden_sizes: list[int]
    actor_lr: float
    critic_lr: float
    gamma: float
    tau: float
    policy_delay: int
    target_noise: float
    noise_clip: float
    gradient_clip: float
    base_encoder_dim: int
    preview_embed_dim: int
    transformer_layers: int
    transformer_heads: int
    transformer_ff_dim: int
    transformer_dropout: float
    action_l2_weight: float = 0.0
    q_policy_weight: float = 1.0

    @property
    def obs_dim(self) -> int:
        return self.obs_dim_base + self.preview_steps * self.preview_token_dim

    @classmethod
    def from_project_config(cls, config: dict) -> "TD3Config":
        rl = config["rl"]
        act_dim = int(rl.get("act_dim", 2))
        history_cfg = dict(config.get("observation", {}).get("history", {}))
        history_steps = int(history_cfg.get("steps", 0)) if history_cfg.get("enabled", False) else 0
        obs_dim_base = 14 + history_steps * (14 + 2 * act_dim)
        return cls(
            obs_dim_base=int(rl.get("obs_dim_base", obs_dim_base)),
            preview_steps=int(config["preview"]["steps"]),
            preview_token_dim=int(rl.get("preview_token_dim", 2)),
            act_dim=act_dim,
            act_limit=float(config["force_limit"]),
            hidden_sizes=list(rl["hidden_sizes"]),
            actor_lr=float(rl["actor_lr"]),
            critic_lr=float(rl["critic_lr"]),
            gamma=float(rl["gamma"]),
            tau=float(rl["tau"]),
            policy_delay=int(rl["policy_delay"]),
            target_noise=float(rl["target_noise"]) * float(config["force_limit"]),
            noise_clip=float(rl["noise_clip"]) * float(config["force_limit"]),
            gradient_clip=float(rl["gradient_clip"]),
            base_encoder_dim=128,
            preview_embed_dim=int(rl["preview_embed_dim"]),
            transformer_layers=int(rl["transformer_layers"]),
            transformer_heads=int(rl["transformer_heads"]),
            transformer_ff_dim=int(rl["transformer_ff_dim"]),
            transformer_dropout=float(rl["transformer_dropout"]),
            action_l2_weight=float(rl.get("action_l2_weight", 0.0)),
            q_policy_weight=float(rl.get("q_policy_weight", 1.0)),
        )


class TD3Agent:
    def __init__(self, cfg: TD3Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.actor = TransformerActor(cfg).to(device)
        self.actor_target = TransformerActor(cfg).to(device)
        self.critic1 = TransformerCritic(cfg).to(device)
        self.critic2 = TransformerCritic(cfg).to(device)
        self.critic1_target = TransformerCritic(cfg).to(device)
        self.critic2_target = TransformerCritic(cfg).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=cfg.critic_lr,
        )
        self.update_count = 0

    def select_action(self, obs: np.ndarray, noise: float = 0.0) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]
        if was_training:
            self.actor.train()
        if noise > 0:
            action = action + np.random.normal(0.0, noise, size=action.shape)
        return np.clip(action, -self.cfg.act_limit, self.cfg.act_limit).astype(np.float32)

    def train_step(
        self,
        replay,
        batch_size: int,
        expert_replay=None,
        bc_weight: float = 0.0,
        bc_batch_size: int | None = None,
    ) -> tuple[float, float | None]:
        self.update_count += 1
        batch = replay.sample(batch_size, self.device)
        critic_loss = self._update_critics(batch)
        actor_loss = None
        if self.update_count % self.cfg.policy_delay == 0:
            expert_batch = None
            if expert_replay is not None and len(expert_replay) > 0 and bc_weight > 0.0:
                expert_batch = expert_replay.sample(int(bc_batch_size or batch_size), self.device)
            actor_loss = self._update_actor(batch, expert_batch=expert_batch, bc_weight=bc_weight)
            self._soft_update()
        return critic_loss, actor_loss

    def _update_critics(self, batch: dict) -> float:
        obs, act, rew, next_obs, done = (
            batch["obs"],
            batch["act"],
            batch["rew"],
            batch["next_obs"],
            batch["done"],
        )
        with torch.no_grad():
            noise = torch.clamp(
                torch.randn_like(act) * self.cfg.target_noise,
                -self.cfg.noise_clip,
                self.cfg.noise_clip,
            )
            next_act = torch.clamp(self.actor_target(next_obs) + noise, -self.cfg.act_limit, self.cfg.act_limit)
            target_q = torch.min(self.critic1_target(next_obs, next_act), self.critic2_target(next_obs, next_act))
            target = rew.unsqueeze(1) + (1 - done.unsqueeze(1)) * self.cfg.gamma * target_q
        q1 = self.critic1(obs, act)
        q2 = self.critic2(obs, act)
        loss = nn.MSELoss()(q1, target) + nn.MSELoss()(q2, target)
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.cfg.gradient_clip)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.cfg.gradient_clip)
        self.critic_optimizer.step()
        return float(loss.item())

    def _update_actor(self, batch: dict, expert_batch: dict | None = None, bc_weight: float = 0.0) -> float:
        actor_action = self.actor(batch["obs"])
        policy_loss = -self.critic1(batch["obs"], actor_action).mean()
        policy_loss = self.cfg.q_policy_weight * policy_loss
        loss = policy_loss
        if self.cfg.action_l2_weight > 0.0:
            action_l2 = torch.mean(torch.square(actor_action / self.cfg.act_limit))
            loss = loss + self.cfg.action_l2_weight * action_l2
        if expert_batch is not None and bc_weight > 0.0:
            expert_action = self.actor(expert_batch["obs"])
            bc_loss = nn.MSELoss()(expert_action / self.cfg.act_limit, expert_batch["act"] / self.cfg.act_limit)
            loss = policy_loss + float(bc_weight) * bc_loss
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.gradient_clip)
        self.actor_optimizer.step()
        return float(loss.item())

    def pretrain_actor_bc(
        self,
        expert_replay,
        epochs: int,
        batch_size: int,
        max_steps_per_epoch: int | None = None,
    ) -> list[float]:
        """Warm-start the actor by imitating the expert buffer actions."""
        losses: list[float] = []
        if expert_replay is None or len(expert_replay) == 0 or epochs <= 0:
            return losses
        steps_per_epoch = max(1, len(expert_replay) // max(1, batch_size))
        if max_steps_per_epoch is not None:
            steps_per_epoch = min(steps_per_epoch, max(1, int(max_steps_per_epoch)))
        for _ in range(int(epochs)):
            epoch_losses = []
            for _ in range(steps_per_epoch):
                batch = expert_replay.sample(batch_size, self.device)
                pred = self.actor(batch["obs"])
                loss = nn.MSELoss()(pred, batch["act"])
                self.actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.gradient_clip)
                self.actor_optimizer.step()
                epoch_losses.append(float(loss.item()))
            losses.append(float(np.mean(epoch_losses)))
        self.actor_target.load_state_dict(self.actor.state_dict())
        return losses

    def _soft_update(self):
        tau = self.cfg.tau
        for net, target in (
            (self.actor, self.actor_target),
            (self.critic1, self.critic1_target),
            (self.critic2, self.critic2_target),
        ):
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": self.cfg.__dict__,
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "critic1_target": self.critic1_target.state_dict(),
                "critic2_target": self.critic2_target.state_dict(),
            },
            path,
        )

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.critic1_target.load_state_dict(ckpt["critic1_target"])
        self.critic2_target.load_state_dict(ckpt["critic2_target"])
