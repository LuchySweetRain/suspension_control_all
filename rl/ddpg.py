from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import TransformerActor, TransformerCritic


@dataclass
class DDPGConfig:
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
    def from_project_config(cls, config: dict) -> "DDPGConfig":
        rl = config["rl"]
        return cls(
            obs_dim_base=14,
            preview_steps=int(config["preview"]["steps"]),
            preview_token_dim=2,
            act_dim=2,
            act_limit=float(config["force_limit"]),
            hidden_sizes=list(rl["hidden_sizes"]),
            actor_lr=float(rl["actor_lr"]),
            critic_lr=float(rl["critic_lr"]),
            gamma=float(rl["gamma"]),
            tau=float(rl["tau"]),
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


class DDPGAgent:
    def __init__(self, cfg: DDPGConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.actor = TransformerActor(cfg).to(device)
        self.actor_target = TransformerActor(cfg).to(device)
        self.critic = TransformerCritic(cfg).to(device)
        self.critic_target = TransformerCritic(cfg).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

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

    def train_step(self, replay, batch_size: int, **_) -> tuple[float, float]:
        batch = replay.sample(batch_size, self.device)
        obs, act, rew, next_obs, done = (
            batch["obs"],
            batch["act"],
            batch["rew"],
            batch["next_obs"],
            batch["done"],
        )
        with torch.no_grad():
            next_act = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, next_act)
            target = rew.unsqueeze(1) + (1 - done.unsqueeze(1)) * self.cfg.gamma * target_q
        q = self.critic(obs, act)
        critic_loss = nn.MSELoss()(q, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.gradient_clip)
        self.critic_optimizer.step()

        actor_action = self.actor(obs)
        actor_loss = -self.critic(obs, actor_action).mean() * self.cfg.q_policy_weight
        if self.cfg.action_l2_weight > 0.0:
            actor_loss = actor_loss + self.cfg.action_l2_weight * torch.mean(
                torch.square(actor_action / self.cfg.act_limit)
            )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.gradient_clip)
        self.actor_optimizer.step()
        self._soft_update()
        return float(critic_loss.item()), float(actor_loss.item())

    def pretrain_actor_bc(self, expert_replay, epochs: int, batch_size: int, max_steps_per_epoch: int | None = None):
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
        for net, target in ((self.actor, self.actor_target), (self.critic, self.critic_target)):
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.data.copy_(self.cfg.tau * p.data + (1 - self.cfg.tau) * tp.data)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": self.cfg.__dict__,
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
            },
            path,
        )

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
