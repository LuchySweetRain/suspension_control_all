from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import TransformerCritic, TransformerGaussianActor


@dataclass
class SACConfig:
    obs_dim_base: int
    preview_steps: int
    preview_token_dim: int
    act_dim: int
    act_limit: float
    hidden_sizes: list[int]
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    gamma: float
    tau: float
    gradient_clip: float
    base_encoder_dim: int
    preview_embed_dim: int
    transformer_layers: int
    transformer_heads: int
    transformer_ff_dim: int
    transformer_dropout: float
    alpha: float = 0.2
    automatic_entropy_tuning: bool = True
    target_entropy: float | None = None
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    @property
    def obs_dim(self) -> int:
        return self.obs_dim_base + self.preview_steps * self.preview_token_dim

    @classmethod
    def from_project_config(cls, config: dict) -> "SACConfig":
        rl = config["rl"]
        sac = rl.get("sac", {})
        act_dim = 2
        return cls(
            obs_dim_base=14,
            preview_steps=int(config["preview"]["steps"]),
            preview_token_dim=2,
            act_dim=act_dim,
            act_limit=float(config["force_limit"]),
            hidden_sizes=list(rl["hidden_sizes"]),
            actor_lr=float(rl["actor_lr"]),
            critic_lr=float(rl["critic_lr"]),
            alpha_lr=float(sac.get("alpha_lr", rl["actor_lr"])),
            gamma=float(rl["gamma"]),
            tau=float(rl["tau"]),
            gradient_clip=float(rl["gradient_clip"]),
            base_encoder_dim=128,
            preview_embed_dim=int(rl["preview_embed_dim"]),
            transformer_layers=int(rl["transformer_layers"]),
            transformer_heads=int(rl["transformer_heads"]),
            transformer_ff_dim=int(rl["transformer_ff_dim"]),
            transformer_dropout=float(rl["transformer_dropout"]),
            alpha=float(sac.get("alpha", 0.2)),
            automatic_entropy_tuning=bool(sac.get("automatic_entropy_tuning", True)),
            target_entropy=float(sac.get("target_entropy", -act_dim)),
            log_std_min=float(sac.get("log_std_min", -20.0)),
            log_std_max=float(sac.get("log_std_max", 2.0)),
        )


class SACAgent:
    def __init__(self, cfg: SACConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.actor = TransformerGaussianActor(cfg).to(device)
        self.critic1 = TransformerCritic(cfg).to(device)
        self.critic2 = TransformerCritic(cfg).to(device)
        self.critic1_target = TransformerCritic(cfg).to(device)
        self.critic2_target = TransformerCritic(cfg).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=cfg.critic_lr,
        )
        self.log_alpha = torch.tensor(np.log(cfg.alpha), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray, noise: float = 0.0) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            action, _ = self.actor.sample(obs_t, deterministic=noise <= 0.0)
        if was_training:
            self.actor.train()
        return action.cpu().numpy()[0].astype(np.float32)

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
            next_act, next_log_prob = self.actor.sample(next_obs)
            target_q = torch.min(self.critic1_target(next_obs, next_act), self.critic2_target(next_obs, next_act))
            target_q = target_q - self.alpha.detach() * next_log_prob
            target = rew.unsqueeze(1) + (1 - done.unsqueeze(1)) * self.cfg.gamma * target_q
        q1 = self.critic1(obs, act)
        q2 = self.critic2(obs, act)
        critic_loss = nn.MSELoss()(q1, target) + nn.MSELoss()(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.cfg.gradient_clip)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.cfg.gradient_clip)
        self.critic_optimizer.step()

        new_act, log_prob = self.actor.sample(obs)
        q_new = torch.min(self.critic1(obs, new_act), self.critic2(obs, new_act))
        actor_loss = (self.alpha.detach() * log_prob - q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.gradient_clip)
        self.actor_optimizer.step()

        if self.cfg.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + float(self.cfg.target_entropy)).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        self._soft_update()
        return float(critic_loss.item()), float(actor_loss.item())

    def _soft_update(self):
        for net, target in ((self.critic1, self.critic1_target), (self.critic2, self.critic2_target)):
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.data.copy_(self.cfg.tau * p.data + (1 - self.cfg.tau) * tp.data)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": self.cfg.__dict__,
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "critic1_target": self.critic1_target.state_dict(),
                "critic2_target": self.critic2_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
            },
            path,
        )

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.critic1_target.load_state_dict(ckpt["critic1_target"])
        self.critic2_target.load_state_dict(ckpt["critic2_target"])
        if "log_alpha" in ckpt:
            self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
