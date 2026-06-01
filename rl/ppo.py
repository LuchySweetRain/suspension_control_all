from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import TransformerGaussianActor, TransformerValue


@dataclass
class PPOConfig:
    obs_dim_base: int
    preview_steps: int
    preview_token_dim: int
    act_dim: int
    act_limit: float
    hidden_sizes: list[int]
    actor_lr: float
    critic_lr: float
    gamma: float
    lam: float
    clip_ratio: float
    train_epochs: int
    minibatch_size: int
    entropy_coef: float
    value_coef: float
    projection_penalty_weight: float
    gradient_clip: float
    base_encoder_dim: int
    preview_embed_dim: int
    transformer_layers: int
    transformer_heads: int
    transformer_ff_dim: int
    transformer_dropout: float
    log_std_min: float = -5.0
    log_std_max: float = 1.0

    @property
    def obs_dim(self) -> int:
        return self.obs_dim_base + self.preview_steps * self.preview_token_dim

    @classmethod
    def from_project_config(cls, config: dict) -> "PPOConfig":
        rl = config["rl"]
        ppo = rl.get("ppo", {})
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
            lam=float(ppo.get("lam", 0.95)),
            clip_ratio=float(ppo.get("clip_ratio", 0.2)),
            train_epochs=int(ppo.get("train_epochs", 8)),
            minibatch_size=int(ppo.get("minibatch_size", rl.get("batch_size", 128))),
            entropy_coef=float(ppo.get("entropy_coef", 0.0)),
            value_coef=float(ppo.get("value_coef", 0.5)),
            projection_penalty_weight=float(ppo.get("projection_penalty_weight", 0.0)),
            gradient_clip=float(rl["gradient_clip"]),
            base_encoder_dim=128,
            preview_embed_dim=int(rl["preview_embed_dim"]),
            transformer_layers=int(rl["transformer_layers"]),
            transformer_heads=int(rl["transformer_heads"]),
            transformer_ff_dim=int(rl["transformer_ff_dim"]),
            transformer_dropout=float(rl["transformer_dropout"]),
            log_std_min=float(ppo.get("log_std_min", -5.0)),
            log_std_max=float(ppo.get("log_std_max", 1.0)),
        )


class PPOAgent:
    def __init__(self, cfg: PPOConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.actor = TransformerGaussianActor(cfg).to(device)
        self.value = TransformerValue(cfg).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=cfg.critic_lr)
        self.bc_anchor_obs: torch.Tensor | None = None
        self.bc_anchor_act: torch.Tensor | None = None

    def select_action(self, obs: np.ndarray, noise: float = 0.0) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            action, _ = self.actor.sample(obs_t, deterministic=noise <= 0.0)
        if was_training:
            self.actor.train()
        return action.cpu().numpy()[0].astype(np.float32)

    def act_for_training(self, obs: np.ndarray) -> tuple[np.ndarray, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, logp = self.actor.sample(obs_t, deterministic=False)
            value = self.value(obs_t)
        return action.cpu().numpy()[0].astype(np.float32), float(logp.item()), float(value.item())

    def log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        mu, log_std = self.actor(obs)
        scaled = torch.clamp(act / self.cfg.act_limit, -0.999999, 0.999999)
        pre_tanh = torch.atanh(scaled)
        std = log_std.exp()
        normal_log_prob = -0.5 * (
            ((pre_tanh - mu) / (std + 1e-8)).pow(2) + 2 * log_std + np.log(2 * np.pi)
        )
        log_prob = normal_log_prob.sum(dim=-1, keepdim=True)
        log_prob -= torch.log(self.cfg.act_limit * (1 - scaled.pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
        entropy = (0.5 + 0.5 * np.log(2 * np.pi) + log_std).sum(dim=-1, keepdim=True)
        return log_prob, entropy

    def set_bc_anchor(self, obs: np.ndarray, act: np.ndarray):
        self.bc_anchor_obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        self.bc_anchor_act = torch.as_tensor(np.asarray(act), dtype=torch.float32, device=self.device)

    def train_trajectory(self, trajectory: dict[str, list], bc_anchor_weight: float = 0.0, bc_anchor_batch_size: int | None = None) -> tuple[float, float]:
        obs = torch.as_tensor(np.asarray(trajectory["obs"]), dtype=torch.float32, device=self.device)
        act = torch.as_tensor(np.asarray(trajectory["act"]), dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(np.asarray(trajectory["logp"]), dtype=torch.float32, device=self.device).unsqueeze(1)
        returns = torch.as_tensor(np.asarray(trajectory["ret"]), dtype=torch.float32, device=self.device).unsqueeze(1)
        adv = torch.as_tensor(np.asarray(trajectory["adv"]), dtype=torch.float32, device=self.device).unsqueeze(1)
        projection_error = torch.as_tensor(
            np.asarray(trajectory.get("projection_error", np.zeros(len(trajectory["obs"])))),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = obs.shape[0]
        actor_losses = []
        value_losses = []
        for _ in range(self.cfg.train_epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.cfg.minibatch_size):
                idx = indices[start : start + self.cfg.minibatch_size]
                logp, entropy = self.log_prob(obs[idx], act[idx])
                ratio = torch.exp(logp - old_logp[idx])
                clipped = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * adv[idx]
                actor_loss = -(torch.min(ratio * adv[idx], clipped)).mean()
                actor_loss = actor_loss - self.cfg.entropy_coef * entropy.mean()
                if self.cfg.projection_penalty_weight > 0.0:
                    actor_loss = actor_loss + self.cfg.projection_penalty_weight * (ratio * projection_error[idx]).mean()
                if bc_anchor_weight > 0.0 and self.bc_anchor_obs is not None and self.bc_anchor_act is not None:
                    anchor_n = int(self.bc_anchor_obs.shape[0])
                    anchor_batch = int(bc_anchor_batch_size or idx.shape[0])
                    anchor_idx = torch.randint(0, anchor_n, (anchor_batch,), device=self.device)
                    anchor_pred, _ = self.actor.sample(self.bc_anchor_obs[anchor_idx], deterministic=True)
                    anchor_loss = nn.MSELoss()(
                        anchor_pred / self.cfg.act_limit,
                        self.bc_anchor_act[anchor_idx] / self.cfg.act_limit,
                    )
                    actor_loss = actor_loss + float(bc_anchor_weight) * anchor_loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.gradient_clip)
                self.actor_optimizer.step()

                value_loss = nn.MSELoss()(self.value(obs[idx]), returns[idx]) * self.cfg.value_coef
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.gradient_clip)
                self.value_optimizer.step()
                actor_losses.append(float(actor_loss.item()))
                value_losses.append(float(value_loss.item()))
        return float(np.mean(value_losses)), float(np.mean(actor_losses))

    def pretrain_actor_bc(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        epochs: int,
        batch_size: int,
        max_steps_per_epoch: int | None = None,
    ) -> list[float]:
        obs = np.asarray(obs, dtype=np.float32)
        act = np.asarray(act, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != self.cfg.obs_dim:
            raise ValueError(f"BC obs must have shape (N, {self.cfg.obs_dim}), got {obs.shape}")
        if act.ndim != 2 or act.shape[1] != self.cfg.act_dim:
            raise ValueError(f"BC act must have shape (N, {self.cfg.act_dim}), got {act.shape}")
        if len(obs) == 0 or epochs <= 0:
            return []
        n = obs.shape[0]
        losses: list[float] = []
        steps_per_epoch = max(1, int(np.ceil(n / max(1, batch_size))))
        if max_steps_per_epoch is not None:
            steps_per_epoch = min(steps_per_epoch, max(1, int(max_steps_per_epoch)))
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        for _ in range(int(epochs)):
            epoch_losses = []
            for _ in range(steps_per_epoch):
                idx = torch.randint(0, n, (int(batch_size),), device=self.device)
                pred, _ = self.actor.sample(obs_t[idx], deterministic=True)
                loss = nn.MSELoss()(pred / self.cfg.act_limit, act_t[idx] / self.cfg.act_limit)
                self.actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.gradient_clip)
                self.actor_optimizer.step()
                epoch_losses.append(float(loss.item()))
            losses.append(float(np.mean(epoch_losses)))
        return losses

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": self.cfg.__dict__,
                "actor": self.actor.state_dict(),
                "value": self.value.state_dict(),
            },
            path,
        )

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.value.load_state_dict(ckpt["value"])


def finish_ppo_trajectory(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float,
    lam: float,
) -> tuple[list[float], list[float]]:
    adv = np.zeros(len(rewards), dtype=np.float32)
    last_gae = 0.0
    values_ext = values + [0.0]
    for t in reversed(range(len(rewards))):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * values_ext[t + 1] * nonterminal - values_ext[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + np.asarray(values, dtype=np.float32)
    return returns.tolist(), adv.tolist()
