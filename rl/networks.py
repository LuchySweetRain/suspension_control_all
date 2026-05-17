from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


def mlp(input_dim: int, hidden_sizes: list[int], output_dim: int, final_tanh: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for hidden in hidden_sizes:
        layers += [nn.Linear(prev, hidden), nn.ReLU()]
        prev = hidden
    layers.append(nn.Linear(prev, output_dim))
    if final_tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PreviewEncoder(nn.Module):
    def __init__(self, token_dim: int, embed_dim: int, layers: int, heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(token_dim, embed_dim)
        self.pos = SinusoidalPositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.pos(self.proj(tokens))
        encoded = self.encoder(x)
        return encoded.mean(dim=1)


@dataclass
class ObsSpec:
    base_dim: int
    preview_steps: int
    preview_token_dim: int

    @property
    def obs_dim(self) -> int:
        return self.base_dim + self.preview_steps * self.preview_token_dim

    def split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = obs[..., : self.base_dim]
        preview = obs[..., self.base_dim :].reshape(
            obs.shape[0], self.preview_steps, self.preview_token_dim
        )
        return base, preview


class TransformerActor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.spec = ObsSpec(cfg.obs_dim_base, cfg.preview_steps, cfg.preview_token_dim)
        self.act_limit = cfg.act_limit
        self.base_encoder = mlp(cfg.obs_dim_base, [cfg.base_encoder_dim, cfg.base_encoder_dim], cfg.base_encoder_dim)
        self.preview_encoder = PreviewEncoder(
            cfg.preview_token_dim,
            cfg.preview_embed_dim,
            cfg.transformer_layers,
            cfg.transformer_heads,
            cfg.transformer_ff_dim,
            cfg.transformer_dropout,
        )
        self.head = mlp(
            cfg.base_encoder_dim + cfg.preview_embed_dim,
            cfg.hidden_sizes,
            cfg.act_dim,
            final_tanh=True,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        base, preview = self.spec.split(obs)
        x = torch.cat([self.base_encoder(base), self.preview_encoder(preview)], dim=-1)
        return self.act_limit * self.head(x)


class TransformerCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.spec = ObsSpec(cfg.obs_dim_base, cfg.preview_steps, cfg.preview_token_dim)
        self.base_encoder = mlp(cfg.obs_dim_base, [cfg.base_encoder_dim, cfg.base_encoder_dim], cfg.base_encoder_dim)
        self.preview_encoder = PreviewEncoder(
            cfg.preview_token_dim,
            cfg.preview_embed_dim,
            cfg.transformer_layers,
            cfg.transformer_heads,
            cfg.transformer_ff_dim,
            cfg.transformer_dropout,
        )
        self.head = mlp(
            cfg.base_encoder_dim + cfg.preview_embed_dim + cfg.act_dim,
            cfg.hidden_sizes,
            1,
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        base, preview = self.spec.split(obs)
        x = torch.cat([self.base_encoder(base), self.preview_encoder(preview), act], dim=-1)
        return self.head(x)

