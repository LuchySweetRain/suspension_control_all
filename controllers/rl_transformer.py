from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .base import Controller
from rl.ddpg import DDPGAgent, DDPGConfig
from rl.ppo import PPOAgent, PPOConfig
from rl.sac import SACAgent, SACConfig
from rl.td3 import TD3Agent, TD3Config


ALGORITHMS = {
    "td3": (TD3Config, TD3Agent, "TD3"),
    "ddpg": (DDPGConfig, DDPGAgent, "DDPG"),
    "sac": (SACConfig, SACAgent, "SAC"),
    "ppo": (PPOConfig, PPOAgent, "PPO"),
}


class RLTransformerController(Controller):
    def __init__(
        self,
        config: dict,
        algorithm: str = "td3",
        checkpoint: str | Path | None = None,
        device: str | None = None,
    ):
        key = algorithm.lower()
        if key not in ALGORITHMS:
            raise ValueError(f"Unknown RL algorithm: {algorithm}")
        config_cls, agent_cls, label = ALGORITHMS[key]
        self.name = label
        self.config = config_cls.from_project_config(config)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.agent = agent_cls(self.config, self.device)
        if checkpoint:
            self.agent.load(Path(checkpoint))

    def reset(self):
        pass

    def compute_action(self, obs: np.ndarray, info: dict) -> np.ndarray:
        return self.agent.select_action(obs, noise=0.0)
