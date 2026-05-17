from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .base import Controller
from rl.td3 import TD3Agent, TD3Config


class RLTransformerTD3Controller(Controller):
    name = "RL"

    def __init__(self, config: dict, checkpoint: str | Path | None = None, device: str | None = None):
        self.config = TD3Config.from_project_config(config)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.agent = TD3Agent(self.config, self.device)
        if checkpoint:
            self.agent.load(Path(checkpoint))

    def reset(self):
        pass

    def compute_action(self, obs: np.ndarray, info: dict) -> np.ndarray:
        return self.agent.select_action(obs, noise=0.0)

