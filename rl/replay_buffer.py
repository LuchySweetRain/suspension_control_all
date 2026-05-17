from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.sample_indices(idx, device)

    def sample_indices(self, idx, device: torch.device):
        idx = np.asarray(idx, dtype=np.int64)
        return {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            "act": torch.as_tensor(self.act[idx], dtype=torch.float32, device=device),
            "rew": torch.as_tensor(self.rew[idx], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "done": torch.as_tensor(self.done[idx], dtype=torch.float32, device=device),
        }

    def __len__(self) -> int:
        return self.size


class MixedReplaySampler:
    """Sample batches from online and expert buffers with a decaying expert ratio."""

    def __init__(
        self,
        online_buffer: ReplayBuffer,
        expert_buffer: ReplayBuffer,
        initial_expert_ratio: float,
        final_expert_ratio: float,
        decay_episodes: int,
    ):
        self.online_buffer = online_buffer
        self.expert_buffer = expert_buffer
        self.initial_expert_ratio = float(initial_expert_ratio)
        self.final_expert_ratio = float(final_expert_ratio)
        self.decay_episodes = max(1, int(decay_episodes))
        self.current_episode = 0

    def set_episode(self, episode: int):
        self.current_episode = max(0, int(episode))

    @property
    def expert_ratio(self) -> float:
        progress = min(1.0, self.current_episode / self.decay_episodes)
        return self.initial_expert_ratio + progress * (self.final_expert_ratio - self.initial_expert_ratio)

    def sample(self, batch_size: int, device: torch.device):
        if len(self.expert_buffer) == 0:
            return self.online_buffer.sample(batch_size, device)
        if len(self.online_buffer) == 0:
            return self.expert_buffer.sample(batch_size, device)

        expert_n = int(round(batch_size * self.expert_ratio))
        expert_n = min(max(expert_n, 0), batch_size)
        online_n = batch_size - expert_n
        parts = []
        if online_n > 0:
            parts.append(self.online_buffer.sample(online_n, device))
        if expert_n > 0:
            parts.append(self.expert_buffer.sample(expert_n, device))
        if len(parts) == 1:
            return parts[0]
        return {
            key: torch.cat([part[key] for part in parts], dim=0)
            for key in parts[0].keys()
        }
