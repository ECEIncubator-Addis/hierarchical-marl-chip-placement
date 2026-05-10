from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor


class RolloutBuffer:
    """On-policy storage with GAE for PPO/A2C style updates."""

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str | torch.device = "cpu",
    ) -> None:
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.capacity, *self.observation_shape), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.log_probs = np.zeros(self.capacity, dtype=np.float32)
        self.returns = np.zeros(self.capacity, dtype=np.float32)
        self.advantages = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        if self.full:
            raise RuntimeError("RolloutBuffer is full. Call reset() after update().")

        self.observations[self.pos] = observation
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.pos += 1
        self.full = self.pos == self.capacity

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def compute_returns_and_advantages(self, last_value: float, last_done: bool) -> None:
        last_gae = 0.0
        last_index = len(self) - 1
        for step in range(last_index, -1, -1):
            if step == last_index:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae

        size = len(self)
        self.returns[:size] = self.advantages[:size] + self.values[:size]

    def as_tensors(self) -> RolloutBatch:
        size = len(self)
        advantages = torch.as_tensor(self.advantages[:size], dtype=torch.float32, device=self.device)
        if size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        return RolloutBatch(
            observations=torch.as_tensor(self.observations[:size], dtype=torch.float32, device=self.device),
            actions=torch.as_tensor(self.actions[:size], dtype=torch.long, device=self.device),
            old_log_probs=torch.as_tensor(self.log_probs[:size], dtype=torch.float32, device=self.device),
            returns=torch.as_tensor(self.returns[:size], dtype=torch.float32, device=self.device),
            advantages=advantages,
            values=torch.as_tensor(self.values[:size], dtype=torch.float32, device=self.device),
        )

    def minibatches(self, batch_size: int, shuffle: bool = True) -> Iterator[RolloutBatch]:
        batch = self.as_tensors()
        size = batch.actions.shape[0]
        indices = torch.randperm(size, device=self.device) if shuffle else torch.arange(size, device=self.device)

        for start in range(0, size, batch_size):
            idx = indices[start : start + batch_size]
            yield RolloutBatch(
                observations=batch.observations[idx],
                actions=batch.actions[idx],
                old_log_probs=batch.old_log_probs[idx],
                returns=batch.returns[idx],
                advantages=batch.advantages[idx],
                values=batch.values[idx],
            )


@dataclass
class ReplayBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Fixed-size replay memory for off-policy discrete-action agents."""

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        device: str | torch.device = "cpu",
    ) -> None:
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.device = torch.device(device)
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.pos] = observation
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_observations[self.pos] = next_observation
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def sample(self, batch_size: int) -> ReplayBatch:
        size = len(self)
        if size < batch_size:
            raise ValueError(f"Not enough samples: requested {batch_size}, buffer has {size}.")

        idx = np.random.randint(0, size, size=batch_size)
        return ReplayBatch(
            observations=torch.as_tensor(self.observations[idx], dtype=torch.float32, device=self.device),
            actions=torch.as_tensor(self.actions[idx], dtype=torch.long, device=self.device),
            rewards=torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=self.device),
            next_observations=torch.as_tensor(self.next_observations[idx], dtype=torch.float32, device=self.device),
            dones=torch.as_tensor(self.dones[idx], dtype=torch.float32, device=self.device),
        )
