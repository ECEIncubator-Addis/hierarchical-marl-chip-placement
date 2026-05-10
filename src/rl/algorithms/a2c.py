from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.rl.algorithms.ppo import HierarchicalActorCritic
    from src.rl.buffer import RolloutBuffer
except ImportError:  # pragma: no cover
    from rl.algorithms.ppo import HierarchicalActorCritic
    from rl.buffer import RolloutBuffer


@dataclass
class A2CConfig:
    learning_rate: float = 7e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dim: int = 256
    hidden_layers: int = 2


class A2CAgent:
    """Synchronous advantage actor-critic with hierarchical discrete actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_macros: int | None = None,
        num_directions: int = 4,
        config: A2CConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config or A2CConfig()
        self.device = torch.device(device)
        self.num_directions = num_directions
        self.num_macros = num_macros or max(1, action_dim // num_directions)
        self.model = HierarchicalActorCritic(
            obs_dim=obs_dim,
            num_macros=self.num_macros,
            num_directions=num_directions,
            hidden_dim=self.config.hidden_dim,
            hidden_layers=self.config.hidden_layers,
        ).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config.learning_rate, eps=1e-5)

    @torch.no_grad()
    def act(self, observation, deterministic: bool = False) -> tuple[int, float, float]:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        action, log_prob, value = self.model.act(obs, deterministic=deterministic)
        return int(action.item()), float(log_prob.item()), float(value.item())

    @torch.no_grad()
    def value(self, observation) -> float:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        _, _, values = self.model.forward(obs)
        return float(values.item())

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        batch = buffer.as_tensors()
        log_probs, entropy, values = self.model.evaluate_actions(batch.observations, batch.actions)
        policy_loss = -(log_probs * batch.advantages).mean()
        value_loss = F.mse_loss(values, batch.returns)
        loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy.mean().detach().cpu()),
        }

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.config.__dict__,
                "num_macros": self.num_macros,
                "num_directions": self.num_directions,
                "obs_dim": self.model.obs_dim,
            },
            path,
        )
