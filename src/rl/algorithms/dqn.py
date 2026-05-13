from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.rl.buffer import ReplayBuffer
except ImportError:  # pragma: no cover
    from rl.buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, hidden_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = obs_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(last_dim, hidden_dim), nn.ReLU()])
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        return self.net(observations)


@dataclass
class DQNConfig:
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    train_frequency: int = 4
    target_update_frequency: int = 1_000
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 50_000
    hidden_dim: int = 256
    hidden_layers: int = 2
    max_grad_norm: float = 10.0


class DQNAgent:
    """DQN baseline over the flattened macro-direction action space."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: DQNConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config or DQNConfig()
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.online = QNetwork(obs_dim, action_dim, self.config.hidden_dim, self.config.hidden_layers).to(self.device)
        self.target = QNetwork(obs_dim, action_dim, self.config.hidden_dim, self.config.hidden_layers).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.steps = 0

    def epsilon(self) -> float:
        fraction = min(1.0, self.steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + fraction * (self.config.epsilon_final - self.config.epsilon_start)

    @torch.no_grad()
    def act(self, observation, deterministic: bool = False) -> tuple[int, float, float]:
        if not deterministic:
            self.steps += 1
        if not deterministic and random.random() < self.epsilon():
            return random.randrange(self.action_dim), 0.0, 0.0

        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        q_values = self.online(obs).squeeze(0)
        action = int(q_values.argmax().item())
        return action, 0.0, float(q_values[action].item())

    def update(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
        if len(replay_buffer) < max(self.config.batch_size, self.config.learning_starts):
            return {}

        batch = replay_buffer.sample(self.config.batch_size)
        q_values = self.online(batch.observations).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(batch.next_observations).argmax(dim=1, keepdim=True)
            next_q = self.target(batch.next_observations).gather(1, next_actions).squeeze(1)
            targets = batch.rewards + self.config.gamma * (1.0 - batch.dones) * next_q

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        if self.steps % self.config.target_update_frequency == 0:
            self.target.load_state_dict(self.online.state_dict())

        return {"loss": float(loss.detach().cpu()), "epsilon": self.epsilon()}

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "config": self.config.__dict__,
                "action_dim": self.action_dim,
                "obs_dim": self.online.net[0].in_features,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "DQNAgent":
        payload = torch.load(path, map_location=device)
        config = DQNConfig(**payload["config"])
        agent = cls(
            obs_dim=payload["obs_dim"],
            action_dim=payload["action_dim"],
            config=config,
            device=device,
        )
        agent.online.load_state_dict(payload["online"])
        agent.target.load_state_dict(payload["target"])
        return agent
