from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

try:
    from src.models.macro_placement_env import MacroPlacementEnv
    from src.rl.algorithms.a2c import A2CAgent, A2CConfig
    from src.rl.algorithms.dqn import DQNAgent, DQNConfig
    from src.rl.algorithms.ppo import PPOAgent, PPOConfig
    from src.rl.buffer import ReplayBuffer, RolloutBuffer
    from src.rl.evaluator import EvaluationResult, evaluate_policy
except ImportError:  # pragma: no cover
    from models.macro_placement_env import MacroPlacementEnv
    from rl.algorithms.a2c import A2CAgent, A2CConfig
    from rl.algorithms.dqn import DQNAgent, DQNConfig
    from rl.algorithms.ppo import PPOAgent, PPOConfig
    from rl.buffer import ReplayBuffer, RolloutBuffer
    from rl.evaluator import EvaluationResult, evaluate_policy


AlgorithmName = Literal["ppo", "a2c", "dqn"]


@dataclass
class TrainerConfig:
    graph_path: str
    algorithm: AlgorithmName = "ppo"
    total_timesteps: int = 100_000
    rollout_steps: int = 2_048
    eval_frequency: int = 10_000
    eval_episodes: int = 3
    seed: int | None = 42
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    save_best_graph: bool = True
    num_directions: int = 4
    algorithm_config: dict[str, Any] = field(default_factory=dict)


class HierarchicalRLTrainer:
    """End-to-end trainer for macro-placement agents."""

    def __init__(self, config: TrainerConfig, env: Any | None = None) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.env = env or MacroPlacementEnv(config.graph_path)
        self.eval_env = MacroPlacementEnv(config.graph_path)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if config.seed is not None:
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            self.env.reset(seed=config.seed)
            self.eval_env.reset(seed=config.seed + 1)

        self.obs_shape = tuple(self.env.observation_space.shape)
        self.obs_dim = int(np.prod(self.obs_shape))
        self.action_dim = int(self.env.action_space.n)
        self.num_macros = getattr(self.env, "num_macros", max(1, self.action_dim // config.num_directions))
        self.agent = self._build_agent()
        self.best_eval_reward = -float("inf")
        self.history: list[dict[str, Any]] = []

    def _build_agent(self) -> Any:
        cfg = self.config.algorithm_config
        if self.config.algorithm == "ppo":
            return PPOAgent(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                num_macros=self.num_macros,
                num_directions=self.config.num_directions,
                config=PPOConfig(**cfg),
                device=self.device,
            )
        if self.config.algorithm == "a2c":
            return A2CAgent(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                num_macros=self.num_macros,
                num_directions=self.config.num_directions,
                config=A2CConfig(**cfg),
                device=self.device,
            )
        if self.config.algorithm == "dqn":
            return DQNAgent(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                config=DQNConfig(**cfg),
                device=self.device,
            )
        raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")

    def train(self) -> list[dict[str, Any]]:
        if self.config.algorithm == "dqn":
            return self._train_dqn()
        return self._train_on_policy()

    def _train_on_policy(self) -> list[dict[str, Any]]:
        observation, _ = self.env.reset(seed=self.config.seed)
        episode_reward = 0.0
        episode_length = 0
        completed_episodes = 0
        global_step = 0

        while global_step < self.config.total_timesteps:
            steps = min(self.config.rollout_steps, self.config.total_timesteps - global_step)
            buffer = RolloutBuffer(
                capacity=steps,
                observation_shape=self.obs_shape,
                gamma=self.agent.config.gamma,
                gae_lambda=self.agent.config.gae_lambda,
                device=self.device,
            )
            last_done = False

            for _ in range(steps):
                action, log_prob, value = self.agent.act(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                done = bool(terminated or truncated)
                buffer.add(observation, action, reward, done, value, log_prob)

                episode_reward += float(reward)
                episode_length += 1
                global_step += 1
                last_done = done
                observation = next_observation

                if done:
                    self.history.append(
                        {
                            "step": global_step,
                            "episode": completed_episodes,
                            "episode_reward": episode_reward,
                            "episode_length": episode_length,
                        }
                    )
                    completed_episodes += 1
                    observation, _ = self.env.reset()
                    episode_reward = 0.0
                    episode_length = 0

            last_value = 0.0 if last_done else self.agent.value(observation)
            buffer.compute_returns_and_advantages(last_value=last_value, last_done=last_done)
            train_metrics = self.agent.update(buffer)
            self._maybe_evaluate(global_step, train_metrics)

        return self.history

    def _train_dqn(self) -> list[dict[str, Any]]:
        cfg: DQNConfig = self.agent.config
        replay = ReplayBuffer(cfg.buffer_size, self.obs_shape, device=self.device)
        observation, _ = self.env.reset(seed=self.config.seed)
        episode_reward = 0.0
        episode_length = 0
        completed_episodes = 0

        for step in range(1, self.config.total_timesteps + 1):
            action, _, _ = self.agent.act(observation)
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated or truncated)
            replay.add(observation, action, reward, next_observation, done)

            train_metrics = {}
            if step % cfg.train_frequency == 0:
                train_metrics = self.agent.update(replay)

            episode_reward += float(reward)
            episode_length += 1
            observation = next_observation

            if done:
                self.history.append(
                    {
                        "step": step,
                        "episode": completed_episodes,
                        "episode_reward": episode_reward,
                        "episode_length": episode_length,
                    }
                )
                completed_episodes += 1
                observation, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

            self._maybe_evaluate(step, train_metrics)

        return self.history

    def _maybe_evaluate(self, step: int, train_metrics: dict[str, float]) -> None:
        if self.config.eval_frequency <= 0 or step % self.config.eval_frequency != 0:
            return

        result = evaluate_policy(self.agent, self.eval_env, episodes=self.config.eval_episodes)
        record = {"step": step, "eval": result.to_dict(), "train": train_metrics}
        self.history.append(record)
        if result.mean_reward > self.best_eval_reward:
            self.best_eval_reward = result.mean_reward
            self._save_checkpoint(step, result)

    def _save_checkpoint(self, step: int, result: EvaluationResult) -> None:
        model_path = self.checkpoint_dir / f"best_{self.config.algorithm}.pt"
        save = getattr(self.agent, "save", None)
        if callable(save):
            save(model_path)

        if self.config.save_best_graph and hasattr(self.eval_env, "graph"):
            graph_path = self.checkpoint_dir / f"best_placement_step_{step}.pt"
            torch.save(
                {
                    "graph": self.eval_env.graph,
                    "metrics": result.to_dict(),
                    "algorithm": self.config.algorithm,
                    "step": step,
                },
                graph_path,
            )


def train(config: TrainerConfig) -> tuple[Any, list[dict[str, Any]]]:
    trainer = HierarchicalRLTrainer(config)
    history = trainer.train()
    return trainer.agent, history
