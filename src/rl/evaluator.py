from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import torch


@dataclass
class EvaluationResult:
    episodes: int
    mean_reward: float
    std_reward: float
    mean_length: float
    best_reward: float
    best_hpwl: float | None
    final_hpwl_mean: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _hpwl(env: Any) -> float | None:
    compute_hpwl = getattr(env, "_compute_hpwl", None)
    if callable(compute_hpwl):
        return float(compute_hpwl())
    return None


def evaluate_policy(
    agent: Any,
    env: Any,
    episodes: int = 5,
    deterministic: bool = True,
    render: bool = False,
) -> EvaluationResult:
    """Run complete episodes and summarize reward plus placement HPWL when available."""

    rewards: list[float] = []
    lengths: list[int] = []
    final_hpwls: list[float] = []
    best_reward = -float("inf")
    best_hpwl: float | None = None

    for _ in range(episodes):
        observation, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            action, *_ = agent.act(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            episode_reward += float(reward)
            episode_length += 1
            if render:
                env.render()

        rewards.append(episode_reward)
        lengths.append(episode_length)
        hpwl = _hpwl(env)
        if hpwl is not None:
            final_hpwls.append(hpwl)
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_hpwl = hpwl

    return EvaluationResult(
        episodes=episodes,
        mean_reward=float(np.mean(rewards)) if rewards else 0.0,
        std_reward=float(np.std(rewards)) if rewards else 0.0,
        mean_length=float(np.mean(lengths)) if lengths else 0.0,
        best_reward=float(best_reward) if rewards else 0.0,
        best_hpwl=best_hpwl,
        final_hpwl_mean=float(np.mean(final_hpwls)) if final_hpwls else None,
    )


@torch.no_grad()
def rollout_placement(agent: Any, env: Any, deterministic: bool = True) -> tuple[Any, EvaluationResult]:
    """Run one episode and return the environment graph in its final placement state."""

    result = evaluate_policy(agent, env, episodes=1, deterministic=deterministic)
    return getattr(env, "graph", None), result
