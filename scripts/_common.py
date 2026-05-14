from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def make_env(config: Any, *, eval_env: bool = True) -> Any:
    env_config = config.eval_env_config if eval_env else config.env_config
    env_config = env_config or config.env_config
    try:
        from configs.rl_envs import MacroPlacementEnv
    except ImportError:
        from src.models.macro_placement_env import MacroPlacementEnv

    try:
        return MacroPlacementEnv(config.graph_path, **env_config)
    except TypeError:
        legacy_config = {key: value for key, value in env_config.items() if key == "max_steps"}
        return MacroPlacementEnv(config.graph_path, **legacy_config)


def load_agent(checkpoint_path: str | Path, algorithm: str, device: str = "cpu") -> Any:
    checkpoint_path = resolve_project_path(checkpoint_path)
    algorithm = algorithm.lower()

    if algorithm == "ppo":
        from src.rl.algorithms.ppo import PPOAgent

        return PPOAgent.load(checkpoint_path, device=device)

    payload = torch.load(checkpoint_path, map_location=device)

    if algorithm == "a2c":
        from src.rl.algorithms.a2c import A2CAgent, A2CConfig

        agent = A2CAgent(
            obs_dim=payload["obs_dim"],
            action_dim=payload["num_macros"] * payload["num_directions"],
            num_macros=payload["num_macros"],
            num_directions=payload["num_directions"],
            config=A2CConfig(**payload["config"]),
            device=device,
            edge_index=payload.get("edge_index"),
        )
        agent.model.load_state_dict(payload["state_dict"])
        return agent

    if algorithm == "dqn":
        from src.rl.algorithms.dqn import DQNAgent, DQNConfig

        action_dim = payload["action_dim"]
        obs_dim = next(iter(payload["online"].values())).shape[1]
        agent = DQNAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=DQNConfig(**payload["config"]),
            device=device,
        )
        agent.online.load_state_dict(payload["online"])
        agent.target.load_state_dict(payload["target"])
        return agent

    raise ValueError(f"Unsupported algorithm: {algorithm}")
