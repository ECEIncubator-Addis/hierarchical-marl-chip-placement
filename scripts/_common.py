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

    if algorithm == "a2c":
        from src.rl.algorithms.a2c import A2CAgent

        return A2CAgent.load(checkpoint_path, device=device)

    if algorithm == "dqn":
        from src.rl.algorithms.dqn import DQNAgent

        return DQNAgent.load(checkpoint_path, device=device)

    raise ValueError(f"Unsupported algorithm: {algorithm}")
