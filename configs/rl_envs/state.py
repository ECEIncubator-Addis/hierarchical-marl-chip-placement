from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import spaces


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass
class PlacementState:
    positions: np.ndarray
    node_features: np.ndarray
    step: int
    max_steps: int

    @property
    def flattened(self) -> np.ndarray:
        return np.concatenate([self.positions, self.node_features], axis=1).astype(np.float32).reshape(-1)


class PlacementStateEncoder:
    """Builds flattened observations from normalized macro positions and features."""

    def __init__(self, graph, include_step_fraction: bool = False) -> None:
        self.include_step_fraction = include_step_fraction
        self.num_macros = int(graph.num_nodes)
        self.node_feature_dim = int(graph.x.shape[1]) if hasattr(graph, "x") and graph.x is not None else 0
        self.features_per_macro = self.node_feature_dim + 2 + int(include_step_fraction)
        self.obs_dim = self.num_macros * self.features_per_macro

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

    def encode(self, graph, step: int = 0, max_steps: int = 1) -> np.ndarray:
        state = self.state(graph, step=step, max_steps=max_steps)
        obs = state.flattened
        if self.include_step_fraction:
            step_fraction = np.full((self.num_macros, 1), min(1.0, step / max(1, max_steps)), dtype=np.float32)
            obs = np.concatenate([state.positions, state.node_features, step_fraction], axis=1).reshape(-1)
        return np.clip(obs.astype(np.float32), 0.0, 1.0)

    def state(self, graph, step: int = 0, max_steps: int = 1) -> PlacementState:
        if not hasattr(graph, "pos") or graph.pos is None:
            raise ValueError("graph must define normalized macro positions in graph.pos.")

        positions = _as_numpy(graph.pos).astype(np.float32)
        node_features = (
            _as_numpy(graph.x).astype(np.float32)
            if hasattr(graph, "x") and graph.x is not None
            else np.empty((positions.shape[0], 0), dtype=np.float32)
        )
        return PlacementState(
            positions=np.clip(positions, 0.0, 1.0),
            node_features=np.clip(node_features, 0.0, 1.0),
            step=int(step),
            max_steps=int(max_steps),
        )
