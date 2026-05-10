from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

try:
    from .constraints import PlacementConstraints
except ImportError:  # pragma: no cover
    from constraints import PlacementConstraints


def canvas_size(graph) -> np.ndarray:
    if hasattr(graph, "canvas_size"):
        canvas = graph.canvas_size
        if isinstance(canvas, torch.Tensor):
            canvas = canvas.detach().cpu().numpy()
        return np.asarray(canvas, dtype=np.float32)
    return np.asarray([400.0, 400.0], dtype=np.float32)


def compute_hpwl(graph, divide_bidirectional_edges: bool = True) -> float:
    if not hasattr(graph, "edge_index") or graph.edge_index is None or graph.edge_index.numel() == 0:
        return 0.0

    pos = graph.pos.detach().cpu().numpy() if isinstance(graph.pos, torch.Tensor) else np.asarray(graph.pos)
    edge_index = graph.edge_index.detach().cpu().numpy() if isinstance(graph.edge_index, torch.Tensor) else np.asarray(graph.edge_index)
    pos_abs = pos * canvas_size(graph)
    src, dst = edge_index
    hpwl = np.abs(pos_abs[src, 0] - pos_abs[dst, 0]) + np.abs(pos_abs[src, 1] - pos_abs[dst, 1])
    total = float(np.sum(hpwl))
    return total / 2.0 if divide_bidirectional_edges else total


@dataclass
class RewardConfig:
    hpwl_scale: float = 0.001
    improvement_scale: float = 0.0
    terminal_bonus_scale: float = 0.0


class PlacementReward:
    """Reward for minimizing HPWL with optional constraint penalties."""

    def __init__(self, config: RewardConfig | None = None, constraints: PlacementConstraints | None = None) -> None:
        self.config = config or RewardConfig()
        self.constraints = constraints or PlacementConstraints()
        self.previous_hpwl: float | None = None

    def reset(self, graph) -> None:
        self.previous_hpwl = compute_hpwl(graph)

    def __call__(self, graph, terminated: bool = False) -> tuple[float, dict[str, float]]:
        hpwl = compute_hpwl(graph)
        penalty = self.constraints.penalty(graph)
        reward = -self.config.hpwl_scale * hpwl - penalty

        if self.previous_hpwl is not None and self.config.improvement_scale:
            reward += self.config.improvement_scale * (self.previous_hpwl - hpwl)

        if terminated and self.config.terminal_bonus_scale:
            reward -= self.config.terminal_bonus_scale * hpwl

        self.previous_hpwl = hpwl
        return float(reward), {"hpwl": float(hpwl), "constraint_penalty": float(penalty)}
