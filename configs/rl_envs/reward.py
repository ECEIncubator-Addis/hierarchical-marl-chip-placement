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
    hpwl_scale: float = 1.0
    improvement_scale: float = 5.0
    overlap_scale: float = 10.0
    density_scale: float = 2.0
    terminal_bonus_scale: float = 5.0


def compute_density_penalty(
    graph,
    bins: int = 10,
) -> float:

    canvas = canvas_size(graph)

    pos = graph.pos.detach().cpu().numpy()
    sizes = graph.x[:, :2].detach().cpu().numpy()

    grid = np.zeros((bins, bins), dtype=np.float32)

    for i in range(graph.num_nodes):

        x_bin = int(np.clip(
            pos[i, 0] * bins,
            0,
            bins - 1,
        ))

        y_bin = int(np.clip(
            pos[i, 1] * bins,
            0,
            bins - 1,
        ))

        area = sizes[i, 0] * sizes[i, 1]

        grid[y_bin, x_bin] += area

    density_std = np.std(grid)
    max_density = np.max(grid)

    return float(
        0.5 * density_std +
        0.5 * max_density
    )
class PlacementReward:
    """Reward for minimizing HPWL with optional constraint penalties."""

    def __init__(self, config: RewardConfig | None = None, constraints: PlacementConstraints | None = None) -> None:
        self.config = config or RewardConfig()
        self.constraints = constraints or PlacementConstraints()
        self.previous_hpwl: float | None = None

    def reset(self, graph) -> None:
        self.previous_hpwl = compute_hpwl(graph)

    def __call__(
        self,
        graph,
        terminated: bool = False,
    ) -> tuple[float, dict[str, float]]:

        hpwl = compute_hpwl(graph)
        overlap_penalty = self.constraints.penalty(graph)
        density_penalty = compute_density_penalty(graph)

        #normalization 
        hpwl_norm = hpwl / 2000.0
        overlap_norm = overlap_penalty / 100000.0
        density_norm = density_penalty / 1000.0

        reward = (
            -self.config.hpwl_scale * hpwl_norm
            -self.config.overlap_scale * overlap_norm
            -self.config.density_scale * density_norm
        )

        #relative improvement reward
        improvement = 0.0

        if self.previous_hpwl is not None:
            improvement = (
                self.previous_hpwl - hpwl
            ) / max(self.previous_hpwl, 1e-6)

            reward += (
                self.config.improvement_scale *
                improvement
            )

        #Terminal bonus for final improvement
        if terminated:
            reward += (
                self.config.terminal_bonus_scale *
                max(improvement, 0.0)
            )

        self.previous_hpwl = hpwl

        return float(reward), {
            "hpwl": float(hpwl),
            "overlap_penalty": float(overlap_penalty),
            "density_penalty": float(density_penalty),
            "improvement": float(improvement),
        }
