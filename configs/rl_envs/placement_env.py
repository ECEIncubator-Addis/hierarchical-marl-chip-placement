from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym

try:
    from .action_space import HierarchicalActionSpace
    from .constraints import BoundaryConstraint, PlacementConstraints
    from .reward import PlacementReward, RewardConfig, compute_hpwl
    from .simulator import PlacementSimulator
    from .state import PlacementStateEncoder
except ImportError:  # pragma: no cover
    from action_space import HierarchicalActionSpace
    from constraints import BoundaryConstraint, PlacementConstraints
    from reward import PlacementReward, RewardConfig, compute_hpwl
    from simulator import PlacementSimulator
    from state import PlacementStateEncoder


class HierarchicalMacroPlacementEnv(gym.Env):
    """Gymnasium environment for hierarchical manager-worker macro placement."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        graph_path: str | Path,
        max_steps: int = 200,
        movement_step: float = 0.05,
        hpwl_scale: float = 0.001,
        randomize_initial_positions: bool = True,
        include_step_fraction: bool = False,
        overlap_weight: float = 0.0,
        density_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.graph_path = str(graph_path)
        self.max_steps = int(max_steps)
        self.randomize_initial_positions = bool(randomize_initial_positions)
        self.current_step = 0

        directions = [
            [0.0, movement_step],
            [0.0, -movement_step],
            [-movement_step, 0.0],
            [movement_step, 0.0],
        ]

        simulator, metadata = PlacementSimulator.from_file(self.graph_path)
        self.metadata_info: dict[str, Any] = metadata
        self.graph = simulator.graph
        self.num_macros = int(self.graph.num_nodes)

        self.hierarchical_action_space = HierarchicalActionSpace(self.num_macros, directions=directions)
        self.manager_action_space = self.hierarchical_action_space.manager_space
        self.worker_action_space = self.hierarchical_action_space.worker_space
        self.action_space = self.hierarchical_action_space.flat_space

        constraints = PlacementConstraints(
            boundary=BoundaryConstraint(low=0.0, high=1.0, clamp=True),
            overlap_weight=overlap_weight,
            density_weight=density_weight,
        )
        self.simulator = PlacementSimulator(self.graph, action_space=self.hierarchical_action_space, constraints=constraints)
        self.state_encoder = PlacementStateEncoder(self.graph, include_step_fraction=include_step_fraction)
        self.observation_space = self.state_encoder.observation_space
        self.reward_model = PlacementReward(RewardConfig(hpwl_scale=hpwl_scale), constraints=constraints)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        randomize = bool(options.get("randomize_initial_positions", self.randomize_initial_positions))
        self.current_step = 0
        self.graph = self.simulator.reset(seed=seed, randomize=randomize)
        self.reward_model.reset(self.graph)
        return self._get_obs(), self._info()

    def step(self, action):
        result = self.simulator.step(action)
        self.graph = result.graph
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False
        reward, reward_info = self.reward_model(self.graph, terminated=terminated)
        info = self._info()
        info.update(reward_info)
        info["manager_macro_index"] = result.action.macro_index
        info["worker_direction_index"] = result.action.direction_index
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        return None

    def _get_obs(self):
        return self.state_encoder.encode(self.graph, step=self.current_step, max_steps=self.max_steps)

    def _compute_hpwl(self) -> float:
        return compute_hpwl(self.graph)

    def _info(self) -> dict[str, Any]:
        return {
            "step": self.current_step,
            "max_steps": self.max_steps,
            "num_macros": self.num_macros,
        }


MacroPlacementEnv = HierarchicalMacroPlacementEnv
