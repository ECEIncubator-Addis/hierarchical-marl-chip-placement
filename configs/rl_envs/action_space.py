from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class HierarchicalAction:
    """Manager selects a macro; worker selects a movement direction."""

    macro_index: int
    direction_index: int


class HierarchicalActionSpace:
    """Codec for hierarchical macro-placement actions.

    The flat action is kept for compatibility with discrete-action RL agents:
    ``action = macro_index * num_directions + direction_index``.
    """

    DEFAULT_DIRECTIONS = np.asarray(
        [
            [0.0, 0.05],
            [0.0, -0.05],
            [-0.05, 0.0],
            [0.05, 0.0],
        ],
        dtype=np.float32,
    )

    def __init__(self, num_macros: int, directions: Sequence[Sequence[float]] | None = None) -> None:
        if num_macros <= 0:
            raise ValueError("num_macros must be positive.")

        self.num_macros = int(num_macros)
        self.directions = np.asarray(directions if directions is not None else self.DEFAULT_DIRECTIONS, dtype=np.float32)
        if self.directions.ndim != 2 or self.directions.shape[1] != 2:
            raise ValueError("directions must have shape (num_directions, 2).")

        self.num_directions = int(self.directions.shape[0])
        self.manager_space = spaces.Discrete(self.num_macros)
        self.worker_space = spaces.Discrete(self.num_directions)
        self.flat_space = spaces.Discrete(self.num_macros * self.num_directions)

    def encode(self, macro_index: int, direction_index: int) -> int:
        self._validate(macro_index, direction_index)
        return int(macro_index) * self.num_directions + int(direction_index)

    def decode(self, action: int | HierarchicalAction | Mapping[str, Any] | Sequence[int]) -> HierarchicalAction:
        if isinstance(action, HierarchicalAction):
            decoded = action
        elif isinstance(action, Mapping):
            decoded = HierarchicalAction(
                macro_index=int(action.get("macro_index", action.get("manager", action.get("macro", 0)))),
                direction_index=int(action.get("direction_index", action.get("worker", action.get("direction", 0)))),
            )
        elif isinstance(action, Sequence) and not isinstance(action, (str, bytes)):
            if len(action) != 2:
                raise ValueError("hierarchical action sequences must be (macro_index, direction_index).")
            decoded = HierarchicalAction(int(action[0]), int(action[1]))
        else:
            action_id = int(action)
            decoded = HierarchicalAction(
                macro_index=action_id // self.num_directions,
                direction_index=action_id % self.num_directions,
            )

        self._validate(decoded.macro_index, decoded.direction_index)
        return decoded

    def delta(self, direction_index: int) -> np.ndarray:
        if not 0 <= int(direction_index) < self.num_directions:
            raise ValueError(f"direction_index out of range: {direction_index}")
        return self.directions[int(direction_index)].copy()

    def sample(self) -> HierarchicalAction:
        return self.decode(int(self.flat_space.sample()))

    def _validate(self, macro_index: int, direction_index: int) -> None:
        if not 0 <= int(macro_index) < self.num_macros:
            raise ValueError(f"macro_index out of range: {macro_index}")
        if not 0 <= int(direction_index) < self.num_directions:
            raise ValueError(f"direction_index out of range: {direction_index}")
