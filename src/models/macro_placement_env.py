import gymnasium as gym
from gymnasium import spaces
import torch
from torch_geometric.data import Data
import numpy as np
from torch.serialization import safe_globals
from typing import Tuple

class MacroPlacementEnv(gym.Env):
    def __init__(self, graph_path: str, max_steps: int = 200):
        super().__init__()
        self.graph, _ = self.load_dataset(graph_path)
        self.original_pos = self.graph.pos.clone()
        self.canvas = self.graph.canvas_size.numpy() if hasattr(self.graph, 'canvas_size') else (400.0, 400.0)
        
        self.num_macros = self.graph.num_nodes
        self.features_per_macro = self.graph.x.shape[1] + 2  # x features + pos (x,y)
        self.obs_dim = self.num_macros * self.features_per_macro  # Flattened for MLP
        
        # Observation: flattened vector (normalized 0-1)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        
        # Simple discrete actions: select macro (0 to N-1) + direction (4 options) → N*4 actions
        self.action_space = spaces.Discrete(self.num_macros * 4)
        
        self.max_steps = max_steps
        self.current_step = 0
    def load_dataset(self, graph_path: str) -> Tuple[Data, dict]:
        with safe_globals([Data]):
            data = torch.load(
                graph_path,
                map_location="cpu",
                weights_only=False
            )
        if isinstance(data, dict) and "graph" in data:
            graph = data["graph"]
            metadata = data.get("metadata", {})
            return graph, metadata

        if isinstance(data, Data):
            return data, {}

        raise ValueError("Unsupported .pt file format")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.graph.pos = torch.rand_like(self.original_pos)  # Random initial positions
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # Concat pos + all node features, then flatten
        pos_norm = self.graph.pos.numpy()  # (N,2) already normalized
        features = self.graph.x.numpy()   # (N,6)
        combined = np.concatenate([pos_norm, features], axis=1)  # (N,8)
        return combined.flatten()  # (N*8,)

    def _compute_hpwl(self):
        if self.graph.edge_index.numel() == 0:
            return 0.0
        pos_abs = self.graph.pos.numpy() * self.canvas
        src, dst = self.graph.edge_index.numpy()
        dx = np.abs(pos_abs[src, 0] - pos_abs[dst, 0])
        dy = np.abs(pos_abs[src, 1] - pos_abs[dst, 1])
        return float(np.sum(dx + dy)) / 2  # Total approx HPWL

    def step(self, action):
        macro_idx = action // 4
        direction = action % 4
        deltas = np.array([[0, 0.05], [0, -0.05], [-0.05, 0], [0.05, 0]])  # N, S, W, E
        delta = deltas[direction]
        
        self.graph.pos[macro_idx] += torch.tensor(delta, dtype=torch.float)
        self.graph.pos = torch.clamp(self.graph.pos, 0.0, 1.0)  # Bounds
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        reward = -self._compute_hpwl() * 0.001  # Scale negative HPWL (adjust factor)
        # Optional: Add density penalty later
        
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        pass  # Use your external visualizer

if __name__ == "__main__":
    env = MacroPlacementEnv("data/preprocessed/real-connection/MemPool_tile/Nangate45/MemPool_tile_Nangate45_graph.pt")
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")  # Should be (20*8,) = (160,)
    action = env.action_space.sample()
    next_obs, reward, term, trunc, info = env.step(action)
    print(f"Reward: {reward:.2f}, Terminated: {term}")