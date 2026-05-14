from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

try:
    from .encoder import FlatPlacementObservationEncoder
except ImportError:  # pragma: no cover
    from encoder import FlatPlacementObservationEncoder


class GNNHierarchicalActorCritic(nn.Module):
    """Actor-critic policy that learns RL representations with the placement GNN.

    Macro logits are produced from node embeddings. Direction logits and values are
    produced from the pooled graph embedding.
    """

    def __init__(
        self,
        num_macros: int,
        features_per_macro: int,
        edge_index: torch.Tensor,
        num_directions: int = 4,
        hidden_channels: int = 128,
        embedding_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_macros = int(num_macros)
        self.features_per_macro = int(features_per_macro)
        self.obs_dim = self.num_macros * self.features_per_macro
        self.num_directions = int(num_directions)
        self.representation = FlatPlacementObservationEncoder(
            num_macros=num_macros,
            features_per_macro=features_per_macro,
            edge_index=edge_index,
            hidden_channels=hidden_channels,
            out_channels=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.macro_head = nn.Linear(embedding_dim, 1)
        self.direction_head = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Tanh(), nn.Linear(embedding_dim, num_directions))
        self.value_head = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Tanh(), nn.Linear(embedding_dim, 1))

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rep = self.representation(observations)
        macro_logits = self.macro_head(rep.node_embeddings).squeeze(-1)
        direction_logits = self.direction_head(rep.graph_embedding)
        values = self.value_head(rep.graph_embedding).squeeze(-1)
        return macro_logits, direction_logits, values

    def distribution(self, observations: torch.Tensor) -> tuple[Categorical, Categorical, torch.Tensor]:
        macro_logits, direction_logits, values = self.forward(observations)
        return Categorical(logits=macro_logits), Categorical(logits=direction_logits), values

    def act(self, observations: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        macro_dist, direction_dist, values = self.distribution(observations)
        if deterministic:
            macros = macro_dist.probs.argmax(dim=-1)
            directions = direction_dist.probs.argmax(dim=-1)
        else:
            macros = macro_dist.sample()
            directions = direction_dist.sample()
        actions = macros * self.num_directions + directions
        log_probs = macro_dist.log_prob(macros) + direction_dist.log_prob(directions)
        return actions, log_probs, values

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        macro_dist, direction_dist, values = self.distribution(observations)
        macros = actions // self.num_directions
        directions = actions % self.num_directions
        log_probs = macro_dist.log_prob(macros) + direction_dist.log_prob(directions)
        entropy = macro_dist.entropy() + direction_dist.entropy()
        return log_probs, entropy, values

    def save_encoder(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.representation.encoder.state_dict(),
                "num_macros": self.num_macros,
                "features_per_macro": self.features_per_macro,
                "edge_index": self.representation.edge_index.detach().cpu(),
            },
            path,
        )


def build_gnn_actor_critic_from_env(env: Any, **kwargs: Any) -> GNNHierarchicalActorCritic:
    graph = getattr(env, "graph", None)
    if graph is None or not hasattr(graph, "edge_index"):
        raise ValueError("Environment must expose graph.edge_index to build a GNN actor-critic.")

    num_macros = int(getattr(env, "num_macros", graph.num_nodes))
    obs_dim = int(env.observation_space.shape[0])
    if obs_dim % num_macros != 0:
        raise ValueError(f"Observation dim {obs_dim} is not divisible by num_macros {num_macros}.")
    return GNNHierarchicalActorCritic(
        num_macros=num_macros,
        features_per_macro=obs_dim // num_macros,
        edge_index=graph.edge_index,
        **kwargs,
    )
