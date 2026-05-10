from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

try:
    from .message_passing import ActivationType, ConvolutionType, MacroPlacementMessagePassing, NormalizationType
except ImportError:  # pragma: no cover
    from message_passing import ActivationType, ConvolutionType, MacroPlacementMessagePassing, NormalizationType


PoolingType = Literal["mean", "max", "sum", "none"]


class SageConvCustom(nn.Module):
    """Backward-compatible GraphSAGE wrapper used by older scripts."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mp = MacroPlacementMessagePassing(
            in_channels=in_dim,
            out_channels=out_dim,
            dropout=dropout,
            conv_type="sage",
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.mp(x, edge_index)


class PlacementGNNLayer(nn.Module):
    """Configurable GNN layer for macro-placement representation learning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1,
        conv_type: ConvolutionType = "sage",
        activation: ActivationType = "relu",
        normalization: NormalizationType = "layer",
        residual: bool = True,
        heads: int = 1,
    ) -> None:
        super().__init__()
        self.block = MacroPlacementMessagePassing(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            conv_type=conv_type,
            activation=activation,
            normalization=normalization,
            residual=residual,
            heads=heads,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.block(x, edge_index)


class GraphReadout(nn.Module):
    """Pools node embeddings into graph embeddings for policy/value heads."""

    def __init__(self, pooling: PoolingType = "mean") -> None:
        super().__init__()
        self.pooling = pooling

    def forward(self, node_embeddings: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        if self.pooling == "none":
            return node_embeddings
        if batch is None:
            return node_embeddings.mean(dim=0, keepdim=True)
        if self.pooling == "mean":
            return global_mean_pool(node_embeddings, batch)
        if self.pooling == "max":
            return global_max_pool(node_embeddings, batch)
        if self.pooling == "sum":
            return global_add_pool(node_embeddings, batch)
        raise ValueError(f"Unsupported pooling: {self.pooling}")
