from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


ConvolutionType = Literal["sage", "gcn", "gat"]
ActivationType = Literal["relu", "gelu", "tanh", "identity"]
NormalizationType = Literal["layer", "batch", "none"]


def build_activation(name: ActivationType) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def build_normalization(name: NormalizationType, channels: int) -> nn.Module:
    if name == "layer":
        return nn.LayerNorm(channels)
    if name == "batch":
        return nn.BatchNorm1d(channels)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported normalization: {name}")


def build_convolution(
    conv_type: ConvolutionType,
    in_channels: int,
    out_channels: int,
    heads: int = 1,
) -> nn.Module:
    if conv_type == "sage":
        return SAGEConv(in_channels=in_channels, out_channels=out_channels, aggr="mean")
    if conv_type == "gcn":
        return GCNConv(in_channels=in_channels, out_channels=out_channels)
    if conv_type == "gat":
        return GATConv(in_channels=in_channels, out_channels=out_channels, heads=heads, concat=False)
    raise ValueError(f"Unsupported convolution type: {conv_type}")


class MacroPlacementMessagePassing(nn.Module):
    """Message-passing block used to build placement-aware node representations."""

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
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.residual_enabled = bool(residual)

        self.conv = build_convolution(conv_type, in_channels, out_channels, heads=heads)
        self.norm = build_normalization(normalization, out_channels)
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.residual_enabled:
            x = x + residual
        return x
