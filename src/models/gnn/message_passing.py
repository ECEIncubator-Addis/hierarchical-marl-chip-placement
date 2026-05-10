# message_passing.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv


class MacroPlacementMessagePassing(nn.Module):
    """
    GraphSAGE-based message passing block
    for macro placement GNN encoding.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):

        super().__init__()

        # GraphSAGE message passing
        self.sage = SAGEConv(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr="mean"
        )

        # Optional normalization
        self.norm = nn.LayerNorm(out_channels)

        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        """
        Parameters
        x : Tensor
            Node features
            Shape: [N, F]

        edge_index : Tensor
            Graph connectivity
            Shape: [2, E]

        Returns
            Tensor: Updated node embeddings
        """

        # Message passing + aggregation
        x = self.sage(x, edge_index)

        # Normalization
        x = self.norm(x)

        # Non-linearity
        x = self.activation(x)

        # Dropout
        x = self.dropout(x)

        return x