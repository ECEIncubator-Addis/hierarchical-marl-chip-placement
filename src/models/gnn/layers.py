import torch.nn as nn
from message_passing import MacroPlacementMessagePassing

class SageConvCustom(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.mp = MacroPlacementMessagePassing(
            in_channels=in_dim,
            out_channels=out_dim
        )

    def forward(self, x, edge_index):
        x = self.mp(x, edge_index)

        return x