import torch.nn as nn
from layers import SageConvCustom

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNEncoder, self).__init__()
        self.conv1 = SageConvCustom(in_channels, hidden_channels)
        self.conv2 = SageConvCustom(hidden_channels, out_channels)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        x = self.global_pool(x.unsqueeze(-1)).squeeze(-1)
        return x