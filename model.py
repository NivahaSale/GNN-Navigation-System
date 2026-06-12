import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class EdgeGNN(nn.Module):

    def __init__(self, node_in, edge_in, hidden):
        super().__init__()

        self.conv1 = SAGEConv(node_in, hidden)
        self.conv2 = SAGEConv(hidden, hidden)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + edge_in, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, edge_index, edge_attr):

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))

        src, dst = edge_index

        edge_features = torch.cat([
            x[src],
            x[dst],
            edge_attr
        ], dim=1)

        return self.edge_mlp(edge_features)