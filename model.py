import torch
import torch.nn as nn
from torch_geometric.nn import NNConv


class EdgeGNN(nn.Module):
    def __init__(self, node_in, edge_in, hidden):
        super().__init__()

        # -----------------------------
        # EDGE NETWORK (CRITICAL FIX)
        # MUST output node_in * hidden
        # -----------------------------
        self.edge_mlp1 = nn.Sequential(
            nn.Linear(edge_in, 128),
            nn.ReLU(),
            nn.Linear(128, node_in * hidden)
        )

        self.conv1 = NNConv(node_in, hidden, self.edge_mlp1, aggr='mean')

        # second layer
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(edge_in, 128),
            nn.ReLU(),
            nn.Linear(128, hidden * hidden)
        )

        self.conv2 = NNConv(hidden, hidden, self.edge_mlp2, aggr='mean')

        self.relu = nn.ReLU()

        # edge prediction head
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden * 2 + edge_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, edge_attr):

        x = self.relu(self.conv1(x, edge_index, edge_attr))
        x = self.relu(self.conv2(x, edge_index, edge_attr))

        src, dst = edge_index

        edge_features = torch.cat([
            x[src],
            x[dst],
            edge_attr
        ], dim=1)

        return self.edge_predictor(edge_features)