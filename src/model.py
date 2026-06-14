import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import NNConv


class GNNRouter(nn.Module):

    def __init__(
        self,
        node_features=3,
        edge_features=2,
        hidden_dim=64
    ):
        super().__init__()

        self.edge_network = nn.Sequential(
            nn.Linear(
                edge_features,
                64
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                node_features * hidden_dim
            )
        )

        self.conv1 = NNConv(
            node_features,
            hidden_dim,
            self.edge_network,
            aggr="mean"
        )

        self.edge_network2 = nn.Sequential(
            nn.Linear(
                edge_features,
                64
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                hidden_dim * hidden_dim
            )
        )

        self.conv2 = NNConv(
            hidden_dim,
            hidden_dim,
            self.edge_network2,
            aggr="mean"
        )

        self.edge_predictor = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                128
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                64
            ),
            nn.ReLU(),

            nn.Linear(
                64,
                1
            )
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr
    ):

        x = self.conv1(
            x,
            edge_index,
            edge_attr
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index,
            edge_attr
        )

        x = F.relu(x)

        src = edge_index[0]
        dst = edge_index[1]

        edge_embeddings = torch.cat(
            [
                x[src],
                x[dst]
            ],
            dim=1
        )

        predictions = self.edge_predictor(
            edge_embeddings
        )

        return predictions.squeeze()