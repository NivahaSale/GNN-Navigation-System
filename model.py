import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class EdgeGNN(nn.Module):

    def __init__(self, node_in, edge_in, hidden):
        super().__init__()

        # -----------------------------
        # Node-level encoder (GraphSAGE)
        # -----------------------------
        self.conv1 = SAGEConv(node_in, hidden)
        self.conv2 = SAGEConv(hidden, hidden)

        self.relu = nn.ReLU()

        # -----------------------------
        # Edge predictor MLP
        # -----------------------------
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + edge_in, 128),
            nn.ReLU(),
            nn.Dropout(0.2),   # 🔥 improves stability
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # -----------------------------
        # Weight initialization (IMPORTANT FIX)
        # -----------------------------
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr):

        # -----------------------------
        # Node embedding learning
        # -----------------------------
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))

        # -----------------------------
        # Edge feature construction
        # -----------------------------
        src, dst = edge_index

        edge_features = torch.cat([
            x[src],          # source node embedding
            x[dst],          # target node embedding
            edge_attr        # original edge features
        ], dim=1)

        # -----------------------------
        # Edge prediction (cost / congestion)
        # -----------------------------
        out = self.edge_mlp(edge_features)

        return out