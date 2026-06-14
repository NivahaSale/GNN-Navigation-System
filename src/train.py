import torch
import torch.nn as nn
import torch.optim as optim

from src.graph import create_graph
from archive.utils import build_graph_tensors
from src.model import GNNRouter


def train():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Loading graph...")
    G = create_graph()

    print("Creating tensors...")

    (
        x,
        edge_index,
        edge_attr,
        edge_labels,
        node_map
    ) = build_graph_tensors(G)

    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    edge_labels = edge_labels.to(device)

    print(f"Nodes: {x.shape[0]}")
    print(f"Edges: {edge_index.shape[1]}")

    model = GNNRouter().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.MSELoss()

    epochs = 50

    print("Training started...\n")

    for epoch in range(epochs):

        model.train()

        predictions = model(
            x,
            edge_index,
            edge_attr
        )

        loss = criterion(
            predictions,
            edge_labels
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {loss.item():.6f}"
        )

    print("\nSaving model...")

    torch.save(
        model.state_dict(),
        "gnn_model.pt"
    )

    print("Saving graph tensors...")

    torch.save(
        {
            "x": x.cpu(),
            "edge_index": edge_index.cpu(),
            "edge_attr": edge_attr.cpu(),
            "node_map": node_map
        },
        "graph_data.pt"
    )

    print("Training complete!")


if __name__ == "__main__":
    train()