import torch
import torch.nn as nn

from gnn_model import GraphSAGERouter

print("Loading graph data...")

data = torch.load(
    "graph_data.pt",
    weights_only=False
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)

data = data.to(device)

model = GraphSAGERouter().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

loss_fn = nn.MSELoss()

epochs = 50

print("\nTraining started...\n")

for epoch in range(epochs):

    model.train()

    pred = model(
        data.x,
        data.edge_index,
        data.edge_attr
    )

    loss = loss_fn(pred, data.y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 5 == 0:

        mae = torch.mean(
            torch.abs(pred - data.y)
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss.item():.4f} | "
            f"MAE: {mae.item():.4f}"
        )

torch.save(
    model.state_dict(),
    "gnn_router.pt"
)

print("\nModel saved as gnn_router.pt")