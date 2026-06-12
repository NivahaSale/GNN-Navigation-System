import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

from pyg_data import convert_to_pyg_data
from subgraph import create_subgraph
from model import EdgeGNN
from graph import create_graph_with_traffic


# -----------------------------
# TRAIN FUNCTION
# -----------------------------
def train():

    print("📡 Loading graph...")

    G = create_graph_with_traffic()
    G_sub = create_subgraph(G)

    # 🔥 IMPORTANT: keep node order fixed ONCE
    data, nodes = convert_to_pyg_data(G_sub)

    # -----------------------------
    # MODEL
    # -----------------------------
    model = EdgeGNN(
        node_in=data.x.shape[1],
        edge_in=data.edge_attr.shape[1],
        hidden=32
    )

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()

    # -----------------------------
    # TARGET: travel time proxy
    # -----------------------------
    length = data.edge_attr[:, 0]
    speed = data.edge_attr[:, 2]

    target = (length / (speed + 1e-6)).unsqueeze(1)

    # 🔥 normalize target (VERY IMPORTANT for stable training)
    target = (target - target.mean()) / (target.std() + 1e-6)

    print("🚀 Training started...")

    epochs = 30

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.edge_attr)

        loss = loss_fn(out, target)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "gnn_model.pt")

    print("✅ Model saved as gnn_model.pt")

    return model, data, G_sub, nodes


# -----------------------------
# ROUTING
# -----------------------------
def test_routing(model, data, G_sub, nodes):

    print("\n🧭 Running AI-based routing...")

    model.eval()

    with torch.no_grad():
        edge_pred = model(
            data.x,
            data.edge_index,
            data.edge_attr
        ).squeeze().cpu().numpy()

    edge_index = data.edge_index.t().cpu().numpy()

    # -----------------------------
    # UPDATE EDGE WEIGHTS
    # -----------------------------
    for i, (u_idx, v_idx) in enumerate(edge_index):

        u = nodes[u_idx]
        v = nodes[v_idx]

        if G_sub.has_edge(u, v):

            edge_data = G_sub.edges[u, v]

            G_sub.edges[u, v]["weight"] = (
                0.7 * edge_data.get("length", 1.0)
                + 0.3 * float(edge_pred[i])
            )

    # -----------------------------
    # ROUTING
    # -----------------------------
    source = nodes[10]
    target = nodes[200]

    try:
        path = nx.shortest_path(
            G_sub,
            source,
            target,
            weight="weight"
        )

        print("✅ AI Route Found!")
        print("Path length:", len(path))
        print("First 10 nodes:", path[:10])

    except Exception as e:
        print("❌ Routing failed:", e)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    model, data, G_sub, nodes = train()
    test_routing(model, data, G_sub, nodes)