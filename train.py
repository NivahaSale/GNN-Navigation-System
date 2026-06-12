import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

from graph import create_graph_with_traffic
from subgraph import create_subgraph
from pyg_data import convert_to_pyg_data
from model import EdgeGNN


def train():

    print("📡 Loading graph...")
    G = create_graph_with_traffic()

    # Keep as undirected simple graph
    G = nx.Graph(G)

    print("📍 Creating subgraph...")
    G_sub = create_subgraph(G)

    print("🔄 Converting to PyG format...")
    data, nodes = convert_to_pyg_data(G_sub)

    data.x = (data.x - data.x.mean()) / (data.x.std() + 1e-6)
    data.edge_attr = (data.edge_attr - data.edge_attr.mean(dim=0)) / (data.edge_attr.std(dim=0) + 1e-6)

    print(" Initializing model...")
    model = EdgeGNN(
        node_in=data.x.shape[1],
        edge_in=data.edge_attr.shape[1],
        hidden=32
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

 
    length = data.edge_attr[:, 0]
    congestion = data.edge_attr[:, 1]
    speed = data.edge_attr[:, 2]

    target = (length / (speed + 1e-6)) * (1 + congestion)

    # Normalize target
    target = (target - target.mean()) / (target.std() + 1e-6)
    target = target.unsqueeze(1)

    print("Training started...")

    for epoch in range(30):

        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.edge_attr)

        loss = loss_fn(out, target)

        loss.backward()
        optimizer.step()

        
        if epoch % 5 == 0:
            print("Pred mean:", out.mean().item())
            print("Target mean:", target.mean().item())

        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "gnn_model.pt")

    return model, data, G_sub, nodes

def test_routing(model, data, G_sub, nodes):

    print("\n🛣️ Routing...")

    model.eval()

    with torch.no_grad():

        pred = model(
            data.x,
            data.edge_index,
            data.edge_attr
        ).squeeze().cpu().numpy()

    edge_index = data.edge_index.t().cpu().numpy()

    for i, (u_i, v_i) in enumerate(edge_index):

        u = nodes[u_i]
        v = nodes[v_i]

        if G_sub.has_edge(u, v):

            edge_data = G_sub.get_edge_data(u, v)
            length = edge_data.get("length", 1.0)
            weight = length + 0.5 * float(pred[i])
            weight = max(0.1, weight)

            G_sub[u][v]["weight"] = weight

    source = nodes[10]
    target = nodes[200]

    path = nx.shortest_path(G_sub, source, target, weight="weight")

    print("Route found:", len(path))
    print(path[:10])


if __name__ == "__main__":
    model, data, G_sub, nodes = train()
    test_routing(model, data, G_sub, nodes)