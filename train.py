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

    G = nx.Graph(G)
    G_sub = create_subgraph(G)

    data, nodes = convert_to_pyg_data(G_sub)

    print("Initializing model...")

    model = EdgeGNN(
        node_in=data.x.shape[1],
        edge_in=data.edge_attr.shape[1],
        hidden=32
    )

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()

    length = data.edge_attr[:, 0]
    congestion = data.edge_attr[:, 1]
    speed = data.edge_attr[:, 2]

    target = (length / (speed + 1e-6)) * (1 + 8.0 * congestion)
    target = (target - target.mean()) / (target.std() + 1e-6)
    target = target.unsqueeze(1)

    print("Training started...")

    for epoch in range(30):

        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.edge_attr)

        out = torch.clamp(out, -3, 3)

        loss = loss_fn(out, target)

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print("Pred mean:", out.mean().item())

        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "gnn_model.pt")

    test_routing(model, data, G_sub, nodes)


def test_routing(model, data, G_sub, nodes):

    print("\nMULTI-ROUTE AI NAVIGATION SYSTEM\n")

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

        if not G_sub.has_edge(u, v):
            continue

        edge_data = G_sub.get_edge_data(u, v)
        if isinstance(edge_data, dict) and 0 in edge_data:
            edge_data = edge_data[0]

        length = float(edge_data.get("length", 1.0))
        congestion = float(edge_data.get("congestion", 0.0))

        learned = float(pred[i])
        learned = max(0.0, learned)

        ai_weight = (
            length * (1 + 10.0 * congestion)  
            + (50.0 * learned)                 
        )

        ai_weight = max(0.1, ai_weight)

        G_sub[u][v]["distance_weight"] = max(0.1, length)
        G_sub[u][v]["congestion_weight"] = max(0.01, congestion)
        G_sub[u][v]["ai_weight"] = ai_weight

    source = nodes[10]
    target = nodes[200]

    fastest = nx.shortest_path(G_sub, source, target, weight="distance_weight")
    traffic = nx.shortest_path(G_sub, source, target, weight="congestion_weight")
    ai = nx.shortest_path(G_sub, source, target, weight="ai_weight")

    print("🟢 FASTEST ROUTE:", len(fastest))
    print("🟠 TRAFFIC ROUTE:", len(traffic))
    print("🔵 AI ROUTE:", len(ai))

    print("\n🔵 AI SAMPLE:", ai[:10])

if __name__ == "__main__":
    train()