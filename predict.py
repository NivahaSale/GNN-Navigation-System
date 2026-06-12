import torch
import networkx as nx
import osmnx as ox

from pyg_data import convert_to_pyg_data
from subgraph import create_subgraph
from model import EdgeGNN


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model(data):

    model = EdgeGNN(
        node_in=data.x.shape[1],
        edge_in=data.edge_attr.shape[1],
        hidden=32
    )

    model.load_state_dict(torch.load("gnn_model.pt", map_location="cpu"))
    model.eval()

    return model


# -----------------------------
# RUN INFERENCE
# -----------------------------
def run_inference(model, data):

    with torch.no_grad():
        edge_predictions = model(
            data.x,
            data.edge_index,
            data.edge_attr
        ).squeeze()

    return edge_predictions.cpu().numpy()


# -----------------------------
# UPDATE GRAPH (FIXED PROPERLY)
# -----------------------------
def update_graph(G_sub, edge_predictions, data, nodes):

    print("🧠 Updating edge weights...")

    edge_index = data.edge_index.t().cpu().numpy()

    for i, (u_idx, v_idx) in enumerate(edge_index):

        u = nodes[u_idx]
        v = nodes[v_idx]

        if G_sub.has_edge(u, v):

            edge_data = G_sub.edges[u, v]

            G_sub.edges[u, v]["weight"] = (
                edge_data.get("length", 1.0)
                + float(edge_predictions[i])
            )

    return G_sub


# -----------------------------
# FIND ROUTE
# -----------------------------
def find_route(G_sub):

    nodes = list(G_sub.nodes())

    source = nodes[10]
    destination = nodes[200]

    try:
        path = nx.shortest_path(
            G_sub,
            source,
            destination,
            weight="weight"
        )

        print("\n🚀 AI ROUTE FOUND")
        print("Route length:", len(path))
        print("First 10 nodes:", path[:10])

        return path

    except nx.NetworkXNoPath:
        print("❌ No route found")
        return None


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():

    print("📡 Loading graph...")

    G = ox.graph_from_place(
        "Hyderabad, Telangana, India",
        network_type="drive"
    )

    # IMPORTANT: ensure simple graph (NOT MultiGraph issues)
    G = nx.Graph(G)

    G_sub = create_subgraph(G)

    # IMPORTANT FIX: keep node order stable
    nodes = list(G_sub.nodes())

    data = convert_to_pyg_data(G_sub)

    print("🤖 Loading trained model...")
    model = load_model(data)

    print("🔮 Running inference...")
    edge_predictions = run_inference(model, data)

    print("🧠 Updating edge weights...")
    G_sub = update_graph(G_sub, edge_predictions, data, nodes)

    print("🧭 Computing best route...")
    route = find_route(G_sub)

    return route


if __name__ == "__main__":
    main()