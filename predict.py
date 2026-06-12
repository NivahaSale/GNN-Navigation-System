import torch
import networkx as nx
import osmnx as ox

from graph import create_graph_with_traffic
from subgraph import create_subgraph
from pyg_data import convert_to_pyg_data
from model import EdgeGNN


def load_model(data):

    model = EdgeGNN(
        node_in=data.x.shape[1],
        edge_in=data.edge_attr.shape[1],
        hidden=32
    )

    model.load_state_dict(torch.load("gnn_model.pt", map_location="cpu"))
    model.eval()

    return model


def run(model, data):

    with torch.no_grad():
        return model(
            data.x,
            data.edge_index,
            data.edge_attr
        ).squeeze().cpu().numpy()


def update_graph(G_sub, pred, data, nodes):

    edge_index = data.edge_index.t().cpu().numpy()

    for i, (u_i, v_i) in enumerate(edge_index):

        u = nodes[u_i]
        v = nodes[v_i]

        if G_sub.has_edge(u, v):

            d = G_sub.get_edge_data(u, v, default={})
            length = d.get("length", 1.0)

            G_sub[u][v]["weight"] = length + float(pred[i])

    return G_sub


def main():

    G = ox.graph_from_place("Hyderabad, Telangana, India", network_type="drive")
    G = nx.Graph(G)

    G_sub = create_subgraph(G)

    data, nodes = convert_to_pyg_data(G_sub)

    model = load_model(data)

    pred = run(model, data)

    G_sub = update_graph(G_sub, pred, data, nodes)

    source = nodes[10]
    target = nodes[200]

    path = nx.shortest_path(G_sub, source, target, weight="weight")

    print("\n FINAL ROUTE")
    print(path[:10])


if __name__ == "__main__":
    main()