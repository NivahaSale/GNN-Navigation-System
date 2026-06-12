import torch
from torch_geometric.data import Data


def safe_float(x, default):

    if isinstance(x, list):
        x = x[0]

    if isinstance(x, str):
        x = x.split()[0]

    try:
        return float(x)
    except:
        return float(default)


def convert_to_pyg_data(G):

    print("Converting to PyG...")

    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    x = torch.tensor(
        [[G.degree[n]] for n in nodes],
        dtype=torch.float
    )

    edge_index = []
    edge_attr = []

    for u, v, data in G.edges(data=True):

        u_i = node_map[u]
        v_i = node_map[v]

        length = safe_float(data.get("length", 1.0), 1.0)
        congestion = safe_float(data.get("congestion", 0.0), 0.0)
        speed = safe_float(data.get("speed_limit", 40.0), 40.0)
        road = safe_float(data.get("road_type_score", 1.0), 1.0)

        feat = [length, congestion, speed, road]

        edge_index.append([u_i, v_i])
        edge_index.append([v_i, u_i])

        edge_attr.append(feat)
        edge_attr.append(feat)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    assert edge_index.shape[1] == edge_attr.shape[0], "Edge mismatch!"
    assert not torch.isnan(edge_attr).any(), "NaN detected in edge_attr!"

    print("PyG ready")
    print("Nodes:", x.shape)
    print("Edges:", edge_index.shape)
    print("Edge features:", edge_attr.shape)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), nodes