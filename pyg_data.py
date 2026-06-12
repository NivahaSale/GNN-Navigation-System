import torch
from torch_geometric.data import Data


def convert_to_pyg_data(G):

    print("Converting to PyG...")

    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}

    x = torch.tensor([[G.degree[n]] for n in nodes], dtype=torch.float)

    edge_index = []
    edge_attr = []

    for u, v, data in G.edges(data=True):

        u_i = node_map[u]
        v_i = node_map[v]

        length = float(data.get("length", 1.0))
        congestion = float(data.get("congestion", 0.0))
        speed = float(data.get("speed_limit", 40.0))
        road = float(data.get("road_type_score", 1.0))

        feat = [length, congestion, speed, road]

        # -----------------------------
        # FIX: SYMMETRIC EDGES MUST MATCH ATTRS
        # -----------------------------
        edge_index.append([u_i, v_i])
        edge_attr.append(feat)

        edge_index.append([v_i, u_i])
        edge_attr.append(feat)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # -----------------------------
    # SAFETY CHECK (IMPORTANT)
    # -----------------------------
    assert edge_index.shape[1] == edge_attr.shape[0], "Edge mismatch fixed failed!"
    assert not torch.isnan(edge_attr).any(), "NaN detected!"

    print("PyG ready")
    print("Nodes:", x.shape)
    print("Edges:", edge_index.shape)
    print("Edge features:", edge_attr.shape)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), nodes