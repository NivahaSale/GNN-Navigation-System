import torch
from torch_geometric.data import Data


def convert_to_pyg_data(G):

    print("🔄 Converting NetworkX graph to PyG format...")

    # IMPORTANT: fixed node order (MUST be reused in predict.py)
    nodes = list(G.nodes())
    node_map = {node: i for i, node in enumerate(nodes)}

    # -----------------------------
    # NODE FEATURES
    # -----------------------------
    x = torch.tensor(
        [[G.degree[node]] for node in nodes],
        dtype=torch.float
    )

    edge_index = []
    edge_attr = []

    for u, v, data in G.edges(data=True):

        u_idx = node_map[u]
        v_idx = node_map[v]

        length = float(data.get("length", 1.0))
        congestion = float(data.get("congestion", 0.0))
        speed = float(data.get("speed_limit", 40.0))
        road_type = float(data.get("road_type_score", 1.0))

        # add both directions (undirected graph)
        edge_index.append([u_idx, v_idx])
        edge_index.append([v_idx, u_idx])

        edge_attr.append([length, congestion, speed, road_type])
        edge_attr.append([length, congestion, speed, road_type])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # -----------------------------
    # SAFETY CHECKS
    # -----------------------------
    assert edge_index.shape[1] == edge_attr.shape[0], "Edge mismatch!"
    assert not torch.isnan(edge_attr).any(), "NaN in edge_attr!"

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    print("✅ Conversion complete!")
    print("Nodes:", x.shape)
    print("Edges:", edge_index.shape)
    print("Edge features:", edge_attr.shape)

    # 🔥 CRITICAL FIX: return node list for mapping
    return data, nodes