import torch


def build_graph_tensors(G):

    nodes = list(G.nodes())

    node_map = {
        node: idx
        for idx, node in enumerate(nodes)
    }

    x = []

    for node in nodes:

        lat = float(G.nodes[node]["y"])
        lon = float(G.nodes[node]["x"])

        degree = float(G.degree(node))

        x.append([
            lat,
            lon,
            degree
        ])

    x = torch.tensor(
        x,
        dtype=torch.float
    )

    edge_index = []
    edge_attr = []
    edge_labels = []

    road_map = {
        "motorway": 5,
        "trunk": 4,
        "primary": 3,
        "secondary": 2,
        "tertiary": 2,
        "residential": 1,
        "service": 1
    }

    for u, v, k, data in G.edges(
        keys=True,
        data=True
    ):

        edge_index.append([
            node_map[u],
            node_map[v]
        ])

        length = float(
            data.get("length", 1)
        )

        highway = data.get(
            "highway",
            "residential"
        )

        if isinstance(highway, list):
            highway = highway[0]

        road_type = road_map.get(
            highway,
            1
        )

        speed = 20 + road_type * 10

        travel_time = (
            length / 1000
        ) / speed * 60

        edge_attr.append([
            length,
            road_type
        ])

        edge_labels.append(
            travel_time
        )

    edge_index = torch.tensor(
        edge_index,
        dtype=torch.long
    ).t().contiguous()

    edge_attr = torch.tensor(
        edge_attr,
        dtype=torch.float
    )

    edge_labels = torch.tensor(
        edge_labels,
        dtype=torch.float
    )

    return (
        x,
        edge_index,
        edge_attr,
        edge_labels,
        node_map
    )