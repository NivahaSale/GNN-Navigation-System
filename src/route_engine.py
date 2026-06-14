import torch
import osmnx as ox
import networkx as nx

from gnn_model import GraphSAGERouter

print("Loading graph...")
G = ox.load_graphml("hyderabad_west.graphml")

print("Loading model...")

model = GraphSAGERouter(edge_features=7)

model.load_state_dict(
    torch.load(
        "gnn_router.pt",
        map_location="cpu"
    )
)

model.eval()

# Encoders

road_map = {
    'motorway': 0,
    'motorway_link': 1,
    'primary': 2,
    'primary_link': 3,
    'secondary': 4,
    'secondary_link': 5,
    'tertiary': 6,
    'tertiary_link': 7,
    'residential': 8,
    'living_street': 9,
    'unclassified': 10
}

weather_map = {
    "clear": 0,
    "rain": 1,
    "heavy_rain": 2
}

incident_map = {
    "none": 0,
    "roadwork": 1,
    "accident": 2
}

# Build node features once

nodes = list(G.nodes())
node_map = {n: i for i, n in enumerate(nodes)}

x = []

for n in nodes:

    lat = float(G.nodes[n]["y"])
    lon = float(G.nodes[n]["x"])
    degree = float(G.degree(n))

    x.append([
        lat,
        lon,
        degree
    ])

x = torch.tensor(x, dtype=torch.float)
def get_route(
    start_lat,
    start_lon,
    end_lat,
    end_lon,
    hour=18,
    weather="clear",
    incident="none"
):

    edge_index = []
    edge_attr = []
    edge_refs = []

    for u, v, k, data in G.edges(keys=True, data=True):

        length = float(
            data.get("length", 50)
        )

        highway = data.get(
            "highway",
            "residential"
        )

        if isinstance(highway, list):
            highway = highway[0]

        highway = str(highway)

        road_type = road_map.get(
            highway,
            8
        )

        hotspot = 0

        edge_index.append([
            node_map[u],
            node_map[v]
        ])

        edge_attr.append([
            length,
            road_type,
            hour,
            0,
            weather_map[weather],
            incident_map[incident],
            hotspot
        ])

        edge_refs.append(
            (u, v, k)
        )

    edge_index = torch.tensor(
        edge_index,
        dtype=torch.long
    ).t()

    edge_attr = torch.tensor(
        edge_attr,
        dtype=torch.float
    )

    with torch.no_grad():

        predictions = model(
            x,
            edge_index,
            edge_attr
        )

    for i, (u, v, k) in enumerate(edge_refs):

        G[u][v][k]["gnn_weight"] = float(
            max(predictions[i], 1)
        )

    source = ox.distance.nearest_nodes(
        G,
        X=start_lon,
        Y=start_lat
    )

    target = ox.distance.nearest_nodes(
        G,
        X=end_lon,
        Y=end_lat
    )

    route = nx.shortest_path(
        G,
        source,
        target,
        weight="gnn_weight"
    )

    route_coords = [
        (
            G.nodes[n]["y"],
            G.nodes[n]["x"]
        )
        for n in route
    ]

    return route_coords