import torch
import pandas as pd
import osmnx as ox
from torch_geometric.data import Data

print("Loading graph...")
G = ox.load_graphml("hyderabad_west.graphml")

print("Loading dataset...")
df = pd.read_csv("traffic_dataset.csv")

# -------------------------
# NODE MAPPING
# -------------------------

nodes = list(G.nodes())
node_map = {n: i for i, n in enumerate(nodes)}

# -------------------------
# NODE FEATURES
# -------------------------

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

# -------------------------
# ENCODINGS
# -------------------------

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
    'clear': 0,
    'rain': 1,
    'heavy_rain': 2
}

incident_map = {
    'none': 0,
    'roadwork': 1,
    'accident': 2
}

# -------------------------
# BUILD GRAPH
# -------------------------

edge_index = []
edge_attr = []
edge_labels = []

print("Building graph tensors...")

for _, row in df.iterrows():

    u = int(row["u"])
    v = int(row["v"])

    if u not in node_map or v not in node_map:
        continue

    edge_index.append([
        node_map[u],
        node_map[v]
    ])

    edge_attr.append([
        float(row["length"]),
        road_map[row["road_type"]],
        float(row["hour"]),
        float(row["day"]),
        weather_map[row["weather"]],
        incident_map[row["incident"]],
        float(row["hotspot"])
    ])

    edge_labels.append(
        float(row["travel_time"])
    )

edge_index = torch.tensor(
    edge_index,
    dtype=torch.long
).t().contiguous()

edge_attr = torch.tensor(
    edge_attr,
    dtype=torch.float
)

y = torch.tensor(
    edge_labels,
    dtype=torch.float
)

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=y
)

torch.save(data, "graph_data.pt")

print("\nSaved graph_data.pt")
print(data)