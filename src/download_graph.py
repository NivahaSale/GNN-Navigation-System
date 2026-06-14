import osmnx as ox
import networkx as nx

print("Downloading graph...")

north = 17.470
south = 17.380
east = 78.420
west = 78.320

G = ox.graph_from_bbox(
    north=north,
    south=south,
    east=east,
    west=west,
    network_type="drive"
)

largest_cc = max(nx.strongly_connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()

print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))

ox.save_graphml(G, "hyderabad_west.graphml")

print("Saved!")