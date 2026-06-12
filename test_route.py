import osmnx as ox
import networkx as nx

bbox = (
    78.3245,  # west
    17.3807,  # south
    78.3444,  # east
    17.4372   # north
)

print("Downloading graph...")

G = ox.graph_from_bbox(
    bbox,
    network_type="drive"
)

print("Graph loaded!")
print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))

G = G.to_undirected()

print("Converted to undirected graph.")

nodes = list(G.nodes)

source = nodes[10]
destination = nodes[200]

print("Source:", source)
print("Destination:", destination)

try:
    route = nx.shortest_path(
        G,
        source,
        destination,
        weight="length"
    )

    print("\nRoute Found")
    print("Route contains", len(route), "nodes")

    print("\nFirst 10 nodes in route:")
    print(route[:10])

except nx.NetworkXNoPath:
    print("No path found")