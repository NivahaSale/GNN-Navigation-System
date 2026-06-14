import osmnx as ox
import networkx as nx


def create_graph():

    print("Downloading Hyderabad road network...")

    center_point = (17.4425, 78.3570)  # Kokapet / Financial District

    G = ox.graph_from_point(
        center_point,
        dist=5000,
        network_type="drive"
    )

    print("Making graph strongly connected...")

    largest_cc = max(
        nx.strongly_connected_components(G),
        key=len
    )

    G = G.subgraph(largest_cc).copy()

    print(f"Nodes: {len(G.nodes)}")
    print(f"Edges: {len(G.edges)}")

    return G