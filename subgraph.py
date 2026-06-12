import networkx as nx
import random


def create_subgraph(G, start_node=None, size=3000):
    """
    Create CONNECTED subgraph (fixes routing failure)
    """

    try:
        print("📍 Creating CONNECTED subgraph...")

        nodes = list(G.nodes())

        # -----------------------------
        # 1. Pick a random starting node
        # -----------------------------
        if start_node is None:
            start_node = random.choice(nodes)

        # -----------------------------
        # 2. BFS expansion (IMPORTANT FIX)
        # -----------------------------
        visited = set()
        queue = [start_node]

        while queue and len(visited) < size:

            node = queue.pop(0)

            if node in visited:
                continue

            visited.add(node)

            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    queue.append(neighbor)

        # -----------------------------
        # 3. Build connected subgraph
        # -----------------------------
        G_sub = G.subgraph(visited).copy()

        print("✅ Connected subgraph created!")
        print("Nodes:", len(G_sub.nodes))
        print("Edges:", len(G_sub.edges))

        return G_sub

    except Exception as e:
        print("❌ Subgraph creation failed:", e)
        return None


# TEST
if __name__ == "__main__":
    import osmnx as ox

    G = ox.graph_from_place(
        "Hyderabad, Telangana, India",
        network_type="drive"
    )

    sub = create_subgraph(G)