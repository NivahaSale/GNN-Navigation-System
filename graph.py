import osmnx as ox
import networkx as nx
import random


def create_graph_with_traffic():

    place = "Hyderabad, Telangana, India"

    print("📡 Downloading road network:", place)

    G = ox.graph_from_place(place, network_type="drive")
    G = G.to_undirected()

    print("🚦 Adding REALISTIC traffic features...")

    for u, v, data in G.edges(data=True):

        length = float(data.get("length", 1.0))

        # 🔥 stronger congestion separation
        hour = random.randint(0, 23)

        if 8 <= hour <= 11 or 17 <= hour <= 20:
            base_congestion = random.uniform(0.6, 1.0)
        else:
            base_congestion = random.uniform(0.1, 0.5)

        noise = random.uniform(0, 0.2)
        congestion = min(1.0, base_congestion + noise)

        data["length"] = length
        data["congestion"] = congestion

        # base weight (fastest route)
        data["weight"] = length

    for n in G.nodes:
        G.nodes[n]["degree"] = float(G.degree[n])

    print("✅ Graph ready (improved traffic separation)")
    print("Nodes:", len(G.nodes))
    print("Edges:", len(G.edges))

    return G