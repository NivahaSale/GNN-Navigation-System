import osmnx as ox
import networkx as nx
import random


def safe_speed(maxspeed):

    if isinstance(maxspeed, list):
        maxspeed = maxspeed[0]

    if isinstance(maxspeed, str):
        maxspeed = maxspeed.split()[0]

    try:
        return float(maxspeed)
    except:
        return 40.0


def road_score(highway):

    if isinstance(highway, list):
        highway = highway[0]

    if highway in ["motorway", "trunk"]:
        return 0.3
    elif highway in ["primary", "secondary"]:
        return 0.6
    else:
        return 0.9


def create_graph_with_traffic():

    place = "Hyderabad, Telangana, India"

    print("📡 Downloading road network:", place)

    # Load graph
    G = ox.graph_from_place(place, network_type="drive")

    # ✅ FIX: stable undirected conversion (NO utils_graph)
    G = G.to_undirected()

    print("🚦 Adding traffic features...")

    for u, v, data in G.edges(data=True):

        length = float(data.get("length", 1.0))
        congestion = random.uniform(0.1, 1.0)

        maxspeed = data.get("maxspeed", 40)
        speed = safe_speed(maxspeed)

        highway = data.get("highway", "residential")
        score = road_score(highway)

        data["length"] = length
        data["congestion"] = congestion
        data["speed_limit"] = speed
        data["road_type_score"] = score

    # Node features
    for n in G.nodes:
        G.nodes[n]["degree"] = float(G.degree[n])

    print("✅ Graph ready")
    print("Nodes:", len(G.nodes))
    print("Edges:", len(G.edges))

    return G

if __name__ == "__main__":
    G = create_graph_with_traffic()