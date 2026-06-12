import osmnx as ox
import networkx as nx
import random


def create_graph_with_traffic():
    try:
        # -----------------------------
        # 1. Use PLACE instead of bbox (RECOMMENDED)
        # -----------------------------
        place_name = "Hyderabad, Telangana, India"

        print("📡 Downloading road network for:", place_name)

        G = ox.graph_from_place(
            place_name,
            network_type="drive"
        )

        print("✅ Graph loaded successfully!")
        print(f"Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")

        # -----------------------------
        # 2. Convert to undirected (for GNN + routing)
        # -----------------------------
        G = G.to_undirected()

        # -----------------------------
        # 3. Add edge features (GNN-ready)
        # -----------------------------
        print("🚦 Adding traffic features...")

        for u, v, key, data in G.edges(keys=True, data=True):

            length = float(data.get("length", 1.0))

            # synthetic congestion (replace later with ML model)
            congestion = random.uniform(0.1, 1.0)

            highway = data.get("highway", "residential")
            if isinstance(highway, list):
                highway = highway[0]

            # road type encoding
            if highway in ["motorway", "trunk"]:
                road_type_score = 0.3
            elif highway in ["primary", "secondary"]:
                road_type_score = 0.6
            else:
                road_type_score = 0.9

            maxspeed = data.get("maxspeed", 40)
            if isinstance(maxspeed, list):
                maxspeed = maxspeed[0]

            try:
                speed_limit = float(maxspeed)
            except:
                speed_limit = 40.0

            # -----------------------------
            # STORE EDGE FEATURES
            # -----------------------------
            data["length"] = length
            data["congestion"] = congestion
            data["road_type_score"] = road_type_score
            data["speed_limit"] = speed_limit

            # fallback routing cost
            data["base_cost"] = length * 0.6 + congestion * 10.0

        # -----------------------------
        # 4. Node features (IMPORTANT for GNN)
        # -----------------------------
        for node in G.nodes:
            G.nodes[node]["degree"] = float(G.degree[node])

        print("✅ Graph ready for GNN training!")

        return G

    except Exception as e:
        print("❌ Error creating graph:", e)
        return None


# -----------------------------
# TEST RUN
# -----------------------------
if __name__ == "__main__":
    G = create_graph_with_traffic()

    if G:
        print("\n===== SUMMARY =====")
        print("Nodes:", len(G.nodes))
        print("Edges:", len(G.edges))

        print("\nSample node:")
        print(list(G.nodes(data=True))[0])

        print("\nSample edge:")
        print(list(G.edges(data=True))[0])