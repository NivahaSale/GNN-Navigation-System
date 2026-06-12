import osmnx as ox
import networkx as nx
import random
from ml_model.inference import get_congestion


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

def generate_realistic_conditions():

    hour = random.randint(0, 23)
    day = random.randint(0, 6)

    
    if hour in [8, 9, 10, 17, 18, 19]:
        base_congestion = random.uniform(0.6, 1.0)
    elif hour in [11, 12, 13, 14, 15]:
        base_congestion = random.uniform(0.3, 0.6)
    else:
        base_congestion = random.uniform(0.1, 0.4)

    
    if day in [5, 6]:
        base_congestion *= 0.7

    
    temp = random.uniform(290, 310)
    rain = random.choices([0, 0, 0.5, 2.0, 5.0])[0]
    snow = 0
    clouds = random.randint(20, 100)

    return temp, rain, snow, clouds, hour, day, base_congestion


def create_graph_with_traffic():

    place = "Hyderabad, Telangana, India"

    print("Downloading road network:", place)

    G = ox.graph_from_place(place, network_type="drive")
    G = G.to_undirected()

    print("Adding REALISTIC traffic features...")

    for u, v, data in G.edges(data=True):

        length = float(data.get("length", 1.0))

        
        temp, rain, snow, clouds, hour, day, base_congestion = generate_realistic_conditions()

        
        congestion = get_congestion(
            temp, rain, snow, clouds, hour, day, 0
        )

        
        congestion = 0.6 * base_congestion + 0.4 * congestion

        maxspeed = data.get("maxspeed", 40)
        speed = safe_speed(maxspeed)

        highway = data.get("highway", "residential")
        score = road_score(highway)

        
        data["weight"] = length + (congestion * 12)

        
        data["length"] = length
        data["congestion"] = congestion
        data["speed_limit"] = speed
        data["road_type_score"] = score
        data["hour"] = hour
        data["day"] = day

    for n in G.nodes:
        G.nodes[n]["degree"] = float(G.degree[n])

    print("Graph ready with realistic traffic")
    print("Nodes:", len(G.nodes))
    print("Edges:", len(G.edges))

    return G


if __name__ == "__main__":
    G = create_graph_with_traffic()