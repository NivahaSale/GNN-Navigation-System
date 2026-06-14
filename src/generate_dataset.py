import os
import random
import pandas as pd
import osmnx as ox
from tqdm import tqdm

# ==========================
# LOAD GRAPH
# ==========================
print("Loading graph...")

GRAPH_FILE = "hyderabad_west.graphml"

if not os.path.exists(GRAPH_FILE):
    raise FileNotFoundError(
        f"{GRAPH_FILE} not found. Run download_graph.py first."
    )

G = ox.load_graphml(GRAPH_FILE)

print(f"Nodes: {len(G.nodes)}")
print(f"Edges: {len(G.edges)}")

# ==========================
# ROAD SPEEDS (km/h)
# ==========================

ROAD_SPEEDS = {
    "motorway": 70,
    "motorway_link": 60,
    "trunk": 60,
    "trunk_link": 50,
    "primary": 50,
    "primary_link": 45,
    "secondary": 40,
    "secondary_link": 35,
    "tertiary": 35,
    "tertiary_link": 30,
    "residential": 25,
    "service": 20,
    "unclassified": 25,
    "living_street": 15
}

# ==========================
# WEATHER FACTORS
# ==========================

WEATHER_FACTORS = {
    "clear": 1.0,
    "rain": 0.8,
    "heavy_rain": 0.6
}

# ==========================
# INCIDENT FACTORS
# ==========================

INCIDENT_FACTORS = {
    "none": 1.0,
    "roadwork": 0.7,
    "accident": 0.4
}

# ==========================
# DAY MAPPING
# ==========================

DAYS = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

# ==========================
# TRAFFIC LOGIC
# ==========================

def apply_rush_hour(speed, hour):

    # Morning peak
    if hour in [8, 9, 10]:
        speed *= 0.55

    # Evening peak
    elif hour in [17, 18, 19, 20]:
        speed *= 0.50

    # Night
    elif hour >= 22 or hour <= 5:
        speed *= 1.15

    return speed


# ==========================
# HOTSPOT LOCATIONS
# ==========================

HOTSPOTS = [

    # Gachibowli
    (17.4400, 78.3489),

    # Financial District
    (17.4188, 78.3415),

    # Kokapet
    (17.3974, 78.3347),

    # Madhapur
    (17.4483, 78.3915),

    # Kondapur
    (17.4698, 78.3678)
]


def near_hotspot(lat, lon):

    for hlat, hlon in HOTSPOTS:

        distance = ((lat - hlat) ** 2 + (lon - hlon) ** 2) ** 0.5

        if distance < 0.02:
            return True

    return False


# ==========================
# DATASET CREATION
# ==========================

rows = []

print("Generating realistic traffic samples...")

for u, v, k, data in tqdm(G.edges(keys=True, data=True)):

    length = float(data.get("length", 50))

    highway = data.get("highway", "residential")

    if isinstance(highway, list):
        highway = highway[0]

    highway = str(highway)

    base_speed = ROAD_SPEEDS.get(highway, 30)

    lat = G.nodes[u]["y"]
    lon = G.nodes[u]["x"]

    hotspot = near_hotspot(lat, lon)

    # Generate multiple traffic scenarios
    for _ in range(3):

        hour = random.randint(0, 23)

        day_name = random.choice(list(DAYS.keys()))
        day = DAYS[day_name]

        weather = random.choice(
            ["clear", "clear", "clear",
             "rain", "heavy_rain"]
        )

        incident = random.choice(
            ["none", "none", "none",
             "roadwork", "accident"]
        )

        speed = float(base_speed)

        # Rush hour
        speed = apply_rush_hour(speed, hour)

        # Weather effect
        speed *= WEATHER_FACTORS[weather]

        # Incident effect
        speed *= INCIDENT_FACTORS[incident]

        # Business district congestion
        if hotspot:
            speed *= 0.80

        # Random variation
        speed *= random.uniform(0.9, 1.1)

        speed = max(speed, 5)

        # Convert km/h → m/s
        speed_mps = speed * 1000 / 3600

        travel_time = length / speed_mps

        rows.append({
            "u": str(u),
            "v": str(v),

            "length": round(length, 2),

            "road_type": highway,

            "hour": hour,

            "day": day,

            "weather": weather,

            "incident": incident,

            "hotspot": int(hotspot),

            "speed_kmh": round(speed, 2),

            "travel_time": round(travel_time, 2)
        })


# ==========================
# SAVE DATASET
# ==========================

df = pd.DataFrame(rows)

OUTPUT_FILE = "traffic_dataset.csv"

df.to_csv(OUTPUT_FILE, index=False)

print("\nDataset generated successfully!")
print(f"Rows: {len(df):,}")
print(f"Saved as: {OUTPUT_FILE}")

print("\nSample:")
print(df.head())