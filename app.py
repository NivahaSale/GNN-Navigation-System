from flask import Flask, render_template, request, jsonify
from src.route_engine import get_route

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict")
def predict():

    try:

        start_lat = float(request.args.get("startLat"))
        start_lon = float(request.args.get("startLon"))

        end_lat = float(request.args.get("endLat"))
        end_lon = float(request.args.get("endLon"))

        hour = int(request.args.get("hour", 18))

        weather = request.args.get(
            "weather",
            "clear"
        )

        incident = request.args.get(
            "incident",
            "none"
        )

        route_coords = get_route(
            start_lat,
            start_lon,
            end_lat,
            end_lon,
            hour,
            weather,
            incident
        )

        # Calculate distance
        total_distance = 0

        for i in range(len(route_coords) - 1):

            lat1, lon1 = route_coords[i]
            lat2, lon2 = route_coords[i + 1]

            dx = (lon2 - lon1) * 111
            dy = (lat2 - lat1) * 111

            total_distance += (dx**2 + dy**2) ** 0.5

        distance_km = round(total_distance, 2)

        # Traffic-aware speed estimate
        if weather == "heavy_rain":
            avg_speed = 20
        elif weather == "rain":
            avg_speed = 28
        else:
            avg_speed = 35

        if incident == "accident":
            avg_speed *= 0.6
        elif incident == "roadwork":
            avg_speed *= 0.8

        travel_time = round(
            (distance_km / avg_speed) * 60,
            2
        )

        return jsonify({
            "route": route_coords,
            "distance": distance_km,
            "time": travel_time
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )