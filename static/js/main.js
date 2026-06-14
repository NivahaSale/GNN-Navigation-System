document.addEventListener(
    "DOMContentLoaded",
    function () {

        const map = L.map("map")
            .setView(
                [17.3850, 78.4867],
                12
            );

        L.tileLayer(
            "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            {
                maxZoom: 19
            }
        ).addTo(map);

        let routeLine = null;

        const markers = [];

        document
            .getElementById("route-btn")
            .addEventListener(
                "click",
                function () {

                    const origin =
                        document
                        .getElementById("origin")
                        .value
                        .split(",");

                    const destination =
                        document
                        .getElementById("destination")
                        .value
                        .split(",");

                    const startLat =
                        parseFloat(origin[0]);

                    const startLon =
                        parseFloat(origin[1]);

                    const endLat =
                        parseFloat(destination[0]);

                    const endLon =
                        parseFloat(destination[1]);

                    const hour =
                        document
                        .getElementById("hour")
                        .value;

                    const weather =
                        document
                        .getElementById("weather")
                        .value;

                    const incident =
                        document
                        .getElementById("incident")
                        .value;

                    const loading =
                        document.createElement("div");

                    loading.id = "loading";

                    loading.innerHTML =
                        "Calculating route...";

                    document.body.appendChild(
                        loading
                    );

                    fetch(
                        `/predict?startLat=${startLat}` +
                        `&startLon=${startLon}` +
                        `&endLat=${endLat}` +
                        `&endLon=${endLon}` +
                        `&hour=${hour}` +
                        `&weather=${weather}` +
                        `&incident=${incident}`
                    )
                    .then(
                        response =>
                        response.json()
                    )
                    .then(
                        data => {

                            loading.remove();

                            if(data.error){

                                alert(
                                    data.error
                                );

                                return;
                            }

                            markers.forEach(
                                m => map.removeLayer(m)
                            );

                            markers.length = 0;

                            if(routeLine){

                                map.removeLayer(
                                    routeLine
                                );
                            }

                            const startMarker =
                                L.marker(
                                    data.route[0]
                                ).addTo(map);

                            markers.push(
                                startMarker
                            );

                            const endMarker =
                                L.marker(
                                    data.route[
                                        data.route.length-1
                                    ]
                                ).addTo(map);

                            markers.push(
                                endMarker
                            );

                            routeLine =
                                L.polyline(
                                    data.route,
                                    {
                                        color:"blue",
                                        weight:5
                                    }
                                ).addTo(map);

                            map.fitBounds(
                                routeLine.getBounds()
                            );

                            document
                            .getElementById(
                                "distance"
                            ).textContent =
                                data.distance;

                            document
                            .getElementById(
                                "time"
                            ).textContent =
                                data.time;
                        }
                    )
                    .catch(
                        err => {

                            loading.remove();

                            console.error(
                                err
                            );

                            alert(
                                "Route failed"
                            );
                        }
                    );
                }
            );

        window.resetMap = function(){

            markers.forEach(
                m => map.removeLayer(m)
            );

            markers.length = 0;

            if(routeLine){

                map.removeLayer(
                    routeLine
                );
            }

            document
            .getElementById(
                "distance"
            ).textContent = "--";

            document
            .getElementById(
                "time"
            ).textContent = "--";
        };
    }
);