def route_stats(G, path):

    total_length = 0
    total_congestion = 0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]

        data = G.get_edge_data(u, v)
        if not data:
            continue

        if isinstance(data, dict) and 0 in data:
            data = data[0]

        total_length += data.get("length", 0)
        total_congestion += data.get("congestion", 0)

    return {
        "length": total_length,
        "congestion": total_congestion,
        "hops": len(path)
    }