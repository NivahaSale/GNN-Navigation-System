from src.route_engine import get_route

route = get_route(
    17.4425,
    78.3772,
    17.3974,
    78.3347,
    hour=18,
    weather="rain",
    incident="none"
)

print("Points:", len(route))
print(route[:5])