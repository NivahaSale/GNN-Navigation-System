import networkx as nx
import random

def create_subgraph(G, size=3000):

    print("Creating CONNECTED subgraph...")

    start = random.choice(list(G.nodes))

    visited = set()
    queue = [start]

    while queue and len(visited) < size:

        node = queue.pop(0)

        if node in visited:
            continue

        visited.add(node)

        for nbr in G.neighbors(node):
            if nbr not in visited:
                queue.append(nbr)

    G_sub = G.subgraph(visited).copy()
    G_sub = nx.Graph(G_sub)

    print("Subgraph ready:", len(G_sub.nodes), "nodes")

    return G_sub