import networkx as nx

def build_sample_graph():
    """
    Creates a simple undirected weighted graph for demonstration.
    """
    G = nx.Graph()

    # (node1, node2, weight)
    edges = [
        (1, 2, 4),
        (1, 3, 2),
        (2, 3, 5),
        (2, 4, 10),
        (3, 5, 3),
        (5, 4, 4)
    ]

    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    return G


def run_dijkstra(G, source, target):
    """
    Runs Dijkstra’s shortest path algorithm between two nodes.
    """
    path = nx.dijkstra_path(G, source, target, weight='weight')
    cost = nx.dijkstra_path_length(G, source, target, weight='weight')
    return path, cost


if __name__ == "__main__":
    print("Running Dijkstra baseline simulation...")
    G = build_sample_graph()
    path, cost = run_dijkstra(G, 1, 4)
    print(f"Shortest path from 1 to 4: {path} (total cost = {cost})")
