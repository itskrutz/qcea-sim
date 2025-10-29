import networkx as nx

class QCEARouting:
    def __init__(self, weights):
        """
        weights = {
            'wl': 0.3,  # latency
            'wb': 0.2,  # bandwidth
            'wp': 0.2,  # packet loss
            'we': 0.2,  # residual energy
            'wc': 0.1   # transmission cost
        }
        """
        self.weights = weights

    def compute_path(self, G, src, dst):
        """Compute path using weighted multi-objective cost function."""
        def link_cost(u, v, data):
            # extract metrics
            L = data.get("prop_delay_ms", 1.0)
            B = data.get("bandwidth_bps", 1e6)
            P = data.get("loss_prob", 0.0)
            E = min(G.nodes[u]['residual_energy_j'], G.nodes[v]['residual_energy_j'])
            C = 1 / B  # proxy for transmission cost per bit

            # normalize roughly and combine
            return (
                self.weights["wl"] * (L / 100.0) +
                self.weights["wb"] * (1 / (B / 1e6 + 1e-6)) +
                self.weights["wp"] * P +
                self.weights["we"] * (1 / (E / 1e4 + 1e-6)) +
                self.weights["wc"] * C
            )

        for u, v, data in G.edges(data=True):
            data["cost"] = link_cost(u, v, data)

        try:
            path = nx.shortest_path(G, src, dst, weight="cost")
            total_cost = nx.shortest_path_length(G, src, dst, weight="cost")
            return path, total_cost
        except nx.NetworkXNoPath:
            return None, float('inf')
