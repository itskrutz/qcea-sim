import random
from src.qcea_routing import QCEARouting
from src.baseline_dijkstra import run_dijkstra
from src.topology import (
    load_graph_yaml,
    simple_dynamic_update,
    deduct_path_energy
)
from src.traffic import generate_flows
from src.metrics import Metrics

def run_simulation(config):
    print("Loading topology...")
    G = load_graph_yaml(config["topology_path"])

    algo = config.get("routing", "qcea")
    print(f"Routing algorithm selected: {algo}")
    weights = config.get("weights", {})
    steps = config.get("time_steps", 10)
    qcea = QCEARouting(weights)

    flows = generate_flows(G, num_flows=config.get("num_flows", 3))
    metrics = Metrics()

    for t in range(steps):
        print(f"\n=== Time Step {t} ===")
        simple_dynamic_update(G, t)

        total_delay, total_tp, total_energy, total_deliv = [], [], [], []

        for flow in flows:
            src, dst = flow["src"], flow["dst"]
            pkt_size = 8 * 1000  # bits

            if algo == "dijkstra":
                path, cost = run_dijkstra(G, src, dst)
            else:
                path, cost = qcea.compute_path(G, src, dst)

            if path is None:
                continue

            delay = cost
            throughput = 1 / delay if delay else 0
            energy_used = deduct_path_energy(G, path, pkt_size)

            total_delay.append(delay)
            total_tp.append(throughput)
            total_energy.append(energy_used)
            total_deliv.append(1.0)  # assume delivered

        metrics.log(sum(total_delay)/len(flows),
                    sum(total_tp)/len(flows),
                    sum(total_energy),
                    sum(total_deliv)/len(flows))

    print("\n=== Simulation Complete ===")
    print(metrics.summary())
    return metrics
