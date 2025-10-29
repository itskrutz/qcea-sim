import random

TRAFFIC_CLASSES = {
    "voip": {"packet_size": 160, "rate": 10, "max_latency": 150},
    "video": {"packet_size": 1400, "rate": 5, "max_latency": 300},
    "besteffort": {"packet_size": 1500, "rate": 2, "max_latency": None}
}

def generate_flows(G, num_flows=5, seed=42):
    random.seed(seed)
    nodes = list(G.nodes())
    flows = []
    for _ in range(num_flows):
        src, dst = random.sample(nodes, 2)
        tclass = random.choice(list(TRAFFIC_CLASSES.keys()))
        flows.append({"src": src, "dst": dst, "class": tclass})
    return flows
