import random

TRAFFIC_CLASSES = {
    "voip": {
        "packet_size": 160, 
        "rate": 10, 
        "max_latency": 150, 
        "priority": 1,
        # VoIP: Latency-sensitive, loss-sensitive, needs stability
        "weights": {"wl": 0.5, "wb": 0.1, "wp": 0.3, "we": 0.1, "wc": 0.0}
    },
    "video": {
        "packet_size": 1400, 
        "rate": 5, 
        "max_latency": 300, 
        "priority": 2,
        # Video: Bandwidth-hungry, latency-tolerant, loss-sensitive
        "weights": {"wl": 0.2, "wb": 0.5, "wp": 0.2, "we": 0.1, "wc": 0.0}
    },
    "besteffort": {
        "packet_size": 1500, 
        "rate": 2, 
        "max_latency": None, 
        "priority": 3,
        # Best Effort: Energy-aware, cost-conscious, no strict QoS
        "weights": {"wl": 0.2, "wb": 0.1, "wp": 0.1, "we": 0.4, "wc": 0.2}
    }
}

def get_traffic_weights(traffic_class):
    """
    Get appropriate routing weights for a traffic class.
    
    Args:
        traffic_class: 'voip', 'video', or 'besteffort'
    
    Returns:
        dict: Weight configuration for routing algorithm
    """
    return TRAFFIC_CLASSES.get(traffic_class, {}).get("weights", {
        "wl": 0.25, "wb": 0.25, "wp": 0.2, "we": 0.2, "wc": 0.1
    })

def generate_flows(G, num_flows=5, seed=42, traffic_mix=None):
    """
    Generate traffic flows with configurable mix ratios.
    
    Args:
        G: NetworkX graph
        num_flows: Number of flows to generate
        seed: Random seed for reproducibility
        traffic_mix: Dict with traffic class ratios, e.g., 
                    {"voip": 0.5, "video": 0.3, "besteffort": 0.2}
                    If None, uses uniform distribution
    
    Returns:
        List of flow dictionaries
    """
    random.seed(seed)
    nodes = list(G.nodes())
    flows = []
    
    # Default to uniform mix if not specified
    if traffic_mix is None:
        traffic_mix = {"voip": 0.33, "video": 0.33, "besteffort": 0.34}
    
    # Create weighted list based on mix ratios
    traffic_list = []
    for tclass, ratio in traffic_mix.items():
        count = max(1, int(num_flows * ratio))
        traffic_list.extend([tclass] * count)
    
    # Fill remaining slots if needed
    while len(traffic_list) < num_flows:
        traffic_list.append(random.choice(list(TRAFFIC_CLASSES.keys())))
    
    # Generate flows
    for i in range(num_flows):
        src, dst = random.sample(nodes, 2)
        tclass = traffic_list[i] if i < len(traffic_list) else random.choice(list(TRAFFIC_CLASSES.keys()))
        flows.append({
            "src": src, 
            "dst": dst, 
            "class": tclass,
            "id": i,
            "timestamp": 0
        })
    
    return flows
