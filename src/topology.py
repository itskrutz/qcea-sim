"""
Topology builder for QCEA-Sim
- Creates a Barabasi-Albert (scale-free) topology
- Attaches per-edge attributes: bandwidth_bps, prop_delay_ms, loss_prob, queue_size_bytes
- Attaches per-node attributes: residual_energy_j, role ('core'|'edge')
- Exposes functions to:
    * create the graph
    * pick border nodes
    * save/load YAML topology
    * apply time-step dynamic updates to link QoS metrics (bandwidth, delay, loss)
Usage:
    from src.topology import create_scale_free_topology, simple_dynamic_update, save_graph_yaml
    G = create_scale_free_topology(n=25, m=2, seed=42)
    save_graph_yaml(G, "config/topology.yaml")
    # in simulation loop:
    simple_dynamic_update(G, t, params)
"""

import random
import math
import yaml
import networkx as nx

# --- Default constants (tweakable) ---
DEFAULT_CORE_ENERGY_J = 1e6      # Joules for "core" nodes (very large)
DEFAULT_EDGE_ENERGY_J = 1e4      # Joules for "edge" nodes (smaller battery)
DEFAULT_BW_CHOICES_MBPS = [1, 10, 50, 100]  # choose from these Mbps
DEFAULT_PROP_DELAY_MS_RANGE = (1.0, 30.0)   # ms
DEFAULT_LOSS_CHOICES = [0.0, 0.0, 0.001, 0.01]   # baseline loss distribution
DEFAULT_QUEUE_BYTES = 20_000     # 20 KB default queue

# Energy consumption constants (useful later)
TX_ENERGY_PER_BIT = 50e-9   # 50 nJ per bit (example)
RX_ENERGY_PER_BIT = 30e-9   # 30 nJ per bit (example)

def create_scale_free_topology(
    n=25,
    m=2,
    seed=42,
    core_fraction=0.2,
    core_energy_j=DEFAULT_CORE_ENERGY_J,
    edge_energy_j=DEFAULT_EDGE_ENERGY_J,
    bw_choices_mbps=None,
    prop_delay_ms_range=None,
    loss_choices=None,
    queue_size_bytes=DEFAULT_QUEUE_BYTES,
):
    """
    Create a Barabasi-Albert (scale-free) undirected graph with attributes.
    Parameters:
      - n: number of nodes
      - m: BA parameter (edges to attach from new node to existing nodes)
      - seed: RNG seed for reproducibility
      - core_fraction: fraction of nodes treated as 'core' (higher energy)
      - core_energy_j, edge_energy_j: initial energies (Joules)
      - bw_choices_mbps: list of Mbps choices for link bw
      - prop_delay_ms_range: (min,max) propagation delay in ms
      - loss_choices: list of loss probabilities to sample from
      - queue_size_bytes: default buffer size per link
    Returns:
      - networkx.Graph with attributes set
    """
    if bw_choices_mbps is None:
        bw_choices_mbps = DEFAULT_BW_CHOICES_MBPS
    if prop_delay_ms_range is None:
        prop_delay_ms_range = DEFAULT_PROP_DELAY_MS_RANGE
    if loss_choices is None:
        loss_choices = DEFAULT_LOSS_CHOICES

    random.seed(seed)
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    G = nx.Graph(G)  # ensure undirected and remove multi edges if any

    # tag roles: pick highest-degree nodes as core (common in networks)
    degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    n_core = max(1, int(math.floor(core_fraction * n)))
    core_nodes = set([node for node, _deg in degree_sorted[:n_core]])

    # assign node attributes
    for node in G.nodes():
        if node in core_nodes:
            role = 'core'
            energy = float(core_energy_j)
        else:
            role = 'edge'
            energy = float(edge_energy_j)
        G.nodes[node]['role'] = role
        G.nodes[node]['residual_energy_j'] = energy

    # assign edge attributes
    for u, v in list(G.edges()):
        bw_mbps = random.choice(bw_choices_mbps)
        G[u][v]['bandwidth_bps'] = float(bw_mbps * 1e6)
        G[u][v]['prop_delay_ms'] = float(random.uniform(*prop_delay_ms_range))
        G[u][v]['loss_prob'] = float(random.choice(loss_choices))
        G[u][v]['queue_size_bytes'] = int(queue_size_bytes)
        # we keep an additional dynamic state slot to enable updates
        G[u][v]['_dynamic_state'] = {
            'bw_mbps_base': bw_mbps,
            'prop_delay_ms_base': G[u][v]['prop_delay_ms'],
            'loss_prob_base': G[u][v]['loss_prob'],
            'last_update_t': 0.0
        }
    return G


# ----------------------
# Dynamic update helpers
# ----------------------

def simple_dynamic_update(G, t, params=None):
    """
    A simple time-step dynamic updater for link QoS:
      - bandwidth: small random walk around base value
      - prop_delay: small jitter + occasional spike
      - loss: occasional burst events
    This function mutates G in place.
    Arguments:
      - G: networkx graph
      - t: current simulation time (float, e.g., seconds)
      - params: dict of parameters (optional)
         keys and defaults:
           'bw_variation_frac' : 0.2      # fraction +/- variation around base bw
           'delay_jitter_ms'   : 2.0      # ms jitter per tick (stddev)
           'loss_burst_prob'   : 0.001    # prob of a burst event per link per tick
           'loss_burst_add'    : 0.05     # added loss prob during a burst
           'burst_duration_s'  : 5.0      # how long bursts last (seconds)
           'seed'              : None
    Returns:
      - None (G updated in place)
    Notes:
      - This is intentionally simple and deterministic via seed for reproducibility.
      - You can replace with a more complex model later (fading, mobility, etc.)
    """
    if params is None:
        params = {}
    bw_var = params.get('bw_variation_frac', 0.2)
    jitter_ms = params.get('delay_jitter_ms', 2.0)
    burst_prob = params.get('loss_burst_prob', 0.001)
    burst_add = params.get('loss_burst_add', 0.05)
    burst_dur = params.get('burst_duration_s', 5.0)
    seed = params.get('seed', None)
    if seed is not None:
        random.seed(seed + int(t))  # vary by time for reproducibility

    for u, v, d in G.edges(data=True):
        ds = d.get('_dynamic_state', None)
        if ds is None:
            continue

        # BANDWIDTH: random walk around base (bounded > 100 kbps)
        base_bw_mbps = ds.get('bw_mbps_base', d['bandwidth_bps']/1e6)
        # small gaussian variation
        delta = random.gauss(0, bw_var * base_bw_mbps * 0.3)
        new_bw_mbps = max(0.001, base_bw_mbps + delta)  # at least 1 kbps
        d['bandwidth_bps'] = new_bw_mbps * 1e6

        # PROPAGATION DELAY: jitter around base, with possible transient spike
        base_delay = ds.get('prop_delay_ms_base', d['prop_delay_ms'])
        jitter = random.gauss(0, jitter_ms)
        # occasional spike
        if random.random() < 0.0005:
            spike = random.uniform(20.0, 200.0)  # ms spike
        else:
            spike = 0.0
        new_delay = max(0.1, base_delay + jitter + spike)
        d['prop_delay_ms'] = new_delay

        # LOSS: occasional burst events
        base_loss = ds.get('loss_prob_base', d['loss_prob'])
        # if currently in a burst window stored in state, reduce remaining time
        burst_state = ds.get('burst_state', None)
        if burst_state and 'ends_at' in burst_state:
            if t >= burst_state['ends_at']:
                # end the burst
                ds.pop('burst_state', None)
                d['loss_prob'] = base_loss
            else:
                # still in burst
                d['loss_prob'] = min(1.0, base_loss + burst_state.get('add', burst_add))
        else:
            # maybe start a new burst
            if random.random() < burst_prob:
                ds['burst_state'] = {'add': burst_add, 'ends_at': t + burst_dur}
                d['loss_prob'] = min(1.0, base_loss + burst_add)
            else:
                # small fluctuation
                d['loss_prob'] = max(0.0, base_loss + random.gauss(0, base_loss * 0.5))

        # update last_update
        ds['last_update_t'] = t


# ----------------------
# Utilities: border nodes and I/O
# ----------------------

def choose_border_nodes(G, method='degree', fraction=0.1):
    """
    Pick border (ingress/egress) nodes to simulate external traffic injection.
    Methods:
      - 'degree': pick top-degree nodes (useful for core border)
      - 'random': choose random set
    Returns list of node ids.
    """
    n = len(G.nodes())
    k = max(1, int(math.floor(fraction * n)))
    if method == 'degree':
        degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        chosen = [node for node, _deg in degree_sorted[:k]]
    else:
        nodes = list(G.nodes())
        chosen = random.sample(nodes, k)
    return chosen

# ...existing code...

def deduct_path_energy(G, path, packet_size_bits):
    """
    Deduct energy from nodes along a path based on packet transmission.
    
    Args:
        G: NetworkX graph
        path: list of node IDs
        packet_size_bits: size of packet in bits
    
    Returns:
        total_energy_consumed (Joules)
    """
    total_energy = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        
        # TX energy at sender node
        tx_energy = TX_ENERGY_PER_BIT * packet_size_bits
        G.nodes[u]['residual_energy_j'] -= tx_energy
        total_energy += tx_energy
        
        # RX energy at receiver node
        rx_energy = RX_ENERGY_PER_BIT * packet_size_bits
        G.nodes[v]['residual_energy_j'] -= rx_energy
        total_energy += rx_energy
        
        # Prevent negative energy
        G.nodes[u]['residual_energy_j'] = max(0, G.nodes[u]['residual_energy_j'])
        G.nodes[v]['residual_energy_j'] = max(0, G.nodes[v]['residual_energy_j'])
    
    return total_energy

def save_graph_yaml(G, path="config/topology.yaml"):
    """
    Save a compact YAML describing nodes and edges and their important attributes.
    This is human-readable and useful for inspection / reproducibility.
    """
    out = {'nodes': {}, 'edges': []}
    for n, d in G.nodes(data=True):
        out['nodes'][int(n)] = {
            'role': d.get('role'),
            'residual_energy_j': float(d.get('residual_energy_j', 0.0))
        }
    for u, v, d in G.edges(data=True):
        out['edges'].append({
            'u': int(u),
            'v': int(v),
            'bandwidth_bps': float(d.get('bandwidth_bps', 0.0)),
            'prop_delay_ms': float(d.get('prop_delay_ms', 0.0)),
            'loss_prob': float(d.get('loss_prob', 0.0)),
            'queue_size_bytes': int(d.get('queue_size_bytes', 0))
        })
    with open(path, 'w') as f:
        yaml.safe_dump(out, f)
    return path


def load_graph_yaml(path="config/topology.yaml"):
    """
    Load the YAML created by save_graph_yaml back into a networkx Graph.
    This will populate the most important attributes (doesn't restore internal dynamic state).
    """
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    G = nx.Graph()
    for n_str, nd in data.get('nodes', {}).items():
        n = int(n_str)
        G.add_node(n)
        G.nodes[n]['role'] = nd.get('role')
        G.nodes[n]['residual_energy_j'] = float(nd.get('residual_energy_j', 0.0))
    for e in data.get('edges', []):
        u = int(e['u']); v = int(e['v'])
        G.add_edge(u, v)
        G[u][v]['bandwidth_bps'] = float(e.get('bandwidth_bps', 0.0))
        G[u][v]['prop_delay_ms'] = float(e.get('prop_delay_ms', 0.0))
        G[u][v]['loss_prob'] = float(e.get('loss_prob', 0.0))
        G[u][v]['queue_size_bytes'] = int(e.get('queue_size_bytes', 0))
        # set a simple dynamic state so update functions can run
        G[u][v]['_dynamic_state'] = {
            'bw_mbps_base': G[u][v]['bandwidth_bps'] / 1e6,
            'prop_delay_ms_base': G[u][v]['prop_delay_ms'],
            'loss_prob_base': G[u][v]['loss_prob'],
            'last_update_t': 0.0
        }
    return G


# ----------------------
# Small test / summary helper
# ----------------------
def topology_summary(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degs = [d for _, d in G.degree()]
    avg_deg = sum(degs) / len(degs) if degs else 0.0
    bw_list = [d['bandwidth_bps'] for _, _, d in G.edges(data=True)]
    avg_bw_mbps = sum(bw_list) / len(bw_list) / 1e6 if bw_list else 0.0
    return {
        'nodes': n,
        'edges': m,
        'avg_degree': avg_deg,
        'avg_bw_mbps': avg_bw_mbps
    }

# If this module is run directly, generate a tiny sample and print summary
if __name__ == "__main__":
    G = create_scale_free_topology(n=25, m=2, seed=42)
    print("Topology created. Summary:")
    print(topology_summary(G))
    border = choose_border_nodes(G, method='degree', fraction=0.12)
    print("Border nodes (top degree):", border)
    save_path = save_graph_yaml(G, "config/topology_sample.yaml")
    print("Saved topology to:", save_path)
    # show one sample link
    u, v = list(G.edges())[0]
    print("Sample edge attrs:", G[u][v])
