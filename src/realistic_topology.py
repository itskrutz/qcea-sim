"""
Enhanced realistic network topologies for P-QCEA evaluation.
Includes multiple topology types with realistic characteristics:
- Hierarchical ISP-like networks
- Data center topologies (Fat-Tree, Leaf-Spine)
- Wireless mesh networks
- Hybrid edge-core architectures
"""

import networkx as nx
import random
import math
import yaml
from src.topology import (
    DEFAULT_CORE_ENERGY_J, 
    DEFAULT_EDGE_ENERGY_J,
    TX_ENERGY_PER_BIT,
    RX_ENERGY_PER_BIT
)


def create_hierarchical_isp_topology(
    num_tier1=3,
    num_tier2_per_tier1=4,
    num_tier3_per_tier2=5,
    seed=42
):
    """
    Create a realistic hierarchical ISP-like topology.
    
    Tier 1: Core routers (full mesh, high capacity)
    Tier 2: Distribution routers (connected to multiple tier1)
    Tier 3: Edge routers (connected to 1-2 tier2)
    
    Args:
        num_tier1: Number of tier 1 (core) nodes
        num_tier2_per_tier1: Tier 2 nodes per tier 1 node
        num_tier3_per_tier2: Tier 3 nodes per tier 2 node
        seed: Random seed
    
    Returns:
        NetworkX graph with realistic ISP structure
    """
    random.seed(seed)
    G = nx.Graph()
    
    # Track node tiers
    tier1_nodes = []
    tier2_nodes = []
    tier3_nodes = []
    
    node_id = 0
    
    # Create Tier 1 (Core) - Full mesh with high bandwidth
    print(f"Creating Tier 1 (Core): {num_tier1} nodes...")
    for i in range(num_tier1):
        G.add_node(node_id)
        G.nodes[node_id]['role'] = 'core'
        G.nodes[node_id]['tier'] = 1
        G.nodes[node_id]['residual_energy_j'] = DEFAULT_CORE_ENERGY_J * 2  # Very high
        tier1_nodes.append(node_id)
        node_id += 1
    
    # Tier 1 full mesh (high-speed backbone)
    for i in range(len(tier1_nodes)):
        for j in range(i+1, len(tier1_nodes)):
            u, v = tier1_nodes[i], tier1_nodes[j]
            G.add_edge(u, v)
            G[u][v]['bandwidth_bps'] = random.choice([1e9, 10e9, 40e9])  # 1-40 Gbps
            G[u][v]['prop_delay_ms'] = random.uniform(0.5, 5.0)  # Low latency
            G[u][v]['loss_prob'] = random.choice([0.0, 0.0001, 0.0005])  # Very low loss
            G[u][v]['queue_size_bytes'] = 1_000_000  # 1 MB
            G[u][v]['link_type'] = 'tier1_backbone'
            _init_dynamic_state(G[u][v])
    
    # Create Tier 2 (Distribution)
    print(f"Creating Tier 2 (Distribution): {num_tier1 * num_tier2_per_tier1} nodes...")
    for tier1_node in tier1_nodes:
        for i in range(num_tier2_per_tier1):
            G.add_node(node_id)
            G.nodes[node_id]['role'] = 'distribution'
            G.nodes[node_id]['tier'] = 2
            G.nodes[node_id]['residual_energy_j'] = DEFAULT_CORE_ENERGY_J
            tier2_nodes.append(node_id)
            
            # Connect to parent tier1 node
            u, v = tier1_node, node_id
            G.add_edge(u, v)
            G[u][v]['bandwidth_bps'] = random.choice([100e6, 1e9, 10e9])  # 100Mbps-10Gbps
            G[u][v]['prop_delay_ms'] = random.uniform(1.0, 10.0)
            G[u][v]['loss_prob'] = random.choice([0.0, 0.001, 0.005])
            G[u][v]['queue_size_bytes'] = 500_000
            G[u][v]['link_type'] = 'tier1_to_tier2'
            _init_dynamic_state(G[u][v])
            
            node_id += 1
    
    # Add some tier2-to-tier2 connections for redundancy
    for i in range(len(tier2_nodes)):
        # Connect to 1-2 nearby tier2 nodes
        num_peers = random.randint(1, 2)
        candidates = [n for n in tier2_nodes if n != tier2_nodes[i]]
        peers = random.sample(candidates, min(num_peers, len(candidates)))
        
        for peer in peers:
            if not G.has_edge(tier2_nodes[i], peer):
                u, v = tier2_nodes[i], peer
                G.add_edge(u, v)
                G[u][v]['bandwidth_bps'] = random.choice([100e6, 1e9])  # Lower than tier1
                G[u][v]['prop_delay_ms'] = random.uniform(2.0, 15.0)
                G[u][v]['loss_prob'] = random.choice([0.001, 0.005, 0.01])
                G[u][v]['queue_size_bytes'] = 200_000
                G[u][v]['link_type'] = 'tier2_peering'
                _init_dynamic_state(G[u][v])
    
    # Create Tier 3 (Edge)
    print(f"Creating Tier 3 (Edge): {len(tier2_nodes) * num_tier3_per_tier2} nodes...")
    for tier2_node in tier2_nodes:
        for i in range(num_tier3_per_tier2):
            G.add_node(node_id)
            G.nodes[node_id]['role'] = 'edge'
            G.nodes[node_id]['tier'] = 3
            G.nodes[node_id]['residual_energy_j'] = DEFAULT_EDGE_ENERGY_J
            tier3_nodes.append(node_id)
            
            # Connect to parent tier2 node
            u, v = tier2_node, node_id
            G.add_edge(u, v)
            G[u][v]['bandwidth_bps'] = random.choice([10e6, 100e6, 1e9])  # 10Mbps-1Gbps
            G[u][v]['prop_delay_ms'] = random.uniform(5.0, 30.0)  # Higher latency
            G[u][v]['loss_prob'] = random.choice([0.001, 0.01, 0.02])  # Higher loss
            G[u][v]['queue_size_bytes'] = 50_000
            G[u][v]['link_type'] = 'tier2_to_tier3'
            _init_dynamic_state(G[u][v])
            
            # Some tier3 nodes connect to backup tier2
            if random.random() < 0.3:  # 30% have redundant connection
                backup_tier2 = random.choice([n for n in tier2_nodes if n != tier2_node])
                if not G.has_edge(node_id, backup_tier2):
                    G.add_edge(node_id, backup_tier2)
                    G[node_id][backup_tier2]['bandwidth_bps'] = random.choice([10e6, 100e6])
                    G[node_id][backup_tier2]['prop_delay_ms'] = random.uniform(10.0, 40.0)
                    G[node_id][backup_tier2]['loss_prob'] = random.choice([0.01, 0.02])
                    G[node_id][backup_tier2]['queue_size_bytes'] = 50_000
                    G[node_id][backup_tier2]['link_type'] = 'tier3_backup'
                    _init_dynamic_state(G[node_id][backup_tier2])
            
            node_id += 1
    
    # Store metadata
    G.graph['topology_type'] = 'hierarchical_isp'
    G.graph['tier1_nodes'] = tier1_nodes
    G.graph['tier2_nodes'] = tier2_nodes
    G.graph['tier3_nodes'] = tier3_nodes
    
    print(f"✓ Created hierarchical ISP topology:")
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  Total edges: {G.number_of_edges()}")
    print(f"  Tier 1 (Core): {len(tier1_nodes)} nodes")
    print(f"  Tier 2 (Distribution): {len(tier2_nodes)} nodes")
    print(f"  Tier 3 (Edge): {len(tier3_nodes)} nodes")
    
    return G


def create_datacenter_fattree_topology(k=4, seed=42):
    """
    Create a Fat-Tree data center topology.
    
    Fat-Tree is widely used in data centers:
    - Core switches at top
    - Aggregation switches in middle
    - Edge switches at bottom
    - Servers connected to edge switches
    
    Args:
        k: Fat-tree parameter (k-ary fat tree)
           k=4 → 16 servers, k=8 → 128 servers
        seed: Random seed
    
    Returns:
        NetworkX graph with data center structure
    """
    random.seed(seed)
    G = nx.Graph()
    
    node_id = 0
    core_nodes = []
    agg_nodes = []
    edge_nodes = []
    server_nodes = []
    
    # Number of pods
    num_pods = k
    num_core = (k//2) ** 2
    switches_per_pod = k
    servers_per_edge = k // 2
    
    print(f"Creating Fat-Tree (k={k}) data center topology...")
    
    # Create core switches
    for i in range(num_core):
        G.add_node(node_id)
        G.nodes[node_id]['role'] = 'core'
        G.nodes[node_id]['tier'] = 'core'
        G.nodes[node_id]['residual_energy_j'] = DEFAULT_CORE_ENERGY_J * 1.5
        core_nodes.append(node_id)
        node_id += 1
    
    # Create pods (aggregation + edge switches)
    for pod in range(num_pods):
        pod_agg = []
        pod_edge = []
        
        # Aggregation switches in this pod
        for i in range(k//2):
            G.add_node(node_id)
            G.nodes[node_id]['role'] = 'aggregation'
            G.nodes[node_id]['tier'] = 'aggregation'
            G.nodes[node_id]['pod'] = pod
            G.nodes[node_id]['residual_energy_j'] = DEFAULT_CORE_ENERGY_J
            pod_agg.append(node_id)
            agg_nodes.append(node_id)
            node_id += 1
        
        # Edge switches in this pod
        for i in range(k//2):
            G.add_node(node_id)
            G.nodes[node_id]['role'] = 'edge'
            G.nodes[node_id]['tier'] = 'edge'
            G.nodes[node_id]['pod'] = pod
            G.nodes[node_id]['residual_energy_j'] = DEFAULT_EDGE_ENERGY_J * 2
            pod_edge.append(node_id)
            edge_nodes.append(node_id)
            node_id += 1
        
        # Connect aggregation to core (each agg connects to k/2 core switches)
        for agg in pod_agg:
            core_subset = random.sample(core_nodes, k//2)
            for core in core_subset:
                if not G.has_edge(agg, core):
                    G.add_edge(agg, core)
                    G[agg][core]['bandwidth_bps'] = 10e9  # 10 Gbps
                    G[agg][core]['prop_delay_ms'] = random.uniform(0.1, 1.0)
                    G[agg][core]['loss_prob'] = 0.0001
                    G[agg][core]['queue_size_bytes'] = 500_000
                    G[agg][core]['link_type'] = 'agg_to_core'
                    _init_dynamic_state(G[agg][core])
        
        # Connect edge to aggregation (full bipartite within pod)
        for edge in pod_edge:
            for agg in pod_agg:
                G.add_edge(edge, agg)
                G[edge][agg]['bandwidth_bps'] = 10e9  # 10 Gbps
                G[edge][agg]['prop_delay_ms'] = random.uniform(0.1, 0.5)
                G[edge][agg]['loss_prob'] = 0.0001
                G[edge][agg]['queue_size_bytes'] = 200_000
                G[edge][agg]['link_type'] = 'edge_to_agg'
                _init_dynamic_state(G[edge][agg])
        
        # Connect servers to edge switches
        for edge in pod_edge:
            for i in range(servers_per_edge):
                G.add_node(node_id)
                G.nodes[node_id]['role'] = 'server'
                G.nodes[node_id]['tier'] = 'server'
                G.nodes[node_id]['pod'] = pod
                G.nodes[node_id]['residual_energy_j'] = DEFAULT_EDGE_ENERGY_J
                server_nodes.append(node_id)
                
                G.add_edge(edge, node_id)
                G[edge][node_id]['bandwidth_bps'] = 1e9  # 1 Gbps
                G[edge][node_id]['prop_delay_ms'] = random.uniform(0.05, 0.2)
                G[edge][node_id]['loss_prob'] = 0.0001
                G[edge][node_id]['queue_size_bytes'] = 100_000
                G[edge][node_id]['link_type'] = 'server_to_edge'
                _init_dynamic_state(G[edge][node_id])
                
                node_id += 1
    
    G.graph['topology_type'] = 'fat_tree'
    G.graph['k'] = k
    G.graph['core_nodes'] = core_nodes
    G.graph['agg_nodes'] = agg_nodes
    G.graph['edge_nodes'] = edge_nodes
    G.graph['server_nodes'] = server_nodes
    
    print(f"✓ Created Fat-Tree (k={k}) topology:")
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  Core switches: {len(core_nodes)}")
    print(f"  Aggregation switches: {len(agg_nodes)}")
    print(f"  Edge switches: {len(edge_nodes)}")
    print(f"  Servers: {len(server_nodes)}")
    print(f"  Total edges: {G.number_of_edges()}")
    
    return G


def create_wireless_mesh_topology(
    num_nodes=30,
    transmission_range=150,
    area_size=500,
    seed=42
):
    """
    Create a wireless mesh network topology.
    
    Realistic for IoT, sensor networks, mobile ad-hoc networks.
    - Random geometric graph (nodes within range connect)
    - Variable link quality based on distance
    - Limited energy (battery-powered)
    
    Args:
        num_nodes: Number of wireless nodes
        transmission_range: Radio transmission range (meters)
        area_size: Size of deployment area (meters)
        seed: Random seed
    
    Returns:
        NetworkX graph with wireless characteristics
    """
    random.seed(seed)
    
    print(f"Creating wireless mesh network: {num_nodes} nodes...")
    
    # Generate random node positions
    positions = {}
    for i in range(num_nodes):
        x = random.uniform(0, area_size)
        y = random.uniform(0, area_size)
        positions[i] = (x, y)
    
    # Create graph based on transmission range
    G = nx.Graph()
    
    for i in range(num_nodes):
        G.add_node(i)
        # Designate 10% as "gateway" nodes (higher energy)
        if i < num_nodes * 0.1:
            G.nodes[i]['role'] = 'gateway'
            G.nodes[i]['residual_energy_j'] = DEFAULT_CORE_ENERGY_J * 0.5
        else:
            G.nodes[i]['role'] = 'sensor'
            G.nodes[i]['residual_energy_j'] = DEFAULT_EDGE_ENERGY_J * 0.5  # Limited battery
        G.nodes[i]['position'] = positions[i]
    
    # Add edges based on distance
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if distance <= transmission_range:
                G.add_edge(i, j)
                
                # Link quality degrades with distance
                signal_quality = 1.0 - (distance / transmission_range)
                
                # Bandwidth varies with signal quality
                if signal_quality > 0.8:
                    bw = random.choice([54e6, 150e6, 300e6])  # High: 54-300 Mbps
                elif signal_quality > 0.5:
                    bw = random.choice([11e6, 54e6, 150e6])   # Medium
                else:
                    bw = random.choice([1e6, 11e6, 54e6])      # Low
                
                G[i][j]['bandwidth_bps'] = bw
                
                # Propagation delay based on distance (speed of light)
                G[i][j]['prop_delay_ms'] = distance / 300.0  # ~1ms per 300m
                
                # Loss increases with distance and interference
                base_loss = 0.001 + (1 - signal_quality) * 0.05
                G[i][j]['loss_prob'] = min(0.1, base_loss + random.uniform(0, 0.01))
                
                G[i][j]['queue_size_bytes'] = 20_000
                G[i][j]['distance_m'] = distance
                G[i][j]['signal_quality'] = signal_quality
                G[i][j]['link_type'] = 'wireless'
                _init_dynamic_state(G[i][j])
    
    # Ensure connectivity (add long-range links if needed)
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        print(f"  Network fragmented into {len(components)} components, adding bridges...")
        
        for i in range(len(components) - 1):
            # Connect largest node from each component
            comp1 = list(components[i])
            comp2 = list(components[i+1])
            
            # Find closest pair
            min_dist = float('inf')
            bridge = None
            for u in comp1:
                for v in comp2:
                    x1, y1 = positions[u]
                    x2, y2 = positions[v]
                    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if dist < min_dist:
                        min_dist = dist
                        bridge = (u, v)
            
            if bridge:
                u, v = bridge
                G.add_edge(u, v)
                G[u][v]['bandwidth_bps'] = 11e6  # Lower capacity for long link
                G[u][v]['prop_delay_ms'] = min_dist / 300.0
                G[u][v]['loss_prob'] = 0.05
                G[u][v]['queue_size_bytes'] = 20_000
                G[u][v]['distance_m'] = min_dist
                G[u][v]['signal_quality'] = 0.5
                G[u][v]['link_type'] = 'wireless_bridge'
                _init_dynamic_state(G[u][v])
    
    G.graph['topology_type'] = 'wireless_mesh'
    G.graph['transmission_range'] = transmission_range
    G.graph['area_size'] = area_size
    G.graph['positions'] = positions
    
    print(f"✓ Created wireless mesh topology:")
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  Total edges: {G.number_of_edges()}")
    print(f"  Avg degree: {2*G.number_of_edges()/G.number_of_nodes():.2f}")
    print(f"  Connected: {nx.is_connected(G)}")
    
    return G


def _init_dynamic_state(edge_data):
    """Initialize dynamic state for an edge."""
    edge_data['_dynamic_state'] = {
        'bw_mbps_base': edge_data['bandwidth_bps'] / 1e6,
        'prop_delay_ms_base': edge_data['prop_delay_ms'],
        'loss_prob_base': edge_data['loss_prob'],
        'last_update_t': 0.0
    }


def save_realistic_topology(G, path="config/realistic_topology.yaml"):
    """Save topology with all metadata."""
    out = {
        'metadata': {
            'topology_type': G.graph.get('topology_type', 'unknown'),
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
        },
        'nodes': {},
        'edges': []
    }
    
    # Save additional graph metadata
    for key, value in G.graph.items():
        if key not in ['topology_type'] and isinstance(value, (int, float, str, bool)):
            out['metadata'][key] = value
    
    for n, d in G.nodes(data=True):
        out['nodes'][int(n)] = {
            'role': d.get('role'),
            'tier': d.get('tier'),
            'residual_energy_j': float(d.get('residual_energy_j', 0.0))
        }
        if 'position' in d:
            out['nodes'][int(n)]['position'] = [float(d['position'][0]), float(d['position'][1])]
    
    for u, v, d in G.edges(data=True):
        edge_info = {
            'u': int(u),
            'v': int(v),
            'bandwidth_bps': float(d.get('bandwidth_bps', 0.0)),
            'prop_delay_ms': float(d.get('prop_delay_ms', 0.0)),
            'loss_prob': float(d.get('loss_prob', 0.0)),
            'queue_size_bytes': int(d.get('queue_size_bytes', 0))
        }
        if 'link_type' in d:
            edge_info['link_type'] = d['link_type']
        if 'distance_m' in d:
            edge_info['distance_m'] = float(d['distance_m'])
        
        out['edges'].append(edge_info)
    
    with open(path, 'w') as f:
        yaml.safe_dump(out, f)
    
    print(f"✓ Saved topology to: {path}")
    return path


if __name__ == "__main__":
    print("\n" + "="*60)
    print("REALISTIC TOPOLOGY GENERATOR")
    print("="*60 + "\n")
    
    # 1. Hierarchical ISP (most realistic for internet routing)
    G_isp = create_hierarchical_isp_topology(
        num_tier1=3,
        num_tier2_per_tier1=3,
        num_tier3_per_tier2=4,
        seed=42
    )
    save_realistic_topology(G_isp, "config/topology_isp.yaml")
    
    print("\n" + "-"*60 + "\n")
    
    # 2. Data center (Fat-Tree)
    G_dc = create_datacenter_fattree_topology(k=4, seed=42)
    save_realistic_topology(G_dc, "config/topology_datacenter.yaml")
    
    print("\n" + "-"*60 + "\n")
    
    # 3. Wireless mesh
    G_mesh = create_wireless_mesh_topology(
        num_nodes=30,
        transmission_range=150,
        area_size=500,
        seed=42
    )
    save_realistic_topology(G_mesh, "config/topology_wireless.yaml")
    
    print("\n" + "="*60)
    print("All realistic topologies generated!")
    print("="*60)