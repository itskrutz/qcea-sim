"""
Complete demonstration of P-QCEA (Predictive QCEA) routing protocol.

This script:
1. Creates a network topology
2. Runs comparative simulations (Dijkstra, QCEA, P-QCEA)
3. Analyzes and visualizes results
4. Demonstrates the benefit of prediction

Usage:
    python demo_pqcea.py
"""

import yaml
import os
import json
import matplotlib.pyplot as plt
import numpy as np

from src.topology import create_scale_free_topology, save_graph_yaml
from src.simulation_pqcea import  ComparativeSimulation
from src.visualize import plot_metrics


def setup_environment():
    """Create necessary directories."""
    os.makedirs('config', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    print("✓ Environment setup complete")


def create_test_topology(topology_type='isp'):
    """Create and save a test topology."""
    print("\n" + "="*60)
    print("STEP 1: Creating Network Topology")
    print("="*60)
    
    if topology_type == 'isp':
        # Hierarchical ISP-like network (most realistic)
        from src.realistic_topology import create_hierarchical_isp_topology, save_realistic_topology
        G = create_hierarchical_isp_topology(
            num_tier1=3,
            num_tier2_per_tier1=3,
            num_tier3_per_tier2=4,
            seed=42
        )
        topo_path = save_realistic_topology(G, "config/topology_isp.yaml")
    
    elif topology_type == 'datacenter':
        # Fat-Tree data center
        from src.realistic_topology import create_datacenter_fattree_topology, save_realistic_topology
        G = create_datacenter_fattree_topology(k=4, seed=42)
        topo_path = save_realistic_topology(G, "config/topology_datacenter.yaml")
    
    elif topology_type == 'wireless':
        # Wireless mesh network
        from src.realistic_topology import create_wireless_mesh_topology, save_realistic_topology
        G = create_wireless_mesh_topology(
            num_nodes=30,
            transmission_range=150,
            area_size=500,
            seed=42
        )
        topo_path = save_realistic_topology(G, "config/topology_wireless.yaml")
    
    else:
        # Default: scale-free network
        G = create_scale_free_topology(
            n=25, m=2, seed=42,
            core_fraction=0.2,
            bw_choices_mbps=[10, 50, 100],
            prop_delay_ms_range=(1.0, 30.0),
            loss_choices=[0.0, 0.001, 0.01]
        )
        topo_path = save_graph_yaml(G, "config/topology_sample.yaml")
    
    print(f"✓ Created {G.number_of_nodes()}-node network")
    print(f"✓ Total links: {G.number_of_edges()}")
    print(f"✓ Avg degree: {2*G.number_of_edges()/G.number_of_nodes():.2f}")
    print(f"✓ Saved to: {topo_path}")
    
    return G


def run_comparative_study(config_path="config/pqcea_config.yaml"):
    """Run comparative simulation study."""
    print("\n" + "="*60)
    print("STEP 2: Running Comparative Simulation")
    print("="*60)
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Run comparative simulation
    sim = ComparativeSimulation(config)
    results = sim.run_comparison()
    
    # Print comparison table
    sim.print_comparison_table()
    
    # Save results
    results_file = 'results/comparison_results.json'
    
    # Convert results to JSON-serializable format
    json_results = {}
    for algo_name, result in results.items():
        json_results[algo_name] = {
            'summary': result['summary'],
            'algorithm': result['algorithm']
        }
        if 'algo_stats' in result:
            json_results[algo_name]['stats'] = result['algo_stats']
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    return results


def visualize_comparison(results):
    """Create visualization comparing all algorithms."""
    print("\n" + "="*60)
    print("STEP 3: Creating Visualizations")
    print("="*60)
    
    # Extract data for plotting
    algorithms = list(results.keys())
    
    # Metric names
    metrics = ['delay', 'throughput', 'energy', 'packet_delivery']
    metric_labels = ['Avg Delay (ms)', 'Avg Throughput', 'Total Energy (J)', 'Delivery Rate']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        # Get values for each algorithm
        values = [results[algo]['summary'][metric] for algo in algorithms]
        
        # Create bar chart
        bars = ax.bar(algorithms, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} Comparison', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x-axis labels
        ax.set_xticklabels(algorithms, rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig('results/plots/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/plots/algorithm_comparison.png")
    
    # Create time-series plots for each algorithm
    for algo_name, result in results.items():
        if 'metrics' in result:
            plot_metrics(result['metrics'])
            plt.savefig(f'results/plots/{algo_name}_timeseries.png', dpi=300)
            print(f"✓ Saved: results/plots/{algo_name}_timeseries.png")
            plt.close('all')


def demonstrate_prediction_benefit():
    """
    Demonstrate specific benefit of prediction with a focused example.
    """
    print("\n" + "="*60)
    print("STEP 4: Demonstrating Prediction Benefit")
    print("="*60)
    
    print("""
    Key Insight: Why Prediction Helps
    
    1. REACTIVE (Traditional QCEA):
       - Measures link quality NOW
       - Routes packet
       - By arrival time, link may be congested!
       - Result: Packet experiences HIGH delay
    
    2. PROACTIVE (P-QCEA with Prediction):
       - Measures link quality NOW
       - PREDICTS link quality in 3 time steps
       - Routes packet to avoid predicted congestion
       - Result: Packet experiences LOWER delay
    
    Example Scenario:
    - Link A: Currently fast (10ms) but degrading (trend: +5ms/step)
    - Link B: Currently slow (25ms) but stable
    
    Traditional QCEA: Chooses Link A (10ms < 25ms)
    P-QCEA Prediction: Link A in 3 steps = 10 + 5*3 = 25ms
                       Chooses Link B instead (stable 25ms)
    
    Result: P-QCEA avoids congestion proactively!
    """)


def generate_research_summary():
    """Generate a summary suitable for research paper."""
    print("\n" + "="*60)
    print("RESEARCH CONTRIBUTION SUMMARY")
    print("="*60)
    
    summary = """
    NOVEL CONTRIBUTION: Predictive Quality-Conscious Energy-Aware Routing
    
    1. PROBLEM ADDRESSED:
       Traditional QoS-aware routing is reactive - it responds to network
       conditions AFTER they occur, leading to suboptimal decisions.
    
    2. PROPOSED SOLUTION:
       P-QCEA integrates lightweight time-series prediction directly into
       the routing cost function to make PROACTIVE decisions.
    
    3. KEY TECHNIQUES:
       a) Exponential smoothing for noise reduction
       b) Linear trend detection via regression
       c) Confidence-based prediction gating
       d) Multi-metric prediction (delay + bandwidth + loss)
    
    4. NOVELTY ASPECTS:
       - Training-free: No dataset or ML model training required
       - Lightweight: Only stores 15 historical samples per link
       - Real-time: Online learning as network operates
       - Confidence-aware: Falls back to current metrics when uncertain
       - Integrated: Prediction seamlessly embedded in cost function
    
    5. EXPECTED BENEFITS:
       - Reduced end-to-end delay (proactive congestion avoidance)
       - Lower energy consumption (better path selection)
       - Improved packet delivery rate (avoiding lossy links)
       - Higher network throughput (reduced retransmissions)
    
    6. EVALUATION METHODOLOGY:
       Comparative study against:
       - Baseline: Dijkstra (shortest path)
       - State-of-art: Traditional QCEA (reactive QoS-aware)
       - Proposed: P-QCEA (proactive with prediction)
    
    7. IMPLEMENTATION:
       - Python-based discrete event simulation
       - Scale-free topology (Barabási-Albert model)
       - Dynamic link quality variations
       - Multiple traffic classes (VoIP, Video, Best-effort)
    """
    
    print(summary)
    
    with open('results/research_summary.txt', 'w') as f:
        f.write(summary)
    
    print("\n✓ Summary saved to: results/research_summary.txt")


def main():
    """Main demo execution."""
    print("\n" + "="*70)
    print(" "*15 + "P-QCEA ROUTING PROTOCOL DEMONSTRATION")
    print("="*70)
    print("\nPredictive Quality-Conscious Energy-Aware Routing")
    print("A Novel Approach to Proactive QoS-Aware Routing\n")
    
    # Setup
    setup_environment()
    
    # Create topology
    create_test_topology()
    
    # Run comparative study
    results = run_comparative_study()
    
    # Visualize results
    visualize_comparison(results)
    
    # Demonstrate prediction benefit
    demonstrate_prediction_benefit()
    
    # Generate research summary
    generate_research_summary()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - config/topology_sample.yaml        (Network topology)")
    print("  - results/comparison_results.json    (Numerical results)")
    print("  - results/plots/                     (Visualizations)")
    print("  - results/research_summary.txt       (Contribution summary)")
    print("\nNext steps:")
    print("  1. Review comparison results")
    print("  2. Analyze performance gains")
    print("  3. Tune prediction parameters")
    print("  4. Extend for your specific research needs")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()