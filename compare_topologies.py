"""
Comprehensive evaluation of P-QCEA across multiple realistic topologies.
Generates publication-ready comparison data.
"""

import yaml
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from src.realistic_topology import (
    create_hierarchical_isp_topology,
    create_datacenter_fattree_topology,
    create_wireless_mesh_topology,
    save_realistic_topology
)
from src.topology import create_scale_free_topology, save_graph_yaml
from src.simulation_pqcea import ComparativeSimulation


def generate_all_topologies():
    """Generate all topology types for comparison."""
    print("\n" + "="*70)
    print("GENERATING REALISTIC TOPOLOGIES")
    print("="*70 + "\n")
    
    topologies = {}
    
    # 1. Hierarchical ISP (Internet-like)
    print("1. Hierarchical ISP Topology...")
    G_isp = create_hierarchical_isp_topology(
        num_tier1=3,
        num_tier2_per_tier1=3,
        num_tier3_per_tier2=4,
        seed=42
    )
    path_isp = save_realistic_topology(G_isp, "config/topology_isp.yaml")
    topologies['isp'] = {
        'graph': G_isp,
        'path': path_isp,
        'name': 'ISP Network',
        'description': 'Hierarchical 3-tier ISP'
    }
    
    print("\n" + "-"*70 + "\n")
    
    # 2. Data Center (Fat-Tree)
    print("2. Data Center Fat-Tree Topology...")
    G_dc = create_datacenter_fattree_topology(k=4, seed=42)
    path_dc = save_realistic_topology(G_dc, "config/topology_datacenter.yaml")
    topologies['datacenter'] = {
        'graph': G_dc,
        'path': path_dc,
        'name': 'Data Center',
        'description': 'Fat-Tree (k=4)'
    }
    
    print("\n" + "-"*70 + "\n")
    
    # 3. Wireless Mesh
    print("3. Wireless Mesh Topology...")
    G_mesh = create_wireless_mesh_topology(
        num_nodes=30,
        transmission_range=150,
        area_size=500,
        seed=42
    )
    path_mesh = save_realistic_topology(G_mesh, "config/topology_wireless.yaml")
    topologies['wireless'] = {
        'graph': G_mesh,
        'path': path_mesh,
        'name': 'Wireless Mesh',
        'description': '30-node mesh network'
    }
    
    print("\n" + "-"*70 + "\n")
    
    # 4. Scale-Free (baseline comparison)
    print("4. Scale-Free Topology...")
    G_sf = create_scale_free_topology(n=35, m=2, seed=42)
    path_sf = save_graph_yaml(G_sf, "config/topology_scalefree.yaml")
    topologies['scalefree'] = {
        'graph': G_sf,
        'path': path_sf,
        'name': 'Scale-Free',
        'description': 'Barabási-Albert (n=35)'
    }
    
    print("\n" + "="*70)
    print(f"✓ Generated {len(topologies)} topologies")
    print("="*70 + "\n")
    
    return topologies


def run_topology_comparison(topologies, num_trials=5):
    """
    Run P-QCEA evaluation on all topologies.
    
    Args:
        topologies: Dict of topology info
        num_trials: Number of trials per topology
    
    Returns:
        Comprehensive results dict
    """
    print("\n" + "="*70)
    print("RUNNING MULTI-TOPOLOGY COMPARISON")
    print("="*70 + "\n")
    
    all_results = {}
    
    for topo_name, topo_info in topologies.items():
        print(f"\n{'='*70}")
        print(f"EVALUATING: {topo_info['name']} ({topo_info['description']})")
        print(f"{'='*70}")
        
        topo_results = {
            'name': topo_info['name'],
            'description': topo_info['description'],
            'graph_stats': {
                'nodes': topo_info['graph'].number_of_nodes(),
                'edges': topo_info['graph'].number_of_edges(),
                'avg_degree': 2 * topo_info['graph'].number_of_edges() / topo_info['graph'].number_of_nodes()
            },
            'trials': []
        }
        
        # Run multiple trials
        for trial in range(num_trials):
            print(f"\n  Trial {trial+1}/{num_trials}...")
            
            # Configuration for this topology
            config = {
                'topology_path': topo_info['path'],
                'simulation_mode': 'comparative',
                'time_steps': 50,  # Shorter for faster comparison
                'num_flows': 5,
                'seed': 42 + trial,
                'run_dijkstra': True,
                'run_qcea': True,
                'run_pqcea': True,
                'prediction_horizon': 3,
                'weights': {
                    'wl': 0.3, 'wb': 0.2, 'wp': 0.2, 'we': 0.2, 'wc': 0.1
                }
            }
            
            # Run simulation
            sim = ComparativeSimulation(config)
            results = sim.run_comparison()
            
            # Extract key metrics
            trial_data = {}
            for algo_name, algo_result in results.items():
                trial_data[algo_name] = algo_result['summary']
            
            topo_results['trials'].append(trial_data)
        
        # Compute statistics across trials
        topo_results['statistics'] = compute_trial_statistics(topo_results['trials'])
        
        all_results[topo_name] = topo_results
        
        print(f"\n✓ Completed {topo_info['name']}")
    
    return all_results


def compute_trial_statistics(trials):
    """Compute mean and std dev across trials."""
    if not trials:
        return {}
    
    algorithms = trials[0].keys()
    stats = {}
    
    for algo in algorithms:
        algo_stats = {}
        metrics = trials[0][algo].keys()
        
        for metric in metrics:
            values = [trial[algo][metric] for trial in trials]
            algo_stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        stats[algo] = algo_stats
    
    return stats


def create_comparison_plots(all_results, save_dir='results/topology_comparison'):
    """Create comprehensive comparison plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70 + "\n")
    
    topologies = list(all_results.keys())
    topo_names = [all_results[t]['name'] for t in topologies]
    
    metrics = ['delay', 'throughput', 'energy', 'packet_delivery']
    metric_labels = {
        'delay': 'Avg Delay (ms)',
        'throughput': 'Avg Throughput',
        'energy': 'Total Energy (J)',
        'packet_delivery': 'Delivery Rate'
    }
    
    # Plot 1: Algorithm comparison per topology (4x3 grid)
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    
    for row, metric in enumerate(metrics):
        for col, algo in enumerate(['dijkstra', 'qcea', 'pqcea']):
            ax = axes[row, col]
            
            means = []
            stds = []
            
            for topo in topologies:
                stats = all_results[topo]['statistics']
                if algo in stats:
                    means.append(stats[algo][metric]['mean'])
                    stds.append(stats[algo][metric]['std'])
                else:
                    means.append(0)
                    stds.append(0)
            
            x = np.arange(len(topo_names))
            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                         color=['#FF6B6B', '#4ECDC4', '#45B7D1'][col])
            
            ax.set_ylabel(metric_labels[metric], fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(topo_names, rotation=15, ha='right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            
            if row == 0:
                ax.set_title(algo.upper(), fontsize=12, fontweight='bold')
            
            # Add value labels
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.1f}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/algorithm_by_topology.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/algorithm_by_topology.png")
    plt.close()
    
    # Plot 2: P-QCEA gains per topology
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Calculate improvement of P-QCEA over Dijkstra
        improvements = []
        for topo in topologies:
            stats = all_results[topo]['statistics']
            if 'pqcea' in stats and 'dijkstra' in stats:
                baseline = stats['dijkstra'][metric]['mean']
                pqcea = stats['pqcea'][metric]['mean']
                
                # For delivery, higher is better; for others, lower is better
                if metric == 'packet_delivery' or metric == 'throughput':
                    improvement = ((pqcea - baseline) / baseline) * 100
                else:
                    improvement = ((baseline - pqcea) / baseline) * 100
                
                improvements.append(improvement)
            else:
                improvements.append(0)
        
        x = np.arange(len(topo_names))
        colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
        bars = ax.bar(x, improvements, color=colors, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel(f'{metric_labels[metric]} Improvement (%)', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(topo_names, rotation=15, ha='right')
        ax.set_title(f'P-QCEA vs Dijkstra: {metric_labels[metric]}', 
                    fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{imp:+.1f}%',
                   ha='center', va='bottom' if imp > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pqcea_improvements.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/pqcea_improvements.png")
    plt.close()
    
    # Plot 3: Topology characteristics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Nodes
    nodes = [all_results[t]['graph_stats']['nodes'] for t in topologies]
    axes[0].bar(topo_names, nodes, color='#3498db', alpha=0.8)
    axes[0].set_ylabel('Number of Nodes', fontsize=11)
    axes[0].set_title('Network Size', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)
    
    # Edges
    edges = [all_results[t]['graph_stats']['edges'] for t in topologies]
    axes[1].bar(topo_names, edges, color='#9b59b6', alpha=0.8)
    axes[1].set_ylabel('Number of Edges', fontsize=11)
    axes[1].set_title('Link Count', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    
    # Avg degree
    degrees = [all_results[t]['graph_stats']['avg_degree'] for t in topologies]
    axes[2].bar(topo_names, degrees, color='#e67e22', alpha=0.8)
    axes[2].set_ylabel('Average Degree', fontsize=11)
    axes[2].set_title('Connectivity', fontsize=12, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/topology_characteristics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/topology_characteristics.png")
    plt.close()


def save_comparison_results(all_results, path='results/topology_comparison_results.json'):
    """Save detailed comparison results."""
    # Convert to JSON-serializable format
    output = {}
    for topo_name, topo_data in all_results.items():
        output[topo_name] = {
            'name': topo_data['name'],
            'description': topo_data['description'],
            'graph_stats': topo_data['graph_stats'],
            'statistics': topo_data['statistics']
        }
    
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Saved results to: {path}")


def print_summary_table(all_results):
    """Print a publication-ready summary table."""
    print("\n" + "="*100)
    print("TOPOLOGY COMPARISON SUMMARY (Mean ± Std)")
    print("="*100)
    
    print(f"\n{'Topology':<20} {'Algorithm':<12} {'Delay (ms)':<15} {'Energy (J)':<15} {'Delivery':<12}")
    print("-"*100)
    
    for topo_name, topo_data in all_results.items():
        stats = topo_data['statistics']
        topo_display = topo_data['name']
        
        for algo in ['dijkstra', 'qcea', 'pqcea']:
            if algo in stats:
                delay = stats[algo]['delay']
                energy = stats[algo]['energy']
                delivery = stats[algo]['packet_delivery']
                
                print(f"{topo_display:<20} {algo.upper():<12} "
                      f"{delay['mean']:>7.2f}±{delay['std']:<5.2f} "
                      f"{energy['mean']:>7.1f}±{energy['std']:<5.1f} "
                      f"{delivery['mean']:>5.3f}±{delivery['std']:<5.3f}")
        
        print("-"*100)
    
    # Overall P-QCEA gains
    print("\nP-QCEA Overall Performance Gains vs Dijkstra:")
    print("-"*100)
    
    for topo_name, topo_data in all_results.items():
        stats = topo_data['statistics']
        if 'pqcea' in stats and 'dijkstra' in stats:
            delay_gain = ((stats['dijkstra']['delay']['mean'] - stats['pqcea']['delay']['mean']) / 
                         stats['dijkstra']['delay']['mean']) * 100
            energy_gain = ((stats['dijkstra']['energy']['mean'] - stats['pqcea']['energy']['mean']) / 
                          stats['dijkstra']['energy']['mean']) * 100
            
            print(f"{topo_data['name']:<20} Delay: {delay_gain:>6.1f}%  Energy: {energy_gain:>6.1f}%")


def main():
    """Main execution."""
    print("\n" + "="*70)
    print(" "*15 + "MULTI-TOPOLOGY P-QCEA EVALUATION")
    print("="*70)
    print("\nComprehensive evaluation across realistic network topologies")
    print("="*70 + "\n")
    
    # Generate topologies
    topologies = generate_all_topologies()
    
    # Run comparison (fewer trials for demo, increase to 30+ for publication)
    num_trials = 3  # Increase to 30+ for research
    print(f"\nRunning {num_trials} trials per topology...")
    all_results = run_topology_comparison(topologies, num_trials)
    
    # Save results
    save_comparison_results(all_results)
    
    # Create plots
    create_comparison_plots(all_results)
    
    # Print summary
    print_summary_table(all_results)
    
    print("\n" + "="*70)
    print("MULTI-TOPOLOGY EVALUATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - config/topology_*.yaml")
    print("  - results/topology_comparison_results.json")
    print("  - results/topology_comparison/*.png")
    print("\nFor publication-ready results, increase num_trials to 30+")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()