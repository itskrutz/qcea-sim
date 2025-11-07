"""
Performance Report Generator for P-QCEA Simulation
Generates comprehensive performance reports and testing statistics
"""

import json
import csv
import os
from datetime import datetime
import numpy as np

# Import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_palette("husl")
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        # Fallback if seaborn not available
        plt.style.use('default')
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def calculate_statistics(metrics_list):
    """Calculate comprehensive statistics from metrics list."""
    stats = {}
    
    if not metrics_list:
        return stats
    
    # Extract all metric values
    metrics_dict = {}
    # Get all available metric names from the first entry if available
    if metrics_list:
        metric_names = list(metrics_list[0].keys())
    else:
        metric_names = ['latency', 'throughput', 'pdr', 'energy_consumed', 
                       'residual_energy', 'jitter', 'prediction_usage_rate', 
                       'avg_confidence', 'avg_path_length', 'avg_path_cost',
                       'avg_link_utilization', 'max_link_utilization',
                       'avg_link_delay_ms', 'avg_link_loss_prob',
                       'avg_node_energy_j', 'min_node_energy_j',
                       'congested_links_count', 'congestion_ratio']
    
    for metric_name in metric_names:
        values = [m.get(metric_name, 0) for m in metrics_list if metric_name in m]
        if values and isinstance(values[0], (int, float)):
            metrics_dict[metric_name] = values
    
    # Calculate statistics for each metric
    for metric_name, values in metrics_dict.items():
        stats[metric_name] = {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'count': len(values)
        }
    
    return stats


def generate_performance_report(dijkstra_metrics, qcea_metrics, pqcea_metrics, 
                                config=None, output_dir='results'):
    """Generate comprehensive comparative performance report."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate statistics for each algorithm
    stats_dijkstra = calculate_statistics(dijkstra_metrics)
    stats_qcea = calculate_statistics(qcea_metrics)
    stats_pqcea = calculate_statistics(pqcea_metrics)
    
    # Calculate improvement percentages (P-QCEA vs others)
    improvements = {}
    
    # Helper function to calculate improvement
    def calc_improvement(better_val, baseline_val, lower_is_better=True):
        if baseline_val == 0:
            return 0.0
        if lower_is_better:
            return ((baseline_val - better_val) / baseline_val) * 100.0
        else:
            return ((better_val - baseline_val) / baseline_val) * 100.0
    
    # Compare metrics
    for metric in ['latency', 'pdr', 'throughput', 'residual_energy', 'jitter']:
        lower_is_better = metric in ['latency', 'jitter']  # Lower is better for these
        
        if metric in stats_pqcea and metric in stats_qcea:
            pqcea_mean = stats_pqcea[metric]['mean']
            qcea_mean = stats_qcea[metric]['mean']
            improvements[f'pqcea_vs_qcea_{metric}'] = calc_improvement(
                pqcea_mean, qcea_mean, lower_is_better
            )
        
        if metric in stats_pqcea and metric in stats_dijkstra:
            pqcea_mean = stats_pqcea[metric]['mean']
            dijkstra_mean = stats_dijkstra[metric]['mean']
            improvements[f'pqcea_vs_dijkstra_{metric}'] = calc_improvement(
                pqcea_mean, dijkstra_mean, lower_is_better
            )
    
    # Create comprehensive report dictionary
    report = {
        'metadata': {
            'timestamp': timestamp,
            'generation_time': datetime.now().isoformat(),
            'simulation_steps': len(pqcea_metrics),
            'config': config or {}
        },
        'statistics': {
            'dijkstra': stats_dijkstra,
            'qcea': stats_qcea,
            'pqcea': stats_pqcea
        },
        'improvements': improvements,
        'summary': {
            'dijkstra': {
                'avg_latency_ms': stats_dijkstra.get('latency', {}).get('mean', 0),
                'avg_pdr_percent': stats_dijkstra.get('pdr', {}).get('mean', 0),
                'final_residual_energy_j': dijkstra_metrics[-1].get('residual_energy', 0) if dijkstra_metrics else 0,
                'avg_jitter_ms': stats_dijkstra.get('jitter', {}).get('mean', 0),
                'avg_throughput_mbps': stats_dijkstra.get('throughput', {}).get('mean', 0)
            },
            'qcea': {
                'avg_latency_ms': stats_qcea.get('latency', {}).get('mean', 0),
                'avg_pdr_percent': stats_qcea.get('pdr', {}).get('mean', 0),
                'final_residual_energy_j': qcea_metrics[-1].get('residual_energy', 0) if qcea_metrics else 0,
                'avg_jitter_ms': stats_qcea.get('jitter', {}).get('mean', 0),
                'avg_throughput_mbps': stats_qcea.get('throughput', {}).get('mean', 0)
            },
            'pqcea': {
                'avg_latency_ms': stats_pqcea.get('latency', {}).get('mean', 0),
                'avg_pdr_percent': stats_pqcea.get('pdr', {}).get('mean', 0),
                'final_residual_energy_j': pqcea_metrics[-1].get('residual_energy', 0) if pqcea_metrics else 0,
                'avg_jitter_ms': stats_pqcea.get('jitter', {}).get('mean', 0),
                'avg_throughput_mbps': stats_pqcea.get('throughput', {}).get('mean', 0),
                'avg_prediction_usage_rate': stats_pqcea.get('prediction_usage_rate', {}).get('mean', 0),
                'avg_prediction_confidence': stats_pqcea.get('avg_confidence', {}).get('mean', 0)
            }
        }
    }
    
    # Save JSON report
    json_path = os.path.join(output_dir, f'performance_report_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save CSV files for each algorithm
    csv_files = []
    for algo_name, metrics in [('dijkstra', dijkstra_metrics), 
                                ('qcea', qcea_metrics), 
                                ('pqcea', pqcea_metrics)]:
        if metrics:
            csv_path = os.path.join(output_dir, f'{algo_name}_metrics_{timestamp}.csv')
            fieldnames = list(metrics[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metrics)
            csv_files.append(csv_path)
    
    # Generate human-readable text report
    txt_path = os.path.join(output_dir, f'performance_report_{timestamp}.txt')
    generate_text_report(report, txt_path)
    
    # Export graphs
    graphs_exported = []
    plots_dir = None
    comprehensive_comparison_png = None
    if PLOTTING_AVAILABLE:
        try:
            graphs_exported, plots_dir = export_graphs(
                dijkstra_metrics, qcea_metrics, pqcea_metrics, 
                output_dir=output_dir, timestamp=timestamp
            )
            # Find the comprehensive_comparison.png path
            for graph_file in graphs_exported:
                if 'comprehensive_comparison.png' in graph_file:
                    comprehensive_comparison_png = graph_file
                    break
            
            report['graphs'] = {
                'directory': plots_dir,
                'files': graphs_exported,
                'comprehensive_comparison': comprehensive_comparison_png
            }
        except Exception as e:
            print(f"Warning: Could not export graphs: {e}")
            import traceback
            traceback.print_exc()
            report['graphs'] = {'error': str(e)}
    else:
        report['graphs'] = {'error': 'Matplotlib not available'}
    
    return {
        'report': report,
        'json_path': json_path,
        'txt_path': txt_path,
        'csv_files': csv_files,
        'graphs': graphs_exported,
        'plots_dir': plots_dir,
        'comprehensive_comparison_png': comprehensive_comparison_png
    }


def export_graphs(dijkstra_metrics, qcea_metrics, pqcea_metrics, output_dir='results', timestamp=None):
    """Export all graphs as PNG and PDF files."""
    if not PLOTTING_AVAILABLE:
        raise ImportError("Matplotlib is not available for graph export")
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plots_dir = os.path.join(output_dir, 'plots', timestamp)
    os.makedirs(plots_dir, exist_ok=True)
    
    time_steps = list(range(1, len(pqcea_metrics) + 1))
    
    # Extract data
    d_latency = [m['latency'] for m in dijkstra_metrics]
    q_latency = [m['latency'] for m in qcea_metrics]
    p_latency = [m['latency'] for m in pqcea_metrics]
    
    d_pdr = [m['pdr'] for m in dijkstra_metrics]
    q_pdr = [m['pdr'] for m in qcea_metrics]
    p_pdr = [m['pdr'] for m in pqcea_metrics]
    
    d_energy = [m['residual_energy'] for m in dijkstra_metrics]
    q_energy = [m['residual_energy'] for m in qcea_metrics]
    p_energy = [m['residual_energy'] for m in pqcea_metrics]
    
    d_jitter = [m.get('jitter', 0.0) for m in dijkstra_metrics]
    q_jitter = [m.get('jitter', 0.0) for m in qcea_metrics]
    p_jitter = [m.get('jitter', 0.0) for m in pqcea_metrics]
    
    p_usage = [m.get('prediction_usage_rate', 0.0) for m in pqcea_metrics]
    p_confidence = [m.get('avg_confidence', 0.0) for m in pqcea_metrics]
    
    # Additional metrics if available
    d_path_len = [m.get('avg_path_length', 0) for m in dijkstra_metrics]
    q_path_len = [m.get('avg_path_length', 0) for m in qcea_metrics]
    p_path_len = [m.get('avg_path_length', 0) for m in pqcea_metrics]
    
    d_cost = [m.get('avg_path_cost', 0) for m in dijkstra_metrics]
    q_cost = [m.get('avg_path_cost', 0) for m in qcea_metrics]
    p_cost = [m.get('avg_path_cost', 0) for m in pqcea_metrics]
    
    # Network state (same for all algorithms)
    avg_util = [m.get('avg_link_utilization', 0.0) for m in pqcea_metrics]
    congestion = [m.get('congestion_ratio', 0.0) * 100 for m in pqcea_metrics]
    
    graphs_exported = []
    
    # 1. Latency Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_steps, d_latency, 'b-', label='Dijkstra', linewidth=2)
    ax.plot(time_steps, q_latency, 'g-', label='QCEA', linewidth=2)
    ax.plot(time_steps, p_latency, 'm-', label='P-QCEA', linewidth=2)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Average Latency (ms)', fontsize=12)
    ax.set_title('Latency Comparison Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(plots_dir, 'latency_comparison.png')
    pdf_path = os.path.join(plots_dir, 'latency_comparison.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    graphs_exported.extend([png_path, pdf_path])
    
    # 2. PDR Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_steps, d_pdr, 'b-', label='Dijkstra', linewidth=2)
    ax.plot(time_steps, q_pdr, 'g-', label='QCEA', linewidth=2)
    ax.plot(time_steps, p_pdr, 'm-', label='P-QCEA', linewidth=2)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Packet Delivery Rate (%)', fontsize=12)
    ax.set_title('Packet Delivery Rate Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(plots_dir, 'pdr_comparison.png')
    pdf_path = os.path.join(plots_dir, 'pdr_comparison.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    graphs_exported.extend([png_path, pdf_path])
    
    # 3. Energy Consumption
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_steps, d_energy, 'b-', label='Dijkstra', linewidth=2)
    ax.plot(time_steps, q_energy, 'g-', label='QCEA', linewidth=2)
    ax.plot(time_steps, p_energy, 'm-', label='P-QCEA', linewidth=2)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Residual Network Energy (J)', fontsize=12)
    ax.set_title('Network Energy Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(plots_dir, 'energy_comparison.png')
    pdf_path = os.path.join(plots_dir, 'energy_comparison.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    graphs_exported.extend([png_path, pdf_path])
    
    # 4. Jitter Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_steps, d_jitter, 'b-', label='Dijkstra', linewidth=2)
    ax.plot(time_steps, q_jitter, 'g-', label='QCEA', linewidth=2)
    ax.plot(time_steps, p_jitter, 'm-', label='P-QCEA', linewidth=2)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Jitter (ms)', fontsize=12)
    ax.set_title('Jitter Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(plots_dir, 'jitter_comparison.png')
    pdf_path = os.path.join(plots_dir, 'jitter_comparison.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    graphs_exported.extend([png_path, pdf_path])
    
    # 5. Path Length Comparison
    if any(d_path_len) or any(q_path_len) or any(p_path_len):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_steps, d_path_len, 'b-', label='Dijkstra', linewidth=2)
        ax.plot(time_steps, q_path_len, 'g-', label='QCEA', linewidth=2)
        ax.plot(time_steps, p_path_len, 'm-', label='P-QCEA', linewidth=2)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Average Path Length (hops)', fontsize=12)
        ax.set_title('Average Path Length Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        png_path = os.path.join(plots_dir, 'path_length_comparison.png')
        pdf_path = os.path.join(plots_dir, 'path_length_comparison.pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close()
        graphs_exported.extend([png_path, pdf_path])
    
    # 6. Prediction Metrics (P-QCEA only)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(time_steps, p_usage, 'm-', linewidth=2)
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Prediction Usage Rate (%)', fontsize=12)
    ax1.set_title('P-QCEA Prediction Usage Rate', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax2.plot(time_steps, p_confidence, 'c-', linewidth=2)
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Average Confidence (%)', fontsize=12)
    ax2.set_title('P-QCEA Prediction Confidence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(plots_dir, 'prediction_metrics.png')
    pdf_path = os.path.join(plots_dir, 'prediction_metrics.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    graphs_exported.extend([png_path, pdf_path])
    
    # 7. Network State Metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1.plot(time_steps, avg_util, 'r-', linewidth=2)
    ax1.set_xlabel('Time Step', fontsize=10)
    ax1.set_ylabel('Avg Link Utilization', fontsize=10)
    ax1.set_title('Average Link Utilization', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time_steps, congestion, 'orange', linewidth=2)
    ax2.set_xlabel('Time Step', fontsize=10)
    ax2.set_ylabel('Congestion Ratio (%)', fontsize=10)
    ax2.set_title('Network Congestion Level', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    avg_delays = [m.get('avg_link_delay_ms', 0.0) for m in pqcea_metrics]
    ax3.plot(time_steps, avg_delays, 'b-', linewidth=2)
    ax3.set_xlabel('Time Step', fontsize=10)
    ax3.set_ylabel('Avg Link Delay (ms)', fontsize=10)
    ax3.set_title('Average Link Delay', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    avg_losses = [m.get('avg_link_loss_prob', 0.0) * 100 for m in pqcea_metrics]
    ax4.plot(time_steps, avg_losses, 'purple', linewidth=2)
    ax4.set_xlabel('Time Step', fontsize=10)
    ax4.set_ylabel('Avg Loss Probability (%)', fontsize=10)
    ax4.set_title('Average Link Loss', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    png_path = os.path.join(plots_dir, 'network_state_metrics.png')
    pdf_path = os.path.join(plots_dir, 'network_state_metrics.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    graphs_exported.extend([png_path, pdf_path])
    
    # 8. Comprehensive Comparison (All metrics in one figure)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    metrics_to_plot = [
        ('Latency (ms)', [d_latency, q_latency, p_latency], ['Dijkstra', 'QCEA', 'P-QCEA'], False),
        ('PDR (%)', [d_pdr, q_pdr, p_pdr], ['Dijkstra', 'QCEA', 'P-QCEA'], False),
        ('Energy (J)', [d_energy, q_energy, p_energy], ['Dijkstra', 'QCEA', 'P-QCEA'], False),
        ('Jitter (ms)', [d_jitter, q_jitter, p_jitter], ['Dijkstra', 'QCEA', 'P-QCEA'], False),
        ('Path Length', [d_path_len, q_path_len, p_path_len], ['Dijkstra', 'QCEA', 'P-QCEA'], False),
        ('Network Utilization', [avg_util], ['All Algorithms'], True)
    ]
    
    colors = ['b', 'g', 'm']
    for idx, (title, data_lists, labels, single_line) in enumerate(metrics_to_plot[:6]):
        ax = axes[idx]
        if single_line:
            ax.plot(time_steps, data_lists[0], 'k-', linewidth=2, label=labels[0])
        else:
            for data, label, color in zip(data_lists, labels, colors):
                ax.plot(time_steps, data, color + '-', linewidth=2, label=label)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=9)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    png_path = os.path.join(plots_dir, 'comprehensive_comparison.png')
    pdf_path = os.path.join(plots_dir, 'comprehensive_comparison.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    graphs_exported.extend([png_path, pdf_path])
    
    print(f"✓ Exported {len(graphs_exported)} graph files to {plots_dir}")
    return graphs_exported, plots_dir


def generate_text_report(report, output_path):
    """Generate human-readable text report."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("P-QCEA SIMULATION PERFORMANCE REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Metadata
        meta = report['metadata']
        f.write(f"Generated: {meta['generation_time']}\n")
        f.write(f"Simulation Steps: {meta['simulation_steps']}\n\n")
        
        # Summary
        f.write("=" * 80 + "\n")
        f.write("EXECUTIVE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        summary = report['summary']
        
        f.write("Algorithm Performance Summary:\n")
        f.write("-" * 80 + "\n")
        
        algorithms = [
            ('dijkstra', 'Dijkstra (Baseline)'),
            ('qcea', 'QCEA (Reactive)'),
            ('pqcea', 'P-QCEA (Predictive)')
        ]
        
        for algo_key, algo_name in algorithms:
            algo_data = summary[algo_key]
            f.write(f"\n{algo_name}:\n")
            f.write(f"  Average Latency:        {algo_data['avg_latency_ms']:.2f} ms\n")
            f.write(f"  Average PDR:            {algo_data['avg_pdr_percent']:.2f} %\n")
            f.write(f"  Final Residual Energy:  {algo_data['final_residual_energy_j']:.2f} J\n")
            f.write(f"  Average Jitter:         {algo_data['avg_jitter_ms']:.2f} ms\n")
            f.write(f"  Average Throughput:     {algo_data['avg_throughput_mbps']:.2f} Mbps\n")
            if algo_key == 'pqcea':
                f.write(f"  Prediction Usage Rate:  {algo_data['avg_prediction_usage_rate']:.2f} %\n")
                f.write(f"  Avg. Prediction Conf:   {algo_data['avg_prediction_confidence']:.2f} %\n")
        
        # Improvements
        f.write("\n" + "=" * 80 + "\n")
        f.write("PERFORMANCE IMPROVEMENTS (P-QCEA vs Others)\n")
        f.write("=" * 80 + "\n\n")
        
        improvements = report['improvements']
        
        # Group by comparison type
        f.write("P-QCEA vs QCEA:\n")
        f.write("-" * 40 + "\n")
        for key, value in sorted(improvements.items()):
            if 'qcea' in key:
                metric = key.replace('pqcea_vs_qcea_', '').replace('_', ' ').title()
                direction = "reduction" if metric.lower() in ['latency', 'jitter'] else "improvement"
                f.write(f"  {metric:25s}: {value:+7.2f}% {direction}\n")
        
        f.write("\nP-QCEA vs Dijkstra:\n")
        f.write("-" * 40 + "\n")
        for key, value in sorted(improvements.items()):
            if 'dijkstra' in key:
                metric = key.replace('pqcea_vs_dijkstra_', '').replace('_', ' ').title()
                direction = "reduction" if metric.lower() in ['latency', 'jitter'] else "improvement"
                f.write(f"  {metric:25s}: {value:+7.2f}% {direction}\n")
        
        # Detailed Statistics
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        stats = report['statistics']
        for algo_key, algo_name in algorithms:
            f.write(f"\n{algo_name} Statistics:\n")
            f.write("-" * 80 + "\n")
            algo_stats = stats[algo_key]
            
            if not algo_stats:
                f.write("  No data available\n")
                continue
                
            for metric, stat_values in sorted(algo_stats.items()):
                f.write(f"\n  {metric.upper().replace('_', ' ')}:\n")
                f.write(f"    Mean:     {stat_values['mean']:10.4f}\n")
                f.write(f"    Median:   {stat_values['median']:10.4f}\n")
                f.write(f"    Std Dev:  {stat_values['std']:10.4f}\n")
                f.write(f"    Min:      {stat_values['min']:10.4f}\n")
                f.write(f"    Max:      {stat_values['max']:10.4f}\n")
                f.write(f"    25th %ile: {stat_values['q25']:9.4f}\n")
                f.write(f"    75th %ile: {stat_values['q75']:9.4f}\n")
                f.write(f"    Count:    {stat_values['count']:10d}\n")