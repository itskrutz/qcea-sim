"""
Enhanced simulation framework for comparing baseline, QCEA, and P-QCEA routing.
Includes comprehensive metrics collection and comparison analysis.
"""

import random
from src.qcea_routing import QCEARouting
from src.pqcea_routing import PQCEARouting, AdaptivePQCEARouting
from src.baseline_dijkstra import run_dijkstra
from src.topology import (
    load_graph_yaml,
    simple_dynamic_update,
    deduct_path_energy
)
from src.traffic import generate_flows
from src.metrics import Metrics
from src.predictor import PredictiveMetrics


class ComparativeSimulation:
    """
    Run comparative simulation between different routing algorithms.
    """
    
    def __init__(self, config):
        self.config = config
        self.results = {}
    
    def run_algorithm(self, G_orig, algo_name, algo_instance, flows, steps):
        """
        Run simulation for a specific algorithm.
        
        Args:
            G_orig: Original graph (will be copied to avoid mutation)
            algo_name: Name of algorithm for tracking
            algo_instance: Routing algorithm instance (or 'dijkstra')
            flows: List of traffic flows
            steps: Number of simulation time steps
            
        Returns:
            dict with metrics and statistics
        """
        import copy
        G = copy.deepcopy(G_orig)
        
        metrics = Metrics()
        pred_metrics = PredictiveMetrics() if algo_name == 'pqcea' else None
        
        print(f"\n{'='*60}")
        print(f"Running {algo_name.upper()} Simulation")
        print(f"{'='*60}")
        
        for t in range(steps):
            if t % 10 == 0:
                print(f"Time Step {t}/{steps}...", end='\r')
            
            # Update network dynamics
            simple_dynamic_update(G, t)
            
            # Update predictor if P-QCEA
            if algo_name == 'pqcea' and hasattr(algo_instance, 'update_link_measurements'):
                algo_instance.update_link_measurements(G)
            
            # Process each flow
            step_delay, step_tp, step_energy, step_deliv = [], [], [], []
            
            for flow in flows:
                src, dst = flow['src'], flow['dst']
                pkt_size_bits = 8 * 1000  # 1 KB packets
                
                # Compute path based on algorithm
                if algo_name == 'dijkstra':
                    try:
                        # Convert to weight for dijkstra
                        for u, v, d in G.edges(data=True):
                            d['weight'] = d.get('prop_delay_ms', 1.0)
                        path, cost = run_dijkstra(G, src, dst)
                        pred_info = None
                    except:
                        path, cost = None, float('inf')
                        pred_info = None
                        
                elif algo_name == 'pqcea':
                    path, cost, pred_info = algo_instance.compute_path(
                        G, src, dst, mode='predictive'
                    )
                else:  # qcea
                    path, cost = algo_instance.compute_path(G, src, dst)
                    pred_info = None
                
                # Skip if no path found
                if path is None or len(path) < 2:
                    step_deliv.append(0.0)
                    continue
                
                # Calculate metrics
                delay = self._calculate_path_delay(G, path)
                throughput = self._calculate_throughput(G, path, pkt_size_bits)
                energy_used = deduct_path_energy(G, path, pkt_size_bits)
                
                step_delay.append(delay)
                step_tp.append(throughput)
                step_energy.append(energy_used)
                step_deliv.append(1.0)  # packet delivered
                
                # Log prediction accuracy if P-QCEA
                if pred_info and pred_metrics:
                    # This would compare predicted vs actual
                    # (simplified here - full implementation would track)
                    pass
            
            # Aggregate step metrics
            avg_delay = sum(step_delay) / len(flows) if step_delay else 0
            avg_tp = sum(step_tp) / len(flows) if step_tp else 0
            total_energy = sum(step_energy)
            avg_delivery = sum(step_deliv) / len(flows)
            
            metrics.log(avg_delay, avg_tp, total_energy, avg_delivery)
        
        print(f"\n{algo_name.upper()} Complete!")
        
        # Compile results
        result = {
            'metrics': metrics,
            'summary': metrics.summary(),
            'algorithm': algo_name
        }
        
        # Add algorithm-specific stats
        if hasattr(algo_instance, 'get_statistics'):
            result['algo_stats'] = algo_instance.get_statistics()
        
        if pred_metrics:
            result['prediction_stats'] = pred_metrics.get_accuracy_stats()
        
        return result
    
    def _calculate_path_delay(self, G, path):
        """Calculate end-to-end delay for a path."""
        total_delay = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            total_delay += G[u][v].get('prop_delay_ms', 1.0)
        return total_delay
    
    def _calculate_throughput(self, G, path, packet_size_bits):
        """Calculate effective throughput for a path."""
        # Bottleneck bandwidth
        min_bw = float('inf')
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            bw = G[u][v].get('bandwidth_bps', 1e6)
            min_bw = min(min_bw, bw)
        
        # Throughput = packet_size / transmission_time
        if min_bw > 0:
            tx_time_s = packet_size_bits / min_bw
            throughput = packet_size_bits / tx_time_s if tx_time_s > 0 else 0
        else:
            throughput = 0
        
        return throughput
    
    def run_comparison(self):
        """
        Run comparative simulation between all algorithms.
        
        Returns:
            dict with results for each algorithm
        """
        print("\n" + "="*60)
        print("COMPARATIVE SIMULATION: Dijkstra vs QCEA vs P-QCEA")
        print("="*60)
        
        # Load topology
        G_orig = load_graph_yaml(self.config['topology_path'])
        
        # Generate flows
        flows = generate_flows(
            G_orig, 
            num_flows=self.config.get('num_flows', 5),
            seed=self.config.get('seed', 42)
        )
        
        steps = self.config.get('time_steps', 100)
        weights = self.config.get('weights', {
            'wl': 0.3, 'wb': 0.2, 'wp': 0.2, 'we': 0.2, 'wc': 0.1
        })
        
        # Prepare algorithms
        algorithms = {}
        
        # 1. Baseline Dijkstra
        if self.config.get('run_dijkstra', True):
            algorithms['dijkstra'] = 'dijkstra'
        
        # 2. QCEA (without prediction)
        if self.config.get('run_qcea', True):
            algorithms['qcea'] = QCEARouting(weights)
        
        # 3. P-QCEA (with prediction)
        if self.config.get('run_pqcea', True):
            algorithms['pqcea'] = PQCEARouting(
                weights,
                prediction_horizon=self.config.get('prediction_horizon', 3),
                enable_prediction=True
            )
        
        # 4. Adaptive P-QCEA (optional)
        if self.config.get('run_adaptive', False):
            algorithms['adaptive_pqcea'] = AdaptivePQCEARouting(
                weights,
                prediction_horizon=self.config.get('prediction_horizon', 3)
            )
        
        # Run each algorithm
        results = {}
        for name, algo in algorithms.items():
            results[name] = self.run_algorithm(G_orig, name, algo, flows, steps)
        
        self.results = results
        return results
    
    def print_comparison_table(self):
        """Print comparison table of results."""
        if not self.results:
            print("No results to display. Run simulation first.")
            return
        
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        
        # Header
        print(f"{'Algorithm':<20} {'Avg Delay (ms)':<15} {'Avg Throughput':<15} "
              f"{'Total Energy (J)':<15} {'Delivery Rate':<15}")
        print("-"*80)
        
        # Data rows
        for name, result in self.results.items():
            summary = result['summary']
            print(f"{name.upper():<20} "
                  f"{summary['delay']:<15.2f} "
                  f"{summary['throughput']:<15.2f} "
                  f"{summary['energy']:<15.2f} "
                  f"{summary['packet_delivery']:<15.3f}")
        
        print("="*80)
        
        # Performance gains
        if 'dijkstra' in self.results and 'pqcea' in self.results:
            print("\nP-QCEA vs Dijkstra Performance Gains:")
            baseline = self.results['dijkstra']['summary']
            pqcea = self.results['pqcea']['summary']
            
            delay_gain = ((baseline['delay'] - pqcea['delay']) / baseline['delay']) * 100
            energy_gain = ((baseline['energy'] - pqcea['energy']) / baseline['energy']) * 100
            
            print(f"  Delay Reduction: {delay_gain:+.2f}%")
            print(f"  Energy Savings: {energy_gain:+.2f}%")
        
        # Prediction statistics
        if 'pqcea' in self.results:
            algo_stats = self.results['pqcea'].get('algo_stats', {})
            if algo_stats:
                print(f"\nP-QCEA Prediction Statistics:")
                print(f"  Paths Computed: {algo_stats.get('paths_computed', 0)}")
                print(f"  Predictions Used: {algo_stats.get('predictions_used', 0)}")
                print(f"  Prediction Usage Rate: {algo_stats.get('prediction_usage_rate', 0)*100:.1f}%")


def run_simulation(config):
    """
    Main simulation entry point (compatible with your existing code).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Metrics object or comparison results
    """
    # Determine mode
    mode = config.get('simulation_mode', 'comparative')
    
    if mode == 'comparative':
        sim = ComparativeSimulation(config)
        results = sim.run_comparison()
        sim.print_comparison_table()
        return results
    else:
        # Single algorithm mode (backwards compatible)
        print("Loading topology...")
        G = load_graph_yaml(config["topology_path"])
        
        algo_name = config.get("routing", "qcea")
        print(f"Routing algorithm selected: {algo_name}")
        
        weights = config.get("weights", {})
        steps = config.get("time_steps", 10)
        
        # Initialize algorithm
        if algo_name == "pqcea":
            algo = PQCEARouting(weights, prediction_horizon=3)
        elif algo_name == "qcea":
            algo = QCEARouting(weights)
        else:
            algo = None  # dijkstra
        
        flows = generate_flows(G, num_flows=config.get("num_flows", 3))
        metrics = Metrics()
        
        for t in range(steps):
            print(f"\n=== Time Step {t} ===")
            simple_dynamic_update(G, t)
            
            # Update predictor if P-QCEA
            if algo_name == "pqcea":
                algo.update_link_measurements(G)
            
            total_delay, total_tp, total_energy, total_deliv = [], [], [], []
            
            for flow in flows:
                src, dst = flow["src"], flow["dst"]
                pkt_size = 8 * 1000
                
                if algo_name == "dijkstra" or algo is None:
                    path, cost = run_dijkstra(G, src, dst)
                elif algo_name == "pqcea":
                    path, cost, _ = algo.compute_path(G, src, dst)
                else:
                    path, cost = algo.compute_path(G, src, dst)
                
                if path is None:
                    continue
                
                delay = cost
                throughput = 1 / delay if delay else 0
                energy_used = deduct_path_energy(G, path, pkt_size)
                
                total_delay.append(delay)
                total_tp.append(throughput)
                total_energy.append(energy_used)
                total_deliv.append(1.0)
            
            metrics.log(
                sum(total_delay)/len(flows),
                sum(total_tp)/len(flows),
                sum(total_energy),
                sum(total_deliv)/len(flows)
            )
        
        print("\n=== Simulation Complete ===")
        print(metrics.summary())
        return metrics