"""
Enhanced metrics collection for P-QCEA evaluation.
Extends base Metrics class with prediction-specific tracking.
"""

import numpy as np
import json
from collections import defaultdict


class EnhancedMetrics:
    """
    Extended metrics collector for P-QCEA evaluation.
    Tracks both routing performance and prediction accuracy.
    """
    
    def __init__(self):
        # Basic routing metrics (compatible with your existing Metrics class)
        self.data = {
            "delay": [],
            "throughput": [],
            "energy": [],
            "packet_delivery": []
        }
        
        # Prediction-specific metrics
        self.prediction_metrics = {
            "prediction_used_count": [],
            "prediction_skipped_count": [],
            "avg_confidence": [],
            "prediction_accuracy": defaultdict(list)
        }
        
        # Per-flow tracking
        self.flow_metrics = []
        
        # Network state tracking
        self.network_state = {
            "avg_node_energy": [],
            "avg_link_utilization": [],
            "congested_links": []
        }
    
    def log(self, delay, throughput, energy, delivery, 
            prediction_info=None, network_state=None):
        """
        Log metrics for a time step.
        
        Args:
            delay: Average delay for this time step
            throughput: Average throughput
            energy: Total energy consumed
            delivery: Packet delivery rate
            prediction_info: Dict with prediction statistics (optional)
            network_state: Dict with network state info (optional)
        """
        # Basic metrics
        self.data["delay"].append(delay)
        self.data["throughput"].append(throughput)
        self.data["energy"].append(energy)
        self.data["packet_delivery"].append(delivery)
        
        # Prediction metrics
        if prediction_info:
            self.prediction_metrics["prediction_used_count"].append(
                prediction_info.get("links_predicted", 0)
            )
            self.prediction_metrics["prediction_skipped_count"].append(
                prediction_info.get("links_current", 0)
            )
            self.prediction_metrics["avg_confidence"].append(
                prediction_info.get("avg_confidence", 0.0)
            )
        
        # Network state
        if network_state:
            self.network_state["avg_node_energy"].append(
                network_state.get("avg_energy", 0.0)
            )
            self.network_state["avg_link_utilization"].append(
                network_state.get("avg_utilization", 0.0)
            )
            self.network_state["congested_links"].append(
                network_state.get("congested_count", 0)
            )
    
    def log_flow(self, flow_id, src, dst, path, delay, energy, delivered, 
                 used_prediction=False):
        """
        Log per-flow statistics.
        
        Args:
            flow_id: Flow identifier
            src, dst: Source and destination
            path: Computed path
            delay: End-to-end delay
            energy: Energy consumed
            delivered: Whether packet was delivered
            used_prediction: Whether prediction was used for this flow
        """
        self.flow_metrics.append({
            "flow_id": flow_id,
            "src": src,
            "dst": dst,
            "path_length": len(path) if path else 0,
            "delay": delay,
            "energy": energy,
            "delivered": delivered,
            "used_prediction": used_prediction
        })
    
    def log_prediction_accuracy(self, metric_name, predicted, actual):
        """
        Log prediction accuracy for analysis.
        
        Args:
            metric_name: Name of metric (e.g., 'delay', 'bandwidth')
            predicted: Predicted value
            actual: Actual observed value
        """
        if actual > 0:
            error = abs(predicted - actual) / actual
            self.prediction_metrics["prediction_accuracy"][metric_name].append(error)
    
    def summary(self):
        """
        Compute summary statistics.
        
        Returns:
            Dict with mean values for all metrics
        """
        summary = {k: np.mean(v) if v else 0.0 for k, v in self.data.items()}
        
        # Add prediction summary
        if self.prediction_metrics["avg_confidence"]:
            summary["avg_prediction_confidence"] = np.mean(
                self.prediction_metrics["avg_confidence"]
            )
            summary["avg_links_predicted"] = np.mean(
                self.prediction_metrics["prediction_used_count"]
            )
        
        return summary
    
    def detailed_summary(self):
        """
        Compute detailed statistics including variance, percentiles, etc.
        
        Returns:
            Dict with comprehensive statistics
        """
        def stats(values):
            if not values:
                return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99))
            }
        
        detailed = {}
        
        # Basic metrics statistics
        for metric, values in self.data.items():
            detailed[metric] = stats(values)
        
        # Prediction statistics
        if self.prediction_metrics["avg_confidence"]:
            detailed["prediction_confidence"] = stats(
                self.prediction_metrics["avg_confidence"]
            )
            
            # Prediction accuracy per metric
            detailed["prediction_accuracy"] = {}
            for metric, errors in self.prediction_metrics["prediction_accuracy"].items():
                if errors:
                    detailed["prediction_accuracy"][metric] = {
                        "mean_error": float(np.mean(errors)),
                        "median_error": float(np.median(errors)),
                        "accuracy": float(1.0 - np.mean(errors))  # 1 - error
                    }
        
        # Flow-level statistics
        if self.flow_metrics:
            flows_with_pred = [f for f in self.flow_metrics if f["used_prediction"]]
            flows_without_pred = [f for f in self.flow_metrics if not f["used_prediction"]]
            
            detailed["flow_comparison"] = {
                "with_prediction": {
                    "count": len(flows_with_pred),
                    "avg_delay": np.mean([f["delay"] for f in flows_with_pred]) if flows_with_pred else 0,
                    "avg_energy": np.mean([f["energy"] for f in flows_with_pred]) if flows_with_pred else 0
                },
                "without_prediction": {
                    "count": len(flows_without_pred),
                    "avg_delay": np.mean([f["delay"] for f in flows_without_pred]) if flows_without_pred else 0,
                    "avg_energy": np.mean([f["energy"] for f in flows_without_pred]) if flows_without_pred else 0
                }
            }
        
        return detailed
    
    def export_to_json(self, filepath):
        """
        Export all metrics to JSON file.
        
        Args:
            filepath: Path to output file
        """
        export_data = {
            "summary": self.summary(),
            "detailed": self.detailed_summary(),
            "timeseries": {
                k: [float(x) for x in v] for k, v in self.data.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def export_to_csv(self, filepath):
        """
        Export time-series data to CSV.
        
        Args:
            filepath: Path to output file
        """
        import csv
        
        # Get all metric names
        headers = ["time_step"] + list(self.data.keys())
        
        # Add prediction metrics if available
        if self.prediction_metrics["avg_confidence"]:
            headers.extend(["prediction_confidence", "links_predicted"])
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            # Write data rows
            n_steps = len(self.data["delay"])
            for i in range(n_steps):
                row = [i]
                for metric in self.data.keys():
                    row.append(self.data[metric][i] if i < len(self.data[metric]) else 0)
                
                if self.prediction_metrics["avg_confidence"]:
                    row.append(self.prediction_metrics["avg_confidence"][i] 
                             if i < len(self.prediction_metrics["avg_confidence"]) else 0)
                    row.append(self.prediction_metrics["prediction_used_count"][i]
                             if i < len(self.prediction_metrics["prediction_used_count"]) else 0)
                
                writer.writerow(row)
    
    def compare_with_baseline(self, baseline_metrics):
        """
        Compare this metrics object with a baseline.
        
        Args:
            baseline_metrics: Another EnhancedMetrics or Metrics object
            
        Returns:
            Dict with comparison statistics
        """
        comparison = {}
        
        for metric in self.data.keys():
            if hasattr(baseline_metrics, 'data') and metric in baseline_metrics.data:
                baseline_vals = baseline_metrics.data[metric]
                current_vals = self.data[metric]
                
                if baseline_vals and current_vals:
                    baseline_mean = np.mean(baseline_vals)
                    current_mean = np.mean(current_vals)
                    
                    if baseline_mean != 0:
                        improvement = ((baseline_mean - current_mean) / baseline_mean) * 100
                    else:
                        improvement = 0
                    
                    comparison[metric] = {
                        "baseline_mean": float(baseline_mean),
                        "current_mean": float(current_mean),
                        "improvement_percent": float(improvement),
                        "absolute_change": float(current_mean - baseline_mean)
                    }
        
        return comparison
    
    def plot_comparison(self, baseline_metrics, save_path=None):
        """
        Create comparison plots against baseline.
        
        Args:
            baseline_metrics: Baseline metrics object
            save_path: Path to save plot (optional)
        """
        import matplotlib.pyplot as plt
        
        metrics_to_plot = ['delay', 'throughput', 'energy', 'packet_delivery']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            if hasattr(baseline_metrics, 'data') and metric in baseline_metrics.data:
                baseline_vals = baseline_metrics.data[metric]
                current_vals = self.data[metric]
                
                time_steps = range(min(len(baseline_vals), len(current_vals)))
                
                ax.plot(time_steps, baseline_vals[:len(time_steps)], 
                       label='Baseline', color='#FF6B6B', linewidth=2, alpha=0.7)
                ax.plot(time_steps, current_vals[:len(time_steps)], 
                       label='P-QCEA', color='#45B7D1', linewidth=2, alpha=0.7)
                
                ax.set_xlabel('Time Step', fontsize=10)
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
                ax.set_title(f'{metric.replace("_", " ").title()} Over Time', 
                           fontsize=12, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


# Backwards compatibility with your existing Metrics class
class Metrics(EnhancedMetrics):
    """
    Alias for compatibility with existing code.
    Behaves exactly like your original Metrics class, but with
    extended capabilities.
    """
    pass