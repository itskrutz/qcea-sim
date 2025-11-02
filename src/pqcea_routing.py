"""
Predictive Quality-Conscious Energy-Aware (P-QCEA) Routing Protocol
Extends QCEA by incorporating link quality prediction for proactive routing decisions.
"""

import networkx as nx
from src.predictor import LinkPredictor


class PQCEARouting:
    """
    P-QCEA routing with integrated link quality prediction.
    
    Key differences from baseline QCEA:
    1. Maintains prediction history for all links
    2. Uses predicted metrics when confidence is high
    3. Falls back to current metrics when prediction is uncertain
    4. Tracks prediction accuracy for analysis
    """
    
    def __init__(self, weights, prediction_horizon=3, enable_prediction=True):
        """
        Args:
            weights: Dict with keys wl, wb, wp, we, wc (same as QCEA)
            prediction_horizon: Steps ahead to predict (default: 3)
            enable_prediction: If False, behaves like regular QCEA
        """
        self.weights = weights
        self.prediction_horizon = prediction_horizon
        self.enable_prediction = enable_prediction
        
        # Initialize predictor
        self.predictor = LinkPredictor(
            history_size=15,
            alpha=0.5,
            confidence_threshold=0.7
        )
        
        # Statistics
        self.stats = {
            'paths_computed': 0,
            'predictions_used': 0,
            'predictions_skipped': 0,
            'fallback_to_current': 0
        }
    
    def update_link_measurements(self, G):
        """
        Update predictor with current link measurements.
        Call this at each time step before computing paths.
        
        Args:
            G: NetworkX graph with current link attributes
        """
        for u, v, data in G.edges(data=True):
            delay = data.get('prop_delay_ms', 1.0)
            bandwidth = data.get('bandwidth_bps', 1e6)
            loss = data.get('loss_prob', 0.0)
            
            self.predictor.update(u, v, delay, bandwidth, loss)
    
    def _get_link_metrics(self, G, u, v, use_prediction=True):
        """
        Get link metrics (either current or predicted).
        
        Args:
            G: NetworkX graph
            u, v: Link endpoints
            use_prediction: Whether to attempt using predictions
            
        Returns:
            tuple: (delay_ms, bandwidth_bps, loss_prob, was_predicted)
        """
        data = G[u][v]
        
        # Get current values as baseline
        current_delay = data.get('prop_delay_ms', 1.0)
        current_bw = data.get('bandwidth_bps', 1e6)
        current_loss = data.get('loss_prob', 0.0)
        
        # Try to get prediction if enabled
        if use_prediction and self.enable_prediction:
            prediction = self.predictor.predict(u, v, steps_ahead=self.prediction_horizon)
            
            if prediction and prediction['use_prediction']:
                # Use predicted values
                self.stats['predictions_used'] += 1
                return (
                    prediction['delay_ms'],
                    prediction['bandwidth_bps'],
                    prediction['loss_prob'],
                    True  # was_predicted flag
                )
            else:
                # Not confident enough, use current
                if prediction:
                    self.stats['predictions_skipped'] += 1
                self.stats['fallback_to_current'] += 1
        
        # Return current values
        return current_delay, current_bw, current_loss, False
    
    def compute_path(self, G, src, dst, mode='predictive'):
        """
        Compute optimal path using P-QCEA algorithm.
        
        Args:
            G: NetworkX graph
            src: Source node
            dst: Destination node
            mode: 'predictive' (use predictions) or 'current' (baseline QCEA)
            
        Returns:
            tuple: (path, total_cost, prediction_info)
        """
        self.stats['paths_computed'] += 1
        use_pred = (mode == 'predictive')
        
        # Track which links used predictions for this path
        prediction_info = {
            'links_predicted': 0,
            'links_current': 0,
            'avg_confidence': 0.0
        }
        
        def link_cost(u, v, data):
            """Calculate link cost using weighted multi-objective function."""
            
            # Get metrics (predicted or current)
            L, B, P, was_predicted = self._get_link_metrics(G, u, v, use_pred)
            
            # Track prediction usage for this path
            if was_predicted:
                prediction_info['links_predicted'] += 1
                pred = self.predictor.predict(u, v, self.prediction_horizon)
                if pred:
                    prediction_info['avg_confidence'] += pred['confidence']['average']
            else:
                prediction_info['links_current'] += 1
            
            # Get node residual energies
            E_u = G.nodes[u].get('residual_energy_j', 1e4)
            E_v = G.nodes[v].get('residual_energy_j', 1e4)
            E = min(E_u, E_v)
            
            # Transmission cost (inverse of bandwidth)
            C = 1.0 / (B / 1e6 + 1e-6)
            
            # Normalize and combine with weights
            cost = (
                self.weights.get('wl', 0.3) * (L / 100.0) +
                self.weights.get('wb', 0.2) * (1.0 / (B / 1e6 + 1e-6)) +
                self.weights.get('wp', 0.2) * P +
                self.weights.get('we', 0.2) * (1.0 / (E / 1e4 + 1e-6)) +
                self.weights.get('wc', 0.1) * C
            )
            
            return cost
        
        # Compute costs for all edges
        for u, v, data in G.edges(data=True):
            data['cost'] = link_cost(u, v, data)
        
        # Find shortest path
        try:
            path = nx.shortest_path(G, src, dst, weight='cost')
            total_cost = nx.shortest_path_length(G, src, dst, weight='cost')
            
            # Calculate average confidence for this path
            if prediction_info['links_predicted'] > 0:
                prediction_info['avg_confidence'] /= prediction_info['links_predicted']
            
            return path, total_cost, prediction_info
            
        except nx.NetworkXNoPath:
            return None, float('inf'), prediction_info
    
    def get_statistics(self):
        """Get routing and prediction statistics."""
        stats = dict(self.stats)
        
        # Add predictor stats
        stats['predictor'] = self.predictor.get_stats()
        
        # Calculate rates
        if stats['paths_computed'] > 0:
            stats['prediction_usage_rate'] = (
                stats['predictions_used'] / stats['paths_computed']
            )
        else:
            stats['prediction_usage_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats = {
            'paths_computed': 0,
            'predictions_used': 0,
            'predictions_skipped': 0,
            'fallback_to_current': 0
        }


class AdaptivePQCEARouting(PQCEARouting):
    """
    Advanced P-QCEA that adapts weights based on network conditions.
    
    This variant automatically adjusts routing weights based on:
    - Network congestion level
    - Average node energy
    - Traffic class requirements
    """
    
    def __init__(self, base_weights, prediction_horizon=3):
        super().__init__(base_weights, prediction_horizon, enable_prediction=True)
        self.base_weights = base_weights.copy()
        self.adaptive_enabled = True
    
    def _assess_network_state(self, G):
        """
        Assess current network conditions.
        
        Returns:
            dict with network state indicators
        """
        # Average link utilization (simplified)
        delays = [d.get('prop_delay_ms', 1.0) for _, _, d in G.edges(data=True)]
        avg_delay = sum(delays) / len(delays) if delays else 1.0
        
        # Average node energy
        energies = [d.get('residual_energy_j', 1e4) for _, d in G.nodes(data=True)]
        avg_energy = sum(energies) / len(energies) if energies else 1e4
        
        # Loss rate
        losses = [d.get('loss_prob', 0.0) for _, _, d in G.edges(data=True)]
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        return {
            'avg_delay_ms': avg_delay,
            'avg_energy_j': avg_energy,
            'avg_loss': avg_loss,
            'congestion_level': avg_delay / 50.0,  # normalized to ~1.0
            'energy_level': avg_energy / 1e4  # normalized
        }
    
    def compute_path(self, G, src, dst, mode='predictive', traffic_class='besteffort'):
        """
        Compute path with adaptive weight adjustment.
        
        Args:
            traffic_class: 'voip', 'video', or 'besteffort'
        """
        if self.adaptive_enabled:
            # Assess network state
            state = self._assess_network_state(G)
            
            # Adapt weights based on conditions
            adapted_weights = self.base_weights.copy()
            
            # High congestion -> prioritize delay
            if state['congestion_level'] > 1.5:
                adapted_weights['wl'] *= 1.5
            
            # Low energy -> prioritize energy
            if state['energy_level'] < 0.3:
                adapted_weights['we'] *= 2.0
            
            # High loss -> prioritize reliability
            if state['avg_loss'] > 0.01:
                adapted_weights['wp'] *= 1.5
            
            # Adjust for traffic class
            if traffic_class == 'voip':
                adapted_weights['wl'] *= 1.8  # VoIP needs low latency
            elif traffic_class == 'video':
                adapted_weights['wb'] *= 1.5  # Video needs bandwidth
            
            # Normalize weights to sum to 1.0
            total = sum(adapted_weights.values())
            for k in adapted_weights:
                adapted_weights[k] /= total
            
            # Temporarily use adapted weights
            original_weights = self.weights
            self.weights = adapted_weights
            
            result = super().compute_path(G, src, dst, mode)
            
            # Restore original weights
            self.weights = original_weights
            
            return result
        else:
            return super().compute_path(G, src, dst, mode)