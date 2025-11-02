"""
Predictive module for P-QCEA routing protocol.
Implements time-series prediction for network link metrics using:
- Exponential smoothing for noise reduction
- Linear regression for trend detection
- Confidence scoring based on prediction reliability
- Multi-metric prediction (delay, bandwidth, loss)
"""

import numpy as np
from collections import deque


class LinkPredictor:
    """Predicts future link quality metrics using lightweight time-series analysis."""
    
    def __init__(self, history_size=15, alpha=0.3, confidence_threshold=0.7):
        """
        Args:
            history_size: Number of historical measurements to keep
            alpha: Smoothing factor for exponential smoothing (0-1)
            confidence_threshold: Minimum confidence to use prediction (0-1)
        """
        self.history_size = history_size
        self.alpha = alpha  # weight for current value in exp smoothing
        self.confidence_threshold = confidence_threshold
        
        # Storage for each link: (u,v) -> {'delay': deque, 'bandwidth': deque, 'loss': deque}
        self.link_history = {}
        self.smoothed_values = {}  # stores last smoothed value for each metric
        
    def _get_link_key(self, u, v):
        """Create consistent link key regardless of direction."""
        return tuple(sorted([u, v]))
    
    def update(self, u, v, delay_ms, bandwidth_bps, loss_prob):
        """
        Record a new measurement for a link.
        
        Args:
            u, v: Link endpoints
            delay_ms: Current propagation delay
            bandwidth_bps: Current bandwidth
            loss_prob: Current loss probability
        """
        key = self._get_link_key(u, v)
        
        # Initialize history if needed
        if key not in self.link_history:
            self.link_history[key] = {
                'delay': deque(maxlen=self.history_size),
                'bandwidth': deque(maxlen=self.history_size),
                'loss': deque(maxlen=self.history_size)
            }
            self.smoothed_values[key] = {
                'delay': delay_ms,
                'bandwidth': bandwidth_bps,
                'loss': loss_prob
            }
        
        # Apply exponential smoothing
        smooth = self.smoothed_values[key]
        smooth['delay'] = self.alpha * delay_ms + (1 - self.alpha) * smooth['delay']
        smooth['bandwidth'] = self.alpha * bandwidth_bps + (1 - self.alpha) * smooth['bandwidth']
        smooth['loss'] = self.alpha * loss_prob + (1 - self.alpha) * smooth['loss']
        
        # Store smoothed values in history
        hist = self.link_history[key]
        hist['delay'].append(smooth['delay'])
        hist['bandwidth'].append(smooth['bandwidth'])
        hist['loss'].append(smooth['loss'])
    
    def _compute_trend(self, values):
        """
        Compute linear trend from recent values.
        
        Args:
            values: List/deque of measurements
            
        Returns:
            slope: Rate of change per time step (can be positive/negative)
        """
        if len(values) < 3:
            return 0.0
        
        # Use last 5 values for trend (or all if less)
        recent = list(values)[-5:]
        n = len(recent)
        x = np.arange(n)
        y = np.array(recent)
        
        # Linear regression: y = mx + b
        x_mean = x.mean()
        y_mean = y.mean()
        
        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()
        
        if denominator < 1e-9:
            return 0.0
        
        slope = numerator / denominator
        return slope
    
    def _compute_confidence(self, values):
        """
        Compute prediction confidence based on measurement stability.
        
        Args:
            values: List/deque of measurements
            
        Returns:
            confidence: Score between 0 and 1 (1 = most confident)
        """
        if len(values) < 3:
            return 0.0
        
        recent = list(values)[-5:]
        
        # Calculate coefficient of variation (normalized std dev)
        std = np.std(recent)
        mean = np.mean(recent)
        
        if mean < 1e-9:
            return 0.5
        
        cv = std / mean
        
        # Convert CV to confidence: lower CV = higher confidence
        # Use exponential decay: conf = e^(-k*cv)
        confidence = np.exp(-2.0 * cv)
        
        return min(1.0, max(0.0, confidence))
    
    def predict(self, u, v, steps_ahead=3):
        """
        Predict link metrics 'steps_ahead' time steps into the future.
        
        Args:
            u, v: Link endpoints
            steps_ahead: Number of time steps to predict ahead
            
        Returns:
            dict with keys:
                - 'delay_ms': predicted delay
                - 'bandwidth_bps': predicted bandwidth
                - 'loss_prob': predicted loss
                - 'confidence': dict with confidence for each metric
                - 'use_prediction': bool, True if confident enough to use
        """
        key = self._get_link_key(u, v)
        
        # If no history, return None
        if key not in self.link_history:
            return None
        
        hist = self.link_history[key]
        
        # Need at least 3 measurements for meaningful prediction
        if len(hist['delay']) < 3:
            return None
        
        # Get current (smoothed) values
        current = self.smoothed_values[key]
        
        # Compute trends
        delay_trend = self._compute_trend(hist['delay'])
        bw_trend = self._compute_trend(hist['bandwidth'])
        loss_trend = self._compute_trend(hist['loss'])
        
        # Compute confidences
        delay_conf = self._compute_confidence(hist['delay'])
        bw_conf = self._compute_confidence(hist['bandwidth'])
        loss_conf = self._compute_confidence(hist['loss'])
        
        # Average confidence
        avg_confidence = (delay_conf + bw_conf + loss_conf) / 3.0
        
        # Make predictions using linear extrapolation
        pred_delay = current['delay'] + delay_trend * steps_ahead
        pred_bw = current['bandwidth'] + bw_trend * steps_ahead
        pred_loss = current['loss'] + loss_trend * steps_ahead
        
        # Clamp predictions to reasonable ranges
        pred_delay = max(0.1, pred_delay)  # at least 0.1 ms
        pred_bw = max(1e3, pred_bw)  # at least 1 kbps
        pred_loss = max(0.0, min(1.0, pred_loss))  # between 0 and 1
        
        return {
            'delay_ms': pred_delay,
            'bandwidth_bps': pred_bw,
            'loss_prob': pred_loss,
            'confidence': {
                'delay': delay_conf,
                'bandwidth': bw_conf,
                'loss': loss_conf,
                'average': avg_confidence
            },
            'use_prediction': avg_confidence >= self.confidence_threshold,
            'trends': {
                'delay': delay_trend,
                'bandwidth': bw_trend,
                'loss': loss_trend
            }
        }
    
    def get_stats(self):
        """Get statistics about the predictor state."""
        return {
            'num_links_tracked': len(self.link_history),
            'avg_history_size': np.mean([len(h['delay']) for h in self.link_history.values()]) if self.link_history else 0,
            'confidence_threshold': self.confidence_threshold
        }


class PredictiveMetrics:
    """Extended metrics collector that tracks prediction accuracy."""
    
    def __init__(self):
        self.prediction_accuracy = {
            'delay': [],
            'bandwidth': [],
            'loss': []
        }
        self.prediction_used_count = 0
        self.total_predictions = 0
    
    def log_prediction(self, predicted, actual, used):
        """
        Log prediction vs actual for analysis.
        
        Args:
            predicted: dict with predicted values
            actual: dict with actual values  
            used: bool, whether prediction was actually used
        """
        self.total_predictions += 1
        if used:
            self.prediction_used_count += 1
        
        # Calculate percentage errors
        for metric in ['delay', 'bandwidth', 'loss']:
            if metric in predicted and metric in actual:
                pred_val = predicted[metric]
                actual_val = actual[metric]
                
                if actual_val > 1e-9:  # avoid division by zero
                    error = abs(pred_val - actual_val) / actual_val
                    self.prediction_accuracy[metric].append(error)
    
    def get_accuracy_stats(self):
        """Get prediction accuracy statistics."""
        stats = {}
        for metric, errors in self.prediction_accuracy.items():
            if errors:
                stats[metric] = {
                    'mean_error': np.mean(errors),
                    'median_error': np.median(errors),
                    'std_error': np.std(errors)
                }
            else:
                stats[metric] = {'mean_error': 0, 'median_error': 0, 'std_error': 0}
        
        stats['prediction_usage_rate'] = (
            self.prediction_used_count / self.total_predictions 
            if self.total_predictions > 0 else 0
        )
        
        return stats