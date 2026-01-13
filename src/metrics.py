"""
Metrics module: Track performance, robustness, and computational efficiency
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
import yaml


@dataclass
class EpisodeMetrics:
    """Container for per-episode metrics"""
    
    # Tracking/cost
    total_cost: float = 0.0
    tracking_error_mse: float = 0.0
    tracking_error_mae: float = 0.0
    
    # Constraint satisfaction
    num_violations: int = 0
    violation_magnitude: float = 0.0
    violations_per_step: List[float] = field(default_factory=list)
    
    # Event triggering
    num_events: int = 0
    event_rate: float = 0.0
    inter_event_times: np.ndarray = None
    
    # Computational
    cpu_times_per_step: List[float] = field(default_factory=list)
    cpu_time_mean: float = 0.0
    cpu_time_std: float = 0.0
    cpu_time_p95: float = 0.0
    cpu_time_total: float = 0.0
    
    # Control
    mean_control_magnitude: float = 0.0
    max_control_magnitude: float = 0.0
    
    # Transient performance
    settling_time: int = 0  # steps to reach ±5% error
    max_overshoot: float = 0.0
    
    # Metadata
    seed: int = 0
    episode_length: int = 0
    plant_name: str = ""


class MetricsCollector:
    """
    Collects metrics during episode execution
    """
    
    def __init__(self, plant_name: str, seed: int = 0):
        self.plant_name = plant_name
        self.seed = seed
        
        # Storage
        self.step_costs = []
        self.tracking_errors = []
        self.constraint_violations = []
        self.violation_mags = []
        self.control_inputs = []
        self.cpu_times = []
        self.events = []  # boolean per step
        self.predictions = []
        self.states = []
        self.references = []
    
    def log_step(self,
                 x: np.ndarray,
                 x_ref: float,
                 u: float,
                 Q: np.ndarray,
                 R: np.ndarray,
                 violation: bool,
                 violation_mag: float,
                 cpu_time: float,
                 triggered: bool,
                 y_pred: np.ndarray = None):
        """
        Log metrics for one simulation step
        
        Args:
            x: state
            x_ref: reference
            u: control input
            Q: tracking cost weight matrix
            R: control cost weight matrix
            violation: constraint violation occurred
            violation_mag: magnitude of violation
            cpu_time: computation time [s]
            triggered: event triggered
            y_pred: predicted state (optional)
        """
        # Tracking error
        error = x[0] - x_ref  # State space dimension (position or temperature)
        
        # Calculate tracking cost (x-ref)^T Q (x-ref)
        # Handle both scalar and matrix Q robustly
        if np.isscalar(Q) or Q.shape == () or Q.shape == (1,) or Q.shape == (1, 1):
            q_val = float(Q) if np.isscalar(Q) else float(Q.flatten()[0])
            tracking_cost = float(q_val * error**2)
        else:
            # Q is a matrix > 1x1
            error_vec = np.atleast_1d(error)
            tracking_cost = float(error_vec @ Q @ error_vec)
            
        # Calculate control cost u^T R u
        if np.isscalar(R) or R.shape == () or R.shape == (1,) or R.shape == (1, 1):
             r_val = float(R) if np.isscalar(R) else float(R.flatten()[0])
             control_cost = float(r_val * u**2)
        else:
             u_vec = np.atleast_1d(u)
             control_cost = float(u_vec @ R @ u_vec)
        
        self.step_costs.append(tracking_cost + control_cost)
        self.tracking_errors.append(float(error))
        self.control_inputs.append(float(u))
        self.cpu_times.append(cpu_time)
        self.events.append(triggered)
        self.states.append(x.copy())
        self.references.append(x_ref)
        
        if violation:
            self.constraint_violations.append(True)
            self.violation_mags.append(violation_mag)
        else:
            self.constraint_violations.append(False)
            self.violation_mags.append(0.0)
        
        if y_pred is not None:
            self.predictions.append(y_pred.copy())
    
    def finalize(self) -> EpisodeMetrics:
        """
        Compute final aggregated metrics
        
        Returns:
            EpisodeMetrics: completed metrics object
        """
        K = len(self.step_costs)
        
        metrics = EpisodeMetrics(
            seed=self.seed,
            plant_name=self.plant_name,
            episode_length=K
        )
        
        # Cost
        metrics.total_cost = float(np.sum(self.step_costs))
        
        # Tracking
        tracking_errors = np.array(self.tracking_errors)
        metrics.tracking_error_mse = float(np.mean(tracking_errors**2))
        metrics.tracking_error_mae = float(np.mean(np.abs(tracking_errors)))
        
        # Violations
        metrics.num_violations = int(np.sum(self.constraint_violations))
        metrics.violation_magnitude = float(np.sum(self.violation_mags))
        metrics.violations_per_step = self.violation_mags
        
        # Events
        events_array = np.array(self.events, dtype=bool)
        metrics.num_events = int(np.sum(events_array))
        metrics.event_rate = float(np.mean(events_array))
        
        # Inter-event times
        triggered_indices = np.where(events_array)[0]
        if len(triggered_indices) > 1:
            inter_events = np.diff(triggered_indices)
            metrics.inter_event_times = inter_events
        else:
            metrics.inter_event_times = np.array([])
        
        # Computational
        cpu_times = np.array(self.cpu_times)
        metrics.cpu_time_mean = float(np.mean(cpu_times))
        metrics.cpu_time_std = float(np.std(cpu_times))
        metrics.cpu_time_p95 = float(np.percentile(cpu_times, 95))
        metrics.cpu_time_total = float(np.sum(cpu_times))
        metrics.cpu_times_per_step = cpu_times.tolist()
        
        # Control
        control_inputs = np.array(self.control_inputs)
        metrics.mean_control_magnitude = float(np.mean(np.abs(control_inputs)))
        metrics.max_control_magnitude = float(np.max(np.abs(control_inputs)))
        
        # Transient (settling time = first time error < 5% of ref range)
        # Simplified: assume ref range ~1 for motor, ~100 for oven
        settling_threshold = 0.05
        settled = np.abs(tracking_errors) < settling_threshold
        if np.any(settled):
            metrics.settling_time = int(np.argmax(settled))
        else:
            metrics.settling_time = K
        
        # Max overshoot
        metrics.max_overshoot = float(np.max(np.abs(tracking_errors)))
        
        return metrics
    
    def save_trajectory(self, filepath: str):
        """
        Save step-by-step trajectory data to JSON for visualization
        
        Args:
            filepath: output JSON file path
        """
        import json
        from pathlib import Path
        
        # Convert numpy arrays to lists for JSON serialization
        trajectory_data = {
            'states': [s.tolist() for s in self.states],
            'controls': self.control_inputs,
            'references': self.references,
            'events': self.events,
            'cpu_times': self.cpu_times,
            'tracking_errors': self.tracking_errors,
            'plant_name': self.plant_name,
            'seed': self.seed,
            'num_steps': len(self.states)
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        print(f"Trajectory saved to {filepath}")


class MetricsAggregator:
    """
    Aggregates metrics across seeds and scenarios
    """
    
    def __init__(self):
        self.all_metrics = []
        self.scenario_results = {}
    
    def add_metrics(self, metrics: EpisodeMetrics):
        """Add episode metrics"""
        self.all_metrics.append(metrics)
    
    def add_batch(self, metrics_list: List[EpisodeMetrics]):
        """Add multiple episode metrics"""
        self.all_metrics.extend(metrics_list)
    
    def aggregate_by_plant(self) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics by plant
        
        Returns:
            {plant_name: {metric_name: value}}
        """
        plants = {}
        
        for metrics in self.all_metrics:
            if metrics.plant_name not in plants:
                plants[metrics.plant_name] = []
            plants[metrics.plant_name].append(metrics)
        
        aggregated = {}
        for plant_name, metrics_list in plants.items():
            agg = self._aggregate_list(metrics_list)
            aggregated[plant_name] = agg
        
        return aggregated
    
    def _aggregate_list(self, metrics_list: List[EpisodeMetrics]) -> Dict[str, float]:
        """Aggregate a list of metrics"""
        if len(metrics_list) == 0:
            return {}
        
        costs = [m.total_cost for m in metrics_list]
        rmses = [m.tracking_error_mse for m in metrics_list]
        violations = [m.num_violations for m in metrics_list]
        cpu_times = [m.cpu_time_mean for m in metrics_list]
        p95_times = [m.cpu_time_p95 for m in metrics_list]
        event_rates = [m.event_rate for m in metrics_list]
        
        return {
            'cost_mean': float(np.mean(costs)),
            'cost_std': float(np.std(costs)),
            'cost_min': float(np.min(costs)),
            'cost_max': float(np.max(costs)),
            
            'rmse_mean': float(np.mean(rmses)),
            'rmse_std': float(np.std(rmses)),
            'rmse_p95': float(np.percentile(rmses, 95)),
            
            'violations_mean': float(np.mean(violations)),
            'violations_pct': float(100 * np.mean([1 if v > 0 else 0 for v in violations])),
            
            'cpu_mean_ms': float(1000 * np.mean(cpu_times)),
            'cpu_std_ms': float(1000 * np.std(cpu_times)),
            'cpu_p95_ms': float(1000 * np.mean(p95_times)),
            
            'event_rate': float(np.mean(event_rates)),
            
            'n_episodes': len(metrics_list),
        }
    
    def get_summary_table(self) -> Dict[str, Any]:
        """Get summary for all metrics"""
        if len(self.all_metrics) == 0:
            return {}
        
        by_plant = self.aggregate_by_plant()
        return by_plant
    
    def to_csv(self, filepath: str):
        """Export to CSV"""
        import csv
        
        if len(self.all_metrics) == 0:
            return
        
        # Header from first metrics object
        header = [
            'seed', 'plant', 'cost', 'rmse', 'violations', 
            'event_rate', 'cpu_mean_ms', 'cpu_p95_ms'
        ]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            
            for m in self.all_metrics:
                writer.writerow({
                    'seed': m.seed,
                    'plant': m.plant_name,
                    'cost': m.total_cost,
                    'rmse': m.tracking_error_mse,
                    'violations': m.num_violations,
                    'event_rate': m.event_rate,
                    'cpu_mean_ms': 1000 * m.cpu_time_mean,
                    'cpu_p95_ms': 1000 * m.cpu_time_p95,
                })
