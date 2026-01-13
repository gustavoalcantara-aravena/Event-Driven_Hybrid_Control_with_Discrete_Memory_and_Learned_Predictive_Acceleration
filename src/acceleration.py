"""
Learning-Aided Predictive Acceleration (LAPA) Strategies: 
- Neural Warm-starting (LAPA-A)
- Predictive Horizon Adaptation (LAPA-B)
"""

import numpy as np
import yaml
from typing import Dict, Tuple, Optional, Callable


class LAPAAccelerator:
    """
    Learning-Aided Predictive Acceleration (LAPA) for MPC solver
    - LAPA-A: Neural Warm-start initialization from LSTM policy
    - LAPA-B: Predictive Horizon Adaptation based on memory state criticality
    """
    
    def __init__(self, config_path: str = "config/acceleration_config.yaml"):
        """
        Initialize LAPA strategies
        
        Args:
            config_path: path to acceleration configuration
        """
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        self.config = cfg['acceleration']
        
        # LAPA-A config (Neural Warmstart)
        warmstart_cfg = self.config.get('warmstart', {})
        self.lapa_a_enabled = warmstart_cfg.get('enabled', False)
        self.lapa_a_policy = None
        
        # LAPA-B config (Adaptive Horizon)
        horizon_cfg = self.config.get('horizon_adaptation', {})
        self.lapa_b_enabled = horizon_cfg.get('enabled', False)
        self.horizon_normal = horizon_cfg.get('N_base', 10)
        self.horizon_critical = horizon_cfg.get('N_max', 15)
        self.delta_horizon = self.horizon_critical - self.horizon_normal
        
        # Computational budget allocation
        budget_cfg = self.config.get('computational_budget', {})
        self.max_time_ms = budget_cfg.get('max_time_ms', 50.0)
        
        # Statistics
        self.stats = {
            'lapa_a_calls': 0,
            'lapa_a_success': 0,
            'lapa_a_avg_reduction': 0.0,
            'lapa_b_calls': 0,
            'horizon_used': [],
        }
    
    def set_lstm_policy(self, predictor) -> None:
        """
        Register LSTM predictor as warm-start policy
        
        Args:
            predictor: LSTMPredictor instance with predict() method
        """
        self.lapa_a_policy = predictor
    
    def compute_warmstart(self,
                         x_current: np.ndarray,
                         ref: np.ndarray,
                         u_last: Optional[np.ndarray],
                         history: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        LAPA-A: Compute neural warm-start initialization from LSTM policy
        
        Args:
            x_current: current state [nx,]
            ref: reference signal [nx,]
            u_last: last control applied [nu,] (for shift-and-pad fallback)
            history: historical state window [H, nx] for LSTM prediction
        
        Returns:
            u_init: warm-start control sequence [N, nu] or None if disabled/failed
        """
        if not self.lapa_a_enabled or self.lapa_a_policy is None:
            return None
        
        try:
            self.stats['lapa_a_calls'] += 1
            
            # If history provided, use LSTM directly
            if history is not None:
                # Construct LSTM input: [history.flatten(), u_last?, ref]
                lstm_input = np.concatenate([
                    history.flatten(),
                    u_last.flatten() if u_last is not None else np.zeros(1),
                    ref.flatten()
                ])
                
                # Predict next action from LSTM policy
                u_policy = self.lapa_a_policy.predict(lstm_input)  # [nu,]
                
                # Create sequence: replicate with exponential decay for next steps
                N = self.horizon_normal + self.delta_horizon  # Max horizon
                u_init = np.zeros((N, len(u_policy)))
                u_init[0] = u_policy
                
                # Decay towards zero for future steps
                for n in range(1, N):
                    decay = np.exp(-0.5 * n / 10)  # Decay constant
                    u_init[n] = u_policy * decay
                
                self.stats['lapa_a_success'] += 1
                
                return u_init
            else:
                # Fallback: shift-and-pad from last control
                if u_last is None:
                    return None
                
                N = self.horizon_normal + self.delta_horizon
                u_init = np.zeros((N, len(u_last)))
                
                # Shift left and pad with zero
                u_init[:-1] = np.tile(u_last, (N-1, 1))
                
                return u_init
        
        except Exception as e:
            print(f"Warning: LAPA-A failed ({e}), MPC will use zero initialization")
            return None
    
    def compute_adaptive_horizon(self,
                                memory_state: Dict,
                                trigger_value: float,
                                trigger_threshold: float) -> int:
        """
        LAPA-B: Compute adaptive horizon based on memory criticality
        
        Args:
            memory_state: output from DiscreteMemoryManager {bits, normal, saturated, critical}
            trigger_value: event trigger evaluation E(x,ŷ,m)
            trigger_threshold: threshold η(m)
        
        Returns:
            horizon: adaptive horizon for MPC solver
        """
        if not self.lapa_b_enabled:
            return self.horizon_normal
        
        self.stats['lapa_b_calls'] += 1
        
        # Extract criticality from memory
        is_critical = memory_state.get('critical', False)
        is_saturated = memory_state.get('saturated', False)
        
        # Determine horizon
        if is_critical:
            # Extend horizon in critical mode for more planning
            horizon = self.horizon_critical
            reason = "critical"
        elif is_saturated:
            # Slightly extend in saturated mode
            horizon = self.horizon_normal + self.delta_horizon // 2
            reason = "saturated"
        else:
            # Normal mode
            horizon = self.horizon_normal
            reason = "normal"
        
        # Optional: further modulation by trigger proximity
        safety_margin = trigger_threshold - trigger_value  # Higher = safer
        
        if safety_margin < 0.2 * trigger_threshold and not is_critical:
            # Approaching threshold even in normal mode
            horizon = min(horizon + 2, self.horizon_critical)
            reason += "_approaching"
        
        self.stats['horizon_used'].append(horizon)
        
        return horizon
    
    def reset_statistics(self) -> None:
        """Reset statistics counters"""
        self.stats = {
            'lapa_a_calls': 0,
            'lapa_a_success': 0,
            'lapa_a_avg_reduction': 0.0,
            'lapa_b_calls': 0,
            'horizon_used': [],
        }
    
    def get_statistics(self) -> Dict:
        """
        Get acceleration statistics
        
        Returns:
            {lapa_a_success_rate, lapa_b_horizon_stats}
        """
        stats = self.stats.copy()
        
        if stats['lapa_a_calls'] > 0:
            stats['lapa_a_success_rate'] = stats['lapa_a_success'] / stats['lapa_a_calls']
        else:
            stats['lapa_a_success_rate'] = 0.0
        
        if stats['horizon_used']:
            stats['lapa_b_avg_horizon'] = float(np.mean(stats['horizon_used']))
            stats['lapa_b_max_horizon'] = int(np.max(stats['horizon_used']))
            stats['lapa_b_min_horizon'] = int(np.min(stats['horizon_used']))
        else:
            stats['lapa_b_avg_horizon'] = self.horizon_normal
            stats['lapa_b_max_horizon'] = self.horizon_normal
            stats['lapa_b_min_horizon'] = self.horizon_normal
        
        return stats


class StrategySelector:
    """
    Utility to select optimal acceleration strategy based on plant/scenario
    """
    
    @staticmethod
    def select_lapa_config(plant_type: str,
                           scenario_difficulty: str = "medium") -> Dict:
        """
        Select LAPA config based on plant characteristics
        
        Args:
            plant_type: "motor" or "oven"
            scenario_difficulty: "easy", "medium", "hard"
        
        Returns:
            config dict with recommended LAPA settings
        """
        
        base_config = {
            'variants': {
                'lapa_a': {'enabled': True},
                'lapa_b': {
                    'enabled': True,
                    'horizon': {'base': 10, 'extended': 15}
                },
            }
        }
        
        # Plant-specific tuning
        if plant_type.lower() == "motor":
            # Fast dynamics: shorter normal horizon OK
            base_config['variants']['lapa_b']['horizon']['base'] = 8
            base_config['variants']['lapa_b']['horizon']['extended'] = 12
        elif plant_type.lower() == "oven":
            # Slow dynamics: longer horizon for better predictability
            base_config['variants']['lapa_b']['horizon']['base'] = 15
            base_config['variants']['lapa_b']['horizon']['extended'] = 20
        
        # Scenario difficulty modulation
        if scenario_difficulty == "hard":
            base_config['variants']['lapa_a']['enabled'] = True
            base_config['variants']['lapa_b']['enabled'] = True
        elif scenario_difficulty == "easy":
            base_config['variants']['lapa_a']['enabled'] = False
            base_config['variants']['lapa_b']['enabled'] = False
        
        return base_config
    
    @staticmethod
    def compute_expected_improvement(stats_baseline: Dict,
                                     stats_with_lapa: Dict) -> Dict:
        """
        Estimate acceleration benefits
        
        Args:
            stats_baseline: metrics without LAPA (from baseline experiments)
            stats_with_lapa: metrics with LAPA enabled
        
        Returns:
            improvement percentages
        """
        
        improvements = {}
        
        # Compute time improvement
        if stats_baseline.get('cpu_time_mean', 0) > 0:
            cpu_improvement = (
                (stats_baseline['cpu_time_mean'] - stats_with_lapa.get('cpu_time_mean', 0)) /
                stats_baseline['cpu_time_mean'] * 100
            )
            improvements['cpu_time_improvement_pct'] = float(cpu_improvement)
        
        # Cost increase (if any)
        if stats_baseline.get('total_cost', 0) > 0:
            cost_increase = (
                (stats_with_lapa.get('total_cost', stats_baseline['total_cost']) - 
                 stats_baseline['total_cost']) /
                stats_baseline['total_cost'] * 100
            )
            improvements['cost_increase_pct'] = float(cost_increase)
        
        return improvements


# Example usage
if __name__ == "__main__":
    print("Testing LAPA Accelerator...")
    
    # Create accelerator
    lapa = LAPAAccelerator()
    
    # Test LAPA-B with normal memory state
    memory_normal = {'critical': False, 'saturated': False, 'normal': True}
    h_normal = lapa.compute_adaptive_horizon(memory_normal, 1.0, 2.0)
    print(f"✓ Normal mode horizon: {h_normal}")
    
    # Test LAPA-B with critical memory state
    memory_critical = {'critical': True, 'saturated': False, 'normal': False}
    h_critical = lapa.compute_adaptive_horizon(memory_critical, 1.0, 2.0)
    print(f"✓ Critical mode horizon: {h_critical}")
    
    # Test warm-start fallback
    u_last = np.array([0.5])
    u_init = lapa.compute_warmstart(
        np.array([0.1, 0.2]),
        np.array([1.0, 0.0]),
        u_last
    )
    print(f"✓ Warm-start shape: {u_init.shape if u_init is not None else None}")
    
    # Statistics
    stats = lapa.get_statistics()
    print(f"✓ Statistics: {stats}")
