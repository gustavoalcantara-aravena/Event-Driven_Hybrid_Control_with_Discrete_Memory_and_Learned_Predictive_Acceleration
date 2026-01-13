"""
Event trigger module: Determines when to activate MPC based on:
- Prediction error
- Constraint margins (risk)
- Memory state (adaptive threshold)
"""

import numpy as np
from typing import Tuple, Dict, Any
import yaml


class EventTrigger:
    """
    Adaptive event-triggered control logic
    Supports E_error and E_risk trigger functions
    """
    
    def __init__(self, config_path: str = "config/trigger_params.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.trigger_cfg = self.config['trigger']
        self.hysteresis_cfg = self.trigger_cfg['hysteresis']
        
        # State
        self.last_trigger_step = -10  # Allow first event
        self.last_E_value = 0.0
        self.min_inter_event = self.trigger_cfg['minimum_inter_event_time']
        
        # Logging
        self.event_log = []
        self.E_history = []
        
    def evaluate_error_trigger(self, 
                              x: np.ndarray, 
                              y_pred: np.ndarray,
                              x_ref: float = None) -> float:
        """
        E_error: prediction error + optional tracking error
        ||x_k - ŷ_{k|k-1}||_2 + α|x_0 - x_ref|
        """
        e_inno = np.linalg.norm(x - y_pred, ord=2)
        if x_ref is not None:
            # Add tracking error to innovation to ensure we don't drift from ref
            # Even if LSTM prediction is "consistent" with drift
            e_track = np.abs(x[0] - x_ref)
            return float(e_inno + 0.1 * e_track) # α=0.1
        return float(e_inno)
    
    def evaluate_risk_trigger(self,
                             x: np.ndarray,
                             constraint_margins: np.ndarray,
                             e_pred: float,
                             e_pred_nominal: float = 1.0) -> float:
        """
        E_risk: constraint margin + prediction penalty
        -min(margins) + prediction_penalty
        
        Args:
            x: current state
            constraint_margins: array of margins for each constraint (signed)
                               >0 means feasible
            e_pred: current prediction error
            e_pred_nominal: nominal prediction error for scaling
        
        Returns:
            E_val: risk metric (high = risky)
        """
        min_margin = np.min(constraint_margins)  # Most restrictive constraint
        margin_risk = -min_margin  # Positive if any constraint violated
        
        # Normalize prediction error contribution
        pred_penalty = 0.3 * (e_pred / max(e_pred_nominal, 1e-6))
        
        E_val = margin_risk + pred_penalty
        return float(max(0.0, E_val))
    
    def trigger(self,
                E_value: float,
                eta: float,
                current_step: int) -> bool:
        """
        Determine if event occurs based on:
        δ_k = 1{E(·) > η}
        with hysteresis and debouncing
        
        Args:
            E_value: current event function value
            eta: adaptive threshold
            current_step: simulation step counter
        
        Returns:
            should_trigger: True if δ_k = 1
        """
        # Minimum inter-event time
        if current_step - self.last_trigger_step < self.min_inter_event:
            return False
        
        # Hysteresis logic
        if self.hysteresis_cfg['enabled']:
            hysteresis_off = self.hysteresis_cfg['hysteresis_ratio'] * eta
            
            if self.last_E_value > eta:  # Was triggering
                threshold_effective = hysteresis_off  # Require lower to switch off
            else:  # Was not triggering
                threshold_effective = eta  # Require eta to switch on
        else:
            threshold_effective = eta
        
        should_trigger = E_value > threshold_effective
        
        # Log
        self.event_log.append({
            'step': current_step,
            'E_value': E_value,
            'eta': eta,
            'threshold_eff': threshold_effective,
            'triggered': should_trigger
        })
        
        self.E_history.append(E_value)
        
        if should_trigger:
            self.last_trigger_step = current_step
        
        self.last_E_value = E_value
        
        return should_trigger
    
    def get_event_rate(self) -> float:
        """Compute event rate (fraction of steps with δ_k=1)"""
        if len(self.event_log) == 0:
            return 0.0
        
        num_events = sum(1 for e in self.event_log if e['triggered'])
        return num_events / len(self.event_log)
    
    def get_inter_event_times(self) -> np.ndarray:
        """Compute inter-event intervals (steps between events)"""
        triggered_steps = [e['step'] for e in self.event_log if e['triggered']]
        
        if len(triggered_steps) < 2:
            return np.array([])
        
        intervals = np.diff(triggered_steps)
        return intervals
    
    def reset(self):
        """Reset trigger state"""
        self.last_trigger_step = -10
        self.last_E_value = 0.0
        self.event_log = []
        self.E_history = []


class AdaptiveTriggerManager:
    """
    Coordinates event trigger with memory state for adaptive thresholds
    """
    
    def __init__(self, 
                 trigger_config: str = "config/trigger_params.yaml",
                 memory_manager=None):
        self.trigger = EventTrigger(trigger_config)
        self.memory_manager = memory_manager
        
        with open(trigger_config, 'r') as f:
            cfg = yaml.safe_load(f)
        self.trigger_cfg = cfg['trigger']
    
    def get_threshold(self, trigger_type: str = "error") -> float:
        """
        Get adaptive threshold based on memory state
        
        Args:
            trigger_type: "error" or "risk"
        
        Returns:
            η(m_k): threshold value
        """
        if trigger_type == "error":
            thresholds = self.trigger_cfg['functions']['E_error']['thresholds']
        else:
            thresholds = self.trigger_cfg['functions']['E_risk']['thresholds']
        
        if self.memory_manager is not None and self.memory_manager.is_critical():
            return thresholds['critical']['value']
        else:
            return thresholds['normal']['value']
    
    def step(self,
             x: np.ndarray,
             y_pred: np.ndarray,
             constraint_margins: np.ndarray,
             current_step: int,
             trigger_type: str = "error",
             x_ref: float = None) -> Dict[str, Any]:
        """
        Execute one step of event evaluation
        """
        # Compute event function
        if trigger_type == "error":
            E_value = self.trigger.evaluate_error_trigger(x, y_pred, x_ref)
        else:
            e_pred = np.linalg.norm(x - y_pred)
            E_value = self.trigger.evaluate_risk_trigger(x, constraint_margins, e_pred)
        
        # Get adaptive threshold
        eta = self.get_threshold(trigger_type)
        
        # Determine trigger
        should_trigger = self.trigger.trigger(E_value, eta, current_step)
        
        return {
            'E_value': E_value,
            'eta': eta,
            'delta': should_trigger,
            'event_rate': self.trigger.get_event_rate(),
        }
    
    def reset(self):
        self.trigger.reset()
