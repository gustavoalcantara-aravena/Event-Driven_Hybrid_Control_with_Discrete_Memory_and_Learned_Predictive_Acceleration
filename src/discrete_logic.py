"""
Discrete memory module: Flip-flops and state machine logic
Implements 3 bits: {normal, saturated, critical}
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import yaml


@dataclass
class MemoryState:
    """Discrete memory state container"""
    bits: np.ndarray  # Shape (B,) with values {0, 1}
    timestamp: int
    active_transitions: List[str] = field(default_factory=list)
    
    def __getitem__(self, name: str) -> int:
        """Access by bit name"""
        names = ['normal', 'saturated', 'critical']
        return self.bits[names.index(name)]
    
    def __repr__(self):
        return f"MemoryState(bits=[{self.bits[0]},{self.bits[1]},{self.bits[2]}] @ t={self.timestamp})"


class DiscreteLogic:
    """
    Manages 3-bit flip-flop memory and transition logic
    Bit 0: normal       (1 = nominal mode)
    Bit 1: saturated    (1 = control saturated)
    Bit 2: critical     (1 = constraint risk / critical mode)
    """
    
    def __init__(self, config_path: str = "config/trigger_params.yaml"):
        """Initialize discrete logic from config"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.num_bits = self.config['memory']['num_bits']
        self.bits_config = self.config['memory']['bits']
        self.transitions_cfg = self.config['transitions']
        self.safety_cfg = self.config['safety']
        
        # Initialize memory state
        self.m_current = np.array([1, 0, 0], dtype=np.int32)  # [normal, saturated, critical]
        self.m_history = []
        self.step_count = 0
        
        # Counters for debouncing
        self.consecutive_saturated = 0
        self.time_in_saturated = 0
        self.time_in_critical = 0
        
        # Logging
        self.log_transitions = self.config['inspection']['log_memory_state']
        self.transition_log = []
        
    def update(self, 
               u: float, 
               x: np.ndarray,
               e_pred: float,
               margin: float,
               E_risk: float = 0.0) -> np.ndarray:
        """
        Update memory state based on conditions
        """
        u_sat_threshold = self.transitions_cfg['saturation']['threshold']
        error_small_threshold = self.transitions_cfg['recovery']['error_threshold']
        
        # Update memory state based on conditions
        m_new = self.m_current.copy()
        
        # ─── CONDITION EVALUATION ────────────────────────────────────────
        
        # Saturation detection (debounce: require 3 consecutive saturated)
        is_saturated = abs(u) > u_sat_threshold
        if is_saturated:
            self.consecutive_saturated += 1
        else:
            self.consecutive_saturated = 0
        
        cond_saturated_set = self.consecutive_saturated >= 3
        
        # Recovery from saturation
        cond_saturated_reset = (
            not is_saturated and 
            e_pred < error_small_threshold and 
            self.time_in_saturated > 100
        )
        
        # Critical conditions (Aligned with Plan: E_risk > 1.0 or low margin)
        # Note: margin is min(margins). If margin < 0.05, we are close to violation.
        cond_critical_set = (
            E_risk > 1.0 or 
            margin < 0.05  # 5% of range safety buffer
        )
        
        # Recovery from critical
        cond_critical_reset = (
            not cond_critical_set and 
            e_pred < error_small_threshold and 
            self.time_in_critical > 150
        )
        
        # ─── BIT UPDATE (SR LATCH LOGIC) ─────────────────────────────────
        
        # Bit 1: saturated (set-reset flip-flop)
        if cond_saturated_set:
            m_new[1] = 1  # Set
            if self.log_transitions:
                self.transition_log.append({
                    'step': self.step_count,
                    'bit': 'saturated',
                    'action': 'SET',
                    'reason': 'consecutive_saturation'
                })
        
        if cond_saturated_reset:
            m_new[1] = 0  # Reset
            self.consecutive_saturated = 0
            self.time_in_saturated = 0
            if self.log_transitions:
                self.transition_log.append({
                    'step': self.step_count,
                    'bit': 'saturated',
                    'action': 'RESET',
                    'reason': 'recovery'
                })
        
        # Bit 2: critical (set-reset flip-flop)
        if cond_critical_set:
            m_new[2] = 1  # Set
            if self.log_transitions:
                self.transition_log.append({
                    'step': self.step_count,
                    'bit': 'critical',
                    'action': 'SET',
                    'reason': 'risk_high'
                })
        
        if cond_critical_reset:
            m_new[2] = 0  # Reset
            self.time_in_critical = 0
            if self.log_transitions:
                self.transition_log.append({
                    'step': self.step_count,
                    'bit': 'critical',
                    'action': 'RESET',
                    'reason': 'safety_recovered'
                })
        
        # Bit 0: normal (complementary to saturated | critical)
        m_new[0] = 1 - np.max([m_new[1], m_new[2]])  # Normal if not saturated AND not critical
        
        # ─── UPDATE COUNTERS ─────────────────────────────────────────────
        
        if m_new[1] == 1:
            self.time_in_saturated += 1
        
        if m_new[2] == 1:
            self.time_in_critical += 1
        
        # ─── STORE STATE ──────────────────────────────────────────────────
        
        self.m_current = m_new
        self.m_history.append(MemoryState(
            bits=m_new.copy(),
            timestamp=self.step_count,
            active_transitions=[
                'saturated_set' if cond_saturated_set else '',
                'saturated_reset' if cond_saturated_reset else '',
                'critical_set' if cond_critical_set else '',
                'critical_reset' if cond_critical_reset else ''
            ]
        ))
        
        self.step_count += 1
        
        return m_new.copy()
    
    def get_memory(self) -> np.ndarray:
        """Get current memory state"""
        return self.m_current.copy()
    
    def get_mode(self) -> str:
        """Get descriptive mode string"""
        if self.m_current[0] == 1:
            return "NORMAL"
        elif self.m_current[1] == 1:
            return "SATURATED"
        elif self.m_current[2] == 1:
            return "CRITICAL"
        else:
            return "UNDEFINED"
    
    def is_critical(self) -> bool:
        """Is system in critical mode?"""
        return self.m_current[2] == 1
    
    def is_saturated(self) -> bool:
        """Is control saturated?"""
        return self.m_current[1] == 1
    
    def reset(self):
        """Reset memory to initial state"""
        self.m_current = np.array([1, 0, 0], dtype=np.int32)
        self.m_history = []
        self.consecutive_saturated = 0
        self.time_in_saturated = 0
        self.time_in_critical = 0
        self.step_count = 0
        self.transition_log = []
    
    def get_transition_log(self) -> List[Dict]:
        """Get transition history"""
        return self.transition_log.copy()
    
    def get_history(self, last_n: int = None) -> List[MemoryState]:
        """Get history of memory states"""
        if last_n is None:
            return self.m_history.copy()
        else:
            return self.m_history[-last_n:].copy()


class DiscreteMemoryManager:
    """
    Higher-level memory manager that coordinates 
    discrete memory with control logic
    """
    
    def __init__(self, config_path: str = "config/trigger_params.yaml"):
        self.logic = DiscreteLogic(config_path)
        self.config = self.logic.config
    
    def get_adaptive_threshold(self, E_type: str = "error") -> float:
        """
        Get adaptive threshold based on current memory state
        
        Args:
            E_type: "error" or "risk"
        
        Returns:
            η(m_k): threshold value
        """
        if E_type == "error":
            thresholds = self.config['trigger']['functions']['E_error']['thresholds']
        else:  # risk
            thresholds = self.config['trigger']['functions']['E_risk']['thresholds']
        
        if self.logic.is_critical():
            return thresholds['critical']['value']
        else:
            return thresholds['normal']['value']
    
    def step(self, u: float, x: np.ndarray, e_pred: float, margin: float, E_risk: float = 0.0) -> Dict[str, Any]:
        """
        Execute one step of memory update
        
        Returns:
            info: dictionary with memory state and transition info
        """
        m_new = self.logic.update(u, x, e_pred, margin, E_risk)
        
        return {
            'memory': m_new,
            'mode': self.logic.get_mode(),
            'is_critical': self.logic.is_critical(),
            'is_saturated': self.logic.is_saturated(),
            'time_in_critical': self.logic.time_in_critical,
            'time_in_saturated': self.logic.time_in_saturated,
        }
    
    def reset(self):
        self.logic.reset()
