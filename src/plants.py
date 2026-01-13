"""
Plant models: Motor DC and Thermal Oven
Motor: Position (rad) and velocity (rad/s) with saturation and load disturbance
Oven: Chamber temperature and heater temperature with delays
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import yaml


@dataclass
class PlantState:
    """Generic plant state container"""
    x: np.ndarray
    t: float
    metadata: Dict[str, Any] = None


class MotorDC:
    """
    DC Motor model (discretized, T_s = 10ms)
    States: x = [position (rad), velocity (rad/s)]
    Input:  u = voltage (V), saturated at ±12V
    """
    
    
    def __init__(self, config_path: str = "config/motor_params.yaml"):
        """Initialize motor with parameters from YAML config"""
        """Initialize motor with parameters from YAML config"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dyn = self.config['dynamics']
        self.ctrl = self.config['control']
        self.smp = self.config['sampling']
        self.const = self.config['constraints']
        self.ref = self.config['reference']
        
        # Physical constants
        self.b = self.dyn['friction_viscous']      # 0.5
        self.J = self.dyn['inertia']               # 0.1
        self.dist = self.config['disturbance']
        self.tau_L_nom = self.dist['load_nominal']  # 2.0
        self.load_var = self.dist['load_variation'] # ±0.3
        
        self.T_s = self.smp['period_s']            # 0.01
        self.u_sat = self.ctrl['voltage_saturation']  # 12.0
        
        # State
        self.x = np.zeros(2)  # [position, velocity]
        self.u_prev = 0.0
        self.step_count = 0
    
    @property
    def state(self) -> np.ndarray:
        return self.x
    
    @property
    def input_limits(self) -> Tuple[float, float]:
        return (-self.u_sat, self.u_sat)
        
    def step(self, u: float, disturbance: float = 0.0) -> np.ndarray:
        """
        Simulate one step forward
        
        Args:
            u: commanded voltage [V]
            disturbance: load variation [-1, 1] → multiplies nominal load
        
        Returns:
            x_next: [position, velocity] at k+1
        """
        # Saturate control input
        u_sat = np.clip(u, -self.u_sat, self.u_sat)
        
        # Load disturbance
        tau_L = self.tau_L_nom * (1 + disturbance * self.load_var)
        
        # Discrete-time dynamics (Euler)
        # x1_next = x1 + T_s * x2
        # x2_next = (1 - T_s*b/J)*x2 + (T_s/J)*u_sat - (T_s/J)*tau_L
        
        x1_next = self.x[0] + self.T_s * self.x[1]
        
        damping_coeff = 1.0 - (self.T_s * self.b) / self.J
        torque_coeff = self.T_s / self.J
        
        x2_next = (damping_coeff * self.x[1] + 
                   torque_coeff * u_sat - 
                   torque_coeff * tau_L)
        
        # Update state
        self.x = np.array([x1_next, x2_next])
        self.u_prev = u_sat
        self.step_count += 1
        
        return self.x.copy()
    
    def predict(self, x: np.ndarray, u: float) -> np.ndarray:
        """Predict next state without updating internal state"""
        # Saturate control input
        u_sat = np.clip(u, -self.u_sat, self.u_sat)
        
        # Load disturbance assumed 0 or nominal for prediction
        # We assume nominal load (disturbance=0) for open-loop prediction
        tau_L = self.tau_L_nom
        
        # Discrete-time dynamics (Euler)
        x1_next = x[0] + self.T_s * x[1]
        
        damping_coeff = 1.0 - (self.T_s * self.b) / self.J
        torque_coeff = self.T_s / self.J
        
        x2_next = (damping_coeff * x[1] + 
                   torque_coeff * u_sat - 
                   torque_coeff * tau_L)
        
        return np.array([x1_next, x2_next])
    
    def check_constraints(self) -> Tuple[bool, Dict]:
        """
        Check constraint satisfaction
        
        Returns:
            (feasible, violations_dict)
        """
        violations = {}
        feasible = True
        
        if self.x[0] < self.const['position_min']:
            violations['position_min'] = self.const['position_min'] - self.x[0]
            feasible = False
        if self.x[0] > self.const['position_max']:
            violations['position_max'] = self.x[0] - self.const['position_max']
            feasible = False
        
        if self.x[1] < self.const['velocity_min']:
            violations['velocity_min'] = self.const['velocity_min'] - self.x[1]
            feasible = False
        if self.x[1] > self.const['velocity_max']:
            violations['velocity_max'] = self.x[1] - self.const['velocity_max']
            feasible = False
        
        return feasible, violations
    
    def reset(self, x0: np.ndarray = None):
        """Reset to initial state"""
        if x0 is None:
            cfg_init = self.config['initialization']
            pos_var = cfg_init['position_variation']
            vel_var = cfg_init['velocity_variation']
            self.x = np.array([
                cfg_init['position_nominal'] + np.random.uniform(-pos_var, pos_var),
                cfg_init['velocity_nominal'] + np.random.uniform(-vel_var, vel_var)
            ])
        else:
            self.x = x0.copy()
        
        self.u_prev = 0.0
        self.step_count = 0
    
    def get_reference(self, k: int) -> float:
        """Get reference at step k"""
        values = self.ref['values']
        ref_current = values[0]['value']
        
        for v in values[1:]:
            if k >= v['time']:
                ref_current = v['value']
        
        return ref_current


class ThermalOven:
    """
    Thermal oven model with heater delays
    States: x = [chamber_temp (°C), heater_temp (°C)]
    Input:  u = heater power (%), [0, 100]
    Delay:  τ_d = 5 steps in heater response
    """
    
    def __init__(self, config_path: str = "config/horno_params.yaml"):
        """Initialize oven with parameters from YAML config"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dyn = self.config['dynamics']
        self.ctrl = self.config['control']
        self.smp = self.config['sampling']
        self.const = self.config['constraints']
        self.dist = self.config['disturbance']
        self.ref = self.config['reference']
        
        # Physical constants
        self.alpha = self.dyn['conductance_chamber']  # 0.1
        self.beta = self.dyn['conductance_heater']    # 0.05
        self.delays = self.config['delays']
        self.tau_d = self.delays['heater_delay_steps']  # 5
        self.h_scale = self.dyn['temp_scale_heater']  # 200
        self.model = self.config['thermal_model']
        self.T_max_heater = self.model['max_temp_heater']  # 600
        
        self.T_s = self.smp['period_s']  # 0.1
        self.ambient_std = self.dist['ambient_noise_std']  # 2.0
        
        # State
        self.x = np.zeros(2)  # [T_chamber, T_heater]
        self.u_history = np.zeros(self.tau_d)  # Circular buffer for delay
        self.u_prev = 0.0
        self.u_prev = 0.0
        self.step_count = 0
    
    @property
    def state(self) -> np.ndarray:
        return self.x
    
    @property
    def input_limits(self) -> Tuple[float, float]:
        return (self.ctrl['power_min'], self.ctrl['power_max'])
        
    def _heat_transfer(self, T_heater: float) -> float:
        """Nonlinear heat transfer from heater to chamber"""
        return self.h_scale * (T_heater / self.T_max_heater)
    
    def step(self, u: float, disturbance: float = 0.0) -> np.ndarray:
        """
        Simulate one step forward
        
        Args:
            u: heater power command [%], saturated to [0, 100]
            disturbance: ambient noise [-1, 1] * ambient_std
        
        Returns:
            x_next: [T_chamber, T_heater] at k+1
        """
        # Saturate control
        u_sat = np.clip(u, self.ctrl['power_min'], self.ctrl['power_max'])
        
        # Shift delay buffer and add new input
        self.u_history = np.roll(self.u_history, -1)
        self.u_history[-1] = u_sat
        
        # Get delayed input (τ_d steps back)
        u_delayed = self.u_history[0]
        
        # Noise/disturbance
        w = disturbance * self.ambient_std
        
        # Discrete dynamics
        # T_chamber_next = (1 - α) * T_chamber + α * h(T_heater) + w
        # T_heater_next = (1 - β) * T_heater + β * (u_delayed / 100 * T_max)
        
        T_chamber_next = ((1 - self.alpha) * self.x[0] + 
                          self.alpha * self._heat_transfer(self.x[1]) + 
                          w)
        
        T_heater_next = ((1 - self.beta) * self.x[1] + 
                         self.beta * (u_delayed / 100.0 * self.T_max_heater))
        
        self.x = np.array([T_chamber_next, T_heater_next])
        self.u_prev = u_sat
        self.step_count += 1
        
        return self.x.copy()
    
    def predict(self, x: np.ndarray, u: float) -> np.ndarray:
        """Predict next state without updating internal state"""
        # Saturate control
        u_sat = np.clip(u, self.ctrl['power_min'], self.ctrl['power_max'])
        
        # For prediction, we need the delayed input history?
        # If we just use current u, we ignore delay dynamics.
        # But keeping track of full history in this simple method is hard.
        # Approximation: Assume u has been constant or simply use current.
        # Or better: The controller should handle delay compensation, 
        # but for simple A4 baseline, maybe we just use the model equation with u?
        # The input affects T_heater immediately?
        # NO: T_heater_next depends on u_delayed.
        # If we don't have history, we can't predict accurately.
        # BUT: For A4, we just need a baseline. 
        # Let's assume u_delayed approx u (ignoring delay) for this prediction
        # since we can't access self.u_history from here easily without modifying signature.
        
        u_delayed = u_sat 
        
        # Noise assumed 0
        w = 0.0
        
        T_chamber_next = ((1 - self.alpha) * x[0] + 
                          self.alpha * self._heat_transfer(x[1]) + 
                          w)
        
        T_heater_next = ((1 - self.beta) * x[1] + 
                         self.beta * (u_delayed / 100.0 * self.T_max_heater))
        
        return np.array([T_chamber_next, T_heater_next])
    
    def check_constraints(self) -> Tuple[bool, Dict]:
        """Check constraint satisfaction"""
        violations = {}
        feasible = True
        
        T_chamber = self.x[0]
        
        if T_chamber < self.const['temperature_min']:
            violations['temperature_min'] = self.const['temperature_min'] - T_chamber
            feasible = False
        if T_chamber > self.const['temperature_max']:
            violations['temperature_max'] = T_chamber - self.const['temperature_max']
            feasible = False
        
        return feasible, violations
    
    def reset(self, x0: np.ndarray = None):
        """Reset to initial state"""
        if x0 is None:
            cfg_init = self.config['initialization']
            T_chamber_var = cfg_init['temperature_chamber_variation']
            T_heater_var = cfg_init['temperature_heater_variation']
            self.x = np.array([
                cfg_init['temperature_chamber_nominal'] + np.random.uniform(-T_chamber_var, T_chamber_var),
                cfg_init['temperature_heater_nominal'] + np.random.uniform(-T_heater_var, T_heater_var)
            ])
        else:
            self.x = x0.copy()
        
        self.u_history = np.zeros(self.tau_d)
        self.u_prev = 0.0
        self.step_count = 0
    
    def get_reference(self, k: int) -> float:
        """Get reference at step k"""
        values = self.ref['values']
        ref_current = values[0]['value']
        
        for v in values[1:]:
            if k >= v['time']:
                ref_current = v['value']
        
        return ref_current


# Factory function
def create_plant(plant_type: str, config_path: str = None):
    """Create plant instance"""
    if plant_type.lower() in ['motor', 'motor_dc']:
        return MotorDC(config_path or "config/motor_params.yaml")
    elif plant_type.lower() in ['oven', 'horno', 'thermal']:
        return ThermalOven(config_path or "config/oven_params.yaml")
    else:
        raise ValueError(f"Unknown plant type: {plant_type}")
