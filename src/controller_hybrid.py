"""
Main hybrid event-driven controller (Algorithm 1)
Coordinates: Plant → LSTM → Trigger → Memory → MPC → LAPA
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any
import yaml
from collections import deque

from src.plants import create_plant
from src.discrete_logic import DiscreteMemoryManager
from src.event_trigger import AdaptiveTriggerManager
from src.metrics import MetricsCollector
from src.mpc_solver import MPCController
from src.lstm_predictor import LSTMPredictor
from src.acceleration import LAPAAccelerator


class HybridEventDrivenController:
    """
    Main controller orchestrating all components
    
    Control Loop:
    1. Read measurement x_k
    2. LSTM prediction ŷ_{k|k-1}
    3. Event evaluation δ_k
    4. Discrete logic m_k
    5. MPC (with LAPA if δ_k=1)
    6. Apply u_k, log metrics
    """
    
    def __init__(self, 
                 plant_type: str,
                 config_dir: str = "config/"):
        """
        Initialize controller
        
        Args:
            plant_type: "motor" or "horno"
            config_dir: directory with YAML configs
        """
        self.plant_type = plant_type
        self.config_dir = config_dir
        
        # Load configs
        self._load_configs()
        
        # Create plant
        self.plant = create_plant(
            plant_type,
            config_path=f"{config_dir}/{plant_type}_params.yaml"
        )
        
        # Create controller modules
        self.memory_manager = DiscreteMemoryManager(
            f"{config_dir}/trigger_params.yaml"
        )
        
        self.trigger_manager = AdaptiveTriggerManager(
            f"{config_dir}/trigger_params.yaml",
            memory_manager=self.memory_manager.logic
        )
        
        # Metrics
        self.metrics = None
        self.step_count = 0
        
        # MPC solver (placeholder)
        self.mpc_solver = None  # Will be initialized when running
        
        # LSTM predictor (placeholder)
        self.lstm_predictor = None  # Will be initialized when running
        
        # LAPA accelerator (placeholder)
        self.lapa = None
        
        # History buffer for LSTM [x, u]
        self.history_buffer = None
        self.history_length = 10  # Default, will update from predictor
    
    def _load_configs(self):
        """Load all configuration files"""
        configs = {}
        for name in ['motor_params', 'oven_params', 'mpc_base', 
                     'lstm_config', 'trigger_params', 'acceleration_config']:
            with open(f"{self.config_dir}/{name}.yaml", 'r') as f:
                configs[name] = yaml.safe_load(f)
        self.configs = configs
    
    def initialize_components(self):
        """Initialize simulation components (MPC, LSTM, LAPA)"""
        if self.mpc_solver is None:
            self.mpc_solver = MPCController(
                plant_type=self.plant_type,
                config_path=f"{self.config_dir}/mpc_base.yaml"
            )
        
        if self.lstm_predictor is None:
            self.lstm_predictor = LSTMPredictor(
                config_path=f"{self.config_dir}/lstm_config.yaml"
            )
            self.history_length = self.lstm_predictor.history_length
            self.history_buffer = deque(maxlen=self.history_length)
        
        if self.lapa is None:
            self.lapa = LAPAAccelerator(
                config_path=f"{self.config_dir}/acceleration_config.yaml"
            )
    
    def reset(self, x0: np.ndarray = None, seed: int = 0):
        """
        Reset controller for new episode
        
        Args:
            x0: initial state (if None, sampled randomly)
            seed: random seed
        """
        np.random.seed(seed)
        
        self.plant.reset(x0)
        if self.memory_manager is not None:
            self.memory_manager.reset()
        if self.trigger_manager is not None:
            self.trigger_manager.reset()
        self.metrics = MetricsCollector(self.plant_type, seed=seed)
        self.step_count = 0
        self.x_pred_mpc = None # Initialize x_pred_mpc for new episode
        
        # Reset history
        if self.lstm_predictor is not None:
             self.history_length = self.lstm_predictor.history_length
             self.history_buffer = deque(maxlen=self.history_length)
        else:
             self.history_buffer = deque(maxlen=10)
    
    def step(self,
             u: float,
             y_pred: np.ndarray = None,
             disturbance: float = 0.0) -> Dict[str, Any]:
        """
        Execute one control step
        
        ALGORITHM 1 Implementation:
        
        1. Read measurement
        2. LSTM prediction
        3. Trigger evaluation
        4. Memory update
        5. MPC or hold
        6. Apply control
        7. Log metrics
        
        Args:
            u: control input (V or %)
            y_pred: predicted state from LSTM (optional)
            disturbance: load/environmental disturbance
        
        Returns:
            info: dictionary with step info
        """
        t_step_start = time.time()
        
        # Get current state
        x_k = self.plant.x.copy()
        ref_k = self.plant.get_reference(self.step_count)
        
        # 1. LSTM PREDICTION
        t_lstm_start = time.time()
        
        # Update history with previous step's data (x_{k-1}, u_{k-1})
        # But we are at step k. We need predictions for x_{k+1}?
        # No, Algorithm 1: predict x_k given history. (y_pred_{k|k-1})
        # So we use history up to k-1.
        # But wait, step() receives x_k.
        # So prediction should have happened BEFORE reading x_k?
        # Or we predict x_{k+1}?
        # Paper says: "At step k, predicting x_k using info I_{k-1}".
        # Then compare x_k (measured) with y_pred (predicted).
        # So we need to calculate y_pred using history accessible at k-1.
        # This means y_pred should be stored from previous step?
        # OR we calculate it now using valid history?
        # If we just arrived at k, history contains x_{k-H}...x_{k-1} and u_{k-H}...u_{k-1}.
        # So we can predict x_k.
        
        # NOTE: u coming into step() is u_{k-1} (applied last step).
        # x_k is result of u_{k-1}.
        
        # Update buffer with (x_{k-1}, u_{k-1})?
        # We don't have x_{k-1} here directly, only x_k.
        # Controller must persist state.
        # BETTER: The history buffer should contain [x_{k-H}...x_{k-1}] and [u_{k-H}...u_{k-1}].
        # We need to add x_{k-1} and u_{k-1} to buffer.
        # But we are stateless between calls unless we save it.
        # Let's assume we update buffer at END of step with current x_k and u_k?
        # Then next step, buffer has up to k-1.
        
        if y_pred is None and self.lstm_predictor is not None and self.history_buffer is not None:
            if len(self.history_buffer) == self.history_length:
                # Construct flattened input: [x_{k-H}, u_{k-H}, ..., x_{k-1}, u_{k-1}]
                flat_input = []
                for x_h, u_h in self.history_buffer:
                    flat_input.extend(x_h)
                    flat_input.append(u_h) # u is scalar float
                
                # Predict
                try:
                    y_pred = self.lstm_predictor.predict(np.array(flat_input))
                except Exception as e:
                    # print(f"LSTM error: {e}")
                    y_pred = x_k # Fallback
            else:
                y_pred = x_k # Not enough history
        else:
            # If LSTM is disabled but we have x_pred_mpc (from last step's plan), use it
            if self.x_pred_mpc is not None:
                 y_pred = self.x_pred_mpc
            else:
                 y_pred = x_k if y_pred is None else y_pred
            
        t_lstm = time.time() - t_lstm_start
        
        # 2. EVENT EVALUATION
        # Compute constraint margins
        constraint_margins = self._compute_constraint_margins(x_k)
        
        # Trigger step
        if self.trigger_manager is not None:
            trigger_result = self.trigger_manager.step(
                x=x_k,
                y_pred=y_pred,
                constraint_margins=constraint_margins,
                current_step=self.step_count,
                trigger_type="error",
                x_ref=ref_k
            )
            
            delta_k = trigger_result['delta']
            E_val = trigger_result['E_value']
            eta = trigger_result['eta']
        else:
            # Default trigger behavior (e.g. always trigger if A4_EventMPC?? No, A4 might disable something else)
            # Actually A4 disables memory/LAPA but keeps trigger?
            # If trigger manager is None, we assume periodic (delta=1 always)?
            # Or assume run_proposed sets it up.
            # Let's assume safely:
            delta_k = 1 # Force update if no trigger manager (fallback to periodic)
            E_val = 0.0
            eta = 0.0
            trigger_result = {'event_rate': 1.0}
            
        # Force initial trigger at step 0 to ensure control starts
        if self.step_count == 0:
            delta_k = 1
        
        # 3. DISCRETE MEMORY UPDATE
        e_pred = np.linalg.norm(x_k - y_pred)
        margin = np.min(constraint_margins)
        
        # Calculate E_risk for memory (even if not used for trigger)
        # We can use trigger_manager's internal trigger instance to calculate it
        if self.trigger_manager is not None:
             E_risk_val = self.trigger_manager.trigger.evaluate_risk_trigger(
                 x=x_k,
                 constraint_margins=constraint_margins,
                 e_pred=e_pred
             )
        else:
            E_risk_val = 0.0
        
        if self.memory_manager is not None:
            memory_info = self.memory_manager.step(
                u=u if self.step_count > 0 else 0.0,
                x=x_k,
                e_pred=e_pred,
                margin=margin,
                E_risk=E_risk_val
            )
            m_k = memory_info['memory']
        else:
            # Default memory state if disabled (always nominal)
            m_k = [1, 0, 0] # bit_0_normal=1
            memory_info = {
                'memory': m_k,
                'mode': 'nominal',
                'is_critical': False,
                'is_saturated': False
            }
        
        # 4. MPC CONTROL (conditional on δ_k)
        t_mpc_start = time.time()
        
        if delta_k == 1:  # Event occurred
            if self.mpc_solver is not None:
                # Solve MPC (with optional LAPA)
                # ─── LAPA LOGIC ───────────────────────────────────────────
                N_k = None
                u_init = None
                
                # LAPA Logit and Predictive Acceleration
                if hasattr(self, 'lapa') and self.lapa is not None:
                     # Get trigger info for lapa (need E_val and eta from earlier step)
                     # We have E_val and eta from scope
                     memory_state = {
                         'critical': memory_info['is_critical'],
                         'saturated': memory_info['is_saturated'],
                         'normal': m_k[0] == 1 # Approximate
                     }
                     N_k = self.lapa.compute_adaptive_horizon(
                         memory_state=memory_state,
                         trigger_value=E_val,
                         trigger_threshold=eta if eta > 0 else 1.0 # Avoid div zero
                     )
                     
                     # LAPA-A: Neural Warm-start
                     # We need history for LSTM policy. 
                     # self.plant doesn't expose history buffer easily unless we track it here.
                     # Simplified: pass None for history (LAPA might fallback to shift)
                     # Or better: controller should track buffer.
                     # For now, let's try to usage u_prev as u_last.
                     u_last_arr = np.array([self.plant.u_prev])
                     u_init = self.lapa.compute_warmstart(
                         x_current=x_k,
                         ref=ref_k if isinstance(ref_k, np.ndarray) else np.array([ref_k, 0.0]), # ref_k is float usually? check plants
                         u_last=u_last_arr
                     )
                
                # Ensure ref is array [n,]
                # Plant.get_reference returns float (position/temp). 
                # Solver expects [Ref, 0] usually?
                # MPCSolver expects ref same shape as x.
                if np.isscalar(ref_k):
                    ref_vec = np.zeros_like(x_k)
                    ref_vec[0] = ref_k
                else:
                    ref_vec = ref_k
                
                # Solve MPC
                mpc_result = self.mpc_solver.compute_control(
                    x=x_k,
                    ref=ref_vec,
                    horizon=N_k,
                    u_init=u_init
                )
                
                u_control = float(mpc_result['u'])
                mpc_converged = mpc_result['converged']
            else:
                # Placeholder: return reference-based control
                u_control = self._default_control(x_k, ref_k)
                mpc_converged = True
        else:  # No event
            u_control = self.plant.u_prev if self.step_count > 0 else 0.0
            mpc_converged = False
        
        t_mpc = time.time() - t_mpc_start
        t_total = time.time() - t_step_start
        
        # 5. APPLY CONTROL & STEP PLANT
        x_next = self.plant.step(u_control, disturbance)
        
        # Check constraints
        feasible, violations = self.plant.check_constraints()
        violation_mag = max(violations.values()) if violations else 0.0
        
        # 6. LOG METRICS
        Q = self.configs['mpc_base']['mpc']['cost'][self.plant_type].get('Q_position', 1.0)
        R = self.configs['mpc_base']['mpc']['cost'][self.plant_type].get('R_control', 0.01)
        Q_mat = np.array([[Q]])
        
        self.metrics.log_step(
            x=x_next,
            x_ref=ref_k,
            u=u_control,
            Q=Q_mat,
            R=R,
            violation=not feasible,
            violation_mag=violation_mag,
            cpu_time=t_total,
            triggered=bool(delta_k),
            y_pred=y_pred
        )
        
        # 7. PREPARE OUTPUT
        # Update history buffer at end of step with current (x_k, u_k)
        if self.history_buffer is not None:
             # u_val is u_k (computed now). x_k is x_k.
             # We store (x_k, u_k) so next step can use it as k-1.
             # u_val comes from MPC or held logic.
             # We need to extract actual applied u.
             # In run_proposed, u_applied is what plant uses.
             # But controller outputs u_k.
             # We assume u_val is what is applied.
             self.history_buffer.append((x_k, float(u_control))) # Changed u_val to u_control
        
        # Update MPC open-loop prediction for next step (for A4/Standard ETM)
        self.x_pred_mpc = self.plant.predict(x_k, float(u_control))

        info = {
            # State
            'x_k': x_k,
            'x_next': x_next,
            'u_k': u_control,
            'ref_k': ref_k,
            
            # Prediction
            'y_pred': y_pred,
            'e_pred': e_pred,
            
            # Events & triggers
            'E_value': E_val,
            'eta': eta,
            'delta_k': delta_k,
            'event_triggered': bool(delta_k),
            
            # Memory
            'memory': m_k,
            'mode': memory_info['mode'],
            'is_critical': memory_info['is_critical'],
            
            # Constraints
            'feasible': feasible,
            'constraint_margins': constraint_margins,
            'violation_mag': violation_mag,
            
            # Computational
            't_lstm': t_lstm,
            't_mpc': t_mpc,
            't_total': t_total,
            'mpc_converged': mpc_converged,
            
            # Other
            'step': self.step_count,
            'event_rate': trigger_result['event_rate'],
        }
        
        self.step_count += 1
        return info
    
    def _compute_constraint_margins(self, x: np.ndarray) -> np.ndarray:
        """
        Compute margins for all constraints
        Margin > 0 means feasible
        
        Args:
            x: state
        
        Returns:
            margins: array of margins for each constraint
        """
        const = self.plant.const
        margins = []
        
        if self.plant_type.lower() in ['motor', 'motor_dc']:
            # Position constraints
            margins.append(x[0] - const['position_min'])  # x1 > -π
            margins.append(const['position_max'] - x[0])  # x1 < π
            # Velocity constraints
            margins.append(x[1] - const['velocity_min'])  # x2 > -20
            margins.append(const['velocity_max'] - x[1])  # x2 < 20
        
        elif self.plant_type.lower() in ['horno', 'oven', 'thermal']:
            # Temperature constraints
            margins.append(x[0] - const['temperature_min'])  # T > 20
            margins.append(const['temperature_max'] - x[0])  # T < 200
        
        return np.array(margins)
    
    def _default_control(self, x: np.ndarray, ref: float) -> float:
        """Simple proportional control fallback"""
        error = ref - x[0]
        u = 0.1 * error  # P gain
        
        # Use input_limits property that both plants have
        u_min, u_max = self.plant.input_limits
        return np.clip(u, u_min, u_max)
    
    def run_episode(self, num_steps: int, seed: int = 0) -> Dict[str, Any]:
        """
        Run complete episode
        
        Args:
            num_steps: simulation steps
            seed: random seed
        
        Returns:
            results: aggregated metrics and logs
        """
        self.reset(seed=seed)
        
        for k in range(num_steps):
            # Simple feedforward: use previous control
            u = self.plant.u_prev if k > 0 else 0.0
            info = self.step(u)
        
        # Finalize metrics
        metrics = self.metrics.finalize()
        
        return {
            'metrics': metrics,
            'completed_steps': num_steps,
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create controller
    controller = HybridEventDrivenController(
        plant_type="motor",
        config_dir="config/"
    )
    
    # Run one episode
    print("Running motor control for 100 steps...")
    controller.reset(seed=42)
    
    for step in range(100):
        u = 5.0 * np.sin(step / 20)  # Simple sinusoidal control
        info = controller.step(u)
        
        if step % 20 == 0:
            print(f"Step {step}: "
                  f"x=[{info['x_next'][0]:.3f}, {info['x_next'][1]:.3f}], "
                  f"u={info['u_k']:.2f}, "
                  f"δ={info['delta_k']}, "
                  f"m={info['mode']}, "
                  f"t={info['t_total']*1000:.2f}ms")
    
    # Get final metrics
    final_metrics = controller.metrics.finalize()
    print(f"\nEpisode complete:")
    print(f"  Total cost: {final_metrics.total_cost:.4f}")
    print(f"  Tracking RMSE: {np.sqrt(final_metrics.tracking_error_mse):.4f}")
    print(f"  Constraint violations: {final_metrics.num_violations}")
    print(f"  Event rate: {final_metrics.event_rate*100:.1f}%")
    print(f"  CPU p95: {final_metrics.cpu_time_p95*1000:.2f}ms")
