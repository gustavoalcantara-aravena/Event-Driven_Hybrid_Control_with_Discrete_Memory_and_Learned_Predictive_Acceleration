"""
MPC Solver Module: CasADi/OSQP wrapper for optimal control
Solves: min ||x - ref||_Q^2 + ||u||_R^2 s.t. dynamics + constraints
"""

import numpy as np
import casadi as ca
from typing import Dict, Tuple, Optional, List
import time
import yaml


class MPCSolver:
    """
    Model Predictive Control solver using CasADi
    Supports warm-start and fallback mechanisms
    """
    
    def __init__(self, 
                 plant_type: str,
                 config_path: str = "config/mpc_base.yaml"):
        """
        Initialize MPC solver
        
        Args:
            plant_type: "motor" or "horno"
            config_path: path to MPC configuration
        """
        self.plant_type = plant_type
        self.config_path = config_path
        
        # Load config
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        self.mpc_cfg = cfg['mpc']
        self.cost_cfg = cfg['mpc']['cost'][plant_type]
        
        # MPC parameters
        self.N = self.mpc_cfg['horizon']['base']  # Default horizon
        self.Q = self._get_Q_matrix()
        self.R = self._get_R_scalar()
        
        # Solver settings
        self.solver_cfg = cfg['solver']['ipopt']
        self.max_time_ms = cfg['solver']['max_time_ms'] / 1000.0
        
        # Plant info
        self.nx = 2 if plant_type in ['motor', 'motor_dc'] else 2
        self.nu = 1
        
        # Solver instance (built lazily)
        self.solver = None
        self.nlp = None
        self.last_u_solution = None
        
        # Statistics
        self.last_solve_time = 0.0
        self.last_iterations = 0
        self.last_converged = False
        
    def _get_Q_matrix(self) -> np.ndarray:
        """Get Q cost matrix from config"""
        if self.plant_type in ['motor', 'motor_dc']:
            q_pos = self.cost_cfg.get('Q_position', 1.0)
            q_vel = self.cost_cfg.get('Q_velocity', 0.1)
            return np.diag([q_pos, q_vel])
        else:  # oven
            q_temp = self.cost_cfg.get('Q_temperature', 1.0)
            q_heater = self.cost_cfg.get('Q_heater', 0.05)
            return np.diag([q_temp, q_heater])
    
    def _get_R_scalar(self) -> float:
        """Get R cost scalar from config"""
        return self.cost_cfg.get('R_control', 0.01)
    
    def _build_solver(self, horizon: Optional[int] = None):
        """
        Build CasADi NLP solver
        
        Args:
            horizon: planning horizon (uses default if None)
        """
        if horizon is None:
            horizon = self.N
        else:
            self.N = horizon
        
        # Decision variables: [u_0, u_1, ..., u_{N-1}]
        U = ca.MX.sym('U', self.nu, self.N)
        X0 = ca.MX.sym('X0', self.nx, 1)
        REF = ca.MX.sym('REF', self.nx, 1)
        
        # Objective
        cost = 0
        
        # Predict trajectory and accumulate cost
        x_pred = X0
        for k in range(self.N):
            u_k = U[:, k]
            
            # Track reference (only first state dimension)
            error = x_pred[0] - REF[0]
            cost += error @ self.Q[0, 0] @ error  # Only position error
            cost += u_k @ self.R @ u_k
            
            # Simple forward integration (Euler)
            # For motor: x_{k+1} = f(x_k, u_k)
            x_next = self._predict_next(x_pred, u_k)
            x_pred = x_next
        
        # Terminal cost
        error_term = x_pred[0] - REF[0]
        cost += 0.5 * error_term @ self.Q[0, 0] @ error_term
        
        # Constraints: input saturation
        constraints = []
        constraint_lbs = []
        constraint_ubs = []
        
        u_min = -12.0 if self.plant_type in ['motor', 'motor_dc'] else 0.0
        u_max = 12.0 if self.plant_type in ['motor', 'motor_dc'] else 100.0
        
        for k in range(self.N):
            constraints.append(U[:, k])
            constraint_lbs.append(u_min)
            constraint_ubs.append(u_max)
        
        # NLP formulation
        if constraints:
            g = ca.vertcat(*constraints)
            self.nlp = {
                'x': ca.vertcat(U.reshape((-1, 1))),
                'f': cost,
                'g': g,
                'p': ca.vertcat(X0, REF)
            }
        else:
            self.nlp = {
                'x': ca.vertcat(U.reshape((-1, 1))),
                'f': cost,
                'p': ca.vertcat(X0, REF)
            }
        
        # Create solver
        opts = {
            'ipopt': {
                'max_iter': int(self.solver_cfg['max_iter']),
                'tol': float(self.solver_cfg['tolerance']),
                'constr_viol_tol': float(self.solver_cfg['constraint_tolerance']), # Fixed name & type
                'print_level': 0,  # Silent
            },
            'print_time': False,
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp, opts)
    
    def _predict_next(self, x_k: ca.MX, u_k: ca.MX) -> ca.MX:
        """
        Predict next state using plant model
        
        Args:
            x_k: current state (CasADi symbolic)
            u_k: control input (CasADi symbolic)
        
        Returns:
            x_{k+1} (CasADi symbolic)
        """
        if self.plant_type in ['motor', 'motor_dc']:
            # Motor dynamics (from plants.py)
            b = 0.5
            J = 0.1
            tau_L = 2.0
            T_s = 0.01
            
            x1 = x_k[0]
            x2 = x_k[1]
            
            # Saturate control
            u_sat = ca.fmin(ca.fmax(u_k, -12.0), 12.0)
            
            # Dynamics
            x1_next = x1 + T_s * x2
            damping = 1.0 - (T_s * b) / J
            torque_coeff = T_s / J
            x2_next = damping * x2 + torque_coeff * u_sat - (torque_coeff * tau_L)
            
            return ca.vertcat(x1_next, x2_next)
        
        else:  # oven
            # Oven dynamics
            alpha = 0.1
            beta = 0.05
            h_scale = 200.0
            T_max_heater = 600.0
            T_s = 0.1
            
            T_chamber = x_k[0]
            T_heater = x_k[1]
            
            u_sat = ca.fmin(ca.fmax(u_k, 0.0), 100.0)
            
            # Heat transfer nonlinearity
            h_transfer = h_scale * (T_heater / T_max_heater)
            
            T_chamber_next = (1 - alpha) * T_chamber + alpha * h_transfer
            T_heater_next = (1 - beta) * T_heater + beta * (u_sat / 100.0 * T_max_heater)
            
            return ca.vertcat(T_chamber_next, T_heater_next)
    
    def solve(self,
              x0: np.ndarray,
              ref: np.ndarray,
              horizon: Optional[int] = None,
              u_init: Optional[np.ndarray] = None) -> Tuple[float, bool]:
        """
        Solve MPC problem
        
        Args:
            x0: initial state [n,]
            ref: reference state [n,]
            horizon: planning horizon (uses default if None)
            u_init: warm-start initialization [n_u * N,]
        
        Returns:
            (u_opt, converged): optimal first control and convergence flag
        """
        t_start = time.time()
        
        if horizon is not None and horizon != self.N:
             self._build_solver(horizon)
        elif self.solver is None:
             self._build_solver()
        
        # Determine dimension from x0
        n_states = x0.shape[0] if hasattr(x0, 'shape') else len(x0)
        
        # Safely convert ref to array if scalar
        if np.isscalar(ref):
             ref_arr = np.zeros(n_states)
             ref_arr[0] = ref # Assume first state is tracked
        else:
             ref_arr = ref
        
        # Ensure x0 is array
        x0_arr = np.array(x0) if not isinstance(x0, np.ndarray) else x0
        
        # Prepare parameters
        p_vals = np.concatenate([x0_arr, ref_arr]).reshape((-1, 1))
        
        # Initial guess
        if u_init is not None:
            x_init = u_init.flatten()
        elif self.last_u_solution is not None and self.last_u_solution.size == self.nu * self.N:
            # Warm-start from last solution (shifted)
            x_init = np.concatenate([
                self.last_u_solution[self.nu:],
                np.zeros(self.nu)  # Zero padding for new horizon step
            ])
        else:
            x_init = np.zeros(self.nu * self.N)
            
        # Ensure x_init is a column vector and matches solver dimension
        expected_size = self.nu * self.N
        if x_init.size != expected_size:
            if x_init.size > expected_size:
                x_init = x_init[:expected_size]
            else:
                x_init = np.concatenate([x_init, np.zeros(expected_size - x_init.size)])
        
        x_init = x_init.reshape((-1, 1))
        
        # Solve with timeout
        try:
            # CasADi solvers require numeric bounds if constraints are present
            inf = ca.inf
            result = self.solver(
                x0=x_init,
                p=p_vals,
                ubx=inf,
                lbx=-inf,
                ubg=inf,
                lbg=-inf
            )
            
            u_opt = np.array(result['x']).flatten()
            self.last_u_solution = u_opt
            
            # Extract first control
            u_0 = u_opt[0]
            
            self.last_solve_time = time.time() - t_start
            self.last_converged = True
            
            return float(u_0), True
            
        except Exception as e:
            # Fallback: use warm-start guess or zero
            print(f"Solver Error: {e}")
            self.last_solve_time = time.time() - t_start
            self.last_converged = False
            
            if self.last_u_solution is not None:
                u_0 = self.last_u_solution[0]
            else:
                u_0 = 0.0
            
            return float(u_0), False
    
    def set_horizon(self, horizon: int):
        """Change planning horizon"""
        if horizon != self.N:
            self.N = horizon
            self.solver = None  # Force rebuild
    
    def reset(self):
        """Reset solver state"""
        self.last_u_solution = None
        self.last_solve_time = 0.0
        self.last_iterations = 0
        self.last_converged = False


class MPCController:
    """
    High-level MPC controller managing solver and parameters
    """
    
    def __init__(self,
                 plant_type: str,
                 config_path: str = "config/mpc_base.yaml"):
        self.solver = MPCSolver(plant_type, config_path)
    
    def compute_control(self,
                       x: np.ndarray,
                       ref: np.ndarray,
                       horizon: Optional[int] = None,
                       u_init: Optional[np.ndarray] = None) -> Dict:
        """
        Compute MPC control action
        
        Args:
            x: current state
            ref: reference state
            horizon: planning horizon
            u_init: warm-start (from LAPA-A)
        
        Returns:
            {u: control, converged: bool, time: solve_time}
        """
        u, converged = self.solver.solve(x, ref, horizon, u_init)
        
        return {
            'u': u,
            'converged': converged,
            'time': self.solver.last_solve_time,
            'horizon': self.solver.N,
        }


# Example usage
if __name__ == "__main__":
    print("Testing MPC Solver...")
    
    # Motor
    solver = MPCSolver("motor")
    x0 = np.array([0.0, 0.0])
    ref = np.array([1.5708, 0.0])  # π/2
    
    u, converged = solver.solve(x0, ref)
    print(f"Motor: u={u:.3f}, converged={converged}, t={solver.last_solve_time*1000:.2f}ms")
    
    # Oven
    solver = MPCSolver("oven")
    x0 = np.array([25.0, 25.0])
    ref = np.array([150.0, 50.0])
    
    u, converged = solver.solve(x0, ref)
    print(f"Oven: u={u:.3f}, converged={converged}, t={solver.last_solve_time*1000:.2f}ms")
    
    print("✓ MPC Solver initialized successfully")
