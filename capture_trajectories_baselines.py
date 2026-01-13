"""
Capture real trajectories from baseline methods for Fig6 visualization
Runs one representative episode per baseline and saves trajectory data
"""

import numpy as np
from pathlib import Path
import sys

from src.plants import create_plant
from src.mpc_solver import MPCController
from src.metrics import MetricsCollector
from src.utils import set_seed

# Baseline implementations (simplified for trajectory capture)
class BaselineRunner:
    def __init__(self, plant_type='motor'):
        self.plant_type = plant_type
        self.plant = create_plant(plant_type, f"config/{plant_type}_params.yaml")
        self.mpc = MPCController(plant_type=plant_type, config_path="config/mpc_base.yaml")
        
    def run_periodic_mpc(self, steps=100, seed=0):
        """B1: Periodic MPC"""
        set_seed(seed)
        self.plant.reset()
        metrics = MetricsCollector(self.plant_type, seed=seed)
        
        period = 10
        u_current = 0.0
        
        for k in range(steps):
            x = self.plant.state.copy()
            ref = self.plant.get_reference(k)
            
            # MPC every 'period' steps
            if k % period == 0:
                result = self.mpc.compute_control(x, ref, horizon=10)
                u_current = result['u']
                cpu_time = result['time']
            else:
                cpu_time = 0.0
            
            # Apply control
            self.plant.step(u_current, disturbance=0.0)
            
            # Log
            Q = 1.0
            R = 0.01
            violation = not self.plant.check_constraints()[0]
            triggered = (k % period == 0)
            
            metrics.log_step(x, ref, u_current, Q, R, violation, 0.0, cpu_time, triggered, None)
        
        return metrics
    
    def run_rl_baseline(self, steps=100, seed=0):
        """B3: Simple RL (proportional control)"""
        set_seed(seed)
        self.plant.reset()
        metrics = MetricsCollector(self.plant_type, seed=seed)
        
        K = np.array([[0.3, 0.1]])  # Proportional gains
        
        for k in range(steps):
            x = self.plant.state.copy()
            ref = self.plant.get_reference(k)
            
            # Simple policy
            ref_vec = np.array([ref, 0.0])
            error = x - ref_vec
            u = -K @ error
            u = np.clip(u[0], self.plant.input_limits[0], self.plant.input_limits[1])
            
            # Apply control
            self.plant.step(u, disturbance=0.0)
            
            # Log
            Q = 1.0
            R = 0.01
            violation = not self.plant.check_constraints()[0]
            
            metrics.log_step(x, ref, u, Q, R, violation, 0.0, 0.0, False, None)
        
        return metrics

def main():
    print("Capturing baseline trajectories...")
    
    for plant in ['motor', 'oven']:
        print(f"\n{'='*60}")
        print(f"Plant: {plant.upper()}")
        print(f"{'='*60}")
        
        runner = BaselineRunner(plant_type=plant)
        
        # B1: Periodic MPC
        print("Running B1_PeriodicMPC...")
        metrics_b1 = runner.run_periodic_mpc(steps=100, seed=0)
        traj_file = Path("trajectories") / plant / "B1_PeriodicMPC.json"
        metrics_b1.save_trajectory(str(traj_file))
        
        # B3: RL Baseline
        print("Running B3_RLnoMemory...")
        metrics_b3 = runner.run_rl_baseline(steps=100, seed=0)
        traj_file = Path("trajectories") / plant / "B3_RLnoMemory.json"
        metrics_b3.save_trajectory(str(traj_file))
    
    print("\n✓ Baseline trajectories captured!")

if __name__ == "__main__":
    main()
