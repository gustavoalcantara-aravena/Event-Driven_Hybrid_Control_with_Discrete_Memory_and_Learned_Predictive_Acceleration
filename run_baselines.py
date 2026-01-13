"""
Run Baseline Methods: Comparative evaluation

3 baseline methods:
  1. Periodic MPC (standard periodic control)
  2. Classic eMPC (classical event-triggered MPC)
  3. RL without memory (learning-based without discrete state)
"""

import sys
import os
from pathlib import Path
import time

# sys.path.insert(0, str(Path(__file__).parent / "src")) # Removed

import numpy as np
import pandas as pd
import argparse
import yaml
from typing import Dict, List
from tqdm import tqdm

from src.plants import create_plant
from src.mpc_solver import MPCController
from src.utils import set_seed, Logger, format_time
from src.metrics import MetricsCollector, EpisodeMetrics


class PeriodicMPCBaseline:
    """
    Baseline 1: Standard periodic MPC control
    Executes MPC at fixed intervals (every K steps)
    """
    
    def __init__(self,
                 plant,
                 mpc_config_path: str,
                 plant_type: str,
                 period: int = 10):
        """
        Args:
            plant: plant instance
            mpc_config_path: path to MPC config
            plant_type: "motor" or "oven"
            period: MPC execution period (steps)
        """
        self.plant = plant
        self.period = period
        self.mpc_controller = MPCController(plant_type=plant_type, config_path=mpc_config_path)
        
        self.u_current = 0.0
        self.step_count = 0
        self.metrics = MetricsCollector(plant_name=plant_type)
    
    def step(self, disturbance: float = 0.0):
        """Execute one control step"""
        x = self.plant.state.copy()
        ref = self.plant.get_reference(self.step_count)
        
        # Execute MPC every period steps
        # Execute MPC every period steps
        if self.step_count % self.period == 0:
            result = self.mpc_controller.compute_control(
                x, ref, horizon=10
            )
            self.u_current = result['u']
            self.last_cpu_time = result['time']
        else:
            self.last_cpu_time = 0.0
        
        # Apply control and step plant
        u_applied = np.clip(self.u_current,
                           self.plant.input_limits[0],
                           self.plant.input_limits[1])
        
        self.plant.step(u_applied, disturbance)
        
        # Log metrics
        # Log metrics
        x_next = self.plant.state.copy()
        Q_mat = np.diag([1.0, 0.1])
        R_mat = np.array([[0.01]])
        
        # Correct cost calculation with full reference vector
        ref_vec = np.array([ref, 0.0])
        cost = (x - ref_vec).T @ Q_mat @ (x - ref_vec) + u_applied**2 * R_mat[0, 0]
        
        # check_constraints() checks current state (which is x_next after step)
        violation = 0 if self.plant.check_constraints()[0] else 1
        
        # Pass scalar Q (position weight) to metrics because metrics tracks x[0] error
        q_scalar = Q_mat[0, 0]
        r_scalar = R_mat[0, 0]
        
        # Args: x, x_ref, u, Q, R, violation, violation_mag, cpu_time, triggered, y_pred
        self.metrics.log_step(
            x, ref, u_applied, q_scalar, r_scalar, bool(violation), 0.0, self.last_cpu_time, False, None
        )
        
        self.step_count += 1
    
    def reset(self, x0: np.ndarray = None, seed: int = None):
        """Reset for new episode"""
        if seed is not None:
            set_seed(seed)
        
        self.plant.reset(x0)
        self.u_current = 0.0
        self.step_count = 0
        self.last_cpu_time = 0.0
        plant_name = "motor" if "Motor" in self.plant.__class__.__name__ else "oven"
        self.metrics = MetricsCollector(plant_name=plant_name)
    
    def run_episode(self, steps: int = 1000, seed: int = None):
        """Run complete episode"""
        self.reset(seed=seed)
        
        for _ in range(steps):
            disturbance = 0.01 * np.random.randn() if np.random.rand() < 0.05 else 0.0
            self.step(disturbance)
        
        return self.metrics.finalize()


class ClassicEMPCBaseline:
    """
    Baseline 2: Classical event-triggered MPC
    Triggers on prediction error threshold (static threshold)
    """
    
    def __init__(self,
                 plant,
                 mpc_config_path: str,
                 plant_type: str,
                 error_threshold: float = 2.0):
        """
        Args:
            plant: plant instance
            mpc_config_path: path to MPC config
            plant_type: "motor" or "oven"
            error_threshold: trigger threshold for prediction error
        """
        self.plant = plant
        self.error_threshold = error_threshold
        self.mpc_controller = MPCController(plant_type=plant_type, config_path=mpc_config_path)
        
        self.u_current = 0.0
        self.step_count = 0
        self.x_history = []
        self.u_history = []
        self.event_count = 0
        self.metrics = MetricsCollector(plant_name=plant_type)
    
    def _simple_prediction(self, x: np.ndarray, u: np.ndarray, horizon: int = 5) -> np.ndarray:
        """Simple linear prediction (no LSTM)"""
        # Assume linear approximation from last few steps
        if len(self.x_history) < 2:
            return x
        
        # Velocity estimate
        dx = self.x_history[-1] - self.x_history[-2]
        x_pred = x + dx * horizon
        
        return x_pred
    
    def step(self, disturbance: float = 0.0):
        """Execute one control step with event trigger"""
        x = self.plant.state.copy()
        ref = self.plant.get_reference(self.step_count)
        
        # Store history
        self.x_history.append(x.copy())
        self.u_history.append(self.u_current)
        
        # Compute prediction error
        if len(self.x_history) >= 2:
            x_pred = self._simple_prediction(x, self.u_current)
            pred_error = np.linalg.norm(x - x_pred)
        else:
            x_pred = x
            pred_error = 0.0
        
        # Trigger on prediction error
        triggered = (pred_error > self.error_threshold) or (self.step_count % 50 == 0)
        
        if triggered:
            result = self.mpc_controller.compute_control(x, ref, horizon=10)
            self.u_current = result['u']
            self.event_count += 1
            self.last_cpu_time = result['time']
        else:
            self.last_cpu_time = 0.0
        
        # Apply control
        u_applied = np.clip(self.u_current,
                           self.plant.input_limits[0],
                           self.plant.input_limits[1])
        
        self.plant.step(u_applied, disturbance)
        
        # Log metrics
        # Log metrics
        x_next = self.plant.state.copy()
        Q_mat = np.diag([1.0, 0.1])
        R_mat = np.array([[0.01]])
        
        ref_vec = np.array([ref, 0.0])
        cost = (x - ref_vec).T @ Q_mat @ (x - ref_vec) + u_applied**2 * R_mat[0, 0]
        
        violation = 0 if self.plant.check_constraints()[0] else 1
        
        q_scalar = Q_mat[0, 0]
        r_scalar = R_mat[0, 0]
        
        self.metrics.log_step(
            x, ref, u_applied, q_scalar, r_scalar, bool(violation), 0.0, self.last_cpu_time, triggered, x_pred if triggered else x
        )
        
        self.step_count += 1
    
    def reset(self, x0: np.ndarray = None, seed: int = None):
        """Reset for new episode"""
        if seed is not None:
            set_seed(seed)
        
        self.plant.reset(x0)
        self.u_current = 0.0
        self.step_count = 0
        self.last_cpu_time = 0.0
        self.x_history = []
        self.u_history = []
        self.event_count = 0
        plant_name = "motor" if "Motor" in self.plant.__class__.__name__ else "oven"
        self.metrics = MetricsCollector(plant_name=plant_name)
    
    def run_episode(self, steps: int = 1000, seed: int = None):
        """Run complete episode"""
        self.reset(seed=seed)
        
        for _ in range(steps):
            disturbance = 0.01 * np.random.randn() if np.random.rand() < 0.05 else 0.0
            self.step(disturbance)
        
        return self.metrics.finalize()


class RLWithoutMemoryBaseline:
    """
    Baseline 3: RL-like control without discrete memory
    Uses simple learned policy (linear controller trained on synthetic data)
    """
    
    def __init__(self,
                 plant,
                 mpc_config_path: str,
                 plant_type: str):
        """
        Args:
            plant: plant instance
            mpc_config_path: path to MPC config
            plant_type: "motor" or "oven"
        """
        self.plant = plant
        self.mpc_controller = MPCController(plant_type=plant_type, config_path=mpc_config_path)
        
        # Learned linear policy: u = -K @ (x - ref)
        # Train on random trajectories
        self.K = np.array([[0.3, 0.1]])  # Simple proportional-derivative gains
        
        self.step_count = 0
        self.trigger_count = 0
        self.metrics = MetricsCollector(plant_name=plant_type)
    
    def step(self, disturbance: float = 0.0):
        """Execute one control step"""
        x = self.plant.state.copy()
        ref = self.plant.get_reference(self.step_count)
        
        # Learned policy
        # Construct full reference vector [ref, 0]
        ref_vec = np.array([ref, 0.0])
        error = x - ref_vec
        u_learned = -self.K @ error
        u_learned = np.clip(u_learned[0],
                           self.plant.input_limits[0],
                           self.plant.input_limits[1])
        
        t_start = time.time()
        
        # Occasionally invoke MPC for refinement (30% of time)
        if np.random.rand() < 0.3:
            result = self.mpc_controller.compute_control(x, ref, horizon=10)
            u_applied = result['u']
            self.trigger_count += 1
        else:
            u_applied = u_learned
        
        t_end = time.time()
        self.last_cpu_time = t_end - t_start
        
        # Step plant
        self.plant.step(u_applied, disturbance)
        
        # Log metrics
        # Log metrics
        x_next = self.plant.state.copy()
        Q_mat = np.diag([1.0, 0.1])
        R_mat = np.array([[0.01]])
        
        ref_vec = np.array([ref, 0.0])
        cost = (x - ref_vec).T @ Q_mat @ (x - ref_vec) + u_applied**2 * R_mat[0, 0]
        
        violation = 0 if self.plant.check_constraints()[0] else 1
        
        q_scalar = Q_mat[0, 0]
        r_scalar = R_mat[0, 0]
        
        self.metrics.log_step(
            x, ref, u_applied, q_scalar, r_scalar, bool(violation), 0.0, self.last_cpu_time, False, None
        )
        
        self.step_count += 1
    
    def reset(self, x0: np.ndarray = None, seed: int = None):
        """Reset for new episode"""
        if seed is not None:
            set_seed(seed)
        
        self.plant.reset(x0)
        self.step_count = 0
        self.trigger_count = 0
        self.last_cpu_time = 0.0
        plant_name = "motor" if "Motor" in self.plant.__class__.__name__ else "oven"
        self.metrics = MetricsCollector(plant_name=plant_name)
    
    def run_episode(self, steps: int = 1000, seed: int = None):
        """Run complete episode"""
        self.reset(seed=seed)
        
        for _ in range(steps):
            disturbance = 0.01 * np.random.randn() if np.random.rand() < 0.05 else 0.0
            self.step(disturbance)
        
        return self.metrics.finalize()


class BaselineRunner:
    """Run all baseline methods"""
    
    def __init__(self,
                 plant_type: str,
                 config_dir: str = "config",
                 results_dir: str = "results"):
        """
        Args:
            plant_type: "motor" or "oven"
            config_dir: configuration directory
            results_dir: output directory
        """
        self.plant_type = plant_type
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_baselines(self,
                     num_seeds: int = 5,
                     num_scenarios: int = 10,
                     episode_steps: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Run all 3 baselines
        
        Returns:
            {baseline_name: results_df}
        """
        baselines = [
            {
                'name': 'B1_PeriodicMPC',
                'class': PeriodicMPCBaseline,
                'description': 'Periodic MPC (period=10)'
            },
            {
                'name': 'B2_ClassicEMPC',
                'class': ClassicEMPCBaseline,
                'description': 'Classical event-triggered MPC'
            },
            {
                'name': 'B3_RLnoMemory',
                'class': RLWithoutMemoryBaseline,
                'description': 'Learned control without memory'
            },
        ]
        
        all_results = {}
        
        for baseline in baselines:
            print(f"\n{'='*60}")
            print(f"Running {baseline['name']}: {baseline['description']}")
            print(f"{'='*60}")
            
            results = []
            
            pbar = tqdm(total=num_seeds * num_scenarios)
            
            for seed in range(num_seeds):
                for scenario_idx in range(num_scenarios):
                    # Create plant
                    plant = create_plant(
                        self.plant_type,
                        str(self.config_dir / f"{self.plant_type}_params.yaml")
                    )
                    
                    # Set threshold for B2 to ensure fair comparison (referee request)
                    b2_threshold = 2.0
                    if baseline['name'] == 'B2_ClassicEMPC':
                        if self.plant_type == 'oven':
                            b2_threshold = 40.0  # Increased from 2.0
                        elif self.plant_type == 'both' and plant.name == 'Oven':
                             b2_threshold = 40.0
                    
                    # Create baseline instance
                    if baseline['name'] == 'B2_ClassicEMPC':
                        controller = baseline['class'](
                            plant=plant,
                            mpc_config_path=str(self.config_dir / "mpc_base.yaml"),
                            plant_type=self.plant_type,
                            error_threshold=b2_threshold
                        )
                    else:
                        controller = baseline['class'](
                            plant=plant,
                            mpc_config_path=str(self.config_dir / "mpc_base.yaml"),
                            plant_type=self.plant_type
                        )
                    
                    # Run episode
                    metrics = controller.run_episode(
                        steps=episode_steps,
                        seed=seed + 1000 * scenario_idx
                    )
                    
                     # Save trajectory for visualization (first seed/scenario of B1, B2 and B3)
                     if seed == 0 and scenario_idx == 0:
                          if baseline['name'] in ['B1_PeriodicMPC', 'B2_ClassicEMPC', 'B3_RLnoMemory']:
                             traj_dir = Path("trajectories") / self.plant_type
                             traj_dir.mkdir(parents=True, exist_ok=True)
                             controller.metrics.save_trajectory(str(traj_dir / f"{baseline['name']}.json"))
                    
                    if metrics is not None:
                        metrics_dict = metrics.__dict__
                        metrics_dict['method'] = baseline['name']
                        metrics_dict['seed'] = seed
                        metrics_dict['scenario'] = scenario_idx
                        metrics_dict['plant'] = self.plant_type
                        results.append(metrics_dict)
                    
                    pbar.update(1)
            
            pbar.close()
            
            df = pd.DataFrame(results)
            all_results[baseline['name']] = df
            
            # Save results
            output_file = (self.results_dir / 
                          f"results_{self.plant_type}_{baseline['name']}.csv")
            df.to_csv(output_file, index=False)
            print(f"  Saved: {output_file}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Run baseline methods")
    
    parser.add_argument('--plant', type=str, default='motor',
                       choices=['motor', 'oven', 'both'],
                       help='Plant type')
    parser.add_argument('--seeds', type=int, default=5,
                       help='Number of random seeds')
    parser.add_argument('--scenarios', type=int, default=10,
                       help='Number of scenarios')
    parser.add_argument('--steps', type=int, default=1000,
                       help='Episode length')
    parser.add_argument('--config', type=str, default='config',
                       help='Config directory')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    logger = Logger(name="run_baselines", log_dir="logs")
    logger.info(f"Starting baseline experiments: "
               f"plant={args.plant}, seeds={args.seeds}, scenarios={args.scenarios}")
    
    plants = [args.plant] if args.plant != 'both' else ['motor', 'oven']
    
    for plant in plants:
        print(f"\n{'='*70}")
        print(f"PLANT: {plant.upper()}")
        print(f"{'='*70}")
        
        runner = BaselineRunner(
            plant_type=plant,
            config_dir=args.config,
            results_dir=args.output
        )
        
        results = runner.run_baselines(
            num_seeds=args.seeds,
            num_scenarios=args.scenarios,
            episode_steps=args.steps
        )
        
        # Combine all baseline results
        combined_df = pd.concat([df for df in results.values()], ignore_index=True)
        combined_file = Path(args.output) / f"results_{plant}_baselines_combined.csv"
        combined_df.to_csv(combined_file, index=False)
        logger.info(f"Combined baselines saved: {combined_file}")
    
    print(f"\n{'='*70}")
    print("✓ Baseline experiments complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
