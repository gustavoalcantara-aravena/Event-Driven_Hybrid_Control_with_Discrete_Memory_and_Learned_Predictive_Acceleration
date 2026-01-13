import numpy as np
import pandas as pd
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import logging
from datetime import datetime

from src.plants import create_plant
from src.controller_hybrid import HybridEventDrivenController
from src.metrics import MetricsCollector
from src.utils import set_seed, generate_disturbance_profile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/run_proposed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_proposed")

class ProposedMethodRunner:
    """
    Runner for Proposed H-Event-MPC Method and Ablations
    """
    
    def __init__(self, plant_type: str = "motor", seeds: int = 5, scenarios: int = 5):
        self.plant_type = plant_type
        self.num_seeds = seeds
        self.num_scenarios = scenarios
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Define variants
        self.variants = [
            {
                'name': 'Proposed',
                'use_lstm': True,
                'use_lapa': True,
                'use_memory': True,
                'description': 'Full proposed method',
                'force_periodic': False
            },
            {
                'name': 'A1_NoMemory',
                'use_lstm': True,
                'use_lapa': True,
                'use_memory': False,
                'description': 'Without discrete memory (Always Normal)',
                'force_periodic': False
            },
            {
                'name': 'A2_NoLAPA',
                'use_lstm': True,
                'use_lapa': False,
                'use_memory': True,
                'description': 'Without Predictive Acceleration (LAPA)',
                'force_periodic': False
            },
            {
                'name': 'A3_Periodic',
                'use_lstm': True,
                'use_lapa': True,
                'use_memory': False,
                'description': 'Periodic MPC (No events)',
                'force_periodic': True
            },
            {
                'name': 'A4_EventMPC',
                'use_lstm': False, 
                'use_lapa': False,
                'use_memory': False,
                'description': 'Basic Event-Triggered MPC (Standard)',
                'force_periodic': False
            },
        ]

    def run_all_variants(self, steps: int = 100):
        """Run all variants and save combined results"""
        all_results = []
        
        # Handle "both" option
        plants_to_run = ["motor", "oven"] if self.plant_type == "both" else [self.plant_type]
        
        for plant in plants_to_run:
            print(f"\n{'='*60}")
            print(f"PLANT: {plant.upper()}")
            print(f"{'='*60}")
            
            for variant in self.variants:
                print(f"\nRunning {variant['name']}: {variant['description']}")
                df = self.run_batch(plant, variant, steps)
                df['plant'] = plant
                all_results.append(df)
        
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            output_file = self.results_dir / f"results_{self.plant_type}_combined.csv"
            combined_df.to_csv(output_file, index=False)
            logger.info(f"Combined results saved: {output_file}")
            print(f"\nAll experiments complete. Saved to {output_file}")
            print(f"\nAll experiments complete. Saved to {output_file}")

    def run_batch(self, plant_name: str, variant_config: Dict, steps: int) -> pd.DataFrame:
        """Run batch of experiments for a variant"""
        results_list = []
        
        total_runs = self.num_seeds * self.num_scenarios
        pbar = tqdm(total=total_runs)
        
        for seed in range(self.num_seeds):
            for scenario in range(self.num_scenarios):
                metrics = self.run_simulation(
                    plant_name, seed, scenario, variant_config, steps
                )
                
                # Add metadata
                metrics['seed'] = seed
                metrics['scenario'] = scenario
                metrics['method'] = variant_config['name']
                results_list.append(metrics)
                pbar.update(1)
        
        pbar.close()
        return pd.DataFrame(results_list)

    def run_simulation(self, plant_name: str, seed: int, scenario: int, variant: Dict, steps: int) -> Dict:
        """Execute single simulation episode"""
        set_seed(seed * 100 + scenario)
        
        # Initialize Controller with actual signature
        controller = HybridEventDrivenController(
            plant_type=plant_name,
            config_dir="config/"
        )
        
        # Initialize sub-components (MPC, LSTM, LAPA)
        controller.initialize_components()
        
        # Apply variant modifications before reset
        if not variant['use_lstm']:
            controller.lstm_predictor = None
        
        if not variant['use_memory']:
            controller.memory_manager = None
            
        if not variant.get('use_lapa', True):
            controller.lapa = None
        
        # Special handling for A3 (Periodic) - disable event triggering
        if variant.get('force_periodic', False):
             controller.trigger_manager = None 
        
        # Reset controller for this episode
        controller.reset(x0=None, seed=seed)
        
        # Simulation Loop
        disturbance_profile = generate_disturbance_profile(steps, scenario)
        
        for k in range(steps):
             dist = disturbance_profile[k]
             controller.step(u=0.0, disturbance=dist)
             
        # Save trajectory for visualization (only for first seed/scenario and main Proposed method)
        if seed == 0 and scenario == 0 and variant['name'] == 'Proposed':
             traj_dir = Path("trajectories") / plant_name
             traj_dir.mkdir(parents=True, exist_ok=True)
             controller.metrics.save_trajectory(str(traj_dir / "Proposed.json"))
             
        # Collect Metrics - finalize() returns EpisodeMetrics dataclass
        metrics_obj = controller.metrics.finalize()
        # Convert dataclass to dict for DataFrame compatibility
        summary = {
            'total_cost': metrics_obj.total_cost,
            'tracking_error_mse': metrics_obj.tracking_error_mse,
            'tracking_error_mae': metrics_obj.tracking_error_mae,
            'num_violations': metrics_obj.num_violations,
            'violation_magnitude': metrics_obj.violation_magnitude,
            'violations_per_step': metrics_obj.violations_per_step,
            'num_events': metrics_obj.num_events,
            'event_rate': metrics_obj.event_rate,
            'inter_event_times': metrics_obj.inter_event_times,
            'cpu_times_per_step': metrics_obj.cpu_times_per_step,
            'cpu_time_mean': metrics_obj.cpu_time_mean,
            'cpu_time_std': metrics_obj.cpu_time_std,
            'cpu_time_p95': metrics_obj.cpu_time_p95,
            'cpu_time_total': metrics_obj.cpu_time_total,
            'mean_control_magnitude': metrics_obj.mean_control_magnitude,
            'max_control_magnitude': metrics_obj.max_control_magnitude,
            'settling_time': metrics_obj.settling_time,
            'max_overshoot': metrics_obj.max_overshoot,
            'seed': metrics_obj.seed,
            'episode_length': metrics_obj.episode_length,
            'plant_name': metrics_obj.plant_name,
            'method': variant['name']
        }
        return summary

def main():
    parser = argparse.ArgumentParser(description="Run Proposed Method Experiments")
    parser.add_argument("--plant", type=str, default="motor", choices=["motor", "oven", "both"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--scenarios", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)
    
    args = parser.parse_args()
    
    logger.info(f"Starting proposed method experiments: plant={args.plant}, seeds={args.seeds}, scenarios={args.scenarios}")
    
    runner = ProposedMethodRunner(args.plant, args.seeds, args.scenarios)
    runner.run_all_variants(args.steps)

if __name__ == "__main__":
    main()
