"""
Capture real trajectory from Proposed method for Fig6 visualization
Runs one representative episode and saves trajectory data
"""

import numpy as np
from pathlib import Path
import sys

from src.plants import create_plant
from src.controller_hybrid import HybridEventDrivenController
from src.utils import set_seed

def main():
    print("Capturing Proposed method trajectory...")
    
    for plant in ['motor', 'oven']:
        print(f"\n{'='*60}")
        print(f"Plant: {plant.upper()}")
        print(f"{'='*60}")
        
        set_seed(0)
        
        # Initialize controller
        controller = HybridEventDrivenController(
            plant_type=plant,
            config_dir="config/"
        )
        
        # Reset for episode
        controller.reset(x0=None, seed=0)
        
        # Run episode
        steps = 100
        for k in range(steps):
            # Controller handles everything internally
            controller.step(u=0.0, disturbance=0.0)
        
        # Save trajectory
        traj_file = Path("trajectories") / plant / "Proposed.json"
        controller.metrics.save_trajectory(str(traj_file))
        
        print(f"✓ Saved trajectory to {traj_file}")
    
    print("\n✓ Proposed method trajectories captured!")

if __name__ == "__main__":
    main()
