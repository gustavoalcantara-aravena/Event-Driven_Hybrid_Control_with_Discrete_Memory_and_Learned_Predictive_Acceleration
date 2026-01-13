
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.controller_hybrid import HybridEventDrivenController
from src.mpc_solver import MPCSolver

def test_integration():
    print("=== INTEGRATION TEST ===")
    
    plants = ['motor', 'oven']
    
    for plant in plants:
        print(f"\n[Testing Plant: {plant.upper()}]")
        try:
            # 1. Instantiate Controller
            print("  Initializing Controller...")
            ctrl = HybridEventDrivenController(
                plant_type=plant,
                config_dir="config"
            )
            
            # 2. Instantiate MPC Solver
            print("  Initializing MPC Solver...")
            mpc = MPCSolver(
                plant_type=plant,
                config_path="config/mpc_base.yaml"
            )
            ctrl.mpc_solver = mpc
            
            # 3. Test Reset
            print("  Resetting...")
            ctrl.reset(seed=42)
            
            # 4. Test Step (Open Loop / Initial)
            print("  Stepping (Full)...")
            info = ctrl.step(u=0.0)
            print(f"  ✓ Full Step Successful")
            
            # 5. Test Ablation (No Memory)
            print("  Testing Ablation (No Memory)...")
            ctrl.memory_manager = None
            info_ablation = ctrl.step(u=0.0)
            print(f"  ✓ Ablation Step Successful. Mode: {info_ablation['mode']}")
            
            # 5. Check LAPA Initialization (Implicit in Controller)
            # Access lapa via hidden attribute or check logs if possible
            # But if step() worked, LAPA didn't crash on init.
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print("\n✅ INTEGRATION TEST PASSED")

if __name__ == "__main__":
    test_integration()
