#!/usr/bin/env python3
"""
Quick test script - verifies core components work
Run: python test_quick.py
"""

import sys
from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent)) # Needed for now as tests are not a package? 
# Better: Just run from project root with -m pytest or similar.
# But keeping sys.path for standalone run, but pointing to src modules.
import numpy as np

print("=" * 80)
print("QUICK TEST: Event-Driven Hybrid Control")
print("=" * 80)

# Test 1: Plant imports and instantiation
print("\n[1/5] Testing Plant Models...")
try:
    from src.plants import MotorDC, ThermalOven
    
    # Motor
    motor = MotorDC("config/motor_params.yaml")
    motor.reset()
    x = motor.step(5.0, 0.0)
    print(f"  ✓ Motor: step() → x = {x}")
    
    # Oven
    oven = ThermalOven("config/horno_params.yaml")
    oven.reset()
    x = oven.step(50.0, 0.0)
    print(f"  ✓ Oven: step() → x = {x}")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 2: Discrete Logic
print("\n[2/5] Testing Discrete Memory (Flip-Flops)...")
try:
    from src.discrete_logic import DiscreteMemoryManager, DiscreteLogic   
    mem = DiscreteMemoryManager("config/trigger_params.yaml")
    m = mem.logic.get_memory()
    print(f"  ✓ Initial state: {m} (normal={m[0]}, saturated={m[1]}, critical={m[2]})")
    
    # Simulate saturation
    for i in range(3):
        mem_info = mem.step(u=11.5, x=np.array([0, 0]), e_pred=0.5, margin=0.1)
    
    print(f"  ✓ After 3 saturated steps: {mem.logic.get_mode()}")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 3: Event Trigger
print("\n[3/5] Testing Event Trigger...")
try:
    from src.event_trigger import EventTrigger, AdaptiveTriggerManager
    
    trigger = AdaptiveTriggerManager("config/trigger_params.yaml")
    
    x = np.array([0.5, 1.0])
    y_pred = np.array([0.4, 0.9])
    margins = np.array([0.5, 1.5, 0.8, 2.0])
    
    result = trigger.step(x, y_pred, margins, 0, trigger_type="error")
    
    print(f"  ✓ E_value={result['E_value']:.4f}, η={result['eta']:.4f}, δ={result['delta']}")
    print(f"  ✓ Event rate: {result['event_rate']*100:.1f}%")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 4: Metrics Collection
print("\n[4/5] Testing Metrics Collector...")
try:
    from src.metrics import MetricsCollector, EpisodeMetrics
    
    metrics = MetricsCollector("motor", seed=42)
    
    # Log 10 steps
    for k in range(10):
        x = np.array([0.1*k, 0.2*k])
        x_ref = 1.0
        u = 2.0
        
        metrics.log_step(
            x=x, x_ref=x_ref, u=u,
            Q=np.array([[1.0]]), R=0.01,
            violation=False, violation_mag=0.0,
            cpu_time=0.005,
            triggered=(k % 3 == 0),
            y_pred=x + np.random.randn(2)*0.01
        )
    
    ep_metrics = metrics.finalize()
    print(f"  ✓ Episode length: {ep_metrics.episode_length} steps")
    print(f"  ✓ Cost: {ep_metrics.total_cost:.4f}")
    print(f"  ✓ Event rate: {ep_metrics.event_rate*100:.1f}%")
    print(f"  ✓ CPU mean: {ep_metrics.cpu_time_mean*1000:.2f}ms, p95: {ep_metrics.cpu_time_p95*1000:.2f}ms")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 5: Main Controller
print("\n[5/5] Testing Hybrid Event-Driven Controller...")
try:
    from src.controller_hybrid import HybridEventDrivenController
    
    controller = HybridEventDrivenController(
        plant_type="motor",
        config_dir="config/"
    )
    
    controller.reset(seed=42)
    
    # Run 20 steps
    for k in range(20):
        u = 3.0 * np.sin(k / 5.0)
        info = controller.step(u)
    
    final_metrics = controller.metrics.finalize()
    print(f"  ✓ Completed {final_metrics.episode_length} steps")
    print(f"  ✓ Tracking RMSE: {np.sqrt(final_metrics.tracking_error_mse):.4f}")
    print(f"  ✓ Constraint violations: {final_metrics.num_violations}")
    print(f"  ✓ Event rate: {final_metrics.event_rate*100:.1f}%")
    print(f"  ✓ CPU p95: {final_metrics.cpu_time_p95*1000:.2f}ms")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nThe core framework is ready. Next steps:")
print("  1. Implement MPC solver (src/mpc_solver.py)")
print("  2. Implement LSTM predictor (src/lstm_predictor.py)")
print("  3. Generate training data and run full experiments")
print("\nRun full pipeline with:")
print("  python experiments/train_lstm.py")
print("  python experiments/run_proposed.py")
print("\n" + "=" * 80)
