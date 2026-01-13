
import numpy as np
import casadi as ca
from src.mpc_solver import MPCSolver

def diagnose():
    print("Starting CasADi Solver Diagnosis...")
    solver = MPCSolver("motor")
    
    # Manually trigger building the solver
    solver._build_solver()
    
    x0 = np.array([0.0, 0.0])
    ref = np.array([1.5708, 0.0])
    p_vals = np.concatenate([x0, ref])
    x_init = np.zeros(solver.nu * solver.N)
    
    print("\nAttempting solver.solve()...")
    try:
        u_opt, converged = solver.solve(
            x0=x0,
            ref=ref
        )
        print(f"Solver result: u={u_opt:.4f}, converged={converged}")
        if u_opt != 0.0:
            print("SUCCESS: Non-zero control applied.")
        else:
            print("WARNING: Control is still zero. Check model/cost.")
    except Exception as e:
        print("\n!!! Solver Call Failed !!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose()
