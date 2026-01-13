
import os
from pathlib import Path
import glob

def main():
    results_dir = Path("results")
    if not results_dir.exists():
        print("Results directory not found!")
        return

    # 1. Identify "baselines_combined" files which cause duplication in evaluate.py
    # evaluate.py loads all "results_{plant}_*.csv". 
    # run_baselines.py produces "results_{plant}_{baseline}.csv" AND "results_{plant}_baselines_combined.csv".
    combined_baselines = list(results_dir.glob("*_baselines_combined.csv"))
    
    print(f"Found {len(combined_baselines)} combined baseline files.")
    for f in combined_baselines:
        print(f"  Removing redundant file: {f.name}")
        f.unlink()

    # 2. Verify we have the proposed combined files
    proposed_combined = list(results_dir.glob("*_combined.csv"))
    # Note: *_combined.csv might match *_baselines_combined.csv if I hadn't deleted them. 
    # But I just deleted them.
    # Actually, proposed files are "results_{plant}_combined.csv" (from run_proposed.py logs).
    
    print("\nVerifying Proposed Method Results:")
    for plant in ['motor', 'oven']:
        expected = results_dir / f"results_{plant}_combined.csv"
        if expected.exists():
            print(f"  [OK] {expected.name}")
        else:
            print(f"  [MISSING] {expected.name} (Did run_proposed.py finish? Check subdirs?)")
            # If missing, try to find in subdirs and copy
            # run_proposed.py saves to subdirs like results/experiment_.../results_...csv
            # But line 505 logs said "Combined results saved: results\results_oven_combined.csv"
            # So it SHOULD be there.

    # 3. Verify Baselines (Individual)
    print("\nVerifying Individual Baselines:")
    baselines = ['B1_PeriodicMPC', 'B2_ClassicEMPC', 'B3_RLnoMemory']
    for plant in ['motor', 'oven']:
        for base in baselines:
            expected = results_dir / f"results_{plant}_{base}.csv"
            if expected.exists():
                print(f"  [OK] {expected.name}")
            else:
                 print(f"  [MISSING] {expected.name}")

if __name__ == "__main__":
    main()
