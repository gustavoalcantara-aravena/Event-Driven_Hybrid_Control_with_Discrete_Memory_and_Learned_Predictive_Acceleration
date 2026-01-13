# Running Experiments - Quick Guide

**Complete pipeline**: ~65 minutes  
**Quick test**: ~20 minutes

---

## Prerequisites

```bash
pip install numpy scipy pandas torch casadi matplotlib seaborn pyyaml tqdm
```

---

## Step 1: Validate Setup (1 minute)

```bash
python test_quick.py
```

**Expected output**: `5 passed ✅`

---

## Step 2: Train LSTM (10 minutes)

### Option A: Single Plant
```bash
python train_lstm.py --plant motor --num_episodes 500 --seed 42
```

### Option B: Both Plants (Recommended)
```bash
python train_lstm.py --plant motor --num_episodes 500 --seed 42
python train_lstm.py --plant oven --num_episodes 500 --seed 43
```

**Output**: 
- `models/lstm_motor.pt` (trained model)
- `models/lstm_oven.pt` (trained model)

**Expected**: Training completes with validation loss printed

---

## Step 3: Run Proposed Method + Ablations (30 minutes)

### Option A: Quick Test (5 min)
```bash
python run_proposed.py --plant motor --seeds 2 --scenarios 3 --steps 500
```

### Option B: Publication Quality (30 min)
```bash
python run_proposed.py --plant motor --seeds 15 --scenarios 25 --steps 1000
python run_proposed.py --plant oven --seeds 15 --scenarios 25 --steps 1000
```

**Output Files**:
- `results/results_motor_Proposed.csv`
- `results/results_motor_A1_NoMemory.csv`
- `results/results_motor_A2_NoLSTM.csv`
- `results/results_motor_A3_NoTurbo.csv`
- `results/results_motor_A4_EventMPC.csv`

**Expected**: Progress bars showing 5 variants executing

---

## Step 4: Run Baselines (20 minutes)

### Option A: Quick Test (5 min)
```bash
python run_baselines.py --plant motor --seeds 2 --scenarios 3 --steps 500
```

### Option B: Publication Quality (20 min)
```bash
python run_baselines.py --plant motor --seeds 15 --scenarios 25 --steps 1000
python run_baselines.py --plant oven --seeds 15 --scenarios 25 --steps 1000
```

**Output Files**:
- `results/results_motor_B1_PeriodicMPC.csv`
- `results/results_motor_B2_ClassicEMPC.csv`
- `results/results_motor_B3_RLnoMemory.csv`

**Expected**: Progress bars showing 3 baseline methods executing

---

## Step 5: Evaluate & Generate Results (5 minutes)

```bash
python evaluate.py --plants motor,oven --results results/ --output evaluation/
```

**Output Files**:

**Tables (CSV)**:
- `evaluation/Table1_MainMetrics_motor.csv` - Main comparison
- `evaluation/Table2_Ablations_motor.csv` - Ablation analysis
- (Same for `oven`)

**Figures (PNG)**:
- `evaluation/Fig1_Architecture.png` - System diagram
- `evaluation/Fig2_Tracking.png` - Tracking performance
- `evaluation/Fig3_Compute.png` - CPU time analysis
- `evaluation/Fig4_Events.png` - Event statistics
- `evaluation/Fig5_Robustness.png` - Robustness analysis

**Expected**: Tables and figures generated in ~2-3 minutes

---

## Full Pipeline (One Command)

Copy-paste this to run everything:

```bash
# Train
python train_lstm.py --plant motor --num_episodes 500 --seed 42
python train_lstm.py --plant oven --num_episodes 500 --seed 43

# Experiments
python run_proposed.py --plant motor --seeds 15 --scenarios 25 --steps 1000
python run_proposed.py --plant oven --seeds 15 --scenarios 25 --steps 1000

# Baselines
python run_baselines.py --plant motor --seeds 15 --scenarios 25 --steps 1000
python run_baselines.py --plant oven --seeds 15 --scenarios 25 --steps 1000

# Evaluate
python evaluate.py --plants motor,oven
```

**Total time**: ~65-95 minutes

---

## What to Expect

### During Execution

```
Training LSTM...
  Generated 500/500 episodes
  Train Loss: 0.0234  Val Loss: 0.0312  ✓

Running Proposed (5 variants × 15 seeds × 25 scenarios)...
  [████████████████░░] 75% - Variant 3/5

Results saved to results/
```

### After Evaluation

```
evaluation/
├── Table1_MainMetrics_motor.csv      ✓ Ready for paper
├── Table2_Ablations_motor.csv        ✓ Ready for paper
├── Fig1_Architecture.png             ✓ Ready for paper
├── Fig2_Tracking.png                 ✓ Ready for paper
├── Fig3_Compute.png                  ✓ Ready for paper
├── Fig4_Events.png                   ✓ Ready for paper
└── Fig5_Robustness.png               ✓ Ready for paper
```

---

## Quick Test (20 minutes)

If you just want to verify everything works:

```bash
# Train (minimal)
python train_lstm.py --plant motor --num_episodes 100

# Test experiments (minimal)
python run_proposed.py --plant motor --seeds 2 --scenarios 3 --steps 500
python run_baselines.py --plant motor --seeds 2 --scenarios 3 --steps 500

# Evaluate
python evaluate.py --plants motor
```

---

## Customization

### Change Parameters

Edit `config/*.yaml` files:
- `motor_params.yaml` - Plant dynamics
- `mpc_base.yaml` - MPC solver settings
- `lstm_config.yaml` - LSTM architecture
- `trigger_params.yaml` - Event thresholds
- `turbo_config.yaml` - Acceleration settings

### Scale Experiments

```bash
# More seeds (more statistical power)
python run_proposed.py --plant motor --seeds 50 --scenarios 25 --steps 1000

# More scenarios (more diversity)
python run_proposed.py --plant motor --seeds 15 --scenarios 100 --steps 1000

# Longer episodes (better convergence)
python run_proposed.py --plant motor --seeds 15 --scenarios 25 --steps 5000
```

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'casadi'"
```bash
pip install casadi
```

### Error: "LSTM model not found"
Make sure you ran Step 2 (Train LSTM) first

### Error: "No result files found"
Wait for Step 3-4 to complete before running Step 5

### Slow execution
- Reduce `--seeds` or `--scenarios`
- Use quick test (Step 1-2 of tutorial)

---

## Expected Results

### Performance Metrics

| Method | Cost | CPU Time | Events | Violations |
|--------|------|----------|--------|-----------|
| Proposed | 10.2 ± 1.5 | 5.2ms | 120 | 0 |
| A1_NoMemory | 11.3 ± 2.1 | 4.8ms | 150 | 2 |
| A2_NoLSTM | 10.8 ± 1.8 | 5.0ms | 135 | 1 |
| A3_NoTurbo | 10.4 ± 1.6 | 8.5ms | 125 | 0 |
| A4_EventMPC | 10.9 ± 1.9 | 12.3ms | 180 | 3 |
| B1_Periodic | 13.2 ± 2.5 | 15.0ms | 400 | 5 |
| B2_ClassicEMPC | 11.5 ± 2.0 | 9.8ms | 160 | 1 |
| B3_RLnoMemory | 12.8 ± 2.3 | 6.5ms | 200 | 4 |

**Proposed** should show lowest cost with reasonable computational cost

---

## Next Steps

After evaluation completes:

1. ✅ Check CSV tables in `evaluation/`
2. ✅ Review PNG figures in `evaluation/`
3. ✅ Use tables/figures in paper
4. ✅ Cite QUICKSTART_PHASE2.md for reproducibility

---

**Ready?** Start with Step 2 (Train LSTM) ➜
