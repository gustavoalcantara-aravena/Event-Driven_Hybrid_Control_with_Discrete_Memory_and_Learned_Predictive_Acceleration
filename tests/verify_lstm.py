
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
# sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lstm_predictor import LSTMPredictor
from src.plants import create_plantaset

def verify_lstm():
    print("=== LSTM VERIFICATION (OPTION B) ===")
    
    # 1. Create Dummy Data
    N = 100 # episodes
    L = 50  # length
    H = 10  # history
    Dx = 2  # state dim
    Du = 1  # control dim
    
    states = [np.random.randn(L, Dx) for _ in range(N)]
    controls = [np.random.randn(L, Du) for _ in range(N)]
    refs = [np.zeros((L, Dx))] * N # Ignored now
    
    print(f"Data: {N} episodes, Length {L}, History {H}")
    print(f"Features: State({Dx}) + Control({Du}) = {Dx+Du}")
    
    # 2. Test Dataset
    dataset = SequenceDataset(states, controls, refs, history_length=H)
    print(f"\n[Dataset Check]")
    print(f"Dataset length: {len(dataset)}")
    sample_x, sample_y = dataset[0]
    print(f"Sample X shape: {sample_x.shape} (Expected: ({H}, {Dx+Du}))")
    print(f"Sample Y shape: {sample_y.shape} (Expected: ({Dx},))")
    
    assert sample_x.shape == (H, Dx+Du), "Dataset X shape mismatch"
    assert sample_y.shape == (Dx,), "Dataset Y shape mismatch"
    print("✓ Dataset shapes correct")
    
    # Check content logic (briefly)
    # x_seq[0] should utilize x[0..H-1] and u[0..H-1]
    # y_seq[0] should be x[H-1]? No, x[H]?
    # In code: target = x_seq[t]. Loop t from H to L-1.
    # First item: t=H. x_hist = x[0:H]. target = x[H].
    # So Y is indeed the NEXT state after the history window. Option B correct.
    
    # 3. Test Training Loop & Normalization
    print(f"\n[Training Check]")
    predictor = LSTMPredictor(config_path="config/lstm_config.yaml") # Will verify config load
    # Force minimal config for test if file doesn't match? 
    # Usually config already exists. We assume standard config.
    
    results = predictor.train_on_trajectories(states, controls, refs, train_fraction=0.8)
    print(f"Training results: {results}")
    
    assert results['train_loss'] < 10.0, "Loss abnormally high (check scale)"
    # With normalization, MSE should be low. 
    # random noise data -> MSE ~ 1.0 (normalized)?
    # Since data is roughly N(0,1), mean=0, std=1. 
    # Constant prediction 0 gives MSE=1. Perfect=0.
    
    # 4. Test Prediction (Standard)
    print(f"\n[Prediction Check - Sequence]")
    # Input: (1, H, Feat)
    x_test_seq = torch.randn(1, H, Dx+Du).numpy()
    # But predict() expects flat vector if legacy??
    # Wait, predict() accepts whatever.
    # If we pass structured (1,H,F), let's see.
    # predict() expects "feature vector [input_size,] (history concatenated)" per docstring.
    # But updated code handles shapes.
    # If we pass (H, Feat) directly (as single sample)?
    # x_tensor shape (H, Feat). ndim=2. 
    # Logic in predict: if ndim=1... else...
    # The updated predict code says:
    # "if x.ndim == 1: reshape..."
    # "else: assumes (Batch, Seq, Feat)?" No, normalize treats x_tensor.
    # If we pass (H, Feat), normalization (FeatureDim) will allow broadcasting?
    # Mean is (Feat,). x is (H, Feat). (H,Feat)-(Feat,) works.
    # Model expects (Batch, Seq, Feat).
    # If input is (H, Feat), batch dimension is missing?
    # The code doesn't auto-unsqueeze dim 0 for 2D input in predict?
    # Let's check predict code again:
    # "if x_norm.ndim == 1: x_norm = x_norm.unsqueeze(0)" -> Adds Batch? No, adds 'Seq'?
    # Wait, if x is Flat (dim=1), it processes it.
    # We should support verifying Flat Input (Legacy/LAPA support).
    
    flat_len = H * (Dx+Du)
    x_flat = np.random.randn(flat_len)
    print(f"Predicting with Flat Vector (Legacy) of shape ({flat_len},)")
    y_pred = predictor.predict(x_flat)
    print(f"Prediction output shape: {y_pred.shape}")
    assert y_pred.shape == (Dx,), f"Output shape mismatch: {y_pred.shape}"
    print("✓ Flat input handling works")
    
    # 5. Check Normalization Logic
    print(f"\n[Normalization Check]")
    print(f"Input Mean shape: {predictor.mean.shape}")
    print(f"Target Mean shape: {predictor.target_mean.shape}")
    assert predictor.mean.shape == (Dx+Du,), "Input mean shape mismatch"
    assert predictor.target_mean.shape == (Dx,), "Target mean shape mismatch"
    print("✓ Normalization stats shapes correct")

    print("\n✅ ALL VERIFICATION CHECKS PASSED")

if __name__ == "__main__":
    verify_lstm()
