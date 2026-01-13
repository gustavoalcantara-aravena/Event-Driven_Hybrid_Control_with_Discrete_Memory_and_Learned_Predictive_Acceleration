"""
LSTM Temporal Predictor: PyTorch-based one-step-ahead prediction
Trained on synthetic trajectories, predicts next state from history
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List
import yaml
from pathlib import Path


class LSTMPredictorModel(nn.Module):
    """
    PyTorch LSTM for one-step-ahead state prediction
    Architecture: Input → LSTM → Dense → Output
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 32,
                 num_layers: int = 2,
                 output_size: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            input_size: dimension of input (position, velocity) * history_length
            hidden_size: LSTM hidden units
            num_layers: number of LSTM layers
            output_size: dimension of output (next state)
            dropout: dropout rate
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Output dense layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: input tensor [batch, seq_len, input_size] or [seq_len, input_size]
        
        Returns:
            output: predicted state [batch, output_size] or [output_size]
        """
        # Forward pass for Option B: Sequence -> Last Step -> Output
        
        # x shape: (Batch, SeqLen, InputSize)
        # Note: If x comes from DataLoader, it's already correct.
        
        # LSTM forward
        # output: (Batch, SeqLen, NumDirections*HiddenSize)
        lstm_out, _ = self.lstm(x)
        
        # Use output from last time step (Option B)
        # We want the state at time T+1 given history up to T
        last_step = lstm_out[:, -1, :]  # [Batch, Hidden]
        
        # Dense layers
        out = torch.relu(self.fc1(last_step))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class SequenceDataset(Dataset):
    """Dataset for LSTM training"""
    
    def __init__(self,
                 states: np.ndarray,
                 controls: np.ndarray,
                 references: np.ndarray,
                 history_length: int = 10):
        """
        Args:
            states: array of shape [num_episodes, episode_length, state_dim]
            controls: array of shape [num_episodes, episode_length, 1]
            references: array of shape [num_episodes, episode_length, state_dim]
            history_length: lookback window
        """
        self.history_length = history_length
        
        # Structure data as (SeqLen, Features)
        # Features = [State, Control]
        # Target = NextState
        
        self.sequences = []
        self.targets = []
        
        # Determine feature size
        # Assuming states and controls are 2D arrays (time, dim)
        
        for ep_idx in range(len(states)):
            x_seq = states[ep_idx]
            u_seq = controls[ep_idx]
            # References ignored for physics prediction logic
            
            for t in range(history_length, len(x_seq) - 1):
                # History window (t-H to t-1)
                # We need x_k and u_k for k in [t-H, t-1]
                
                # Slicing: [t-history_length : t] gives length H
                x_hist = x_seq[t-history_length:t]  # (H, Dx)
                u_hist = u_seq[t-history_length:t]  # (H, Du)
                
                # Concatenate features per time step: (H, Dx+Du)
                seq = np.concatenate([x_hist, u_hist], axis=1)
                
                target = x_seq[t]  # We predict CURRENT state x_t using history up to t-1?
                # Wait, paper says y_pred_{k|k-1} = x_k.
                # So we use x_{k-H}...x_{k-1} and u_{k-H}...u_{k-1} to predict x_k.
                
                self.sequences.append(seq)
                self.targets.append(target)
        
        self.sequences = np.array(self.sequences) # (N, H, Feat)
        self.targets = np.array(self.targets)     # (N, Out)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.sequences[idx]).float(),
            torch.from_numpy(self.targets[idx]).float()
        )


class LSTMPredictor:
    """
    High-level LSTM predictor with training and prediction interface
    """
    
    def __init__(self,
                 config_path: str = "config/lstm_config.yaml",
                 device: str = "cpu"):
        """
        Initialize predictor
        
        Args:
            config_path: path to LSTM config
            device: "cpu" or "cuda"
        """
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        self.config = cfg
        self.device = torch.device(device)
        
        # Architecture params
        input_cfg = self.config.get('input', {})
        arch_cfg = self.config.get('architecture', {})
        
        self.history_length = int(input_cfg.get('history_window_size', 10))
        self.input_size = None  # Set after first call to train
        self.hidden_size = int(arch_cfg.get('hidden_units', 32))
        self.num_layers = int(arch_cfg.get('num_layers', 2))
        self.output_size = int(arch_cfg.get('output_size', 2))
        self.dropout = float(arch_cfg.get('dropout_rate', 0.1))
        
        # Model
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        # Statistics for normalization
        self.mean = None
        self.std = None
        self.normalization_params_frozen = False
        
        # History for training
        self.train_losses = []
        self.val_losses = []
        
    def _build_model(self, input_size: int):
        """Build model with given input size"""
        if self.model is None:
            self.input_size = input_size
            self.model = LSTMPredictorModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=self.output_size,
                dropout=self.dropout
            ).to(self.device)
            
            # Optimizer
            train_cfg = self.config.get('training', {})
            lr = float(train_cfg.get('learning_rate', 1e-3))
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr
            )
    
    def train_on_trajectories(self,
                             states_list: List[np.ndarray],
                             controls_list: List[np.ndarray],
                             references_list: List[np.ndarray],
                             train_fraction: float = 0.70) -> Dict:
        """
        Train LSTM on synthetic trajectory data
        
        Args:
            states_list: list of state trajectories [num_episodes, [len_k, state_dim]]
            controls_list: list of control trajectories
            references_list: list of reference trajectories
            train_fraction: fraction for training (rest is validation)
        
        Returns:
            {train_loss, val_loss, epochs}
        """
        # Create dataset
        states_arr = np.array(states_list)
        controls_arr = np.array(controls_list)
        references_arr = np.array(references_list)
        
        dataset = SequenceDataset(
            states_arr,
            controls_arr,
            references_arr,
            history_length=self.history_length
        )
        
        # Build model
        # Build model
        # Input size is the feature dimension (Dx + Du)
        # sequences shape: (N, H, Feat)
        self._build_model(dataset.sequences.shape[2])
        
        # Compute normalization statistics
        # Mean/Std per feature across all steps and samples
        # reshape to (-1, Feat) to compute stats
        flat_seqs = dataset.sequences.reshape(-1, dataset.sequences.shape[2])
        self.mean = np.mean(flat_seqs, axis=0)
        self.std = np.std(flat_seqs, axis=0) + 1e-6
        
        # Target statistics
        self.target_mean = np.mean(dataset.targets, axis=0)
        self.target_std = np.std(dataset.targets, axis=0) + 1e-6
        
        self.normalization_params_frozen = True
        
        # Split dataset
        n_train = int(len(dataset) * train_fraction)
        n_val = len(dataset) - n_train
        
        train_data, val_data = torch.utils.data.random_split(
            dataset,
            [n_train, n_val]
        )
        
        train_loader = DataLoader(
            train_data,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_data,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )
        
        # Training loop
        train_cfg = self.config.get('training', {})
        num_epochs = int(train_cfg.get('num_epochs', 100))
        batch_size = int(train_cfg.get('batch_size', 64))
        
        early_stop_cfg = train_cfg.get('early_stopping', {})
        patience = int(early_stop_cfg.get('patience', 10))
        min_delta = float(early_stop_cfg.get('min_delta', 1e-4))
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train epoch
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self._validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{num_epochs}: "
                      f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return {
            'train_loss': float(np.mean(self.train_losses)),
            'val_loss': float(np.mean(self.val_losses[-10:])),
            'epochs': epoch + 1
        }
    
    def _train_epoch(self, loader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Normalize Inputs
            # batch_x: (Batch, Seq, Feat)
            # mean/std: (Feat,)
            mean_tensor = torch.from_numpy(self.mean).float().to(self.device)
            std_tensor = torch.from_numpy(self.std).float().to(self.device)
            
            batch_x_norm = (batch_x - mean_tensor) / std_tensor
            
            # Normalize Targets
            # batch_y: (Batch, Out)
            target_mean_tensor = torch.from_numpy(self.target_mean).float().to(self.device)
            target_std_tensor = torch.from_numpy(self.target_std).float().to(self.device)
            
            batch_y_norm = (batch_y - target_mean_tensor) / target_std_tensor
            
            # Forward
            self.optimizer.zero_grad()
            y_pred_norm = self.model(batch_x_norm)
            
            # Assert shapes
            assert y_pred_norm.shape == batch_y_norm.shape, \
                f"Shape mismatch: {y_pred_norm.shape} vs {batch_y_norm.shape}"
            
            loss = self.criterion(y_pred_norm, batch_y_norm)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Log raw loss (scaled back? No, keep normalized loss per user advice)
            total_loss += loss.item() * len(batch_x)
        
        return total_loss / len(loader.dataset)
    
    def _validate_epoch(self, loader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Normalize Inputs
                mean_tensor = torch.from_numpy(self.mean).float().to(self.device)
                std_tensor = torch.from_numpy(self.std).float().to(self.device)
                batch_x_norm = (batch_x - mean_tensor) / std_tensor
                
                # Normalize Targets
                target_mean_tensor = torch.from_numpy(self.target_mean).float().to(self.device)
                target_std_tensor = torch.from_numpy(self.target_std).float().to(self.device)
                batch_y_norm = (batch_y - target_mean_tensor) / target_std_tensor
                
                y_pred_norm = self.model(batch_x_norm)
                loss = self.criterion(y_pred_norm, batch_y_norm)
                total_loss += loss.item() * len(batch_x)
        
        return total_loss / len(loader.dataset)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict next state
        
        Args:
            x: feature vector [input_size,] (history concatenated)
        
        Returns:
            y_pred: predicted state [output_size,]
        """
        self.model.eval()
        
        # Handle Flat Vector Input from legacy/external calls
        # We need to reshape the flat vector into (1, SeqLen, Features)
        # Expected flat size = HistoryLength * InputSize
        # Where InputSize = Feature Dimension (Dx + Du)
        
        # Infer dimensions
        feature_dim = self.input_size
        hist_len = self.history_length
        # Note: if x contains Reference (old style), we must drop it?
        # Current fix assumes x matches the new Feature structure [x, u]
        # If acceleration.py passes [x, u, r], we have a problem.
        # Minimal assumption: x is just flattened [x_seq, u_seq].
        
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().to(self.device)
            
            # Reshape logic
            if x.ndim == 1:
                # Assuming simple flat: [h*feat]
                # Check divisibility
                if x.shape[0] % feature_dim == 0:
                     seq_len = x.shape[0] // feature_dim
                     if seq_len == hist_len:
                         # Reshape to (1, Seq, Feat)
                         x_tensor = x_tensor.view(1, seq_len, feature_dim)
                     else:
                         # Fallback or error? pass as is if weird?
                         print(f"Warning: Predict input shape {x.shape} doesn't match history {hist_len} * feat {feature_dim}")
                         x_tensor = x_tensor.view(1, -1, feature_dim) # try best guess
                else:
                    # Maybe it's already structured?
                    x_tensor = x_tensor.unsqueeze(0).unsqueeze(1) # Treat as 1 step?
            
            # Normalize
            if self.mean is not None:
                mean_tensor = torch.from_numpy(self.mean).float().to(self.device)
                std_tensor = torch.from_numpy(self.std).float().to(self.device)
                x_norm = (x_tensor - mean_tensor) / std_tensor
            else:
                x_norm = x_tensor
            
            y_pred_norm = self.model(x_norm)
            
            # Denormalize output
            if getattr(self, 'target_mean', None) is not None:
                t_mean = torch.from_numpy(self.target_mean).float().to(self.device)
                t_std = torch.from_numpy(self.target_std).float().to(self.device)
                y_pred = y_pred_norm * t_std + t_mean
            else:
                y_pred = y_pred_norm
            
            # Squeeze batch
            if y_pred.shape[0] == 1:
                y_pred = y_pred.squeeze(0)
        
        return y_pred.cpu().numpy()
    
    def save(self, filepath: str):
        """Save model to disk"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state': self.model.state_dict(),
            'mean': self.mean,
            'std': self.std,
            'target_mean': getattr(self, 'target_mean', None),
            'target_std': getattr(self, 'target_std', None),
            'config': self.config,
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Restore config
        self.config = checkpoint['config']
        
        # Build model
        if self.model is None:
            # Infer input size from mean
            input_size = len(checkpoint['mean'])
            self._build_model(input_size)
        
        # Restore weights
        self.model.load_state_dict(checkpoint['model_state'])
        self.mean = checkpoint['mean']
        self.std = checkpoint['std']
        self.target_mean = checkpoint.get('target_mean')
        self.target_std = checkpoint.get('target_std')
        self.normalization_params_frozen = True
        
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    print("Testing LSTM Predictor...")
    
    # Create dummy training data
    num_episodes = 100
    episode_length = 500
    state_dim = 2
    
    states_list = [np.random.randn(episode_length, state_dim) for _ in range(num_episodes)]
    controls_list = [np.random.randn(episode_length, 1) for _ in range(num_episodes)]
    references_list = [np.ones((episode_length, state_dim)) for _ in range(num_episodes)]
    
    # Train
    predictor = LSTMPredictor()
    result = predictor.train_on_trajectories(
        states_list, controls_list, references_list, train_fraction=0.8
    )
    
    print(f"✓ LSTM trained: {result}")
    
    # Predict
    x_test = np.random.randn(predictor.input_size)
    y_pred = predictor.predict(x_test)
    print(f"✓ Prediction: input shape {x_test.shape}, output shape {y_pred.shape}")
