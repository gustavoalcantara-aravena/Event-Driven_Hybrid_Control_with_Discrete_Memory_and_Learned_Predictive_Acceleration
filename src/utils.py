"""
Utility functions: Normalization, seeding, logging
"""

import os
import sys
import random
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Any
import json
from datetime import datetime


class Normalizer:
    """
    Z-score normalization (standardization)
    Maps data to mean=0, std=1
    """
    
    def __init__(self, 
                 mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None):
        """
        Initialize with pre-computed statistics
        
        Args:
            mean: per-feature mean
            std: per-feature standard deviation
        """
        self.mean = mean
        self.std = std
    
    def fit(self, data: np.ndarray) -> 'Normalizer':
        """
        Compute statistics from data
        
        Args:
            data: array [n_samples, n_features]
        
        Returns:
            self for chaining
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8  # Avoid division by zero
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data
        
        Args:
            data: array to normalize
        
        Returns:
            normalized data
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        De-normalize data
        
        Args:
            data: normalized data
        
        Returns:
            original scale data
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        return data * self.std + self.mean
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            data: array to fit and normalize
        
        Returns:
            normalized data
        """
        self.fit(data)
        return self.transform(data)
    
    def save(self, filepath: str):
        """Save normalization statistics"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(filepath,
                 mean=self.mean,
                 std=self.std)
        
        print(f"Normalizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load normalization statistics"""
        data = np.load(filepath)
        self.mean = data['mean']
        self.std = data['std']
        
        print(f"Normalizer loaded from {filepath}")
    
    def __repr__(self) -> str:
        if self.mean is None:
            return "Normalizer(not fitted)"
        return f"Normalizer(mean={self.mean.shape}, std={self.std.shape})"


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed: random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class Logger:
    """
    Structured logging for experiment tracking
    """
    
    def __init__(self,
                 name: str = "experiment",
                 log_dir: str = "logs",
                 level: int = logging.INFO):
        """
        Initialize logger
        
        Args:
            name: logger name
            log_dir: directory for log files
            level: logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def debug(self, msg: str, **kwargs):
        """Log debug message"""
        self.logger.debug(msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        """Log info message"""
        self.logger.info(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message"""
        self.logger.warning(msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log error message"""
        self.logger.error(msg, **kwargs)
    
    def log_metrics(self, metrics: dict, prefix: str = ""):
        """Log metrics dictionary"""
        msg = prefix if prefix else "Metrics:"
        msg += " | ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in metrics.items()])
        self.info(msg)
    
    def log_config(self, config: dict, name: str = "Config"):
        """Log configuration dictionary"""
        self.info(f"\n{name}:")
        self.info(json.dumps(config, indent=2, default=str))


class ExperimentTracker:
    """
    Track experiment metadata and results
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize tracker
        
        Args:
            log_dir: directory for metadata
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata = {
            'start_time': datetime.now().isoformat(),
            'experiments': []
        }
    
    def log_experiment(self,
                      name: str,
                      config: dict,
                      results: dict):
        """
        Log single experiment
        
        Args:
            name: experiment identifier
            config: configuration dict
            results: results dict
        """
        exp_record = {
            'name': name,
            'config': config,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.metadata['experiments'].append(exp_record)
    
    def save(self, filepath: Optional[str] = None):
        """
        Save metadata to JSON
        
        Args:
            filepath: output path (default: logs/metadata.json)
        """
        if filepath is None:
            filepath = self.log_dir / "metadata.json"
        
        self.metadata['end_time'] = datetime.now().isoformat()
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        print(f"Experiment metadata saved to {filepath}")


def merge_dicts(*dicts) -> dict:
    """
    Merge multiple dictionaries (later keys override earlier ones)
    
    Args:
        *dicts: variable number of dict arguments
    
    Returns:
        merged dictionary
    """
    result = {}
    for d in dicts:
        if d is not None:
            result.update(d)
    return result


def dict_to_namespace(d: dict):
    """
    Convert dict to namespace object for attribute access
    
    Args:
        d: dictionary
    
    Returns:
        namespace object
    """
    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def __repr__(self):
            return str(self.__dict__)
    
    return Namespace(**d)


def format_time(seconds: float) -> str:
    """
    Format seconds to readable string
    
    Args:
        seconds: number of seconds
    
    Returns:
        formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


# Example usage
if __name__ == "__main__":
    print("Testing utils...")
    
    # Test normalizer
    data = np.random.randn(100, 5)
    normalizer = Normalizer()
    normalized = normalizer.fit_transform(data)
    
    print(f"✓ Normalizer: mean={np.mean(normalized, axis=0)}, "
          f"std={np.std(normalized, axis=0)}")
    
    # Test seeding
    set_seed(42)
    a = np.random.randn(5)
    
    set_seed(42)
    b = np.random.randn(5)
    
    print(f"✓ Seeding: arrays equal = {np.allclose(a, b)}")
    
    # Test logger
    logger = Logger(name="test")
    logger.info("Test message")
    logger.log_metrics({'loss': 0.123, 'accuracy': 0.95})
    
    print(f"✓ Logger created")
    
    # Test experiment tracker
    tracker = ExperimentTracker()
    tracker.log_experiment(
        "exp1",
        {'lr': 0.001},
        {'accuracy': 0.95}
    )
    
    print(f"✓ Experiment tracker created")


def generate_disturbance_profile(steps: int, seed: int = None) -> np.ndarray:
    """
    Generate random disturbance profile
    
    Args:
        steps: experiment length
        seed: random seed
    
    Returns:
        disturbance: array [-1, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Match run_baselines.py: 0.01 * randn() with 5% probability
    profile = np.zeros(steps)
    for i in range(steps):
        if np.random.rand() < 0.05:
            profile[i] = 0.01 * np.random.randn()
    
    return profile

