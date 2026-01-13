
import unittest
import numpy as np
import sys
import os
from pathlib import Path
from collections import deque

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.controller_hybrid import HybridEventDrivenController
from src.lstm_predictor import LSTMPredictor

# Mock Plant to avoid creating full configs
class MockPlant:
    def __init__(self):
        self.x = np.array([0.0, 0.0])
        self.u_prev = 0.0
        self.x = np.array([0.0, 0.0])
        self.u_prev = 0.0
        # Keys matching MotorDC
        self.const = {
            'u_min': -10.0, 'u_max': 10.0,
            'position_min': -100.0, 'position_max': 100.0,
            'velocity_min': -50.0, 'velocity_max': 50.0
        }
    
    def get_reference(self, step):
        return 0.0
    
    def step(self, u, dist=0.0):
        self.x = self.x + 0.1 # dummy dynamics
        self.u_prev = u
        return self.x
    
    def check_constraints(self):
        return True, {}
        
    def reset(self, x0=None):
        self.x = np.array([0.0, 0.0])
        self.u_prev = 0.0

class MockLSTMPredictor:
    def __init__(self):
        self.history_length = 5
        self.input_size = 15 # 5*(2+1)
        
    def predict(self, x):
        # x should be flat vector of size input_size
        if x.shape[0] != self.input_size:
            raise ValueError(f"Input shape mismatch: expected {self.input_size}, got {x.shape[0]}")
        return np.array([0.5, 0.5])

class TestBufferFix(unittest.TestCase):
    def setUp(self):
        # Create controller with mocks
        self.controller = HybridEventDrivenController("motor", "config")
        
        # Inject mocks
        self.controller.plant = MockPlant()
        self.controller.lstm_predictor = MockLSTMPredictor()
        
        # Manual reset to trigger buffer init
        self.controller.reset()
        
    def test_buffer_initialization(self):
        """Test buffer is created with correct size"""
        self.assertIsNotNone(self.controller.history_buffer)
        self.assertEqual(self.controller.history_buffer.maxlen, 5)
        self.assertEqual(len(self.controller.history_buffer), 0)
        
    def test_buffer_growth_and_prediction(self):
        """Test buffer fills up and triggers prediction correctly"""
        
        # Step 1-4: Buffer filling
        for k in range(4):
            # Step calls: get x_k -> predict (fail) -> ... -> append (x_k, u)
            info = self.controller.step(u=1.0)
            # Prediction fallback to x_k because buffer not full
            np.testing.assert_array_equal(info['y_pred'], info['x_k'])
            self.assertEqual(len(self.controller.history_buffer), k+1)
            
        # Step 5: Buffer full (len=5)
        info = self.controller.step(u=1.0)
        # Now prediction should work inside step because buffer was len=4 going in?
        # WAIT. Logic check:
        # Step start: read x_k.
        # Predict: needs buffer len == history_length.
        # Current buffer len is 4.
        # So predict still fallback?
        # End of step: append. Len becomes 5.
        
        self.assertEqual(len(self.controller.history_buffer), 5)
        # Prediction was likely fallback
        np.testing.assert_array_equal(info['y_pred'], info['x_k'])
        
        # Step 6: Buffer full entering step
        info = self.controller.step(u=1.0)
        # Now predict should use LSTM
        # Mock returns [0.5, 0.5]
        np.testing.assert_array_equal(info['y_pred'], np.array([0.5, 0.5]))
        self.assertEqual(len(self.controller.history_buffer), 5) # Maxlen constraint
        
    def test_variable_scope_fix(self):
        """Verify u_control variable name fix"""
        # If u_control variable name was wrong, this would raise UnboundLocalError
        try:
            self.controller.step(u=0.0)
        except UnboundLocalError as e:
            self.fail(f"Variable scope error: {e}")

if __name__ == '__main__':
    unittest.main()
