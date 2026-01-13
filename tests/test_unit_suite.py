
import unittest
import numpy as np
import os
import sys
import shutil
from pathlib import Path

# Add src to path
# sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plants import create_plant, MotorDC, ThermalOven
from src.lstm_predictor import LSTMPredictor, SequenceDataset
from src.mpc_solver import MPCSolver
from src.controller_hybrid import HybridEventDrivenController
import torch

class TestPlants(unittest.TestCase):
    def test_motordc_init_and_step(self):
        plant = create_plant("motor", "config/motor_params.yaml")
        self.assertIsInstance(plant, MotorDC)
        self.assertEqual(plant.u_prev, 0.0)
        
        x0 = plant.x.copy()
        x_next = plant.step(u=1.0)
        self.assertEqual(x_next.shape, (2,))
        self.assertNotEqual(plant.u_prev, 0.0) # Should update u_prev
        
    def test_oven_init_and_step(self):
        plant = create_plant("oven", "config/oven_params.yaml")
        self.assertIsInstance(plant, ThermalOven)
        self.assertEqual(plant.u_prev, 0.0)
        
        # Test step matches signature
        x_next = plant.step(u=50.0)
        self.assertEqual(x_next.shape, (2,))
        self.assertEqual(plant.u_prev, 50.0)

class TestLSTM(unittest.TestCase):
    def setUp(self):
        # Create minimal config for testing
        self.config_path = "config/lstm_config.yaml"
        self.predictor = LSTMPredictor(self.config_path)

    def test_dataset_shapes(self):
        # Mock data: 10 episodes, length 20, state_dim 2, control_dim 1
        states = [np.random.randn(20, 2) for _ in range(10)]
        controls = [np.random.randn(20, 1) for _ in range(10)]
        
        dataset = SequenceDataset(states, controls, history_length=5, references=None)
        self.assertGreater(len(dataset), 0)
        
        x, y = dataset[0]
        # x shape: (seq_len, features) -> (5, 3) where 3 = state(2)+control(1)
        self.assertEqual(x.shape, (5, 3))
        # y shape: (output_dim,) -> (2,) for state prediction
        self.assertEqual(y.shape, (2,))

    def test_predict_flow(self):
        # Force build model
        self.predictor._build_model(input_size=3) 
        # Fake normalization params
        self.predictor.mean = np.zeros(3)
        self.predictor.std = np.ones(3)
        self.predictor.target_mean = np.zeros(2)
        self.predictor.target_std = np.ones(2)
        
        # Flattened input test (history=5 * feat=3 = 15)
        observation = np.random.randn(15)
        y_pred = self.predictor.predict(observation)
        self.assertEqual(y_pred.shape, (2,))

class TestMPCSolver(unittest.TestCase):
    def test_solver_init_oven(self):
        # Should accept "oven"
        solver = MPCSolver("oven", "config/mpc_base.yaml")
        self.assertEqual(solver.plant_type, "oven")
        self.assertEqual(solver.cost_cfg['Q_temperature'], 1.0) # Check config load

    def test_solve_return_type(self):
        solver = MPCSolver("motor", "config/mpc_base.yaml")
        x0 = np.zeros(2)
        ref = np.array([1.0, 0.0])
        u, converged = solver.solve(x0, ref)
        self.assertIsInstance(u, float)
        self.assertIsInstance(converged, bool)

class TestControllerHybrid(unittest.TestCase):
    def test_controller_ablations(self):
        # Test with NO memory (Ablation A1)
        ctrl = HybridEventDrivenController("motor", "config")
        ctrl.memory_manager = None 
        
        ctrl.reset() # Should not crash
        info = ctrl.step(u=0.0)
        self.assertEqual(info['mode'], 'nominal') # Default fallback
        
    def test_controller_no_trigger(self):
        ctrl = HybridEventDrivenController("motor", "config")
        ctrl.trigger_manager = None
        
        ctrl.reset()
        info = ctrl.step(u=0.0)
        self.assertEqual(info['delta_k'], 1) # Default fallback

if __name__ == '__main__':
    unittest.main()
