# Event-Driven Hybrid Control with Discrete Memory and Learned Predictive Acceleration

## 🏫 Academic Research

This project is part of a research initiative developed by the **University of Santiago de Chile (USACH)**, in the field of automatic control and hybrid systems.

**This work was supported by a research grant 2025 of the Technological Faculty, USACH**

---

## 📋 Project Description

This repository contains the implementation of an event-driven hybrid controller that integrates:

- **Model Predictive Control (MPC)**: Real-time trajectory optimization
- **Discrete Logic**: State machines for control decisions
- **LSTM Prediction**: Recurrent neural network for system dynamics prediction
- **Learned Acceleration**: Learning techniques to optimize computational performance
- **Event Triggering**: Efficient mechanism to reduce communication and unnecessary computations

The system was validated on two test plants:
- **DC Motor**: Speed control system
- **Thermal Oven**: Temperature regulation system

---

## 🎯 Key Features

✅ **Efficient Hybrid Control**: Combination of continuous MPC with discrete logic  
✅ **Neural Prediction**: LSTM trained to capture complex dynamics  
✅ **Event Triggering**: Reduces computational load through selective activation  
✅ **Learned Acceleration**: Parameter optimization using ML techniques  
✅ **Experimental Validation**: Results on real plants  
✅ **Reproducibility**: Modular and well-documented code  

---

## 📁 Project Structure

```
.
├── src/                          # Main controller modules
│   ├── controller_hybrid.py       # Main hybrid controller
│   ├── mpc_solver.py             # MPC solver (CasADi)
│   ├── lstm_predictor.py         # LSTM predictor
│   ├── event_trigger.py          # Event triggering mechanism
│   ├── discrete_logic.py         # Discrete logic and state machines
│   ├── acceleration.py           # Learned acceleration
│   ├── plants.py                 # Plant models (Motor, Oven)
│   ├── metrics.py                # Evaluation metrics
│   └── utils.py                  # General utilities
│
├── config/                       # Configuration files
├── models/                       # Trained LSTM models
├── results/                      # Experiment results
├── trajectories/                 # Captured trajectories
├── tests/                        # Test suite
├── evaluation/                   # Evaluation scripts
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
├── run_proposed.py               # Script to run proposed controller
├── run_baselines.py              # Script to run baselines
└── evaluate.py                   # Complete evaluation script
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/gustavoalcantara-aravena/Event-Driven_Hybrid_Control_with_Discrete_Memory_and_Learned_Predictive_Acceleration.git
cd Event-Driven_Hybrid_Control_with_Discrete_Memory_and_Learned_Predictive_Acceleration.git
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Run the Proposed Controller
```bash
python run_proposed.py --plant motor --duration 100 --config config/motor_config.yaml
```

### Run Baselines
```bash
python run_baselines.py --plant oven --duration 100
```

### Complete Evaluation
```bash
python evaluate.py --compare-all
```

### Unit Tests
```bash
pytest tests/ -v
```

---

## 📊 Main Results

The proposed controller demonstrates:

- **Event Reduction**: ~40-60% fewer triggers than periodic MPC
- **Performance Improvement**: Better reference tracking vs baselines
- **Computational Efficiency**: Lower CPU load while maintaining control quality
- **Robustness**: Consistent performance under perturbations

See `results/` for detailed plots and tables.

---

## 📚 Main Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| NumPy | ≥1.24 | Numerical computing |
| SciPy | ≥1.10 | Scientific algorithms |
| CasADi | ≥3.5.5 | Optimization and MPC |
| PyTorch | ≥2.0.0 | LSTM neural network |
| Matplotlib | ≥3.7 | Visualization |
| Pandas | ≥2.0 | Data analysis |

---

## 🧪 Testing

The project includes a comprehensive test suite:

```bash
# Quick tests
pytest tests/test_quick.py -v

# Integration tests
pytest tests/test_integration.py -v

# Full coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📖 Documentation

- **Technical documentation**: See `docs/` for implementation details
- **Configuration**: See `config/` for control parameters
- **Examples**: Scripts in root directory (`run_proposed.py`, `run_baselines.py`)

---

## 👥 Authors

Developed at the **University of Santiago de Chile (USACH)**

---

## 📄 License

This project is open source. Consult the LICENSE file for more details.

---

## 🤝 Contributions

Contributions are welcome. Please:

1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📞 Contact

For questions or suggestions about this project, contact:
- **Institution**: University of Santiago de Chile (USACH)

---

## 🔗 References

This work implements concepts from:
- Model Predictive Control (MPC)
- Hybrid control systems
- Machine learning for control
- Real-time optimization

---

**Last updated**: January 2026
