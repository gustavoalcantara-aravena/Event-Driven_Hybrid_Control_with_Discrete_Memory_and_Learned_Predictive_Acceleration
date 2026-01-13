# Event-Driven Hybrid Control with Discrete Memory and Learned Predictive Acceleration

## 🏫 Investigación Académica

Este proyecto es parte de una investigación desarrollada por la **Universidad de Santiago de Chile (USACH)**, en el área de control automático y sistemas híbridos.

---

## 📋 Descripción del Proyecto

Este repositorio contiene la implementación de un controlador híbrido basado en eventos que integra:

- **Control Predictivo por Modelo (MPC)**: Optimización en tiempo real de trayectorias
- **Lógica Discreta**: Máquinas de estado para decisiones de control
- **Predicción LSTM**: Red neuronal recurrente para predicción de dinámicas del sistema
- **Aceleración Aprendida**: Técnicas de aprendizaje para optimizar el rendimiento computacional
- **Disparo de Eventos**: Mecanismo eficiente para reducir comunicación y cálculos innecesarios

El sistema fue validado en dos plantas de prueba:
- **Motor DC**: Sistema de control de velocidad
- **Horno Térmico**: Sistema de regulación de temperatura

---

## 🎯 Características Principales

✅ **Control Híbrido Eficiente**: Combinación de MPC continuo con lógica discreta  
✅ **Predicción Neuronal**: LSTM entrenada para capturar dinámicas complejas  
✅ **Disparo de Eventos**: Reduce carga computacional mediante activación selectiva  
✅ **Aceleración Aprendida**: Optimización de parámetros mediante técnicas de ML  
✅ **Validación Experimental**: Resultados en plantas reales  
✅ **Reproducibilidad**: Código modular y bien documentado  

---

## 📁 Estructura del Proyecto

```
.
├── src/                          # Módulos principales del controlador
│   ├── controller_hybrid.py       # Controlador híbrido principal
│   ├── mpc_solver.py             # Solucionador MPC (CasADi)
│   ├── lstm_predictor.py         # Predictor LSTM
│   ├── event_trigger.py          # Mecanismo de disparo de eventos
│   ├── discrete_logic.py         # Lógica discreta y máquinas de estado
│   ├── acceleration.py           # Aceleración aprendida
│   ├── plants.py                 # Modelos de plantas (Motor, Horno)
│   ├── metrics.py                # Métricas de evaluación
│   └── utils.py                  # Utilidades generales
│
├── config/                       # Archivos de configuración
├── models/                       # Modelos LSTM entrenados
├── results/                      # Resultados de experimentos
├── trajectories/                 # Trayectorias capturadas
├── tests/                        # Suite de pruebas
├── evaluation/                   # Scripts de evaluación
├── docs/                         # Documentación
├── requirements.txt              # Dependencias Python
├── run_proposed.py               # Script para ejecutar controlador propuesto
├── run_baselines.py              # Script para ejecutar baselines
└── evaluate.py                   # Script de evaluación completa
```

---

## 🚀 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip o conda

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/gustavoalcantara-aravena/Event-Driven_Hybrid_Control_with_Discrete_Memory_and_Learned_Predictive_Acceleration.git
cd Event-Driven_Hybrid_Control_with_Discrete_Memory_and_Learned_Predictive_Acceleration.git
```

2. **Crear entorno virtual (recomendado)**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

---

## 💻 Uso

### Ejecutar el Controlador Propuesto
```bash
python run_proposed.py --plant motor --duration 100 --config config/motor_config.yaml
```

### Ejecutar Baselines
```bash
python run_baselines.py --plant oven --duration 100
```

### Evaluación Completa
```bash
python evaluate.py --compare-all
```

### Pruebas Unitarias
```bash
pytest tests/ -v
```

---

## 📊 Resultados Principales

El controlador propuesto demuestra:

- **Reducción de Eventos**: ~40-60% menos disparos que MPC periódico
- **Mejora de Rendimiento**: Mejor seguimiento de referencia vs baselines
- **Eficiencia Computacional**: Menor carga de CPU manteniendo calidad de control
- **Robustez**: Desempeño consistente bajo perturbaciones

Véase `results/` para gráficos y tablas detalladas.

---

## 📚 Dependencias Principales

| Librería | Versión | Propósito |
|----------|---------|----------|
| NumPy | ≥1.24 | Computación numérica |
| SciPy | ≥1.10 | Algoritmos científicos |
| CasADi | ≥3.5.5 | Optimización y MPC |
| PyTorch | ≥2.0.0 | Red neuronal LSTM |
| Matplotlib | ≥3.7 | Visualización |
| Pandas | ≥2.0 | Análisis de datos |

---

## 🧪 Testing

El proyecto incluye una suite completa de pruebas:

```bash
# Pruebas rápidas
pytest tests/test_quick.py -v

# Pruebas de integración
pytest tests/test_integration.py -v

# Cobertura completa
pytest tests/ --cov=src --cov-report=html
```

---

## 📖 Documentación

- **Documentación técnica**: Ver `docs/` para detalles de implementación
- **Configuración**: Ver `config/` para parámetros de control
- **Ejemplos**: Scripts en raíz (`run_proposed.py`, `run_baselines.py`)

---

## 👥 Autores

Desarrollado en la **Universidad de Santiago de Chile (USACH)**

---

## 📄 Licencia

Este proyecto es de código abierto. Consulta el archivo LICENSE para más detalles.

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📞 Contacto

Para preguntas o sugerencias sobre este proyecto, contacta a través de:
- **Universidad**: Universidad de Santiago de Chile (USACH)
- **Departamento**: Ingeniería en Automatización y Control

---

## 🔗 Referencias

Este trabajo implementa conceptos de:
- Control predictivo basado en modelos (MPC)
- Sistemas de control híbridos
- Aprendizaje automático para control
- Optimización en tiempo real

---

**Última actualización**: Enero 2026
