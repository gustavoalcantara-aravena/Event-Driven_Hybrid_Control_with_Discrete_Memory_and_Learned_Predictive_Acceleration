# Guía de Ejecución y Resultados Esperados

Esta guía detalla qué sucede al ejecutar cada componente del sistema experimental, qué archivos se generan y cómo interpretar los resultados visuales y tabulares. Está diseñada para entender el flujo completo sin conocimientos previos profundos del código.

## 1. Flujo de Ejecución

El experimento se divide en tres etapas secuenciales:

### Paso 1: Ejecutar Líneas Base (`run_baselines.py`)
**Comando:** `python run_baselines.py --plant both --seeds 15 --scenarios 25`

*   **¿Qué hace?**
    Simula métodos de control estándar (clásicos) para tener un punto de comparación. Ejecuta 3 estrategias:
    1.  **B1_PeriodicMPC**: Un controlador robusto pero lento que recalcula siempre (frecuencia fija).
    2.  **B2_ClassicEMPC**: Un controlador por eventos estándar (sin memoria ni aprendizaje).
    3.  **B3_RLnoMemory**: Un controlador simple y rápido (tipo proporcional/PID) sin predicción compleja.

*   **¿Qué verás en la consola?**
    Verás barras de progreso indicando la planta (`MOTOR`, `OVEN`) y el método actual.
    ```text
    Running B1_PeriodicMPC: Periodic MPC...
    100%|██████████| 375/375 [00:50<00:00, 7.37it/s]
    ```

*   **Archivos guardados en `results/`:**
    *   `results_motor_B1_PeriodicMPC.csv`
    *   `results_motor_B2_ClassicEMPC.csv`
    *   `results_motor_B3_RLnoMemory.csv`
    *   (Igual para `oven`...)

---

### Paso 2: Ejecutar Método Propuesto (`run_proposed.py`)
**Comando:** `python run_proposed.py --plant both --seeds 15 --scenarios 25`

*   **¿Qué hace?**
    Ejecuta nuestro algoritmo "Inteligente" (Híbrido con Memoria y LAPA) y sus variaciones (Ablaciones) para probar qué parte del invento funciona mejor.
    1.  **Proposed**: El método completo (Memoria + LAPA + LSTM).
    2.  **A1_NoMemory**: Sin la memoria discreta.
    3.  **A2_NoLAPA**: Sin la aceleración de cálculo (LSTMPredictor + Warming).
    4.  **A4_EventMPC**: Versión básica de disparo por eventos (similar a B2 pero interna).

*   **¿Qué verás en la consola?**
    Similar al anterior, barras de progreso para cada variante.
    ```text
    Running Proposed (15 seeds × 25 scenarios)...
    Running A1_NoMemory...
    ```

*   **Archivos guardados en `results/`:**
    *   `results_motor_combined.csv`: Contiene TODAS las variantes juntas.
    *   `results_oven_combined.csv`: Idem para el horno.

---

### Paso 3: Evaluar y Graficar (`evaluate.py`)
**Comando:** `python evaluate.py --plants "motor, oven"`

*   **¿Qué hace?**
    Toma todos los archivos CSV generados en los pasos 1 y 2, calcula estadísticas (promedios, desviaciones), y genera reportes bonitos (tablas Excel y gráficos PNG).

*   **¿Qué verás en la consola?**
    Resúmenes de texto confirmando la carga de datos y la creación de archivos.
    ```text
    Generating Tables...
    ✓ Table1_MainMetrics_motor.csv
    Generating Figures...
    ✓ Fig1_Architecture.png
    ```

*   **Archivos guardados en `evaluation/`:** (Ver sección de detalle más abajo).

---

---

## 2. Interpretación de Resultados (`evaluation/`)

Aquí está "la carne" del experimento. En la carpeta `evaluation/` encontrarás **4 tablas CSV** y **8 figuras PNG**.

---

### 📊 A. TABLAS DE DATOS (Archivos CSV)

#### **Table1_MainMetrics_[plant].csv** - Comparativa Principal ⭐⭐⭐⭐⭐

**Propósito:** Comparar el método propuesto contra los 3 baselines clásicos.

**Columnas:**

| Columna | Significado | ¿Qué es mejor? | Interpretación |
| :--- | :--- | :--- | :--- |
| **Method** | Nombre del método | - | Identifica cada controlador |
| **Cost (↓)** | Costo total (error + esfuerzo de control) | **Menor** | Eficiencia global del control |
| **Tracking MSE (↓)** | Error cuadrático medio de seguimiento | **Menor** | Precisión en seguir la referencia |
| **Violations (↓)** | Número de violaciones de restricciones | **Cero ideal** | Seguridad del sistema |
| **Compute [ms] (↓)** | Tiempo de CPU por paso | **Menor** | Viabilidad en tiempo real |
| **Event Rate (↓)** | Fracción de pasos con eventos | **Menor** | Eficiencia comunicacional |

**Qué buscar:**
- ✅ **Proposed** debe tener **costo similar o menor** que B1/B2
- ✅ **Event Rate** de Proposed debe ser **< 0.5** (menos del 50% de comunicaciones)
- ✅ **Violations = 0** para todos (seguridad garantizada)

---

#### **Table2_Ablations_[plant].csv** - Estudio de Ablación ⭐⭐⭐⭐

**Propósito:** Justificar la necesidad de cada componente del método propuesto.

**Métodos comparados:**
- **Proposed**: Método completo (LSTM + LAPA + Memory)
- **A1_NoMemory**: Sin memoria discreta
- **A2_NoLAPA**: Sin aceleración LAPA
- **A3_Periodic**: Forzado periódico (sin eventos)
- **A4_EventMPC**: Event-MPC básico (sin LSTM/LAPA/Memory)

**Qué buscar:**
- ✅ **Proposed** debe tener el **mejor balance** costo/eventos
- ✅ Cada ablación debe mostrar **degradación** en alguna métrica
- ✅ Esto demuestra que **cada componente aporta valor**

---

#### **Table3_StatisticalTests.csv** - Tests de Significancia Estadística ⭐⭐⭐⭐⭐

**Propósito:** Validar científicamente que las diferencias observadas son reales, no casualidad.

**Columnas:**
- **Plant**: motor o oven
- **Comparison**: "Proposed vs [Baseline]"
- **[metric]**: p-value con marcador de significancia

**Marcadores de significancia:**
- `***` → p < 0.001 (altamente significativo)
- `**` → p < 0.01 (muy significativo)
- `*` → p < 0.05 (significativo)
- `ns` → no significativo

**Qué buscar:**
- ✅ **p-values < 0.05** en métricas clave (cost, tracking_error)
- ✅ Esto significa que **Proposed es estadísticamente mejor**, no suerte

**Nota:** Con solo 1 seed (dry run), verás `p=nan ns`. Con 15 seeds, obtendrás valores reales.

---

#### **Table4_ComputationalBudget.csv** - Análisis de Viabilidad en Tiempo Real ⭐⭐⭐

**Propósito:** Demostrar que el método es implementable en hardware real.

**Columnas:**
- **Mean CPU (ms)**: Tiempo promedio de cómputo
- **P95 CPU (ms)**: Tiempo en el peor caso (percentil 95)
- **RT Limit (ms)**: Límite de tiempo real del sistema
  - Motor: 10 ms (control a 100 Hz)
  - Oven: 100 ms (control a 10 Hz)
- **RT Feasible**: ✓ Yes / ✗ No
- **Safety Margin (%)**: Margen de seguridad restante

**Qué buscar:**
- ✅ **Todos los métodos** deben tener `RT Feasible = ✓ Yes`
- ✅ **Safety Margin > 50%** es ideal (robusto ante variaciones)
- ✅ **Proposed** debe ser competitivo con baselines en CPU time

---

### 🎨 B. GRÁFICOS (Archivos PNG)

#### **Fig1_Architecture.png** - Diagrama del Sistema ⭐⭐⭐

**Qué muestra:** Diagrama de bloques del controlador híbrido propuesto.

**Componentes visualizados:**
- Planta (Motor/Oven)
- Sensor
- LSTM Predictor
- Event Trigger
- Discrete Memory
- MPC Solver
- LAPA Accelerator

**Utilidad:**
- 📄 **Para el paper:** Figura conceptual en la sección de Metodología
- 🎓 **Para entender:** Flujo de datos y decisiones del sistema

---

#### **Fig2_Tracking.png** - Rendimiento de Seguimiento ⭐⭐⭐⭐⭐

**Qué muestra:** Boxplots comparando error de seguimiento y costo total.

**Cómo leer los boxplots:**
- **Línea central**: Mediana (valor típico)
- **Caja**: Rango intercuartílico (50% de los datos)
- **Bigotes**: Rango completo (excluyendo outliers)
- **Puntos**: Outliers (casos extremos)

**Qué buscar:**
- ✅ **Caja de Proposed más baja** que baselines → Mejor rendimiento
- ✅ **Caja más estrecha** → Mayor consistencia/robustez
- ✅ **Sin outliers** → Comportamiento predecible

**Interpretación:**
- Si Proposed está significativamente más abajo en "Cost", **ganamos en eficiencia global**
- Si está más abajo en "Tracking Error", **ganamos en precisión**

---

#### **Fig3_Compute.png** - Eficiencia Computacional ⭐⭐⭐⭐

**Qué muestra:** Tiempo de CPU por método (barras agrupadas).

**Barras:**
- **Azul**: Tiempo promedio (Mean CPU)
- **Naranja**: Tiempo peor caso (P95 CPU)

**Qué buscar:**
- ✅ **Proposed con LAPA** debe ser **más rápido** que B1_PeriodicMPC
- ✅ **P95 < 10ms** para motor, **< 100ms** para oven (viabilidad tiempo real)
- ✅ **Diferencia pequeña** entre Mean y P95 → Comportamiento predecible

**Interpretación:**
- Si Proposed es más rápido que B1, **LAPA está funcionando**
- Si es similar a B2/B3, **no sacrificamos velocidad por calidad**

---

#### **Fig4_Events.png** - Estadísticas de Comunicación ⭐⭐⭐⭐⭐

**Qué muestra:** Frecuencia y patrón de eventos disparados.

**Subplots:**
1. **Event Rate**: Fracción de pasos con eventos (0-1)
2. **Number of Events**: Conteo total de eventos
3. **Mean Inter-Event Time**: Tiempo promedio entre eventos

**Qué buscar:**
- ✅ **Proposed** debe tener **Event Rate < 0.5** (menos del 50%)
- ✅ **Menor que B1** (que es periódico, ~10% si period=10)
- ✅ **Inter-Event Time alto** → Sistema descansa más, ahorra recursos

**Interpretación:**
- **Event Rate bajo** = Eficiencia comunicacional
- **Adaptativo** (varía según planta/escenario) = Inteligente, no fijo

---

#### **Fig5_Robustness.png** - Análisis de Robustez ⭐⭐⭐⭐

**Qué muestra:** Variabilidad del costo a través de diferentes escenarios.

**Ejes:**
- **X**: Número de escenario (diferentes condiciones de ruido/perturbación)
- **Y**: Costo total
- **Líneas**: Cada método

**Qué buscar:**
- ✅ **Línea de Proposed relativamente plana** → Robusto ante perturbaciones
- ✅ **Sin picos grandes** → No hay "escenarios de falla"
- ✅ **Consistentemente por debajo** de baselines → Dominancia

**Interpretación:**
- Si la línea es plana, **el método es robusto**
- Si tiene picos, **identificar qué escenarios son difíciles**

---

#### **Fig6_Trajectories.png** - Trayectorias Temporales ⭐⭐⭐⭐⭐

**Qué muestra:** Evolución temporal de estado y control en un episodio representativo.

**Subplots (2×2):**
- **Fila 1**: Motor (State + Control)
- **Fila 2**: Oven (State + Control)

**Elementos:**
- **Línea negra punteada**: Referencia (objetivo a seguir)
- **Líneas de colores**: Trayectorias de cada método
- **Círculos negros**: Eventos disparados (comunicaciones)

**Qué buscar:**
- ✅ **Trayectorias convergen rápido** a la referencia → Buen transitorio
- ✅ **Oscilación mínima** alrededor de la referencia → Estabilidad
- ✅ **Eventos concentrados** al inicio (transitorio) → Adaptativo
- ✅ **Control suave** (sin cambios bruscos) → Eficiencia energética

**Interpretación:**
- **Convergencia rápida** = Buen settling time
- **Pocos eventos en régimen permanente** = Eficiencia
- **Control suave** = Menor desgaste de actuadores

---

#### **Fig7_ParetoFront.png** - Frontera de Pareto (Trade-off) ⭐⭐⭐⭐⭐

**Qué muestra:** Relación entre costo (rendimiento) y event rate (comunicación).

**Ejes:**
- **X**: Event Rate (eventos/paso)
- **Y**: Total Cost (costo total)
- **Tamaño de burbuja**: CPU time (más grande = más lento)

**Qué buscar:**
- ✅ **Proposed en esquina inferior izquierda** → Mejor trade-off
- ✅ **Dominancia de Pareto**: Ningún método es mejor en ambas dimensiones
- ✅ **Burbuja pequeña** → Computacionalmente eficiente

**Interpretación:**
- **Esquina inferior izquierda** = Óptimo (bajo costo, pocos eventos)
- Si Proposed domina, **es la mejor opción** para sistemas con restricciones de comunicación

---

#### **Fig10_RadarChart.png** - Comparación Multidimensional ⭐⭐⭐⭐

**Qué muestra:** Desempeño en 5 métricas simultáneamente (normalizado 0-1).

**Ejes (5 dimensiones):**
1. **Cost**: Costo total (invertido: 1 = mejor)
2. **Tracking**: Error de seguimiento (invertido)
3. **Violations**: Violaciones (invertido)
4. **CPU**: Tiempo de cómputo (invertido)
5. **Events**: Tasa de eventos (invertido)

**Cómo leer:**
- **Área mayor** = Mejor desempeño global
- **Forma regular** = Balanceado en todas las métricas
- **Picos** = Fortalezas específicas

**Qué buscar:**
- ✅ **Proposed con área mayor** que baselines → Dominancia global
- ✅ **Forma pentagonal regular** → Método balanceado
- ✅ **Sin "valles"** → Sin debilidades críticas

**Interpretación:**
- **Área grande** = Método superior en múltiples dimensiones
- **Forma balanceada** = No sacrifica una métrica por otra

---

## 3. Resumen para "No Expertos"

### ✅ Checklist de Validación

1. **Ejecución exitosa:**
   - [ ] `run_baselines.py` terminó sin errores
   - [ ] `run_proposed.py` terminó sin errores
   - [ ] `evaluate.py` generó 4 tablas + 8 figuras

2. **Validación de resultados:**
   - [ ] **Table1**: Proposed tiene costo ≤ baselines
   - [ ] **Table3**: p-values < 0.05 en métricas clave
   - [ ] **Table4**: Todos los métodos son RT Feasible
   - [ ] **Fig2**: Boxplot de Proposed está más abajo
   - [ ] **Fig4**: Event Rate de Proposed < 0.5
   - [ ] **Fig6**: Trayectorias convergen suavemente
   - [ ] **Fig7**: Proposed en esquina inferior izquierda
   - [ ] **Fig10**: Proposed tiene área mayor

3. **Interpretación científica:**
   - **Si todas las casillas están marcadas:** ✅ **Método validado exitosamente**
   - **Si faltan algunas:** ⚠️ Revisar configuración o parámetros
   - **Si muchas fallan:** ❌ Problema fundamental en el diseño

---

## 4. Preguntas Frecuentes

**P: ¿Cuánto tiempo toma ejecutar todo?**
- Dry run (1 seed, 1 scenario): ~5 minutos
- Experimento completo (15 seeds, 25 scenarios): ~1-2 horas

**P: ¿Qué hago si un método tiene violations > 0?**
- Revisar restricciones en `config/[plant]_params.yaml`
- Ajustar pesos Q/R en `config/mpc_base.yaml`

**P: ¿Qué significa "p=nan ns" en Table3?**
- Muestra insuficiente (dry run con 1 seed)
- Ejecutar con ≥10 seeds para tests válidos

**P: ¿Puedo usar estas figuras en mi paper?**
- ✅ **Sí, todas son publication-ready**
- Fig6 usa datos experimentales reales
- Incluir caption explicativo en cada figura
