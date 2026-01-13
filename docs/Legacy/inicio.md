# Prompt para LLM (IDE) — Diseño de experimentación
## Título de la propuesta
**Event-Driven Hybrid Control with Discrete Memory and Learned Predictive Acceleration**

---

## Rol y objetivo
Actúa como **coautor técnico** (control + aprendizaje + sistemas híbridos) y ayúdame a **diseñar la experimentación completa** para validar una arquitectura de control híbrido **event-driven** que integra:

1) **Memoria discreta verificable** (tipo flip-flops, PLC-friendly)  
2) **Predicción temporal aprendida** (p.ej., LSTM)  
3) **Aceleración predictiva estructural (“Turbo”)** que reduce cómputo **solo cuando es necesario**, manteniendo restricciones y mejorando transitorios.

Tu salida debe ser un **plan experimental publicable** (nivel Q2/Q1), con: marco matemático mínimo pero sólido, plantas, baselines, métricas, ablations, protocolo de evaluación, y artefactos reproducibles (tablas/figuras/pseudocódigo). Evita vaguedades.

---

## 1) Problema (brecha que atacamos) — Para orientar la experimentación
Existe una brecha entre:
- **Controladores aprendidos**: flexibles, pero opacos e impredecibles.
- **Control industrial clásico**: seguro y determinista, pero rígido y conservador.

Puntos críticos a atacar experimentalmente:
- El control por aprendizaje suele carecer de **memoria verificable y persistente**.
- El control periódico puede ser **costoso** en cómputo/energía.
- Event-triggered clásico usa **umbrales locales**, sin memoria predictiva rica.
- El deep learning rara vez se integra con **lógica discreta formal** (PLC/IEC-like).

La experimentación debe demostrar que la arquitectura propuesta **cierra la brecha**: seguridad/interpretabilidad (discreto) + desempeño/adaptación (aprendido) + eficiencia computacional (event-driven + turbo).

---

## 2) Idea central (definición operacional para experimentar)
Controlador híbrido event-driven que combina:
- **Memoria discreta explícita y verificable**: \( m_k \in \{0,1\}^{B} \)
- **Predicción aprendida** para tendencia/historial: \(\hat{x}\) desde LSTM (u otro predictor temporal)
- **Mecanismo Turbo** de aceleración estructural que:
  - no reemplaza el controlador (MPC),
  - lo **acelera condicionalmente** y de forma segura,
  - con fallback al MPC cuando corresponda.

No es “LSTM + control”; es:
**memoria explícita + aprendizaje + eventos + aceleración segura**.

---

## 3) Marco matemático (mínimo pero sólido) — Debe quedar implementable
### 3.1 Variables y señales (definir antes de ecuaciones)
Define con claridad:
- \(k\): índice de tiempo discreto.
- \(x_k \in \mathbb{R}^{n}\): estado continuo de la planta.
- \(u_k \in \mathbb{R}^{m}\): control aplicado.
- \(w_k\): perturbaciones/incertidumbre.
- \(m_k \in \{0,1\}^{B}\): **memoria discreta persistente** (flip-flops / PLC-friendly).
- \(s_k\): estado híbrido total.
- \(\hat{x}_{k|k-1}\): predicción a un paso basada en historial (LSTM).
- \(\delta_k \in \{0,1\}\): variable de disparo por eventos.
- \(\eta(m_k)\): umbral/condición dependiente del modo discreto.
- \(E(\cdot)\): función de evento (basada en error/tendencia/historial y memoria).
- \(\phi_k\): eventos/lógica que actualiza la memoria discreta.
- \(\pi_{\text{MPC}}(\cdot)\): política del MPC (solución del problema de optimización).
- \(T_{m_k}(\cdot)\): operador Turbo (aceleración).

### 3.2 Dinámica del sistema y estado híbrido
Planta (forma general):
\[
x_{k+1} = f(x_k, u_k, w_k)
\]

Estado híbrido:
\[
s_k = (x_k, m_k)
\]

Dinámica discreta (supervisión/flip-flops):
\[
m_{k+1} = g(m_k, \phi_k)
\]

### 3.3 Disparo por eventos con memoria predictiva (no Markoviano)
El evento no depende solo de error instantáneo, sino de predicción + memoria:
\[
\delta_k = \mathbf{1}\left\{ E\big(x_k, \hat{x}_{k|k-1}, m_k\big) > \eta(m_k) \right\}
\]

Requisito experimental: mostrar que \(\delta_k\) incorpora **historial/tendencia** (vía \(\hat{x}\)) y que \(\eta\) depende del **estado discreto**.

### 3.4 Control condicionado (event-driven)
Control con retención si no hay evento:
\[
u_k =
\begin{cases}
u_{k-1}, & \delta_k = 0 \\
\pi_{\text{MPC}}(x_k, \hat{x}, m_k), & \delta_k = 1
\end{cases}
\]

### 3.5 Aceleración predictiva estructural (“Turbo”)
Define un operador de aceleración:
\[
T_{m_k} : (\hat{x}, m_k) \rightarrow \text{decisión rápida}
\]

“Turbo” debe **cambiar la estructura del cómputo** (no solo parámetros), y debe tener fallback seguro al MPC.

Ejemplos implementables (elige 1–2 para el paper; el plan debe permitir comparar):
- **Turbo-A (Warm-start aprendido)**: LSTM propone una secuencia inicial \(\{u\}\) para el solver de MPC → menos iteraciones.
- **Turbo-B (Horizonte adaptativo)**: el horizonte \(N_k\) depende de \(m_k\) y del riesgo predicho → más capacidad solo en transitorios críticos.
- **Turbo-C (Fast policy + verificación)**: política rápida propone \(u_k\); lógica discreta valida restricciones; fallback al MPC.

---

## 4) Qué debe producir tu respuesta (entregables del plan experimental)
Tu respuesta debe incluir, como mínimo:

1) **Selección de plantas** (dos) y justificación.
2) **Definición del MPC base** (costos, restricciones, solver, horizonte).
3) **Diseño del predictor temporal** (LSTM): datos, entradas/salidas, entrenamiento.
4) **Diseño de memoria discreta \(m_k\)**: bits, semántica, reglas \(g(\cdot)\), eventos \(\phi_k\).
5) **Diseño del trigger \(E(\cdot)\) y \(\eta(m_k)\)** (con alternativas razonables).
6) **Implementación de Turbo** (A/B/C) con detalles operativos.
7) **Baselines obligatorios** (mínimos):
   - MPC periódico (clásico).
   - eMPC clásico (event-triggered MPC con umbrales locales).
   - Control aprendido **sin** memoria discreta explícita.
8) **Métricas** (con definiciones exactas y cómo se calculan):
   - Costo/error (tracking, regulación, etc.).
   - Violaciones de restricciones (conteo y magnitud).
   - Tasa de eventos (frecuencia de \(\delta_k=1\)).
   - Tiempo de cómputo (media y p95) y, si aplica, energía/CPU.
   - Robustez ante perturbaciones (escenarios).
9) **Ablation clave**:
   - sin flip-flops (sin \(m_k\)).
   - sin aceleración (sin Turbo).
   - sin eventos (control periódico siempre).
10) **Protocolo de evaluación**: seeds, escenarios, splits, tuning, significancia estadística.
11) **Artefactos publicables**:
   - Tabla de métricas principal.
   - Tabla de ablation.
   - Curvas (tracking), histograma/boxplot de cómputo, tasa de eventos, violaciones.
   - Diagrama de arquitectura y pseudocódigo del algoritmo.

---

## 5) Plantas (deben forzar el valor de la propuesta)
Define **dos plantas** con intención experimental clara:

### Planta A: No lineal con restricciones duras
Requisitos:
- Dinámica no lineal (p.ej., saturaciones, fricción, o no linealidad conocida).
- Restricciones duras: \(u\) acotado, \(x\) acotado, o restricciones de seguridad.
- Debe estresar “satisfacción de restricciones” + eficiencia de eventos.

### Planta B: Transitorios severos (ideal motor o proceso térmico)
Requisitos:
- Transitorios rápidos/perturbaciones bruscas.
- Permite mostrar que Turbo asigna cómputo “solo en crisis”.
- Debe evidenciar trade-off desempeño vs cómputo.

Tu respuesta debe proponer **instancias concretas** (opciones) para cada planta, por ejemplo:
- Motor (modelo discreto) con saturaciones y cargas variables.
- Proceso térmico con retardos/inercias y límites de temperatura.
- Sistema tipo tanque/acople no lineal, con restricciones de nivel/caudal.
(Elige y especifica parámetros típicos y restricciones numéricas para que sea implementable.)

---

## 6) Baselines (definirlos de forma “justa”)
Para cada baseline especifica:
- Qué información usa (¿predicción? ¿memoria? ¿eventos?).
- Qué se mantiene constante para comparación justa:
  - mismo modelo de planta,
  - mismo costo del MPC (si aplica),
  - mismas restricciones,
  - mismo solver y tolerancias (si se comparan MPCs),
  - mismo presupuesto de cómputo (si se analiza eficiencia).

Baselines mínimos:
1) **MPC periódico**: resuelve cada paso (o cada \(T_s\) fijo).
2) **eMPC clásico**: evento por umbral local (error instantáneo), sin predictor rico.
3) **Control aprendido sin memoria discreta**: política (RL o imitativa) o red que produce \(u_k\), con/sin verificación; pero **sin** \(m_k\) explícito.

Incluye al menos un baseline “aprendido + verificación” (si Turbo-C se usa) para separar el efecto de verificación vs memoria.

---

## 7) Diseño detallado del controlador propuesto (para implementar)
### 7.1 Memoria discreta \(m_k\)
- Define \(B\) (número de bits) y su semántica:
  - modos de operación,
  - enclavamientos,
  - saturaciones persistentes,
  - estados degradados.
- Define reglas de actualización \(g(m_k,\phi_k)\) en forma de tabla o lógica (if/then).
- Define cómo \(\phi_k\) se construye desde señales (p.ej., saturación sostenida, violaciones cercanas, tendencia riesgosa predicha).
- Requisito: que sea **auditable** (explicar cómo se inspecciona y valida).

### 7.2 Predictor (LSTM)
Especifica:
- Entradas: ventanas de \([x, u]\), errores, referencias, etc.
- Salida: \(\hat{x}_{k|k-1}\) o predicción multi-step.
- Pérdida de entrenamiento y normalización.
- Datos: cómo generar trayectorias (simulación con perturbaciones y referencias).
- Splits: train/val/test por seeds y escenarios.
- Criterios de early stopping y regularización.

### 7.3 Trigger por eventos \(E(\cdot)\) y umbral \(\eta(m_k)\)
Propón al menos **2 formas** de \(E\):
- Basada en error de predicción/innovación: \(\|x_k - \hat{x}_{k|k-1}\|\).
- Basada en riesgo de violación: distancia a restricciones (barrier / margen).
Y diseña \(\eta(m_k)\) dependiente del modo:
- modo normal: umbral alto (menos eventos),
- modo crítico: umbral bajo (más eventos).

### 7.4 Turbo (elige 1–2 variantes y define cómo medir su aporte)
Para cada Turbo seleccionado:
- Qué reduce exactamente (iteraciones, horizonte, evaluaciones).
- Cuándo se activa (dependiente de \(m_k\) y/o riesgo).
- Cómo asegura fallback al MPC.

---

## 8) Métricas (definiciones exactas y reporte)
Debes definirlas formalmente y cómo agregarlas:

1) **Costo/Tracking error**
- MSE/IAE/ITAE o costo MPC acumulado.
- Reporte: media ± std, y percentiles en escenarios.

2) **Violaciones de restricciones**
- Conteo de episodios con violación.
- Magnitud integrada de violación (área).
- Separar: violaciones en \(x\) vs \(u\).

3) **Tasa de eventos**
\[
\rho = \frac{1}{K}\sum_{k=1}^{K}\delta_k
\]
y distribución de intervalos entre eventos.

4) **Tiempo de cómputo**
- Por paso: media y p95.
- Por episodio: total.
- Si hay solver: iteraciones promedio y p95.

5) **Robustez**
- Barrido de perturbaciones (niveles de ruido, cargas, cambios paramétricos).
- Reportar degradación relativa vs nominal.

Incluye una métrica compuesta opcional tipo “desempeño por cómputo”:
\[
\text{Score} = \alpha \cdot \text{Error} + \beta \cdot \text{Violaciones} + \gamma \cdot \text{Cómputo}
\]
(con \(\alpha,\beta,\gamma\) justificables), pero sin ocultar métricas base.

---

## 9) Ablation (estructura mínima exigible)
Diseña el ablation para aislar efectos:

A1) **Sin flip-flops**: remover \(m_k\) (o fijarlo) y mantener el resto.
A2) **Sin Turbo**: usar MPC normal cuando \(\delta_k=1\), sin aceleración.
A3) **Sin eventos**: forzar \(\delta_k=1\) siempre (periódico).
A4) (Recomendado) **Sin predictor rico**: reemplazar LSTM por predictor simple (AR/kalman) para medir aporte del aprendizaje.

Cada ablation debe reportar el mismo set de métricas.

---

## 10) Protocolo experimental (para “pasar revisión”)
Define:
- Número de seeds (≥ 10 ideal).
- Número de escenarios por planta (≥ 20 con perturbaciones variadas).
- Longitud de episodio \(K\) y condiciones iniciales.
- Barrido de hiperparámetros:
  - \(B\) (bits),
  - \(\eta\) por modo,
  - horizonte MPC base \(N\),
  - configuración Turbo (p.ej., factor de reducción de iteraciones/horizonte),
  - tamaño de ventana LSTM, capas, hidden size.
- Estrategia de tuning **justa** (mismo presupuesto para todos).
- Pruebas estadísticas:
  - test no paramétrico o intervalos por bootstrap para comparar medianas/p95 de cómputo.
- Reproducibilidad:
  - versiones, seeds, configuración, hardware, scripts.

---

## 11) Qué figuras/tablas deben salir sí o sí
Especifica exactamente:

**Tabla 1 (principal)**: por planta y método:
- Error/costo, violaciones, tasa de eventos, cómputo (media y p95).

**Tabla 2 (ablation)**:
- Propuesto vs A1/A2/A3/(A4) con deltas relativos (%).

**Figura 1**: diagrama de arquitectura (flujo \(x\to\) predictor \(\to\) trigger \(\to\) MPC/Turbo, y lazo de memoria discreta).
**Figura 2**: trayectoria de tracking (mejor caso y caso crítico).
**Figura 3**: cómputo por paso (boxplot) y p95 resaltado.
**Figura 4**: tasa de eventos y distribución de inter-event times.
**Figura 5 (opcional)**: violaciones acumuladas vs tiempo en escenarios críticos.

---

## 12) Pseudocódigo (el LLM debe generarlo)
Pide explícitamente un pseudocódigo tipo “Algoritmo 1”:

- Inicialización: \(x_0, u_{-1}, m_0\)
- Loop:
  1) Predicción \(\hat{x}_{k|k-1}\)
  2) Evaluación evento \(\delta_k\)
  3) Si \(\delta_k=0\): hold \(u_k=u_{k-1}\)
  4) Si \(\delta_k=1\): aplicar Turbo (si corresponde) y/o resolver MPC
  5) Actualizar memoria \(m_{k+1}=g(m_k,\phi_k)\)
  6) Aplicar \(u_k\), avanzar planta
  7) Log de métricas (cómputo, violaciones, etc.)

Debe quedar claro dónde se mide el tiempo y cómo se activa Turbo.

---

## 13) Instrucciones de salida (formato estricto)
Devuélveme el plan con esta estructura y con listas accionables:

1. **Resumen ejecutivo** (5–8 líneas, centrado en hipótesis medibles).
2. **Marco matemático final** (ecuaciones + definiciones).
3. **Plantas** (A y B, con parámetros y restricciones numéricas propuestas).
4. **Métodos comparados** (propuesto + baselines + ablations).
5. **Implementación** (MPC, predictor, memoria, trigger, Turbo).
6. **Métricas** (definiciones y cómputo).
7. **Protocolo** (scenarios, tuning, seeds, estadística).
8. **Tablas/Figuras** (contenido exacto).
9. **Checklist reproducible** (lo que debe estar en el repo).
10. **Riesgos y mitigaciones** (amenazas a validez y cómo controlarlas).

No uses frases genéricas (“se puede”, “sería bueno”). Da decisiones concretas, alternativas con pros/contras, y valores iniciales recomendados.

---

## 14) Hipótesis a validar (deben guiar el diseño)
Incluye explícitamente hipótesis falsables del tipo:

H1: La memoria discreta explícita reduce violaciones y mejora interpretabilidad vs control aprendido sin memoria.  
H2: El event-driven reduce tasa de cómputo sin degradar costo y restricciones vs MPC periódico.  
H3: Turbo reduce p95 de cómputo y mantiene (o mejora) desempeño en transitorios vs eMPC clásico.  
H4: \(\eta(m_k)\) adaptativo produce mejor trade-off (eventos vs desempeño) que umbral fijo.

---

## 15) Nota sobre “suficiente teoría” para Q1/Q2
Asume que no necesitamos estabilidad global dura; basta con:
- invariancia de restricciones (satisfacción),
- y argumentos de estabilidad práctica / ISS bajo supuestos razonables.

La experimentación debe reflejar esto: foco en restricciones, robustez, y desempeño/cómputo.

---

## Comienza ahora
Genera el plan completo siguiendo las secciones anteriores, proponiendo implementaciones concretas para cada componente (plantas, MPC, LSTM, memoria, trigger y Turbo), junto con tablas/figuras y pseudocódigo.
