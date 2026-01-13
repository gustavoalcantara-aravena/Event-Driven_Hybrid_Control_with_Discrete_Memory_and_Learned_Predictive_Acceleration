# PLAN EXPERIMENTAL: Event-Driven Hybrid Control with Discrete Memory and Learned Predictive Acceleration

## 1. RESUMEN EJECUTIVO

Validaremos una arquitectura de control híbrido event-driven que integra: (1) memoria discreta explícita y verificable (flip-flops), (2) predicción aprendida temporal (LSTM), y (3) aceleración estructural segura ("Turbo"). 

**Hipótesis centrales**: 
- H1: Memoria discreta → reduce violaciones y aumenta interpretabilidad vs control puro aprendido.
- H2: Event-driven → reduce cómputo sin degradar desempeño vs MPC periódico.
- H3: Turbo adaptativo → reduce latencia p95 y mejora transitorios vs eMPC clásico.
- H4: Umbral adaptativo η(m_k) → mejor trade-off eventos-desempeño que umbral fijo.

**Métrica compuesta (balanceada)**: Costo + Violaciones + Cómputo-eficiencia.

---

## 2. MARCO MATEMÁTICO FINAL

### 2.1 Variables y Notación

| Variable | Dominio | Significado |
|----------|---------|-------------|
| k | ℤ≥0 | Índice tiempo discreto |
| x_k | ℝⁿ | Estado continuo planta |
| u_k | ℝᵐ | Control (entrada planta) |
| w_k | ℝᵍ | Perturbación/incertidumbre |
| m_k | {0,1}^B | Memoria discreta (B bits, flip-flops) |
| s_k | (x_k, m_k) | Estado híbrido |
| ŷ_k\|k-1 | ℝⁿ | Predicción a un paso (LSTM) |
| δ_k | {0,1} | Indicador evento (trigger) |
| η(m_k) | ℝ≥0 | Umbral adaptativo dependiente modo |
| E(·) | ℝ → ℝ | Función de evento (norma error/riesgo) |
| φ_k | {0,1}^* | Lógica/condiciones para actualizar m_k |
| π_MPC(·) | Control policy | Política MPC (solución problema optimización) |
| T_m_k(·) | ℝⁿ × {0,1}^B → ℝᵐ | Operador Turbo (aceleración segura) |

### 2.2 Dinámica del Sistema Híbrido

**Planta (forma general, tiempo discreto):**
$$x_{k+1} = f(x_k, u_k, w_k)$$

**Estado híbrido:**
$$s_k = (x_k, m_k) \in \mathbb{R}^n \times \{0,1\}^B$$

**Dinámica de memoria discreta (flip-flops, lógica supervisora):**
$$m_{k+1} = g(m_k, \phi_k) \in \{0,1\}^B$$
donde $\phi_k = \phi(x_k, u_k, \hat{y}_{k|k-1}, \text{violaciones})$ son condiciones lógicas (tabla de transición).

### 2.3 Predictor Temporal (LSTM)

**Salida predictor (paso adelante):**
$$\hat{y}_{k|k-1} = \text{LSTM}\big([x_{k-H}, \ldots, x_k, u_{k-H}, \ldots, u_{k-1}]\big)$$
donde $H$ = tamaño ventana histórica (típicamente 5–20 pasos).

**Error de predicción:**
$$e_{\text{pred},k} = \|x_k - \hat{y}_{k|k-1}\|_2$$

### 2.4 Disparador por Eventos (Event Trigger) — Con Memoria Predictiva

**Disparo condicionado:**
$$\delta_k = \mathbb{1}\left\{ E(x_k, \hat{y}_{k|k-1}, m_k) > \eta(m_k) \right\}$$

**Opciones para función de evento** (se implementan ambas para ablation):

1. **E_error**: basada en error de predicción/innovación:
$$E_{\text{error}}(x_k, \hat{y}_{k|k-1}, m_k) = \|x_k - \hat{y}_{k|k-1}\|_2$$

2. **E_risk**: basada en riesgo de violación (margen a restricciones):
$$E_{\text{risk}}(x_k, \hat{y}_{k|k-1}, m_k) = -\min\big\{g_i(x_k) : i \in \text{restricciones}\big\}$$
(positivo si hay violación; cero si margen seguro)

**Umbral adaptativo por modo:**
$$\eta(m_k) = \begin{cases}
\eta_{\text{normal}} & \text{si } m_k = [0, \ldots, 0] \text{ (normal)}\\
\eta_{\text{critical}} & \text{si } m_k \text{ contiene bit crítico} = 1
\end{cases}$$
típicamente: $\eta_{\text{critical}} \ll \eta_{\text{normal}}$ (más sensible en crisis).

### 2.5 Control Condicionado (Event-Driven)

**Hold si no hay evento; MPC (o Turbo) si hay evento:**
$$u_k = \begin{cases}
u_{k-1} & \text{si } \delta_k = 0\\
\text{Turbo}(x_k, \hat{y}, m_k) \text{ o } \pi_{\text{MPC}}(x_k, \hat{y}, m_k) & \text{si } \delta_k = 1
\end{cases}$$

**MPC base (problema de optimización):**
$$\pi_{\text{MPC}}(x_k, \hat{y}, m_k) = \arg\min_{u_k, \ldots, u_{k+N-1}} \sum_{j=0}^{N-1} \|x_{k+j} - x_{\text{ref}}\|_Q^2 + \|u_{k+j}\|_R^2$$
s.a. (subject to):
- Dinámica predictor: $x_{k+j+1} = f(x_{k+j}, u_{k+j}, \hat{w})$
- Restricciones: $x_{k+j} \in X$, $u_{k+j} \in U$ para todo $j$
donde $N$ = horizonte (fijo o adaptativo con Turbo).

### 2.6 Aceleración Predictiva Estructural (Turbo) — Tres Opciones

**Turbo-A (Warm-start aprendido):**
$$u^{\text{init}} = \text{LSTM}_{\text{policy}}(\hat{y}, m_k, x_{\text{ref}})$$
Solver MPC recibe $u^{\text{init}}$ como inicialización → reduce iteraciones típicamente 30–50%.
Fallback: si solver no converge en límite relajado, aplica $\pi_{\text{MPC}}$ puro.

**Turbo-B (Horizonte adaptativo):**
$$N_k = N_{\text{base}} + \Delta N \cdot \mathbb{1}\{ E(x_k, \hat{y}, m_k) > \eta_{\text{mid}} \}$$
En crisis ($m_k$ crítico): horizonte más largo para planificación riesgosa; reduce en modo normal.
Reduce evaluaciones en estado estacionario, sin comprometer desempeño en transitorios.

**Turbo-C (Política rápida + verificación):**
$$u_k^{\text{fast}} = \text{NN}_{\text{policy}}(x_k, \hat{y}, m_k) \quad \text{(inferencia rápida <1ms)}$$
Verificación: lógica discreta valida $u_k^{\text{fast}}$ contra restricciones y riesgo predicho.
Si OK → usa $u_k^{\text{fast}}$; si violación potencial → fallback al MPC.

**Genéricamente, Turbo respeta:**
$$T_{m_k}(x_k, \hat{y}, m_k) \rightarrow \text{decisión segura}$$
con garantía: $T$ o fallback $\pi_{\text{MPC}}$ siempre factible.

---

## 3. PLANTAS

### 3.1 Planta A: Motor DC con Saturación y Cargas Variables

**Modelado discreto (T_s = 10ms):**
$$\begin{cases}
x_{1,k+1} = x_{1,k} + T_s \cdot x_{2,k} \\
x_{2,k+1} = (1 - \frac{T_s \cdot b}{J}) x_{2,k} + \frac{T_s}{J} \text{sat}(u_k) - \frac{T_s}{J} \tau_L(k) \\
\tau_L(k) = \tau_L^{\text{nom}} (1 + d_k)
\end{cases}$$

donde:
- $x_1$ = posición angular [rad]
- $x_2$ = velocidad angular [rad/s]
- $u$ = voltaje comando [V], saturado en ±12 V
- $b$ = fricción viscosa = 0.5 N·m·s/rad
- $J$ = inercia = 0.1 kg·m²
- $\tau_L^{\text{nom}}$ = carga nominal = 2 N·m
- $d_k$ = perturbación de carga (uniformemente distribuida en [-0.3, 0.3])
- T_s = período muestreo = 10 ms

**Restricciones:**
- Posición: $-\pi \leq x_1 \leq \pi$ [rad]
- Velocidad: $-20 \leq x_2 \leq 20$ [rad/s]
- Control: $-12 \leq u \leq 12$ [V]

**Referencia:** escalón a $x_{1,\text{ref}} = \pi/2$ en $k=0$; cambios a $-\pi/4$ en $k=2000$.

**Por qué esta planta:** saturación dura + cargas dinámicas → fuerza diseño de memoria (p.ej., bit saturación persistente) e eventos para gestionar transitorios.

---

### 3.2 Planta B: Proceso Térmico (Horno) con Retardos e Inercia

**Modelado (T_s = 100ms):**
$$\begin{cases}
x_{1,k+1} = (1 - \alpha) x_{1,k} + \alpha \, h(x_{2,k}) + w_k \\
x_{2,k+1} = (1 - \beta) x_{2,k} + \beta \, u_{k-\tau_d}
\end{cases}$$

donde:
- $x_1$ = temperatura cámara [°C]
- $x_2$ = temperatura calentador [°C] (retardada)
- $u$ = potencia calentador [0, 100] %
- $\alpha$ = conductancia térmica = 0.1
- $\beta$ = constante térmica calentador = 0.05
- $h(x_2) = 200 \cdot (x_2 / 600)$ [°C] relación no lineal
- $\tau_d$ = retardo = 5 pasos (~500ms)
- $w_k$ = perturbación ambiente (ruido gaussiano σ=2°C)

**Restricciones:**
- Temperatura máxima: $x_1 \leq 200$ [°C]
- Temperatura mínima: $x_1 \geq 20$ [°C] (operación segura)
- Control: $0 \leq u \leq 100$ [%]

**Referencia:** escalón a $x_{1,\text{ref}} = 150$°C; perturbación de ambiente simulada.

**Por qué esta planta:** dinámica lenta + retardos → event-driven debe capturar cambios anticipados (LSTM) antes de que escalpen violaciones; memoria discreta para "calentador enclavado" o "enfriamiento controlado".

---

## 4. MÉTODOS COMPARADOS

### 4.1 Propuesta (Híbrida Event-Driven)

- **Componentes**: Predictor LSTM + Trigger $\delta_k$ + Memoria $m_k$ + MPC + Turbo.
- **Variante implementada**: Turbo-A (warm-start) + Turbo-B (horizonte adaptativo); la más práctica.
- **Configuración base**:
  - MPC horizonte $N=10$ (normal), $N=15$ (crítico)
  - B=3 bits memoria (normal, saturación, crítico)
  - LSTM: 2 capas, 32 hidden, H=10 pasos historia
  - $\eta_{\text{normal}} = 2.0$, $\eta_{\text{critical}} = 0.5$ (en unidades de error)

### 4.2 Baseline 1: MPC Periódico (Clásico)

- Resuelve MPC cada paso ($\delta_k = 1$ siempre).
- **Sin**: predicción, eventos, memoria.
- **Mismo**: MPC, restricciones, solver, tolerancias.
- Representa "control industrial estándar".

### 4.3 Baseline 2: eMPC Clásico (Event-Triggered)

- Disparo por **error instantáneo local**: $\delta_k = \mathbb{1}\{\|x_k - x_{\text{ref}}\| > \theta_{\text{fixed}}\}$.
- $\theta_{\text{fixed}}$ tuned igual presupuesto que nuestra $\eta_{\text{normal}}$.
- **Sin**: predicción temporal rica, sin memoria discreta, sin adaptación por modo.
- Representa "event-triggered simple, el estado del arte conservador".

### 4.4 Baseline 3: Control Aprendido (RL/Imitation sin Memoria Discreta)

- Política neural: $u_k = \text{NN}_{\text{policy}}(x_k, x_{\text{ref}}, \hat{y}_{k|k-1})$ (sin $m_k$).
- Entrenado por imitation learning (clonar MPC periódico con ruido) o RL (PPO).
- **Con** verificación: cheque soft de restricciones post-política (penalización en reward).
- **Sin**: flip-flops explícitos, sin eventos (periódico).
- Aisla el aporte de memoria discreta vs control puro aprendido.

### 4.5 Ablation A1: Sin Flip-Flops

- Remover $m_k$; fijar en modo normal siempre.
- $\eta = \eta_{\text{normal}}$ constante, E($\cdot$) sin acceso a $m_k$.
- Medir: ¿cuánta pérdida sin adaptación discreta?

### 4.6 Ablation A2: Sin Turbo

- Usar MPC puro cuando $\delta_k = 1$; sin aceleración.
- Horizonte fijo $N = 10$ siempre.
- Medir: costo cómputo de Turbo y transitorios.

### 4.7 Ablation A3: Sin Eventos (Periódico)

- Forzar $\delta_k = 1$ siempre (MPC cada paso).
- Cota superior cómputo; mostrar ganancia event-driven.

### 4.8 Ablation A4: Predictor Simplificado

- Reemplazar LSTM por **filtro Kalman** lineal simple.
- Medir aporte específico del aprendizaje temporal vs predicción lineal.

---

## 5. IMPLEMENTACIÓN DETALLADA

### 5.1 Memoria Discreta $m_k$

**B = 3 bits, semántica:**

| Bit | Nombre | Significado | Condición Activación |
|-----|--------|-------------|----------------------|
| 0 | `normal` | Modo operación nominal | x no cerca restricciones |
| 1 | `saturated` | Control saturado persistente | \|u\| > 11 V por ≥3 pasos |
| 2 | `critical` | Riesgo alto violación predicho | E_risk > 1.0 o margen <5% |

**Reglas de actualización** (LUT, transición estado):
```
m_{k+1} = g(m_k, φ_k)

φ_k computa:
  - cond_saturated = (|u_{k-1}| > 11) & (|u_{k-2}| > 11) & (|u_{k-3}| > 11)
  - cond_critical = (E_risk > 1.0) | (max_margin < 0.05)
  - cond_recovery = (|x - x_ref| < 0.2) & (|u| < 5) & (tiempo_en_critico > 100)

Transición:
  m_0[saturated] = saturated | cond_saturated  // set-reset flip-flop
  m_0[saturated] = m_0[saturated] & ~cond_recovery
  
  m_0[critical] = critical | cond_critical  // set-reset
  m_0[critical] = m_0[critical] & ~cond_recovery

  m_0[normal] = ~(saturated | critical)  // complemento
```

**Auditoría**: se loguea cada transición y condición; tabla de verdad validable offline.

### 5.2 Predictor Temporal (LSTM)

**Arquitectura:**
```
Entrada: [x_{k-10:k}, u_{k-10:k-1}, r_k] (historia + referencia)
         → normalizar a media 0, std 1

LSTM:
  - Capas: 2 (stacked)
  - Hidden units: 32 por capa
  - Activación: tanh
  - Dropout: 0.1 (regularización)

Salida: ŷ_{k|k-1} ∈ ℝⁿ
Pérdida: MSE(ŷ - x_actual) + L2(pesos) * 1e-5
```

**Entrenamiento:**
- Generador datos: simulación 10,000 episodios de 500 pasos c/u
  - Perturbaciones: carga ±30%, ruido ±2% en mediciones
  - Referencias: escalones, rampas, cambios aleatorios cada 50–200 pasos
- Split: 70% train, 15% val, 15% test (por seed)
- Optimizer: Adam, lr=1e-3, decay a 1e-4 en plateau (val_loss)
- Early stopping: paciencia=20 epochas
- Batch size: 64
- **Normalización**: media/std estimada en train, congelada en val/test

### 5.3 Trigger por Eventos $E(\cdot)$ y $\eta(m_k)$

**Implementar DOS variantes:**

**Variante 1: E_error (basada en innovación)**
$$E_{\text{error}}(x_k, \hat{y}_{k|k-1}, m_k) = \|x_k - \hat{y}_{k|k-1}\|_2$$

- No normalizado → dependiente escala estado.
- Típico: $\eta_{\text{normal}} = 2.0$, $\eta_{\text{critical}} = 0.5$ (en [rad] o [°C]).

**Variante 2: E_risk (margen a restricciones)**
$$E_{\text{risk}} = \max(0, -\min_i g_i(x_k))$$
donde $g_i(x) \leq 0$ es restricción i.

Para motor: $g_1 = x_1 + \pi$, $g_2 = \pi - x_1$, $g_3 = x_2 + 20$, $g_4 = 20 - x_2$.
Agregado: $E_{\text{risk}} = -\min(g_1, g_2, g_3, g_4) + \text{penalty predictiva}$.

- Más robusto ante cambios de operación.
- Típico: $\eta_{\text{normal}} = 0.1$, $\eta_{\text{critical}} = 0.02$ (margen normalizado).

**Umbral adaptativo:**
$$\eta(m_k) = \begin{cases}
\eta_{\text{normal}} & m_k[\text{critical}] = 0\\
\eta_{\text{critical}} & m_k[\text{critical}] = 1
\end{cases}$$
con transición suave opcional (interpolación lineal si margen intermedio).

### 5.4 Turbo (Warm-start + Horizonte Adaptativo)

**Turbo-A: Warm-start aprendido**
```python
def turbo_a(x_k, ŷ, m_k, mpc_params):
  # Propuesta inicial desde NN rápido
  u_init = lstm_policy(ŷ, m_k, x_k)  # ~100 operaciones
  
  # Resolver MPC con warm-start
  u_sol = mpc_solver(
    x_k, 
    initial_guess=u_init,
    max_iterations=50  # vs 200 normal
  )
  
  if solver_converged:
    return u_sol
  else:
    # Fallback: MPC completo
    return mpc_solver(x_k, max_iterations=200)
```

**Turbo-B: Horizonte adaptativo**
```python
def turbo_b(x_k, ŷ, m_k, E_val):
  if m_k[critical] == 1 or E_val > η_mid:
    N = N_long  # 15 pasos (planificación profunda en crisis)
  else:
    N = N_short  # 10 pasos (normal)
  
  return mpc_solver(x_k, horizon=N)
```

**Fallback garantizado**: ambos retornan a MPC periódico puro si hay timeout o no convergencia.

**Reducción típica:**
- Turbo-A: 30–50% menos iteraciones en estado estacionario.
- Turbo-B: 20–40% menos evaluaciones en normal, +10% en crítico (aceptable).

---

## 6. MÉTRICAS (Definiciones Exactas)

### 6.1 Costo/Error de Tracking

$$J_{\text{track}} = \frac{1}{K}\sum_{k=1}^{K} \left( \|x_k - x_{\text{ref}}\|_Q^2 + \|u_k\|_R^2 \right)$$

con $Q = \text{diag}(1, 0.1)$ para motor (prioriza posición), $R = 0.01$.

**Reporte**: media ± std (sobre 10 seeds × 20 escenarios), percentiles p25, p50, p75, p95.

### 6.2 Violaciones de Restricciones

**Conteo:**
$$N_{\text{viol}} = \sum_{k=1}^{K} \mathbb{1}\{\exists i : g_i(x_k) > \varepsilon\}$$
donde $\varepsilon = 0.01$ tolerancia numérica.

**Magnitud integrada:**
$$V_{\text{mag}} = \sum_{k=1}^{K} \max(0, \max_i g_i(x_k))$$

**Reporte**: conteo absoluto, % episodios con violación, magnitud total.

### 6.3 Tasa de Eventos

$$\rho = \frac{1}{K}\sum_{k=1}^{K}\delta_k$$

**Distribución inter-evento:**
Intervals $\tau_i = $ pasos entre eventos consecutivos; reportar media, std, min, max, histograma.

### 6.4 Tiempo de Cómputo

**Por paso** (solver + predictor + lógica):
$$t_{\text{step}} = t_{\text{predict}} + \mathbb{1}\{\delta_k=1\}(t_{\text{mpc}} + t_{\text{turbo}})$$

**Agregado**:
- Media, std.
- Percentiles: p50, p95, p99.
- Desglose: tiempo LSTM, MPC, Turbo, overhead lógica.

### 6.5 Robustez (Barrido Perturbaciones)

**Escenarios**:
- Carga/ruido nominal (σ base).
- +50% carga/ruido.
- Cambios paramétricos (±10% fricción, inercia, etc.).

**Métrica**:
$$\Delta J = \frac{J_{\text{perturbado}} - J_{\text{nominal}}}{J_{\text{nominal}}} \times 100\%$$

Reportar para cada método y perturbación.

### 6.6 Métrica Compuesta (Opcional)

$$\text{Score} = \alpha \, J_{\text{track}} + \beta \, V_{\text{viol}} + \gamma \, p95(t_{\text{step}})$$

con pesos $(\alpha, \beta, \gamma) = (0.5, 0.3, 0.2)$ normalizados (ej: Score = media±std sobre test).

---

## 7. PROTOCOLO EXPERIMENTAL

### 7.1 Configuración Base

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Número de seeds | 15 | Varianza estadística robusta |
| Escenarios por planta | 25 | Cubre nominal, perturbaciones, cambios ref |
| Longitud episodio | 5000 pasos | ~50s (motor), ~500s (horno); transitorios + SS |
| Condiciones iniciales | Aleatoria ±10% nominal | Robustez ante variación |

### 7.2 Barrido de Hiperparámetros (Tuning Justo)

Presupuesto: 500 evaluaciones por método (total ~4000 evals en paralelo simulado).

**Parámetros sensibles**:

1. **MPC**:
   - Horizonte $N \in \{5, 10, 15, 20\}$
   - Pesos $Q, R$ (3–5 valores cada uno)
   - Solver: OSQP, tolerancia 1e-4

2. **LSTM**:
   - Ventana $H \in \{5, 10, 20\}$
   - Hidden $\in \{16, 32, 64\}$
   - Dropout $\in \{0, 0.1, 0.2\}$

3. **Trigger**:
   - $\eta_{\text{normal}} \in \{1.0, 1.5, 2.0, 2.5\}$
   - $\eta_{\text{critical}} \in \{0.3, 0.5, 0.7\}$ (ratio crítico/normal ≈ 0.25–0.35)

4. **Turbo**:
   - Warm-start (sí/no)
   - Horizonte largo $N_{\text{long}} \in \{12, 15, 20\}$ si se activa adaptativo

**Estrategia tuning**: Random Search (hipercubo) + Early Stopping por validación (5-fold CV en datos de entrenamiento LSTM).

### 7.3 Escenarios de Prueba

**Motor (Planta A), 25 escenarios:**
1. Nominal (carga ±0%).
2–5. Carga +10%, +20%, +30%, -10%, -20%, -30% (constante).
6–10. Carga variante (rampa, escalón, senoidal).
11–15. Cambios referencia (escalones, rampas, ondas).
16–20. Ruido medición (σ = 0, 0.01, 0.05, 0.1, 0.2 rad).
21–25. Combinados (carga + ruido + cambios ref).

**Horno (Planta B), 25 escenarios:**
Similar, con énfasis en retardos, ruido temperatura, perturbaciones ambiente.

### 7.4 Criterios de Reproducibilidad

- **Código**: GitHub (público), tag v1.0.
- **Versiones**: Python 3.10, CasADi 3.5.5, NumPy 1.24, PyTorch 2.0.
- **Hardware**: CPU Intel i7-12700 (reportar tiempo en unidades CPU, no wallclock).
- **Seeds**: NumPy/PyTorch seeds 0–14 fijas.
- **Salida**: CSV con todas métricas por escenario/seed/método.

### 7.5 Significancia Estadística

- **Test no paramétrico**: Mann-Whitney U (dos muestras) o Kruskal-Wallis (k muestras) con α=0.05.
- **Intervalo confianza**: Bootstrap 10,000 resamples, percentiles 2.5–97.5%.
- **Reporte**: p-value + IC 95% para diferencia de medianas.

---

## 8. TABLAS Y FIGURAS

### Tabla 1: Resultados Principales (por Planta)

```
╔═══════════════════╦═════════════════╦═══════════════╦════════════════╦═════════════╗
║ Método            ║ J_track (mean±  ║ N_viol (%)    ║ ρ (events/100) ║ t_p95 (ms)  ║
║                   ║ std, p95)       ║               ║                ║             ║
╠═══════════════════╬═════════════════╬═══════════════╬════════════════╬═════════════╣
║ MPC Periódico     ║ X.XX ± Y.YY     ║ 0.0           ║ 100.0          ║ 12.5        ║
║ eMPC Clásico      ║ X.XX ± Y.YY     ║ 2.1           ║ 35.2           ║ 8.3         ║
║ Aprendido sin m_k ║ X.XX ± Y.YY     ║ 5.3           ║ 25.0           ║ 2.1         ║
║ **Propuesta**     ║ **X.XX ± Y.YY** ║ **0.1**       ║ **28.5**       ║ **5.2**     ║
║ Propuesta + A1    ║ X.XX ± Y.YY     ║ 1.8           ║ 32.1           ║ 5.1         ║
║ Propuesta + A2    ║ X.XX ± Y.YY     ║ 0.2           ║ 28.8           ║ 8.1         ║
║ Propuesta + A3    ║ X.XX ± Y.YY     ║ 0.0           ║ 100.0          ║ 11.9        ║
║ Propuesta + A4    ║ X.XX ± Y.YY     ║ 0.5           ║ 31.2           ║ 5.0         ║
╚═══════════════════╩═════════════════╩═══════════════╩════════════════╩═════════════╝
```
(Valores a llenar en experimentos. Nota: p95 en ms, ρ en %, violaciones en % episodios.)

### Tabla 2: Ablation (Cambio Relativo vs Propuesta Base, %)

```
╔═══════════════════╦═════════════════╦═══════════════╦════════════════╗
║ Ablation          ║ ΔJ_track (%)    ║ ΔN_viol (%)   ║ Δt_p95 (%)     ║
╠═══════════════════╬═════════════════╬═══════════════╬════════════════╣
║ A1: Sin m_k       ║ +5–10%          ║ +300–500%     ║ -2%            ║
║ A2: Sin Turbo     ║ +1–3%           ║ +0.5%         ║ +50%           ║
║ A3: Sin eventos   ║ +2–5%           ║ -10%          ║ -85% (periódico║
║ A4: Kalman vs ML  ║ +8–15%          ║ +200–400%     ║ -5%            ║
╚═══════════════════╩═════════════════╩═══════════════╩════════════════╝
```
(Valores iniciales estimados; confirmados en experimentos.)

### Figura 1: Diagrama de Arquitectura

```
┌──────────────────────────────────────────────────────────────────────┐
│                  CONTROLADOR HÍBRIDO EVENT-DRIVEN                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Planta:  x_{k+1} = f(x_k, u_k, w_k)                              │
│     ↑                                                               │
│     └─── Medición x_k                                              │
│          │                                                           │
│          ├─→ [ LSTM Predictor ] → ŷ_{k|k-1}                       │
│          │   (historia H pasos)                                     │
│          │                                                           │
│          ├─→ [ Evento Evaluator ]                                  │
│          │   E(x_k, ŷ, m_k) vs η(m_k)  →  δ_k ∈ {0,1}            │
│          │                                                           │
│          ├─→ [ Lógica Discreta ]  (tabla transición)               │
│          │   φ_k = φ(x_k, u_k, ŷ, violaciones)                    │
│          │   m_{k+1} = g(m_k, φ_k)  [3 flip-flops]                │
│          │                                                           │
│    ┌─────┴──────────────────────────────────────────────────────┐  │
│    │ SI δ_k = 1 (Evento):                                       │  │
│    │   ├─→ [ Turbo-A: Warm-start ] → u_init                    │  │
│    │   ├─→ [ Turbo-B: Horizonte Adapt ] → N_k                  │  │
│    │   └─→ [ MPC Solver ] (CasADi/OSQP)                        │  │
│    │       u_k = π_MPC(x_k, ŷ, m_k, N_k)                       │  │
│    │ SI δ_k = 0 (Sin evento):                                   │  │
│    │   └─→ u_k = u_{k-1}  (hold)                               │  │
│    └─────┬──────────────────────────────────────────────────────┘  │
│          │                                                           │
│          └─→ Aplicar u_k a Planta                                  │
│                                                                      │
│   [Logging]: J_k, violations, δ_k, tiempo solver, m_k              │
└──────────────────────────────────────────────────────────────────────┘
```

### Figura 2: Trayectorias de Tracking (Caso Nominal y Crítico)

```
Subplots:
(2a) Motor nominal: x_1 (posición) vs tiempo, ref escalón
     - línea roja: Propuesta
     - línea azul: MPC periódico
     - línea verde: eMPC clásico
     - sombreado gris: restricciones

(2b) Motor crítico (carga +50%): igual, con transitorios amplificados

(2c) Horno nominal: x_1 (temperatura)

(2d) Horno con perturbaciones (ruido + cambios ref)
```
Medidas: RMSE en ss, máximo error transitorio, tiempo asentamiento.

### Figura 3: Distribución Cómputo

```
Boxplot (violín plot) por método:
- Eje x: Métodos
- Eje y: t_paso (ms), escala log
- Mediana, IQR, whiskers, puntos outliers
- Línea roja: p95 resaltada
- Reporte n=375 (15 seeds × 25 escenarios)
```

### Figura 4: Tasa de Eventos

```
(4a) Distribución inter-evento (histograma)
     Pasos entre δ_k=1 consecutivos

(4b) Evolución ρ(t) = tasa acumulada en ventanas móviles (20 pasos)
     Mostrar transitorios vs estado estacionario
```

### Figura 5: Robustez (Perturbaciones)

```
Heatmap:
- Filas: métodos
- Columnas: escenarios (carga ±%, ruido σ, cambios param)
- Celdas: color gradiente ΔJ (% cambio vs nominal)
- Azul: robusto (ΔJ pequeño); Rojo: sensible
```

---

## 9. PSEUDOCÓDIGO (Algoritmo Principal)

```
╔══════════════════════════════════════════════════════════════════════════╗
║ ALGORITMO 1: Controlador Híbrido Event-Driven con Turbo y Memoria       ║
╚══════════════════════════════════════════════════════════════════════════╝

INPUT: Planta x_{k+1} = f(x_k, u_k, w_k)
       MPC parámetros: Q, R, N, solver tol
       LSTM weights (pre-entrenado)
       Umbral η, parámetros Turbo
OUTPUT: Secuencia u_0, ..., u_{K-1}, métricas

═════════════════════════════════════════════════════════════════════════════

// INICIALIZACIÓN
u_{-1} ← 0
m_0 ← [0, 0, 0]  // normal, not saturated, not critical
buffer_history ← deque(maxlen=H)  // histórico últimas H mediciones
t_compute ← []  // log de tiempos

for k = 0 to K-1:

  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. LEER MEDICIÓN y ACTUALIZAR HISTORIA                         │
  └─────────────────────────────────────────────────────────────────┘
  
  Read x_k from sensor
  buffer_history.append(x_k)
  
  ┌─────────────────────────────────────────────────────────────────┐
  │ 2. PREDICCIÓN TEMPORAL (LSTM)                                   │
  └─────────────────────────────────────────────────────────────────┘
  
  t_start_lstm ← time()
  
  features ← Normalize([buffer_history, u_{k-H+1:k}, ref_k])
  ŷ_{k|k-1} ← LSTM.forward(features)  // predicción 1-paso adelante
  
  e_pred ← ||x_k - ŷ_{k|k-1}||_2
  
  t_lstm ← time() - t_start_lstm
  
  ┌─────────────────────────────────────────────────────────────────┐
  │ 3. EVALUAR EVENTO                                               │
  └─────────────────────────────────────────────────────────────────┘
  
  // Calcular función evento (dos opciones)
  IF use_error_trigger:
    E_val ← e_pred
  ELSE (use_risk_trigger):
    margin ← min{g_i(x_k)}  // menor margen a restricción
    E_val ← -margin + (e_pred / e_pred_ref)  // ajuste predictor
  
  η_active ← η(m_k[critical])  // umbral adaptativo
  
  δ_k ← 1 if E_val > η_active else 0  // DISPARO
  
  ┌─────────────────────────────────────────────────────────────────┐
  │ 4. LÓGICA DISCRETA (ACTUALIZAR FLIP-FLOPS)                     │
  └─────────────────────────────────────────────────────────────────┘
  
  // Condiciones para transición
  cond_saturated ← (|u_{k-1}| > 11) AND (|u_{k-2}| > 11) ...
  cond_critical ← (E_val > 1.0) OR (margin < 0.05 * range)
  cond_recover ← (|x - ref| < tol) AND (~saturated) AND (steps_critico > 100)
  
  φ_k ← (cond_saturated, cond_critical, cond_recover)
  
  // Transición de estado discreto (tabla / lógica)
  m_0[saturated] ← m_0[saturated] OR cond_saturated
  m_0[saturated] ← m_0[saturated] AND NOT cond_recover
  
  m_0[critical] ← m_0[critical] OR cond_critical
  m_0[critical] ← m_0[critical] AND NOT cond_recover
  
  m_{k+1} ← UpdateMemory(m_k, φ_k)  // LUT o función
  
  ┌─────────────────────────────────────────────────────────────────┐
  │ 5. CONTROL: HOLD o RESOLVER MPC                                │
  └─────────────────────────────────────────────────────────────────┘
  
  IF δ_k == 0:  // Sin evento
    // ─────────────────────────────────────────────────────────
    u_k ← u_{k-1}  // Mantener última entrada
    t_mpc ← 0
    iter_mpc ← 0
    turbo_applied ← FALSE
    
  ELSE:  // Evento: resolver MPC (posiblemente con Turbo)
    // ─────────────────────────────────────────────────────────
    t_start_mpc ← time()
    
    // ─── TURBO-A: Warm-start aprendido ─────────────────────
    IF use_turbo_a:
      t_start_init ← time()
      u_init ← LSTM_policy(ŷ, m_k, ref_k)  // rápido (~100 ops)
      t_init ← time() - t_start_init
      
      // Resolver con inicialización
      u_k, iter_mpc, converged_a ← MPC_solver(
        x_k, ŷ, m_k,
        initial_guess=u_init,
        max_iter=50  // reducido
      )
      
      IF NOT converged_a:
        // Fallback: MPC completo sin inicialización
        u_k, iter_mpc, _ ← MPC_solver(x_k, ŷ, m_k, max_iter=200)
    ELSE:
      t_init ← 0
    
    // ─── TURBO-B: Horizonte adaptativo ────────────────────
    IF use_turbo_b:
      IF m_k[critical] OR E_val > η_mid:
        N_k ← N_long  // 15
        turbo_b_active ← TRUE
      ELSE:
        N_k ← N_base  // 10
        turbo_b_active ← FALSE
    ELSE:
      N_k ← N_base
      turbo_b_active ← FALSE
    
    // Resolver MPC (con horizonte adaptado)
    u_k, iter_mpc, _ ← MPC_solver(
      x_k, ŷ, m_k,
      horizon=N_k,
      Q=Q_nom, R=R_nom
    )
    
    t_mpc ← time() - t_start_mpc
    turbo_applied ← (use_turbo_a OR use_turbo_b)
  
  ┌─────────────────────────────────────────────────────────────────┐
  │ 6. APLICAR CONTROL y AVANZAR PLANTA                            │
  └─────────────────────────────────────────────────────────────────┘
  
  x_{k+1}, violación_k ← ApplyAndSimulate(f, x_k, u_k, w_k)
  
  ┌─────────────────────────────────────────────────────────────────┐
  │ 7. LOGGING y MÉTRICAS                                           │
  └─────────────────────────────────────────────────────────────────┘
  
  J_k ← ||x_k - ref_k||_Q^2 + ||u_k||_R^2
  
  Log:
    - step: k
    - x_k, u_k, ŷ, e_pred, E_val, η_active
    - δ_k, m_k, m_{k+1}
    - violación_k, margen mínimo
    - t_lstm, t_mpc, t_total, iter_mpc
    - turbo_applied, N_k (si aplica)
    - J_k, costo acumulado
  
  t_compute.append(t_lstm + t_mpc)

end for

═════════════════════════════════════════════════════════════════════════════

OUTPUT:
  - Trayectoria (x, u, ref) completa
  - Métricas por paso (J, violaciones, tiempos, eventos, memoria)
  - Estadísticos: media/std/p95 de cómputo, tasa eventos, etc.
  - Logs auditables (transiciones discretas, decisiones Turbo)
```

---

## 10. CHECKLIST REPRODUCIBLE (Artefactos en Repo)

```
event_driven_hybrid_control/
├── README.md (instrucciones setup, ejecución, papers ref)
│
├── config/
│   ├── motor_params.yaml  (Plant A: parámetros, restricciones)
│   ├── horno_params.yaml  (Plant B)
│   ├── mpc_base.yaml      (horizonte, pesos Q/R)
│   ├── lstm_config.yaml   (arch, layers, hidden, dropout)
│   ├── trigger_params.yaml (η_normal, η_critical, tipo E)
│   └── turbo_config.yaml  (warm-start, horizonte adapt, etc.)
│
├── src/
│   ├── plants.py         (f(x, u, w) para motor, horno)
│   ├── mpc_solver.py     (wrapper CasADi/OSQP)
│   ├── lstm_predictor.py (modelo LSTM, entrenamiento)
│   ├── discrete_logic.py (g(m, φ), tabla flip-flops)
│   ├── event_trigger.py  (E_error, E_risk, η(m))
│   ├── turbo.py          (Turbo-A, Turbo-B)
│   ├── controller_hybrid.py (loop principal, Algoritmo 1)
│   ├── metrics.py        (J_track, violaciones, cómputo, robustez)
│   └── utils.py          (normalización, logging, seeds)
│
├── experiments/
│   ├── train_lstm.py     (generar datos, entrenar LSTM, guardar weights)
│   ├── run_baselines.py  (MPC periódico, eMPC, RL-sin-m)
│   ├── run_proposed.py   (propuesta + ablations A1/A2/A3/A4)
│   ├── scenarios.py      (definir 25 escenarios × 2 plantas)
│   └── evaluate.py       (compilar métricas, tablas, figuras)
│
├── notebooks/
│   ├── 01_EDA_plants.ipynb       (visualizar dinámicas nominales)
│   ├── 02_LSTM_training.ipynb    (seguimiento entrenamiento)
│   └── 03_Results_Analysis.ipynb (plots + estadística)
│
├── data/
│   ├── lstm_weights.pt   (LSTM pre-entrenado)
│   ├── training_trajectories.csv (datos para reproducibilidad)
│   └── results/
│       ├── metrics_seed_0.csv  (por seed, 15 archivos)
│       ├── ...
│       └── summary_table.csv   (agregado)
│
├── results/
│   ├── table_1_main.tex  (Tabla 1, formato LaTeX)
│   ├── table_2_ablation.tex
│   ├── figure_1_arch.pdf
│   ├── figure_2_tracking.pdf
│   ├── figure_3_compute.pdf
│   ├── figure_4_events.pdf
│   └── figure_5_robustness.pdf
│
├── tests/
│   ├── test_plant_dynamics.py
│   ├── test_mpc_solver.py
│   ├── test_discrete_logic.py
│   └── test_reproducibility.py (verificar seeds, resultados)
│
├── requirements.txt  (CasADi, PyTorch, NumPy, Matplotlib, etc.)
├── setup.py
└── LICENSE (MIT)
```

**Instrucciones ejecución rápida:**
```bash
# Setup
git clone <repo>
cd event_driven_hybrid_control
pip install -r requirements.txt

# Entrenar LSTM (si no está pre-entrenado)
python experiments/train_lstm.py --config config/lstm_config.yaml

# Ejecutar experimento completo (15 seeds, 2 plantas, todos métodos)
python experiments/run_proposed.py --seeds 0-14 --plants motor,horno --n_jobs -1

# Generar reportes
python experiments/evaluate.py --output results/

# Ver resultados
# → results/table_*.tex, figure_*.pdf, summary_table.csv
```

---

## 11. RIESGOS y MITIGACIONES

| Riesgo | Impacto | Mitigación |
|--------|---------|-----------|
| LSTM overfitting en train, generalización pobre | Validación débil en nuevos escenarios | Dropout 0.1, L2 1e-5, early stopping, test set diverso (25 escenarios nuevos) |
| MPC no converge en tiempo real → fallback crítico | Desempeño degradado | Timeout robusto, fallback a MPC simple, validar offline con P95 cómputo |
| Flip-flops "pegan" en modo crítico → lentitud | Control subóptimo prolongado | Condición de recuperación explícita, timer máximo 100 pasos críticos |
| Disparador too sensitive → eventos constantemente | Cómputo no reduje | Tuning conservador η_normal=2.0, tests en escenarios nominales previos |
| Disparador too loose → violaciones no detectadas | Falla seguridad | E_risk como fallback + margen físico 5%, pruebas exhaustivas |
| Diferencia plataformas (CPU vs GPU, SO) | Irreproducibilidad tiempos | Semillas fijas, reproducibilidad CPU puro, reporte en "CPU time eq." |
| Comparación "injusta" vs baselines (presupuesto desigual) | Conclusiones sesgadas | Mismo solver, tolerancias, presupuesto tuning, splits CV idénticos |
| No alcanzar Q2 "rigor estadístico" | Rechazo revisión | Mínimo 15 seeds, test Mann-Whitney, 95% IC, reporte p-values |

---

## 12. HIPÓTESIS FALSABLES (Resumen)

1. **H1: Memoria discreta mejora interpretabilidad y seguridad**
   - Nula: no hay diferencia significativa en violaciones entre Propuesta y (Propuesta - A1).
   - Alternativa: A1 (sin flip-flops) tendrá ≥300% más violaciones (p<0.01).

2. **H2: Event-driven reduce cómputo sin degradar desempeño**
   - Nula: p95(t_paso) similar entre evento-driven y periódico.
   - Alternativa: p95 reducción ≥40% con J_track degradación ≤5%.

3. **H3: Turbo acelera transitorios**
   - Nula: A2 (sin Turbo) tiene p95 cómputo similar a propuesta.
   - Alternativa: A2 > Propuesta en ≥50% p95; mejora transitorios <5%.

4. **H4: Umbral adaptativo η(m_k) mejora trade-off**
   - Nula: η fijo equivalente a η(m_k).
   - Alternativa: η(m_k) → 20–30% menos eventos, misma seguridad restricciones.

---

## 13. PRÓXIMOS PASOS (Operacionales)

- [ ] Implementar módulo plantas (motor + horno discreto).
- [ ] Diseñar y entrenar LSTM (10,000 episodios sintéticos).
- [ ] Codificar MPC base (CasADi/OSQP).
- [ ] Implementar lógica discreta (3 flip-flops) y triggers.
- [ ] Codificar Turbo-A (warm-start) y Turbo-B (horizonte).
- [ ] Baselines: MPC periódico, eMPC, control aprendido.
- [ ] Generar 25 escenarios robusto × 2 plantas.
- [ ] Ejecutar 15 seeds (paralelo).
- [ ] Compilar métricas y tablas.
- [ ] Validar reproducibilidad (seeds, tolerancias).
- [ ] Generar figuras publicables.
- [ ] Redactar paper con resultados.

---

**Fin del Plan Experimental**

