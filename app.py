# Pronosticador de Cambio Climatico y Fenomenos Atmosfericos
# Basado en Kaper & Engler (2013), Mathematics and Climate, SIAM.
# Portafolio docente - Calculo Diferencial e Integral / Simuladores STEM.
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.stats import genextreme

# ----------------------------------------------------------------------------
# Configuración general
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pronosticador Climático — Matemáticas y Clima",
    page_icon="🌎",
    layout="wide",
)

SIGMA_SB = 5.67e-8      # Constante de Stefan-Boltzmann [W m^-2 K^-4]
Q_BAR = 342.0           # Insolación media global (S0/4) [W m^-2]
SEG_ANO = 3.1536e7      # Segundos por año

st.title("🌎 Pronosticador de Cambio Climático y Fenómenos Atmosféricos")
st.caption(
    "Modelos matemáticos de Kaper & Engler (2013), *Mathematics and Climate* (SIAM): "
    "balance de energía, caos de Lorenz, El Niño–Oscilación del Sur (ENSO), "
    "ciclos de Milankovitch (Fourier) y teoría de valores extremos."
)

with st.sidebar:
    st.header("📖 Sobre el simulador")
    st.markdown(
        """
        Cada pestaña implementa un **modelo conceptual** del libro, con sus
        ecuaciones, parámetros ajustables y pronósticos interactivos.

        **Uso didáctico (Cálculo Diferencial e Integral):**
        - EDOs y puntos de equilibrio
        - Estabilidad y bifurcaciones
        - Integración numérica (Euler, RK45)
        - Series y transformada de Fourier
        - Ajuste de distribuciones

        **Aviso:** herramienta educativa. No reemplaza los pronósticos
        oficiales (IDEAM, NOAA/CPC, OMM).
        """
    )
    st.divider()
    st.markdown("**Referencia:** Kaper, H. & Engler, H. (2013). *Mathematics and Climate*. SIAM, Philadelphia.")

# ----------------------------------------------------------------------------
# Utilidades compartidas
# ----------------------------------------------------------------------------
def albedo(T, modo, a_fijo=0.3):
    """Albedo planetario: constante o función suave de T (retroalimentación hielo-albedo)."""
    if modo == "Fijo":
        return a_fijo
    # Transición suave: 0.60 (cubierto de hielo) -> 0.30 (libre de hielo), centrada en 268 K
    return 0.45 - 0.15 * np.tanh((T - 268.0) / 4.0)

def f_ebm(T, Q, eps, dF, modo, a_fijo):
    """Balance neto de energía [W/m^2]: entrante - saliente + forzante."""
    return Q * (1.0 - albedo(T, modo, a_fijo)) - eps * SIGMA_SB * T**4 + dF

def equilibrios(Q, eps, dF, modo, a_fijo):
    """Raíces de f_ebm y su estabilidad (f'<0 => estable)."""
    Ts = np.linspace(220, 320, 600)
    fs = f_ebm(Ts, Q, eps, dF, modo, a_fijo)
    raices = []
    for i in range(len(Ts) - 1):
        if fs[i] * fs[i + 1] < 0:
            r = brentq(f_ebm, Ts[i], Ts[i + 1], args=(Q, eps, dF, modo, a_fijo))
            if not raices or abs(r - raices[-1]) > 1e-6:
                raices.append(r)
    out = []
    for r in raices:
        h = 0.05
        der = (f_ebm(r + h, Q, eps, dF, modo, a_fijo)
               - f_ebm(r - h, Q, eps, dF, modo, a_fijo)) / (2 * h)
        out.append((r, der < 0))
    return out

@st.cache_data
def integra_ebm(T0, anos, dF_final, tipo, Q, eps, C, modo, a_fijo, dt=0.02):
    """C dT/dt = f_ebm(T) ; dF(t) constante o en rampa."""
    n = int(anos / dt)
    ts = np.empty(n + 1); Ts = np.empty(n + 1)
    T = T0
    for k in range(n + 1):
        t = k * dt
        ts[k] = t; Ts[k] = T
        dF = dF_final if tipo == "Constante" else dF_final * min(t / anos, 1.0)
        T += dt * f_ebm(T, Q, eps, dF, modo, a_fijo) * SEG_ANO / C
    return ts, Ts

@st.cache_data
def diagrama_bifurcacion(eps, dF):
    """Curva S de equilibrios T* en función de la constante solar (con albedo-T)."""
    factores = np.linspace(0.6, 1.4, 80)
    est, inest = [], []
    for fq in factores:
        for (r, estable) in equilibrios(fq * Q_BAR, eps, dF, "Hielo-albedo", 0.3):
            (est if estable else inest).append((fq, r - 273.15))
    return est, inest

# ----------------------------------------------------------------------------
# TAB 1 — Balance de energía y cambio climático (Cap. 2)
# ----------------------------------------------------------------------------
def tab_ebm():
    st.header("🌡️ Modelo de Balance de Energía (EBM) — Cambio climático")
    st.latex(r"C\,\frac{dT}{dt} \;=\; \bar{Q}\,\bigl(1-\alpha(T)\bigr) \;-\; \varepsilon\,\sigma\, T^{4} \;+\; \Delta F(t)")

    c1, c2, c3 = st.columns(3)
    with c1:
        fq = st.slider("Constante solar (× valor actual)", 0.80, 1.20, 1.00, 0.01)
        eps = st.slider("Emisividad ε (efecto invernadero)", 0.40, 1.00, 0.61, 0.01,
                        help="ε=1 es cuerpo negro sin atmósfera; ε≈0.61 reproduce ≈15 °C actuales.")
    with c2:
        modo = st.radio("Albedo", ["Hielo-albedo", "Fijo"], horizontal=True)
        a_fijo = st.slider("Albedo fijo", 0.10, 0.60, 0.30, 0.01, disabled=(modo != "Fijo"))
    with c3:
        dF = st.slider("Forzante extra ΔF [W/m²] (p. ej. gases)", 0.0, 8.0, 0.0, 0.1)
        C = st.select_slider("Capacidad calorífica C [10⁸ J/m²K]",
                             options=[1.0, 2.0, 4.2, 6.0, 8.0], value=4.2) * 1e8

    Q = fq * Q_BAR
    eqs = equilibrios(Q, eps, dF, modo, a_fijo)

    # --- Gráfico 1: curvas de energía y equilibrios
    Ts = np.linspace(220, 320, 500)
    entrante = Q * (1 - albedo(Ts, modo, a_fijo)) + dF
    saliente = eps * SIGMA_SB * Ts**4
    fig1 = go.Figure()
    fig1.add_scatter(x=Ts - 273.15, y=entrante, name="Energía entrante + ΔF",
                     line=dict(color="#e67e22"))
    fig1.add_scatter(x=Ts - 273.15, y=saliente, name="Energía saliente εσT⁴",
                     line=dict(color="#2980b9"))
    for (r, estable) in eqs:
        fig1.add_scatter(x=[r - 273.15], y=[eps * SIGMA_SB * r**4], mode="markers",
                         marker=dict(size=13, color="#27ae60" if estable else "#c0392b",
                                     symbol="circle" if estable else "x"),
                         name="Equilibrio estable" if estable else "Equilibrio inestable",
                         showlegend=True)
    fig1.update_layout(title="Intersecciones = estados de equilibrio del clima",
                       xaxis_title="Temperatura media global [°C]", yaxis_title="Flujo [W/m²]",
                       height=430, margin=dict(t=50))
    st.plotly_chart(fig1, use_container_width=True)

    cols = st.columns(len(eqs))
    for col, (r, estable) in zip(cols, eqs):
        col.metric(("✅ Estable" if estable else "⚠️ Inestable"), f"{r-273.15:+.1f} °C")

    # --- Gráfico 2: bifurcación
    if modo == "Hielo-albedo":
        est, inest = diagrama_bifurcacion(eps, dF)
        fig2 = go.Figure()
        if est:
            xs, ys = zip(*est)
            fig2.add_scatter(x=xs, y=ys, mode="markers",
                             marker=dict(size=4, color="#27ae60"), name="Ramas estables")
        if inest:
            xs, ys = zip(*inest)
            fig2.add_scatter(x=xs, y=ys, mode="markers",
                             marker=dict(size=4, color="#c0392b"), name="Rama inestable")
        fig2.add_vline(x=fq, line_dash="dash", line_color="gray")
        fig2.update_layout(title="Diagrama de bifurcación: la 'curva S' (Tierra bola de nieve ↔ clima actual)",
                           xaxis_title="Constante solar (× actual)", yaxis_title="T* [°C]",
                           height=400, margin=dict(t=50))
        st.plotly_chart(fig2, use_container_width=True)

    # --- Gráfico 3: evolución temporal
    st.subheader("Pronóstico transitorio T(t)")
    c1, c2 = st.columns(2)
    with c1:
        anos = st.slider("Horizonte [años]", 10, 200, 80, 10)
    with c2:
        tipo = st.radio("Forzante", ["Constante", "Rampa"], horizontal=True)
    T0 = 288.4
    ts, Tsim = integra_ebm(T0, anos, dF, tipo, Q, eps, C, modo, a_fijo)
    fig3 = go.Figure()
    fig3.add_scatter(x=2026 + ts, y=Tsim - 273.15, name="T(t)",
                     line=dict(color="#8e44ad", width=3))
    fig3.update_layout(xaxis_title="Año", yaxis_title="Temperatura [°C]", height=350,
                       margin=dict(t=30))
    st.plotly_chart(fig3, use_container_width=True)
    st.info(f"Temperatura final proyectada: **{Tsim[-1]-273.15:.2f} °C** "
            f"(cambio de **{Tsim[-1]-T0:+.2f} °C** respecto al equilibrio preindustrial de este modelo).")

# ----------------------------------------------------------------------------
# TAB 2 — CO₂ y escenarios a 2100 (Caps. 2 y 10)
# ----------------------------------------------------------------------------
def tab_co2():
    st.header("📈 Escenarios de CO₂ y temperatura a 2100")
    st.latex(r"\Delta F(t) = 5.35\,\ln\frac{C(t)}{C_0},"
             r"\qquad C\,\frac{d\Delta T}{dt} = \Delta F - \gamma(1-g)\,\Delta T")
    st.markdown(
        "Modelo de **respuesta lineal** anclado al clima actual. El factor de retroalimentación "
        "$g$ (vapor de agua, hielo, nubes) amplifica la respuesta: "
        "sensibilidad climática de equilibrio ECS = 3.71/(γ(1−g)) °C por duplicar CO₂."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        esc = st.selectbox("Escenario de emisiones",
                           ["Lineal (+2.5 ppm/año, ritmo actual)",
                            "Cuadrático (emisiones crecientes)",
                            "Estabilización (mitigación fuerte)"])
    with c2:
        g = st.slider("Retroalimentación g", 0.0, 0.85, 0.63, 0.01)
        ecs = 3.71 / (3.32 * (1 - g))
        st.metric("Sensibilidad climática (ECS)", f"{ecs:.1f} °C por 2×CO₂")
    with c3:
        horizonte = st.slider("Horizonte [años desde 2026]", 20, 200, 74, 2)

    t = np.arange(horizonte + 1)
    if esc.startswith("Lineal"):
        Cco2 = 420 + 2.5 * t
    elif esc.startswith("Cuadr"):
        Cco2 = 420 + 2.5 * t + 0.020 * t**2
    else:
        Cco2 = 550 - 130 * np.exp(-t / 45.0)
    dF = 5.35 * np.log(Cco2 / 278.0)

    # Integración del modelo de respuesta lineal
    lam = 3.32 * (1 - g); C = 4.2e8; dt = 0.05
    tf = np.arange(0, horizonte + 1e-9, dt)
    dFi = np.interp(tf, t, dF)
    dT = np.zeros(len(tf))
    for k in range(1, len(tf)):
        dT[k] = dT[k - 1] + dt * (dFi[k - 1] - lam * dT[k - 1]) * SEG_ANO / C

    fig = go.Figure()
    fig.add_scatter(x=2026 + tf, y=dT, name="ΔT(t)",
                    line=dict(color="#c0392b", width=3))
    for nivel, txt in [(1.5, "Meta 1.5 °C"), (2.0, "Meta 2.0 °C")]:
        fig.add_hline(y=nivel, line_dash="dash",
                      line_color="#27ae60" if nivel < 2 else "#e67e22",
                      annotation_text=txt)
    fig.update_layout(title=f"Calentamiento proyectado — {esc.split('(')[0].strip()}",
                      xaxis_title="Año", yaxis_title="ΔT respecto a 2026 [°C]",
                      height=430, margin=dict(t=50))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("CO₂ final", f"{Cco2[-1]:.0f} ppm")
    c2.metric("Forzante final", f"{dF[-1]:.1f} W/m²")
    c3.metric(f"ΔT en {2026 + horizonte}", f"+{dT[-1]:.2f} °C")
    st.caption("Valores relativos a 2026 (el planeta ya acumula ≈1.2–1.3 °C sobre la era preindustrial).")

# ----------------------------------------------------------------------------
# TAB 3 — Caos de Lorenz (Cap. 7)
# ----------------------------------------------------------------------------
@st.cache_data
def integra_lorenz(s, r, b, dx, tmax=40.0):
    def rhs(t, u):
        x, y, z = u
        return [s * (y - x), x * (r - z) - y, x * y - b * z]
    teval = np.arange(0, tmax, 0.01)
    s1 = solve_ivp(rhs, (0, tmax), [1.0, 1.0, 1.0], t_eval=teval, rtol=1e-9, atol=1e-12)
    s2 = solve_ivp(rhs, (0, tmax), [1.0 + dx, 1.0, 1.0], t_eval=teval, rtol=1e-9, atol=1e-12)
    return s1.t, s1.y, s2.y

def tab_lorenz():
    st.header("🌀 Ecuaciones de Lorenz — ¿por qué hay límite de pronóstico?")
    st.latex(r"\dot{x}=\sigma(y-x),\qquad \dot{y}=x(\rho-z)-y,\qquad \dot{z}=xy-\beta z")
    c1, c2, c3, c4 = st.columns(4)
    with c1: s = st.slider("σ (Prandtl)", 1.0, 20.0, 10.0, 0.5)
    with c2: r = st.slider("ρ (Rayleigh)", 5.0, 45.0, 28.0, 0.5,
                           help="ρ>24.74: convección caótica (bifurcación de Hopf).")
    with c3: b = st.slider("β", 1.0, 4.0, 8/3, 0.1)
    with c4: dx0 = st.select_slider("Perturbación inicial Δx₀",
                                    options=[1e-9, 1e-6, 1e-3, 1e-1], value=1e-9)

    t, y1, y2 = integra_lorenz(s, r, b, dx0)
    sep = np.linalg.norm(y1 - y2, axis=0)

    c1, c2 = st.columns(2)
    with c1:
        idx = np.arange(0, len(t), 3)
        fig3d = go.Figure(go.Scatter3d(x=y1[0, idx], y=y1[1, idx], z=y1[2, idx],
                                       mode="lines", line=dict(color=t[idx], colorscale="Viridis", width=2)))
        fig3d.update_layout(title="Atractor de Lorenz", height=480, margin=dict(t=40),
                            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
        st.plotly_chart(fig3d, use_container_width=True)
    with c2:
        figsep = go.Figure()
        figsep.add_scatter(x=t, y=sep, line=dict(color="#c0392b"), name="|Δ(t)|")
        figsep.update_yaxes(type="log")
        figsep.update_layout(title="Efecto mariposa: crecimiento exponencial del error (escala log)",
                             xaxis_title="Tiempo", yaxis_title="|Δ(t)| (log)",
                             height=480, margin=dict(t=40))
        st.plotly_chart(figsep, use_container_width=True)
    st.info("**Lección clave:** el error inicial crece ~exponencialmente (exponente de Lyapunov > 0), "
            "por eso el *pronóstico del tiempo* tiene horizonte de ~2 semanas. "
            "Pero las **estadísticas** del atractor (el clima) son estables y sí proyectables: "
            "esa es la base de los pronósticos climáticos de las otras pestañas.")

# ----------------------------------------------------------------------------
# TAB 4 — ENSO (Cap. 16): pronóstico probabilístico
# ----------------------------------------------------------------------------
@st.cache_data
def ensamble_recarga(T0, h0, R, gh, b, r, sig, meses=24, N=3000, dt=0.1, seed=7):
    """Oscilador de recarga de Jin con forzante estocástico (ensamble Monte Carlo)."""
    rng = np.random.default_rng(seed)
    n = int(meses / dt)
    T = np.full(N, T0); h = np.full(N, h0)
    outT = np.empty((n + 1, N)); outT[0] = T
    for k in range(n):
        T += dt * (R * T + gh * h - 0.002 * T**3) + sig * np.sqrt(dt) * rng.standard_normal(N)
        h += dt * (-r * h - b * T)
        outT[k + 1] = T
    return outT

@st.cache_data
def oscilador_retardo(a, tau, Tend=150.0, dt=0.005):
    """dT/dt = a[T(t) - T(t-τ)] - T^3  (Suarez & Schopf)."""
    lag = int(round(tau / dt)); n = int(Tend / dt)
    buf = np.full(n + lag + 2, 0.4)
    for i in range(lag, n + lag + 1):
        Tt = buf[i]; Td = buf[i - lag]
        buf[i + 1] = np.clip(Tt + dt * (a * (Tt - Td) - Tt**3), -6, 6)
    return dt, buf[lag + 1:]

def clasifica_enso(oni):
    a = abs(oni)
    if a < 0.5:
        return "Neutral", "#7f8c8d"
    cat = ("débil" if a < 1.0 else "moderado" if a < 1.5
           else "fuerte" if a < 2.0 else "muy fuerte")
    return (f"El Niño {cat}", "#c0392b") if oni > 0 else (f"La Niña {cat}", "#2980b9")

def tab_enso():
    st.header("🌊 El Niño – Oscilación del Sur (ENSO)")
    sub1, sub2 = st.tabs(["Pronóstico probabilístico (oscilador de recarga de Jin)",
                          "Oscilador con retardo (Suarez–Schopf)"])

    # ---------- Pronóstico ----------
    with sub1:
        st.latex(r"\frac{dT}{dt}=R\,T+\gamma\,h-cT^{3}+\sigma\,\xi(t),"
                 r"\qquad \frac{dh}{dt}=-r\,h-b\,T")
        st.markdown(
            "$T$: anomalía de temperatura superficial en Niño 3.4 [°C]; "
            "$h$: anomalía de contenido de calor (termoclina) del Pacífico ecuatorial. "
            "El ensamble Monte Carlo convierte el ruido atmosférico en **probabilidades** "
            "de El Niño / Neutral / La Niña, como hacen los pronósticos del CPC/IRI."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            oni = st.number_input("Anomalía actual Niño 3.4 [°C] (consulte CPC/IRI)",
                                  -3.0, 3.0, 0.3, 0.1)
            estado, color = clasifica_enso(oni)
            st.markdown(f"**Estado actual:** <span style='color:{color};font-size:1.2em'>{estado}</span>",
                        unsafe_allow_html=True)
        with c2:
            h0 = st.slider("Anomalía de termoclina h₀", -2.0, 2.0, 0.0, 0.1)
            meses = st.slider("Horizonte [meses]", 6, 24, 18, 1)
        with c3:
            sig = st.slider("Ruido atmosférico σ", 0.05, 0.40, 0.15, 0.01)
            N = st.select_slider("Trayectorias del ensamble", options=[500, 1500, 3000], value=1500)

        # Parámetros calibrados: período ≈ 33 meses, amplitud típica ≈ 1 °C
        R, gh, b, r = 0.01, 0.30, 0.12, 0.03
        outT = ensamble_recarga(oni, h0, R, gh, b, r, sig, meses=meses, N=N)
        dt = 0.1
        idx = (np.arange(meses + 1) / dt).astype(int)
        mensual = outT[idx]                       # (meses+1, N)
        p_nino = (mensual > 0.5).mean(axis=1)
        p_nina = (mensual < -0.5).mean(axis=1)
        p_neut = 1 - p_nino - p_nina
        pct = np.percentile(mensual, [5, 25, 50, 75, 95], axis=1)
        xm = np.arange(meses + 1)

        figp = go.Figure()
        figp.add_scatter(x=np.r_[xm, xm[::-1]], y=np.r_[pct[0], pct[4][::-1]],
                         fill="toself", fillcolor="rgba(41,128,185,0.15)",
                         line=dict(width=0), name="Percentiles 5–95")
        figp.add_scatter(x=np.r_[xm, xm[::-1]], y=np.r_[pct[1], pct[3][::-1]],
                         fill="toself", fillcolor="rgba(41,128,185,0.3)",
                         line=dict(width=0), name="Percentiles 25–75")
        figp.add_scatter(x=xm, y=pct[2], line=dict(color="#1a5276", width=3), name="Mediana")
        figp.add_hline(y=0.5, line_dash="dash", line_color="#c0392b", annotation_text="El Niño (+0.5 °C)")
        figp.add_hline(y=-0.5, line_dash="dash", line_color="#2980b9", annotation_text="La Niña (−0.5 °C)")
        figp.update_layout(title=f"Pluma del pronóstico ENSO a {meses} meses ({N} trayectorias)",
                           xaxis_title="Meses desde hoy", yaxis_title="Anomalía Niño 3.4 [°C]",
                           height=430, margin=dict(t=50))
        st.plotly_chart(figp, use_container_width=True)

        figb = go.Figure()
        figb.add_bar(x=xm, y=p_nino, name="El Niño", marker_color="#c0392b")
        figb.add_bar(x=xm, y=p_neut, name="Neutral", marker_color="#95a5a6")
        figb.add_bar(x=xm, y=p_nina, name="La Niña", marker_color="#2980b9")
        figb.update_layout(barmode="stack", title="Probabilidad por mes (umbral ONI ±0.5 °C)",
                           xaxis_title="Meses desde hoy", yaxis_title="Probabilidad",
                           yaxis_range=[0, 1], height=380, margin=dict(t=50))
        st.plotly_chart(figb, use_container_width=True)

        st.warning("**Contexto Colombia:** en términos generales, El Niño favorece condiciones más "
                   "secas y cálidas en gran parte del país (riesgo de sequías e incendios), mientras "
                   "La Niña favorece lluvias por encima de lo normal (riesgo de inundaciones y "
                   "deslizamientos). Consulte siempre los pronósticos oficiales del IDEAM.")

    # ---------- Retardo ----------
    with sub2:
        st.latex(r"\frac{dT}{dt}=a\,\bigl[T(t)-T(t-\tau)\bigr]-T^{3}")
        st.markdown("El retardo τ representa el tiempo de viaje de las ondas Kelvin/Rossby "
                    "en el Pacífico: la señal del pasado regresa y sostiene la oscilación.")
        c1, c2 = st.columns(2)
        with c1: a = st.slider("a (acoplamiento)", 0.1, 2.0, 1.0, 0.05)
        with c2: tau = st.slider("τ (retardo)", 0.5, 5.0, 2.0, 0.1
