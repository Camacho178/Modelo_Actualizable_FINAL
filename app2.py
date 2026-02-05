import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# ============================================
# CONFIGURACIÓN GENERAL
# ============================================

st.set_page_config(page_title="Impulso — Bienestar y Riesgo", layout="wide")

# ============================================
# ESTILOS GLOBALES IMPULSO (VERSIÓN CORREGIDA)
# ============================================

st.markdown("""
    <style>
        /* Eliminar padding superior del header invisible de Streamlit */
        header[data-testid="stHeader"] {
            height: 0px !important;
            padding: 0px !important;
            margin: 0px !important;
        }


            /* ======== TOP BAR FULL WIDTH ======== */
        .top-nav {
            background-color: #16337b;
            padding: 18px 30px;
            color: white;
            font-size: 22px;
            font-weight: 600;
            border-radius: 0 0 16px 16px;
            margin-bottom: 0px;
            width: 100vw;              /* Fuerza ancho completo */
            margin-left: calc(-50vw + 50%); /* Centra el contenedor */
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* ======== TABS ESTILO IMPULSO ======== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
            padding-left: 0;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: #16337b !important;   /* Color base */
            color: white !important;                /* Texto blanco */
            padding: 10px 18px !important;
            border-radius: 10px 10px 0 0 !important;
            font-weight: 600 !important;
            border: none !important;
        }

        /* ======== TAB SELECCIONADO ======== */
        .stTabs [aria-selected="true"] {
            background-color: white !important;     /* Fondo blanco */
            color: #16337b !important;              /* Texto azul */
            border-bottom: 3px solid #16337b !important;
        }

        /* ======== FIX GLOBAL PARA FONDO ======== */
        html, body, .stApp {
            background-color: #f0f0f0 !important;
            font-family: 'Montserrat', sans-serif;
        }

        /* ======== TOP BAR ======== */
        .top-nav {
            background-color: #16337b;
            padding: 18px 30px;
            color: white;
            font-size: 22px;
            font-weight: 600;
            border-radius: 0 0 16px 16px;
            margin-bottom: 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .top-nav-right {
            font-size: 14px;
            font-weight: 400;
            opacity: 0.9;
        }

        /* ======== KPI CARDS ======== */
        .kpi-card {
            background: white;
            padding: 18px 20px;
            border-radius: 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
            border-top: 4px solid #25b5e8;
            height: 130px;
        }

        .kpi-title {
            font-size: 13px;
            color: #555;
            font-weight: 600;
        }

        .kpi-value {
            font-size: 30px;
            font-weight: 700;
            color: #16337b;
            margin-top: 6px;
        }

        .kpi-sub {
            font-size: 13px;
            color: #25b5e8;
            font-weight: 600;
            margin-top: 2px;
        }

        /* ======== ALERTAS ======== */
        .alert-box {
            background-color: #dbe3eb;
            padding: 12px 14px;
            border-radius: 10px;
            margin-bottom: 8px;
            border-left: 6px solid #16337b;
            font-size: 13px;
            color: #16337b;
        }

        /* ======== TITULOS ======== */
        .section-title {
            font-size: 20px;
            font-weight: 700;
            color: #16337b;
            margin-top: 25px;
            margin-bottom: 10px;
        }

        /* ======== RECOMENDACIONES ======== */
        .rec-summary-box {
            background: white;
            padding: 18px;
            border-radius: 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
            text-align: center;
        }

        .rec-summary-value {
            font-size: 28px;
            font-weight: 700;
            color: #16337b;
        }

        .rec-summary-label {
            font-size: 13px;
            color: #777;
            margin-top: -6px;
        }

        .rec-card {
            background: white;
            padding: 20px;
            border-radius: 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
            margin-bottom: 15px;
            border-left: 6px solid #25b5e8;
        }

        .rec-title {
            font-size: 18px;
            font-weight: 700;
            color: #16337b;
            margin-bottom: 6px;
        }

        .rec-meta {
            font-size: 13px;
            color: #555;
            margin-bottom: 10px;
        }

        .rec-description {
            font-size: 14px;
            color: #333;
            margin-bottom: 12px;
        }

        .rec-tag {
            display: inline-block;
            background-color: #dbe3eb;
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 12px;
            margin-right: 6px;
            color: #16337b;
            font-weight: 600;
        }

        .stButton>button {
            background-color: #25b5e8;
            color: white;
            border-radius: 20px;
            border: none;
            padding: 6px 16px;
            font-weight: 600;
        }

    </style>
""", unsafe_allow_html=True)

# ============================================
# 1. GENERAR DATASET SINTÉTICO
# ============================================

np.random.seed(123)
n = 800

stress = np.random.randint(1, 6, n)
burnout = np.random.randint(1, 6, n)
workload = np.random.randint(60, 150, n)
absenteeism = np.random.randint(0, 80, n)
anxiety = np.random.randint(1, 6, n)

risk_score = (
    0.35 * (burnout/5) +
    0.25 * (stress/5) +
    0.20 * (workload/150) +
    0.10 * (absenteeism/80) +
    0.10 * (anxiety/5) +
    np.random.normal(0, 0.05, n)
)

risk_score = np.clip(risk_score, 0, 1)

risk_level = pd.cut(
    risk_score,
    bins=[0, 0.33, 0.66, 1],
    labels=["Bajo", "Medio", "Alto"]
)

departments = np.random.choice(
    ["Operaciones", "Ventas", "IT", "RRHH", "Finanzas"],
    size=n,
    p=[0.3, 0.25, 0.2, 0.15, 0.1]
)

performance = np.random.normal(80, 8, n)
performance = np.clip(performance, 40, 100)

df = pd.DataFrame({
    "department": departments,
    "stress": stress,
    "burnout": burnout,
    "workload": workload,
    "absenteeism": absenteeism,
    "anxiety": anxiety,
    "risk_score": risk_score,
    "risk_level": risk_level,
    "performance": performance
})

# ============================================
# 2. ENTRENAR MODELO
# ============================================

X = df[["stress", "burnout", "workload", "absenteeism", "anxiety"]]
y = df["risk_level"]

preprocess = ColumnTransformer([
    ("num", "passthrough", X.columns)
])

X_processed = preprocess.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=123, stratify=y
)

rf = RandomForestClassifier(n_estimators=600, random_state=123)
rf.fit(X_train, y_train)

# ============================================
# 3. TOP BAR + NAVEGACIÓN
# ============================================

st.markdown("""
<div class='top-nav'>
    <div>Impulso — Bienestar y Riesgo</div>
    <div class='top-nav-right'>Últimos 30 días</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["Dashboard", "Empleados", "Análisis de Riesgo", "Factores de Riesgo", "Segmentación", "Recomendaciones"])

# ============================================
# 4. TAB: DASHBOARD
# ============================================

with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Score de Desempeño</div>
                <div class='kpi-value'>{df['performance'].mean():.1f}/100</div>
                <div class='kpi-sub'>+5.2%</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        high_risk_pct = (df["risk_level"] == "Alto").mean() * 100
        st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Nivel de Riesgo</div>
                <div class='kpi-value' style='color:#E74C3C;'>ALTO</div>
                <div class='kpi-sub'>{high_risk_pct:.1f}% empleados</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Índice de Sobrecarga</div>
                <div class='kpi-value'>{df['workload'].mean():.1f}/150</div>
                <div class='kpi-sub'>Riesgo Alto</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Tasa de Rotación (simulada)</div>
                <div class='kpi-value'>12.8%</div>
                <div class='kpi-sub'>+2.3%</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Alertas</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='alert-box'>{(df['risk_level']=='Alto').sum()} empleados con alto riesgo de burnout en Operaciones</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='alert-box'>{(df['workload']>130).sum()} empleados con carga de trabajo superior al 130%</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='alert-box'>{(df['performance']>95).sum()} empleados cumplieron 100% de objetivos este mes</div>", unsafe_allow_html=True)

# ============================================
# 5. TAB: EMPLEADOS
# ============================================

with tabs[1]:
    st.markdown("<div class='section-title'>Empleados</div>", unsafe_allow_html=True)
    st.dataframe(df)

# ============================================
# 6. TAB: ANÁLISIS DE RIESGO
# ============================================

with tabs[2]:
    st.markdown("<div class='section-title'>Distribución del Riesgo</div>", unsafe_allow_html=True)
    fig_hist = px.histogram(df, x="risk_score", nbins=20, color="risk_level")
    st.plotly_chart(fig_hist, use_container_width=True)

# ============================================
# 7. TAB: FACTORES DE RIESGO
# ============================================

with tabs[3]:
    st.markdown("<div class='section-title'>Factores de Riesgo</div>", unsafe_allow_html=True)
    importances = rf.feature_importances_
    feature_names = preprocess.get_feature_names_out()
    imp_df = pd.DataFrame({"variable": feature_names, "importancia": importances}).sort_values("importancia", ascending=False)
    fig_imp = px.bar(imp_df, x="importancia", y="variable", orientation="h")
    st.plotly_chart(fig_imp, use_container_width=True)

# ============================================
# 8. TAB: SEGMENTACIÓN
# ============================================

with tabs[4]:
    st.markdown("<div class='section-title'>Segmentación</div>", unsafe_allow_html=True)
    seg_df = df.groupby(["department", "risk_level"]).size().reset_index(name="count")
    fig_seg = px.bar(seg_df, x="department", y="count", color="risk_level", barmode="stack")
    st.plotly_chart(fig_seg, use_container_width=True)

# ============================================
# 9. TAB: RECOMENDACIONES (SOLUCIÓN DEFINITIVA)
# ============================================

with tabs[5]:

    st.markdown("<div class='section-title'>Plan de Acción Inmediato</div>", unsafe_allow_html=True)

    # Resumen superior
    c1, c2, c3, c4 = st.columns(4)

    c1.html("""
    <div class='rec-summary-box'>
        <div class='rec-summary-value'>6</div>
        <div class='rec-summary-label'>Total Recomendaciones</div>
    </div>
    """)

    c2.html("""
    <div class='rec-summary-box'>
        <div class='rec-summary-value'>3</div>
        <div class='rec-summary-label'>Pendientes</div>
    </div>
    """)

    c3.html("""
    <div class='rec-summary-box'>
        <div class='rec-summary-value'>2</div>
        <div class='rec-summary-label'>En Progreso</div>
    </div>
    """)

    c4.html("""
    <div class='rec-summary-box'>
        <div class='rec-summary-value'>1</div>
        <div class='rec-summary-label'>Completadas</div>
    </div>
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tarjeta 1
    st.html("""
    <div class='rec-card'>
        <span class='rec-tag'>Alta Prioridad</span>
        <span class='rec-tag'>En Progreso</span>
        <span class='rec-tag'>Correctiva</span>
        <span class='rec-tag'>Bienestar</span>

        <div class='rec-title'>Implementar Programa de Gestión de Estrés en Operaciones</div>

        <div class='rec-meta'>El departamento de operaciones muestra niveles críticos de estrés (78/100).</div>

        <div class='rec-description'>
            Se recomienda implementar talleres de manejo de estrés, pausas activas y acceso a apoyo psicológico.
        </div>

        <div class='rec-meta'>
            <b>Dirigido a:</b> Líderes de Operaciones, Recursos Humanos<br>
            <b>Impacto Esperado:</b> Reducción del 25-30% en niveles de estrés en 3 meses<br>
            <b>Responsable:</b> María Torres<br>
            <b>Fecha de creación:</b> 15 Jun 2026
        </div>
    </div>
    """)

    # Tarjeta 2
    st.html("""
    <div class='rec-card'>
        <span class='rec-tag'>Alta Prioridad</span>
        <span class='rec-tag'>Pendiente</span>
        <span class='rec-tag'>Operativa</span>

        <div class='rec-title'>Redistribuir Carga de Trabajo en Subdivisión Norte</div>

        <div class='rec-description'>
            Se detectó una sobrecarga laboral superior al 130% en múltiples equipos.
        </div>

        <div class='rec-meta'>
            <b>Responsable:</b> Carlos Ruiz<br>
            <b>Impacto Esperado:</b> Reducción del 20% en carga laboral<br>
            <b>Fecha de creación:</b> 12 Jun 2026
        </div>
    </div>
    """)

    # Tarjeta 3
    st.html("""
    <div class='rec-card'>
        <span class='rec-tag'>Alta Prioridad</span>
        <span class='rec-tag'>Pendiente</span>
        <span class='rec-tag'>Bienestar</span>

        <div class='rec-title'>Programa de Prevención de Burnout para Alto Riesgo</div>

        <div class='rec-description'>
            Se identificaron múltiples empleados con riesgo alto sostenido durante 3 meses consecutivos.
        </div>

        <div class='rec-meta'>
            <b>Responsable:</b> Ana Martín<br>
            <b>Impacto Esperado:</b> Reducción del 30% en burnout<br>
            <b>Fecha de creación:</b> 10 Jun 2026
        </div>
    </div>
    """)
