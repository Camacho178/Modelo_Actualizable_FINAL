import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import base64
from pathlib import Path

# ============================================
# CONFIGURACIÓN GENERAL
# ============================================

st.set_page_config(page_title="Impulso — Bienestar y Riesgo", layout="wide")

# ============================================
# CONFIG DE GRÁFICOS (VISIBILIDAD DE TEXTO)
# ============================================

def apply_plotly_style(fig, font_color="#16337b", grid_color="#e6e6e6"):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color, size=14),
        legend=dict(font=dict(color=font_color), title_font=dict(color=font_color)),
        transition=dict(duration=500, easing="cubic-in-out"),
    )
    fig.update_xaxes(
        title_font=dict(color=font_color),
        tickfont=dict(color=font_color),
        showgrid=True,
        gridcolor=grid_color,
        zeroline=False,
    )
    fig.update_yaxes(
        title_font=dict(color=font_color),
        tickfont=dict(color=font_color),
        showgrid=True,
        gridcolor=grid_color,
        zeroline=False,
    )
    return fig


def render_dash_card(title, value, sub, delta, icon_text="I", icon_bg="#e8f4fb", accent="#25b5e8"):
    st.markdown(
        f"""
<div class='dash-card'>
    <div class='dash-delta' style='color:{accent};'>{delta}</div>
    <div class='dash-icon' style='background:{icon_bg}; color:{accent};'>{icon_text}</div>
    <div class='dash-title'>{title}</div>
    <div class='dash-value'>{value}</div>
    <div class='dash-sub'>{sub}</div>
</div>
        """,
        unsafe_allow_html=True
    )


# ============================================
# LOGO
# ============================================

APP_DIR = Path(__file__).resolve().parent


def load_logo_base64():
    candidates = [
        "Herramienta.png",
        "herramienta.png",
        "Herramienta.PNG",
        "herramienta.PNG",
    ]
    for name in candidates:
        path = APP_DIR / name
        if path.exists():
            return base64.b64encode(path.read_bytes()).decode("utf-8")

    for path in APP_DIR.iterdir():
        if path.is_file() and path.stem.lower() == "herramienta" and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
            return base64.b64encode(path.read_bytes()).decode("utf-8")
    return ""


logo_b64 = load_logo_base64()

# ============================================
# ESTILOS GLOBALES IMPULSO
# ============================================

st.markdown("""
<style>

    /* FONDO TRANSPARENTE (LIMPIO) */
    html, body, .stApp {
        background: transparent !important;
        font-family: 'Montserrat', sans-serif;
    }

    .stAppViewContainer, .main, .block-container {
        background: transparent !important;
    }

    /* SUAVE GRADIENTE MUY SUTIL (casi invisible) */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background: radial-gradient(1200px 600px at 10% 0%, rgba(37,181,232,0.06), transparent 60%),
                    radial-gradient(900px 500px at 90% 10%, rgba(22,51,123,0.05), transparent 60%);
        pointer-events: none;
        z-index: 0;
    }

    .block-container { position: relative; z-index: 1; }

    /* ELIMINAR TODA LA SEPARACIÓN SUPERIOR */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    .stAppViewContainer {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    .main {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* FONDO GENERAL */

    /* TOP BAR FULL WIDTH */
    .top-nav {
        background-color: #16337b;
        padding: 24px 40px;
        color: white;
        font-size: 34px;
        font-weight: 700;
        border-radius: 0 0 16px 16px;
        margin-bottom: 25px;

        width: 100vw !important;
        margin-left: calc(-50vw + 50%) !important;

        display: flex;
        justify-content: flex-start;
        align-items: center;
        gap: 14px;

        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    .top-nav-right {
        font-size: 16px;
        font-weight: 400;
        opacity: 0.9;
    }

    .logo-mark {
        height: 44px;
        width: auto;
        display: block;
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding-left: 0;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #16337b !important;
        color: white !important;
        padding: 10px 18px !important;
        border-radius: 10px 10px 0 0 !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.25s ease-in-out;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #1d449c !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #16337b !important;
        border-bottom: 3px solid #16337b !important;
    }

    /* KPI CARDS */
    .kpi-card {
        background: white;
        padding: 18px 20px;
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-top: 4px solid #25b5e8;
        height: 130px;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }

    /* ENTRADA ANIMADA EN CARDS */
    .kpi-card, .rec-card, .rec-summary-box, .alert-box, .dash-card, .alert-strip {
        animation: floatIn 0.6s ease both;
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

    /* DASHBOARD CARDS */
    .dash-card {
        background: white;
        padding: 16px 18px;
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e9edf3;
        min-height: 120px;
        position: relative;
        overflow: hidden;
    }

    .dash-icon {
        width: 30px;
        height: 30px;
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 12px;
        margin-bottom: 8px;
    }

    .dash-title {
        font-size: 12px;
        color: #555;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .dash-value {
        font-size: 22px;
        font-weight: 700;
        color: #16337b;
    }

    .dash-sub {
        font-size: 12px;
        color: #7a8aa0;
        margin-top: 2px;
    }

    .dash-delta {
        position: absolute;
        top: 12px;
        right: 14px;
        font-size: 11px;
        font-weight: 700;
    }

    /* STATUS CARDS (EMPLEADOS) */
    .status-card {
        background: white;
        padding: 16px 18px;
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e9edf3;
        min-height: 110px;
    }

    .status-title {
        font-size: 12px;
        color: #7a8aa0;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        font-weight: 700;
        color: #16337b;
    }

    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #2ecc71;
        box-shadow: 0 0 0 4px rgba(46, 204, 113, 0.16);
        display: inline-block;
    }

    .status-dot.inactive {
        background: #b5b5b5;
        box-shadow: 0 0 0 4px rgba(181, 181, 181, 0.18);
    }

    .status-value {
        font-size: 28px;
        font-weight: 700;
        color: #16337b;
        margin-top: 6px;
    }

    .status-sub {
        font-size: 12px;
        color: #7a8aa0;
        margin-top: 2px;
    }

    /* EMPLEADOS */
    .employee-card {
        background: white;
        padding: 12px 14px;
        border-radius: 12px;
        border: 1px solid #e9edf3;
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
    }

    .employee-card.selected {
        border-color: #25b5e8;
        box-shadow: 0 6px 14px rgba(37,181,232,0.18);
    }

    .employee-left {
        display: flex;
        align-items: center;
        gap: 10px;
        min-width: 0;
    }

    .employee-avatar {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: #edf3ff;
        color: #16337b;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 13px;
        flex: 0 0 auto;
        position: relative;
    }

    .presence-dot {
        position: absolute;
        right: -2px;
        bottom: -2px;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #2ecc71;
        border: 2px solid white;
        box-shadow: 0 0 0 2px rgba(46, 204, 113, 0.18);
    }

    .presence-dot.inactive {
        background: #b5b5b5;
        box-shadow: 0 0 0 2px rgba(181, 181, 181, 0.2);
    }

    .employee-name {
        font-size: 13px;
        font-weight: 700;
        color: #16337b;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 220px;
    }

    .employee-role {
        font-size: 12px;
        color: #7a8aa0;
    }

    .risk-pill {
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 999px;
        font-weight: 700;
        display: inline-block;
    }

    .risk-low {
        background: #e8f7f1;
        color: #1f7a5c;
    }

    .risk-mid {
        background: #fff4e8;
        color: #8b6a00;
    }

    .risk-high {
        background: #ffe9ee;
        color: #a63545;
    }

    .profile-card {
        background: white;
        padding: 16px 18px;
        border-radius: 14px;
        border: 1px solid #e9edf3;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 12px;
    }

    .profile-title {
        font-size: 18px;
        font-weight: 700;
        color: #16337b;
    }

    .profile-sub {
        font-size: 12px;
        color: #7a8aa0;
        margin-top: 2px;
        margin-bottom: 8px;
    }

    /* FORM INPUTS */
    .stTextInput label, .stSelectbox label, .stSlider label {
        color: #16337b !important;
        font-weight: 600 !important;
    }

    .stTextInput input {
        color: #16337b !important;
        background: #f5f7fb !important;
        border: 1px solid #dbe3eb !important;
    }

    .stTextInput input::placeholder {
        color: #7a8aa0 !important;
        opacity: 1 !important;
    }

    .stSelectbox [data-baseweb="select"] > div {
        background: #f5f7fb !important;
        border-color: #dbe3eb !important;
        color: #16337b !important;
    }

    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] input {
        color: #16337b !important;
    }

    /* MULTISELECT (SEGMENTACIÓN) */
    .stMultiSelect label {
        color: #16337b !important;
        font-weight: 600 !important;
    }

    .stMultiSelect [data-baseweb="select"] > div {
        background: #f5f7fb !important;
        border-color: #dbe3eb !important;
        color: #16337b !important;
    }

    .stMultiSelect [data-baseweb="select"] span,
    .stMultiSelect [data-baseweb="select"] input {
        color: #16337b !important;
    }

    .stMultiSelect [data-baseweb="tag"] {
        background: #eaf0f6 !important;
        color: #16337b !important;
        border: 1px solid #d6e2ef !important;
        box-shadow: none !important;
    }

    .stMultiSelect [data-baseweb="tag"] span,
    .stMultiSelect [data-baseweb="tag"] svg {
        color: #16337b !important;
    }

    /* ALERTAS */
    .alert-box {
        background-color: #dbe3eb;
        padding: 12px 14px;
        border-radius: 10px;
        margin-bottom: 8px;
        border-left: 6px solid #16337b;
        font-size: 13px;
        color: #16337b;
        transition: transform 0.25s ease;
    }

    .alert-box:hover {
        transform: translateX(6px);
    }

    .alert-strip {
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 10px;
        font-size: 13px;
        border: 1px solid transparent;
    }

    .alert-danger {
        background: #fff3f4;
        border-color: #ff6b81;
        color: #a63545;
    }

    .alert-warning {
        background: #fff9e8;
        border-color: #f2c94c;
        color: #8b6a00;
    }

    .alert-info {
        background: #f0f6ff;
        border-color: #6aa6ff;
        color: #1f4b8f;
    }

    /* TITULOS */
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #16337b;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    /* RECOMENDACIONES */
    .rec-summary-box {
        background: white;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
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
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 15px;
        border-left: 6px solid #25b5e8;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    .rec-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
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

    /* SEGMENTACIÓN */
    .seg-hero-card {
        background: white;
        border: 1px solid #e9edf3;
        border-radius: 14px;
        padding: 14px 16px;
        min-height: 110px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }

    .seg-hero-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .seg-hero-title {
        font-size: 18px;
        font-weight: 700;
        margin-top: 6px;
    }

    .seg-hero-sub {
        font-size: 12px;
        color: #6f7f96;
        margin-top: 2px;
    }

    .seg-hero-card.seg-risk {
        background: #fff3f4;
        border-color: #ffd3da;
    }

    .seg-hero-card.seg-risk .seg-hero-label,
    .seg-hero-card.seg-risk .seg-hero-title {
        color: #a63545;
    }

    .seg-hero-card.seg-sat {
        background: #f1fbf5;
        border-color: #cfeedd;
    }

    .seg-hero-card.seg-sat .seg-hero-label,
    .seg-hero-card.seg-sat .seg-hero-title {
        color: #1f7a5c;
    }

    .seg-hero-card.seg-cap {
        background: #fff8e7;
        border-color: #f5e0a6;
    }

    .seg-hero-card.seg-cap .seg-hero-label,
    .seg-hero-card.seg-cap .seg-hero-title {
        color: #8b6a00;
    }

    .seg-card {
        background: white;
        border: 1px solid #e9edf3;
        border-radius: 14px;
        padding: 14px 16px;
        position: relative;
        min-height: 175px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }

    .seg-card-top {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .seg-card-icon {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: #edf3ff;
        color: #16337b;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 14px;
    }

    .seg-card-title {
        font-size: 14px;
        font-weight: 700;
        color: #16337b;
    }

    .seg-card-sub {
        font-size: 12px;
        color: #7a8aa0;
    }

    .seg-risk-badge {
        position: absolute;
        top: 14px;
        right: 16px;
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 12px;
        color: white;
    }

    .seg-risk-high {
        background: #ff6b81;
    }

    .seg-risk-mid {
        background: #f2c94c;
        color: #714f00;
    }

    .seg-risk-low {
        background: #2ecc71;
    }

    .seg-metrics {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px 12px;
        margin-top: 12px;
        font-size: 12px;
        color: #5d6c80;
    }

    .seg-metrics b {
        display: block;
        font-size: 13px;
        color: #16337b;
    }

    .seg-sub-card {
        background: white;
        border: 1px solid #e9edf3;
        border-radius: 12px;
        padding: 14px 16px;
    }

    .seg-sub-title {
        font-size: 13px;
        font-weight: 700;
        color: #16337b;
        margin-bottom: 6px;
    }

    .seg-sub-row {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: #5d6c80;
        margin-bottom: 4px;
    }

    .seg-progress {
        height: 6px;
        background: #e9edf3;
        border-radius: 999px;
        overflow: hidden;
        margin-top: 8px;
    }

    .seg-progress-bar {
        height: 100%;
        background: #25b5e8;
        border-radius: 999px;
    }

    .seg-capacity-high {
        color: #a63545;
        font-weight: 700;
    }

    .seg-capacity-ok {
        color: #1f7a5c;
        font-weight: 700;
    }

    /* FACTORES DE RIESGO */
    .fr-summary {
        background: #eef4ff;
        border: 1px solid #d7e3ff;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }

    .fr-summary-title {
        font-size: 14px;
        font-weight: 700;
        color: #16337b;
        margin-bottom: 4px;
    }

    .fr-summary-text {
        font-size: 12px;
        color: #5d6c80;
        margin-bottom: 12px;
    }

    .fr-summary-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
    }

    .fr-summary-item {
        background: white;
        border: 1px solid #e9edf3;
        border-radius: 12px;
        padding: 10px 12px;
        text-align: center;
    }

    .fr-summary-value {
        font-size: 18px;
        font-weight: 700;
        color: #16337b;
    }

    .fr-summary-label {
        font-size: 11px;
        color: #7a8aa0;
    }

    .fr-factor-card {
        border-radius: 14px;
        padding: 14px 16px;
        border: 1px solid #e9edf3;
        min-height: 120px;
        position: relative;
    }

    .fr-factor-top {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
    }

    .fr-icon {
        width: 30px;
        height: 30px;
        border-radius: 10px;
        background: white;
        border: 1px solid #e9edf3;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 700;
    }

    .fr-factor-title {
        font-size: 13px;
        font-weight: 700;
        color: #16337b;
    }

    .fr-factor-desc {
        font-size: 12px;
        color: #6f7f96;
    }

    .fr-factor-pct {
        position: absolute;
        right: 16px;
        top: 14px;
        font-weight: 700;
        font-size: 16px;
    }

    .fr-detail-card {
        border: 1px solid #e9edf3;
        border-radius: 14px;
        overflow: hidden;
        margin-bottom: 12px;
        background: white;
    }

    .fr-detail-header {
        padding: 12px 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-left: 4px solid transparent;
    }

    .fr-detail-title {
        font-size: 14px;
        font-weight: 700;
        color: #16337b;
    }

    .fr-detail-sub {
        font-size: 12px;
        color: #6f7f96;
    }

    .fr-detail-pct {
        font-size: 18px;
        font-weight: 700;
    }

    .fr-metrics {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        padding: 10px 16px 8px 16px;
    }

    .fr-metric-chip {
        background: #f7f9fc;
        border: 1px solid #e6edf5;
        border-radius: 10px;
        padding: 8px 10px;
        font-size: 12px;
        color: #5d6c80;
    }

    .fr-progress {
        padding: 0 16px 12px 16px;
    }

    .fr-progress-track {
        height: 8px;
        background: #e9edf3;
        border-radius: 999px;
        overflow: hidden;
        position: relative;
    }

    .fr-progress-bar {
        height: 100%;
        border-radius: 999px;
    }

    .fr-progress-label {
        font-size: 11px;
        color: #7a8aa0;
        margin-top: 6px;
        display: flex;
        justify-content: space-between;
    }

    /* ANÁLISIS DE RIESGO */
    .risk-card {
        background: white;
        border: 1px solid #e9edf3;
        border-radius: 14px;
        padding: 14px 16px;
        min-height: 120px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }

    .risk-card-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
    }

    .risk-icon {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: #edf3ff;
        color: #16337b;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 14px;
    }

    .risk-tag {
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
    }

    .risk-tag.high {
        background: #ffe9ee;
        color: #a63545;
    }

    .risk-tag.med {
        background: #fff4e8;
        color: #8b6a00;
    }

    .risk-tag.low {
        background: #e8f7f1;
        color: #1f7a5c;
    }

    .risk-card-value {
        font-size: 22px;
        font-weight: 700;
        color: #16337b;
    }

    .risk-card-label {
        font-size: 12px;
        color: #5d6c80;
        margin-top: 4px;
    }

    .risk-card-sub {
        font-size: 11px;
        color: #7a8aa0;
        margin-top: 2px;
    }

    .risk-panel {
        background: white;
        border: 1px solid #e9edf3;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }

    .risk-panel-title {
        font-size: 14px;
        font-weight: 700;
        color: #16337b;
        margin-bottom: 10px;
    }

    .risk-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 8px;
    }

    .risk-table th {
        text-align: left;
        font-size: 11px;
        color: #7a8aa0;
        font-weight: 700;
        padding: 4px 10px;
    }

    .risk-table td {
        background: #f7f9fc;
        border: 1px solid #e6edf5;
        padding: 10px 12px;
        border-radius: 10px;
        font-size: 12px;
        color: #16337b;
    }

    .risk-chip {
        padding: 3px 8px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        display: inline-block;
    }

    .risk-chip.high {
        background: #ffe9ee;
        color: #a63545;
    }

    .risk-chip.med {
        background: #fff4e8;
        color: #8b6a00;
    }

    .risk-chip.low {
        background: #e8f7f1;
        color: #1f7a5c;
    }

    .risk-anim {
        animation: riseIn 0.65s ease both;
    }

    .risk-delay-1 { animation-delay: 0.05s; }
    .risk-delay-2 { animation-delay: 0.12s; }
    .risk-delay-3 { animation-delay: 0.18s; }
    .risk-delay-4 { animation-delay: 0.24s; }

    @keyframes riseIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* RECOMENDACIONES (ACORDEON) */
    details.rec-accordion {
        background: white;
        border-radius: 14px;
        border: 1px solid #e9edf3;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 12px;
        overflow: hidden;
    }

    details.rec-accordion[open] {
        box-shadow: 0 8px 18px rgba(0,0,0,0.12);
    }

    details.rec-accordion > summary {
        list-style: none;
        cursor: pointer;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        font-weight: 700;
        color: #16337b;
    }

    details.rec-accordion > summary::-webkit-details-marker {
        display: none;
    }

    .rec-summary-left {
        display: flex;
        flex-direction: column;
        gap: 4px;
        min-width: 0;
    }

    .rec-summary-meta {
        font-size: 12px;
        color: #7a8aa0;
        font-weight: 600;
    }

    .rec-priority {
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        white-space: nowrap;
    }

    .priority-high {
        background: #ffe9ee;
        color: #a63545;
    }

    .priority-medium {
        background: #fff4e8;
        color: #8b6a00;
    }

    .priority-low {
        background: #e8f7f1;
        color: #1f7a5c;
    }

    .rec-details {
        padding: 0 18px 16px 18px;
        border-top: 1px solid #eef2f7;
    }

    .rec-details .rec-description {
        margin: 10px 0 12px 0;
        color: #333;
        font-size: 14px;
    }

    .rec-tags {
        margin-top: 10px;
        margin-bottom: 6px;
    }

    .rec-meta-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px 18px;
        font-size: 13px;
        color: #555;
    }

    .rec-meta-grid b {
        color: #16337b;
    }

    .owner-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-weight: 600;
        color: #16337b;
    }

    .owner-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #2ecc71;
        box-shadow: 0 0 0 3px rgba(46, 204, 113, 0.16);
        display: inline-block;
    }

    .owner-dot.inactive {
        background: #b5b5b5;
        box-shadow: 0 0 0 3px rgba(181, 181, 181, 0.18);
    }

    /* TABLA EMPLEADOS */
    .stDataFrame {
        background-color: white !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
        padding: 10px !important;
    }

    /* BOTONES */
    .stButton>button {
        background-color: #25b5e8;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 6px 16px;
        font-weight: 600;
    }

    /* ANIMACIÓN */
    .fade-in {
        animation: fadeIn 0.8s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes floatIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ANIMACIÓN ENTRADA GRÁFICOS */
    .stPlotlyChart {
        animation: chartIn 0.7s ease both;
    }

    @keyframes chartIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* GRAFICOS TRANSPARENTES */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

</style>
""", unsafe_allow_html=True)

# ============================================
# 1. GENERAR DATASET
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

# Metadatos simulados de empleados
first_names = [
    "Ana", "Luis", "Carlos", "María", "Jorge", "Sofía", "Diego", "Lucía", "Pedro", "Valeria",
    "Miguel", "Carmen", "Fernando", "Paula", "Raúl", "Elena", "Javier", "Gabriela", "Andrés", "Natalia"
]
last_names = [
    "García", "Martínez", "López", "Hernández", "González", "Pérez", "Sánchez", "Romero", "Torres", "Vega",
    "Ruiz", "Flores", "Castro", "Ríos", "Mendoza", "Ortega", "Núñez", "Navarro", "Silva", "Morales"
]
roles_by_dept = {
    "Operaciones": ["Supervisor de Producción", "Coordinador de Operaciones", "Analista de Procesos"],
    "Ventas": ["Ejecutivo de Ventas", "Consultor Comercial", "Key Account"],
    "IT": ["Desarrollador Senior", "Analista de Datos", "Ingeniero de Sistemas"],
    "RRHH": ["Analista de RRHH", "Business Partner", "Especialista en Bienestar"],
    "Finanzas": ["Analista Financiero", "Controller", "Planeación Financiera"],
}

name_rng = np.random.default_rng(321)
first = name_rng.choice(first_names, n)
last = name_rng.choice(last_names, n)
df["employee_name"] = [f"{f} {l}" for f, l in zip(first, last)]
df["employee_role"] = [name_rng.choice(roles_by_dept[d]) for d in df["department"]]
df["employee_id"] = np.arange(1, n + 1)
status_rng = np.random.default_rng(2026)
df["active_status"] = status_rng.choice(["Activo", "Inactivo"], size=n, p=[0.86, 0.14])

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
# 3. TOP BAR
# ============================================

if logo_b64:
    header_html = f"""
<div class='top-nav'>
    <img src="data:image/png;base64,{logo_b64}" class="logo-mark" alt="Impulso" />
</div>
"""
else:
    header_html = """
<div class='top-nav'>
    <div>Impulso</div>
</div>
"""

st.markdown(header_html, unsafe_allow_html=True)

tabs = st.tabs([
    "Dashboard",
    "Empleados",
    "Análisis de Riesgo",
    "Factores de Riesgo",
    "Segmentación",
    "Recomendaciones"
])

# ============================================
# 4. DASHBOARD
# ============================================

with tabs[0]:
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)

    h_left, h_right = st.columns([3, 1])
    with h_left:
        st.markdown("<div class='section-title'>Dashboard General</div>", unsafe_allow_html=True)
    with h_right:
        a1, a2 = st.columns([1, 1])
        with a1:
            st.button("Exportar Reporte", key="dash_export")
        with a2:
            st.selectbox(
                "Periodo",
                ["Últimos 30 días", "Últimos 90 días", "Último año"],
                index=0,
                key="dash_period"
            )

    perf_avg = float(df["performance"].mean())
    high_risk_pct = float((df["risk_level"] == "Alto").mean() * 100)
    avg_risk = float(df["risk_score"].mean())
    workload_avg = float(df["workload"].mean())
    rotation = float(df["absenteeism"].mean() / 80 * 20)
    productivity = float(np.clip(perf_avg + 6, 0, 100))
    compliance = float((df["performance"] >= 85).mean() * 100)
    overload_index = float(workload_avg / 150 * 100)
    tasks_done = int((perf_avg / 100) * 3000)
    efficiency = float(np.clip(100 - df["stress"].mean() * 8, 0, 100))

    if avg_risk < 0.33:
        risk_label = "BAJO"
        risk_color = "#25b5e8"
    elif avg_risk < 0.66:
        risk_label = "MEDIO"
        risk_color = "#16337b"
    else:
        risk_label = "ALTO"
        risk_color = "#E74C3C"

    r1 = st.columns(4)
    with r1[0]:
        render_dash_card("Score de Desempeño", f"{perf_avg:.1f}/100", "Promedio general", "+5.2%", "D", "#e8f4ff", "#25b5e8")
    with r1[1]:
        render_dash_card("Nivel de Riesgo", risk_label, f"{high_risk_pct:.1f}% empleados", "+1.2%", "R", "#ffe9ee", risk_color)
    with r1[2]:
        render_dash_card("Tasa de Rotación", f"{rotation:.1f}%", "Anual", "+2.3%", "T", "#fff4e8", "#f2994a")
    with r1[3]:
        render_dash_card("Productividad", f"{productivity:.0f}%", "Promedio", "+1.8%", "P", "#e9fbf7", "#1f9d8f")

    r2 = st.columns(4)
    with r2[0]:
        render_dash_card("Cumplimiento Objetivos", f"{compliance:.0f}%", "En meta", "+4.0%", "C", "#edf0ff", "#6c7cff")
    with r2[1]:
        render_dash_card("Índice de Sobrecarga", f"{overload_index:.0f}/100", "Carga de trabajo", "+1.2%", "S", "#ffecec", "#ff6b81")
    with r2[2]:
        render_dash_card("Tareas Completadas", f"{tasks_done:,}", "Este mes", "+324", "K", "#e8f8ff", "#25b5e8")
    with r2[3]:
        render_dash_card("Eficiencia Operativa", f"{efficiency:.0f}%", "Operativa", "+3.1%", "E", "#eaf9f1", "#2ecc71")

    st.markdown("<div class='section-title'>Alertas</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='alert-strip alert-danger'>{(df['risk_level']=='Alto').sum()} empleados con alto riesgo de burnout en Operaciones<br><span style='font-size:11px; opacity:0.7;'>Hace 15 min</span></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='alert-strip alert-warning'>{(df['workload']>130).sum()} empleados con carga de trabajo superior al 130%<br><span style='font-size:11px; opacity:0.7;'>Hace 1 hora</span></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='alert-strip alert-info'>{(df['performance']>95).sum()} empleados cumplieron 100% de objetivos este mes<br><span style='font-size:11px; opacity:0.7;'>Hace 2 horas</span></div>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='section-title'>Tendencia y Comparativa</div>", unsafe_allow_html=True)

    c_left, c_right = st.columns(2)

    with c_left:
        periods = 6
        months = pd.date_range(end=pd.Timestamp.today(), periods=periods, freq="M")
        month_map = {
            "Jan": "Ene", "Feb": "Feb", "Mar": "Mar", "Apr": "Abr", "May": "May", "Jun": "Jun",
            "Jul": "Jul", "Aug": "Ago", "Sep": "Sep", "Oct": "Oct", "Nov": "Nov", "Dec": "Dic"
        }
        month_labels = [f"{month_map.get(m.strftime('%b'), m.strftime('%b'))} {m.strftime('%y')}" for m in months]

        rng = np.random.default_rng(5)

        def make_trend(base, noise=1.6):
            trend = np.linspace(base - 4, base + 4, periods) + rng.normal(0, noise, periods)
            return np.clip(trend, 0, 100)

        trend_df = pd.DataFrame({
            "Mes": month_labels,
            "Rendimiento": make_trend(perf_avg),
            "Riesgo": make_trend(avg_risk * 100),
        })

        trend_long = trend_df.melt("Mes", var_name="Indicador", value_name="Indice")
        fig_trend = px.line(trend_long, x="Mes", y="Indice", color="Indicador", markers=True)
        fig_trend.update_traces(line_shape="spline")
        fig_trend.update_layout(
            height=360,
            xaxis=dict(title="Mes"),
            yaxis=dict(title="Índice (0-100)", range=[0, 100]),
            legend_title_text="Indicador",
            hovermode="x unified"
        )
        apply_plotly_style(fig_trend)
        st.plotly_chart(fig_trend, use_container_width=True)

    with c_right:
        dept_perf = df.groupby("department", as_index=False)["performance"].mean()
        fig_dept = px.bar(
            dept_perf,
            x="department",
            y="performance",
            color="department",
            color_discrete_sequence=["#25b5e8", "#16337b", "#ff6b81", "#6c7cff", "#2ecc71"]
        )
        fig_dept.update_layout(
            height=360,
            xaxis=dict(title="Departamento"),
            yaxis=dict(title="Rendimiento Promedio")
        )
        apply_plotly_style(fig_dept)
        fig_dept.update_traces(showlegend=False)
        st.plotly_chart(fig_dept, use_container_width=True)

    st.markdown("<div class='section-title'>Comparación por Área</div>", unsafe_allow_html=True)

    dept_metrics = (
        df.groupby("department", as_index=False)
        .agg(
            performance=("performance", "mean"),
            risk=("risk_score", "mean"),
            workload=("workload", "mean"),
            absenteeism=("absenteeism", "mean"),
        )
    )
    dept_metrics["risk_idx"] = dept_metrics["risk"] * 100
    dept_metrics["workload_idx"] = dept_metrics["workload"] / 150 * 100
    dept_metrics["abs_idx"] = dept_metrics["absenteeism"] / 80 * 100

    c_left, c_right = st.columns(2)

    with c_left:
        fig_bubble = px.scatter(
            dept_metrics,
            x="performance",
            y="workload_idx",
            size="risk_idx",
            color="risk_idx",
            text="department",
            color_continuous_scale=["#25b5e8", "#16337b", "#ff6b81"],
            size_max=38,
            hover_data={
                "performance": ":.1f",
                "workload_idx": ":.1f",
                "risk_idx": ":.1f",
                "abs_idx": ":.1f",
            },
        )
        fig_bubble.update_traces(textposition="top center")
        fig_bubble.update_layout(
            height=360,
            xaxis=dict(title="Rendimiento Promedio"),
            yaxis=dict(title="Carga de Trabajo (0-100)"),
            coloraxis_showscale=False,
        )
        apply_plotly_style(fig_bubble)
        st.plotly_chart(fig_bubble, use_container_width=True)

    with c_right:
        comp_long = dept_metrics.melt(
            id_vars="department",
            value_vars=["performance", "risk_idx", "workload_idx", "abs_idx"],
            var_name="Indicador",
            value_name="Indice",
        )
        comp_long["Indicador"] = comp_long["Indicador"].map({
            "performance": "Rendimiento",
            "risk_idx": "Riesgo",
            "workload_idx": "Sobrecarga",
            "abs_idx": "Ausentismo",
        })
        fig_comp = px.bar(
            comp_long,
            x="department",
            y="Indice",
            color="Indicador",
            barmode="group",
            color_discrete_sequence=["#25b5e8", "#16337b", "#ff6b81", "#6c7cff"],
        )
        fig_comp.update_layout(
            height=360,
            xaxis=dict(title="Departamento"),
            yaxis=dict(title="Índice (0-100)", range=[0, 100]),
        )
        apply_plotly_style(fig_comp)
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# 5. EMPLEADOS
# ============================================

with tabs[1]:
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Empleados</div>", unsafe_allow_html=True)

    total_employees = int(df.shape[0])
    active_count = int((df["active_status"] == "Activo").sum())
    inactive_count = int((df["active_status"] == "Inactivo").sum())
    active_pct = (active_count / total_employees * 100) if total_employees else 0
    inactive_pct = (inactive_count / total_employees * 100) if total_employees else 0

    s1, s2, s3 = st.columns([1.6, 1, 1])
    s1.markdown(
        f"""
<div class='status-card'>
    <div class='status-title'>Total Empleados</div>
    <div class='status-value'>{total_employees}</div>
    <div class='status-sub'>En la base de datos</div>
</div>
        """,
        unsafe_allow_html=True
    )
    s2.markdown(
        f"""
<div class='status-card'>
    <div class='status-pill'><span class='status-dot'></span>Activos</div>
    <div class='status-value'>{active_count}</div>
    <div class='status-sub'>{active_pct:.1f}% de la base</div>
</div>
        """,
        unsafe_allow_html=True
    )
    s3.markdown(
        f"""
<div class='status-card'>
    <div class='status-pill'><span class='status-dot inactive'></span>Inactivos</div>
    <div class='status-value'>{inactive_count}</div>
    <div class='status-sub'>{inactive_pct:.1f}% de la base</div>
</div>
        """,
        unsafe_allow_html=True
    )

    f1, f2, f3 = st.columns([2, 1, 1])
    search_query = f1.text_input("Buscar empleado", placeholder="Buscar empleado...", key="emp_search")
    dept_options = ["Todos"] + sorted(df["department"].unique())
    selected_dept = f2.selectbox("Departamento", dept_options, index=0, key="emp_dept")
    risk_options = ["Todos", "Bajo", "Medio", "Alto"]
    selected_risk = f3.selectbox("Nivel de riesgo", risk_options, index=0, key="emp_risk")

    df_emp = df.copy()
    if selected_dept != "Todos":
        df_emp = df_emp[df_emp["department"] == selected_dept]
    if selected_risk != "Todos":
        df_emp = df_emp[df_emp["risk_level"] == selected_risk]
    if search_query:
        q = search_query.strip().lower()
        df_emp = df_emp[
            df_emp["employee_name"].str.lower().str.contains(q) |
            df_emp["employee_role"].str.lower().str.contains(q)
        ]

    left, right = st.columns([1, 2])

    with left:
        st.markdown("<div class='section-title'>Lista de Empleados</div>", unsafe_allow_html=True)
        list_df = df_emp.sort_values(["risk_score", "performance"], ascending=[False, False])

        if list_df.empty:
            st.info("No hay empleados con los filtros seleccionados.")
        else:
            if "selected_emp_id" not in st.session_state or st.session_state.selected_emp_id not in list_df["employee_id"].values:
                st.session_state.selected_emp_id = int(list_df.iloc[0]["employee_id"])

            selected_emp_id = int(st.session_state.selected_emp_id)

            for row in list_df.head(10).itertuples(index=False):
                name_parts = row.employee_name.split()
                initials = name_parts[0][0] + (name_parts[-1][0] if len(name_parts) > 1 else "")
                presence_class = "inactive" if row.active_status == "Inactivo" else ""
                risk_class = "risk-low" if row.risk_level == "Bajo" else "risk-mid" if row.risk_level == "Medio" else "risk-high"
                card_class = "employee-card selected" if row.employee_id == selected_emp_id else "employee-card"

                card_html = f"""
<div class='{card_class}'>
    <div class='employee-left'>
        <div class='employee-avatar'>{initials}<span class='presence-dot {presence_class}'></span></div>
        <div>
            <div class='employee-name'>{row.employee_name}</div>
            <div class='employee-role'>{row.employee_role} · {row.department}</div>
        </div>
    </div>
    <span class='risk-pill {risk_class}'>{row.risk_level} Riesgo</span>
</div>
                """
                card_col, btn_col = st.columns([5, 1])
                card_col.markdown(card_html, unsafe_allow_html=True)
                if btn_col.button("Ver", key=f"emp_{row.employee_id}"):
                    st.session_state.selected_emp_id = int(row.employee_id)

    with right:
        if df_emp.empty:
            st.info("Selecciona un empleado para ver su perfil detallado.")
        else:
            selected_id = st.session_state.get("selected_emp_id", int(df_emp.iloc[0]["employee_id"]))
            if selected_id not in df_emp["employee_id"].values:
                selected_id = int(df_emp.iloc[0]["employee_id"])
                st.session_state.selected_emp_id = selected_id

            emp = df_emp[df_emp["employee_id"] == selected_id].iloc[0]
            risk_class = "risk-low" if emp["risk_level"] == "Bajo" else "risk-mid" if emp["risk_level"] == "Medio" else "risk-high"
            emp_status_class = "inactive" if emp["active_status"] == "Inactivo" else ""

            st.markdown(
                f"""
<div class='profile-card'>
    <div class='profile-title'>{emp['employee_name']}</div>
    <div class='profile-sub'>{emp['employee_role']} · {emp['department']}</div>
    <div class='profile-sub'>
        <span class='status-pill'><span class='status-dot {emp_status_class}'></span>{emp['active_status']}</span>
    </div>
    <span class='risk-pill {risk_class}'>Riesgo {emp['risk_level']}</span>
</div>
                """,
                unsafe_allow_html=True
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"""
<div class='kpi-card'>
    <div class='kpi-title'>Score de Riesgo</div>
    <div class='kpi-value'>{emp['risk_score']*100:.1f}/100</div>
    <div class='kpi-sub'>Individual</div>
</div>
            """, unsafe_allow_html=True)
            m2.markdown(f"""
<div class='kpi-card'>
    <div class='kpi-title'>Desempeño</div>
    <div class='kpi-value'>{emp['performance']:.1f}/100</div>
    <div class='kpi-sub'>Productividad</div>
</div>
            """, unsafe_allow_html=True)
            m3.markdown(f"""
<div class='kpi-card'>
    <div class='kpi-title'>Sobrecarga</div>
    <div class='kpi-value'>{emp['workload']/150*100:.1f}%</div>
    <div class='kpi-sub'>Carga actual</div>
</div>
            """, unsafe_allow_html=True)
            m4.markdown(f"""
<div class='kpi-card'>
    <div class='kpi-title'>Ausentismo</div>
    <div class='kpi-value'>{emp['absenteeism']/80*100:.1f}%</div>
    <div class='kpi-sub'>Índice estimado</div>
</div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='section-title'>Perfil de Riesgo</div>", unsafe_allow_html=True)
            c_left, c_right = st.columns(2)

            with c_left:
                factor_df = pd.DataFrame({
                    "Factor": ["Estrés", "Burnout", "Sobrecarga", "Ausentismo", "Ansiedad"],
                    "Indice": [
                        emp["stress"] / 5 * 100,
                        emp["burnout"] / 5 * 100,
                        emp["workload"] / 150 * 100,
                        emp["absenteeism"] / 80 * 100,
                        emp["anxiety"] / 5 * 100,
                    ],
                })
                fig_radar = px.line_polar(factor_df, r="Indice", theta="Factor", line_close=True)
                fig_radar.update_traces(fill="toself", line_color="#25b5e8", fillcolor="rgba(37,181,232,0.2)")
                fig_radar.update_layout(
                    height=320,
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False,
                )
                apply_plotly_style(fig_radar)
                st.plotly_chart(fig_radar, use_container_width=True)

            with c_right:
                compare_df = pd.DataFrame({
                    "Métrica": ["Desempeño", "Riesgo", "Sobrecarga", "Ausentismo"],
                    "Empleado": [
                        emp["performance"],
                        emp["risk_score"] * 100,
                        emp["workload"] / 150 * 100,
                        emp["absenteeism"] / 80 * 100,
                    ],
                    "Promedio": [
                        df_emp["performance"].mean(),
                        df_emp["risk_score"].mean() * 100,
                        df_emp["workload"].mean() / 150 * 100,
                        df_emp["absenteeism"].mean() / 80 * 100,
                    ],
                })
                compare_long = compare_df.melt("Métrica", var_name="Grupo", value_name="Indice")
                fig_compare = px.bar(
                    compare_long,
                    x="Métrica",
                    y="Indice",
                    color="Grupo",
                    barmode="group",
                    color_discrete_map={"Empleado": "#16337b", "Promedio": "#dbe3eb"},
                )
                fig_compare.update_layout(
                    height=320,
                    yaxis=dict(title="Índice (0-100)", range=[0, 100]),
                )
                apply_plotly_style(fig_compare)
                st.plotly_chart(fig_compare, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# 6. ANÁLISIS DE RIESGO
# ============================================

with tabs[2]:
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Análisis de Riesgo</div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    dept_options = ["Todos"] + sorted(df["department"].unique())
    selected_dept = f1.selectbox("Departamento", dept_options, index=0, key="risk_dept_ana")
    risk_options = ["Todos", "Bajo", "Medio", "Alto"]
    selected_risk = f2.selectbox("Nivel de riesgo", risk_options, index=0, key="risk_level_ana")
    period_options = ["Últimos 6 meses", "Últimos 12 meses"]
    selected_period = f3.selectbox("Periodo", period_options, index=0, key="risk_period_ana")

    df_risk = df.copy()
    if selected_dept != "Todos":
        df_risk = df_risk[df_risk["department"] == selected_dept]
    if selected_risk != "Todos":
        df_risk = df_risk[df_risk["risk_level"] == selected_risk]

    risk_score = float(df_risk["risk_score"].mean() * 100) if not df_risk.empty else 0
    high_risk_pct = float((df_risk["risk_level"] == "Alto").mean() * 100) if not df_risk.empty else 0
    high_risk_count = int((df_risk["risk_level"] == "Alto").sum()) if not df_risk.empty else 0
    workload_avg = float(df_risk["workload"].mean()) if not df_risk.empty else 0
    abs_rate = float(df_risk["absenteeism"].mean() / 80 * 100) if not df_risk.empty else 0

    def risk_tag(score):
        if score >= 66:
            return "Alto", "high"
        if score >= 33:
            return "Medio", "med"
        return "Bajo", "low"

    score_label, score_class = risk_tag(risk_score)
    abs_label, abs_class = risk_tag(abs_rate)
    hr_label, hr_class = risk_tag(high_risk_pct)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        f"""
<div class='risk-card risk-anim risk-delay-1'>
    <div class='risk-card-top'>
        <div class='risk-icon'>R</div>
        <span class='risk-tag {score_class}'>{score_label}</span>
    </div>
    <div class='risk-card-value'>{risk_score:.0f}/100</div>
    <div class='risk-card-label'>Score de Riesgo General</div>
    <div class='risk-card-sub'>+3.2% este mes</div>
</div>
        """,
        unsafe_allow_html=True
    )
    k2.markdown(
        f"""
<div class='risk-card risk-anim risk-delay-2'>
    <div class='risk-card-top'>
        <div class='risk-icon'>B</div>
        <span class='risk-tag {score_class}'>Alto</span>
    </div>
    <div class='risk-card-value'>{prob_burnout:.0f}%</div>
    <div class='risk-card-label'>Probabilidad de Burnout</div>
    <div class='risk-card-sub'>+5% este mes</div>
</div>
        """,
        unsafe_allow_html=True
    )
    k3.markdown(
        f"""
<div class='risk-card risk-anim risk-delay-3'>
    <div class='risk-card-top'>
        <div class='risk-icon'>A</div>
        <span class='risk-tag {hr_class}'>{high_risk_pct:.0f}%</span>
    </div>
    <div class='risk-card-value'>{high_risk_count}</div>
    <div class='risk-card-label'>Empleados en Alto Riesgo</div>
    <div class='risk-card-sub'>+45 empleados</div>
</div>
        """,
        unsafe_allow_html=True
    )
    k4.markdown(
        f"""
<div class='risk-card risk-anim risk-delay-4'>
    <div class='risk-card-top'>
        <div class='risk-icon'>T</div>
        <span class='risk-tag {abs_class}'>{abs_label}</span>
    </div>
    <div class='risk-card-value'>{abs_rate:.1f}%</div>
    <div class='risk-card-label'>Tasa de Ausentismo</div>
    <div class='risk-card-sub'>+1.2% este mes</div>
</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='section-title'>Evolución Histórica del Riesgo</div>", unsafe_allow_html=True)

    periods = 6 if selected_period == "Últimos 6 meses" else 12
    months = pd.date_range(end=pd.Timestamp.today(), periods=periods, freq="M")
    month_map = {
        "Jan": "Ene", "Feb": "Feb", "Mar": "Mar", "Apr": "Abr", "May": "May", "Jun": "Jun",
        "Jul": "Jul", "Aug": "Ago", "Sep": "Sep", "Oct": "Oct", "Nov": "Nov", "Dec": "Dic"
    }
    month_labels = [f"{month_map.get(m.strftime('%b'), m.strftime('%b'))} {m.strftime('%y')}" for m in months]

    rng = np.random.default_rng(11)

    def make_trend(base, noise=1.8):
        trend = np.linspace(base - 4, base + 4, periods) + rng.normal(0, noise, periods)
        return np.clip(trend, 0, 100)

    trend_df = pd.DataFrame({
        "Mes": month_labels,
        "Riesgo general": make_trend(risk_score),
        "Estrés": make_trend(df_risk["stress"].mean() / 5 * 100 if not df_risk.empty else 0),
        "Burnout": make_trend(df_risk["burnout"].mean() / 5 * 100 if not df_risk.empty else 0),
        "Sobrecarga": make_trend(df_risk["workload"].mean() / 150 * 100 if not df_risk.empty else 0),
    })

    trend_long = trend_df.melt("Mes", var_name="Indicador", value_name="Indice")

    fig_trend = px.line(
        trend_long,
        x="Mes",
        y="Indice",
        color="Indicador",
        markers=True
    )
    fig_trend.update_traces(line_shape="spline")
    fig_trend.update_layout(
        height=420,
        xaxis=dict(title="Mes"),
        yaxis=dict(title="Índice (0-100)", range=[0, 100]),
        legend_title_text="Indicador",
        hovermode="x unified"
    )
    apply_plotly_style(fig_trend)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("<div class='section-title'>Distribución y Categorías</div>", unsafe_allow_html=True)

    c_left, c_right = st.columns(2)

    with c_left:
        risk_dist = (
            df_risk["risk_level"]
            .value_counts(normalize=True)
            .reindex(["Bajo", "Medio", "Alto"])
            .fillna(0)
            .reset_index()
        )
        risk_dist.columns = ["Nivel", "Porcentaje"]
        risk_dist["Porcentaje"] = risk_dist["Porcentaje"] * 100
        fig_donut = px.pie(
            risk_dist,
            names="Nivel",
            values="Porcentaje",
            hole=0.65,
            color="Nivel",
            color_discrete_map={"Bajo": "#25b5e8", "Medio": "#16337b", "Alto": "#ff6b81"}
        )
        fig_donut.update_traces(
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>"
        )
        fig_donut.update_layout(height=360, legend_title_text="Nivel de riesgo")
        apply_plotly_style(fig_donut)
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_right:
        cat_df = pd.DataFrame({
            "Categoría": ["Psicosocial", "Burnout", "Organizacional", "Ausentismo", "Operativo"],
            "Indice": [45, 40, 35, 28, 22],
        })
        fig_cat = px.bar(
            cat_df,
            x="Indice",
            y="Categoría",
            orientation="h",
            color="Indice",
            color_continuous_scale=["#dbe9ff", "#16337b"]
        )
        fig_cat.update_layout(
            height=360,
            xaxis=dict(title="Indice", range=[0, 60]),
            yaxis=dict(title=""),
            coloraxis_showscale=False
        )
        apply_plotly_style(fig_cat)
        st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("<div class='section-title'>Análisis de Horas Extra</div>", unsafe_allow_html=True)

    overtime_rows = [
        {"dept": "Operaciones", "avg": "15.5h", "max": "28h", "affected": 85, "risk": "Alto"},
        {"dept": "Ventas", "avg": "8.2h", "max": "18h", "affected": 52, "risk": "Medio"},
        {"dept": "IT", "avg": "6.5h", "max": "14h", "affected": 38, "risk": "Bajo"},
        {"dept": "RRHH", "avg": "2.5h", "max": "8h", "affected": 12, "risk": "Bajo"},
        {"dept": "Finanzas", "avg": "4.2h", "max": "12h", "affected": 28, "risk": "Bajo"},
    ]

    row_html = ""
    for row in overtime_rows:
        chip_class = "high" if row["risk"] == "Alto" else "med" if row["risk"] == "Medio" else "low"
        row_html += f"""
<tr>
    <td>{row['dept']}</td>
    <td>{row['avg']}</td>
    <td>{row['max']}</td>
    <td>{row['affected']}</td>
    <td><span class='risk-chip {chip_class}'>{row['risk']}</span></td>
</tr>
        """

    table_html = f"""
<div class='risk-panel risk-anim risk-delay-2'>
    <div class='risk-panel-title'>Análisis de Horas Extra</div>
    <table class='risk-table'>
        <thead>
            <tr>
                <th>Departamento</th>
                <th>Promedio Semanal</th>
                <th>Máximo Registrado</th>
                <th>Empleados Afectados</th>
                <th>Nivel de Riesgo</th>
            </tr>
        </thead>
        <tbody>
            {row_html}
        </tbody>
    </table>
</div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# 7. FACTORES DE RIESGO
# ============================================

with tabs[3]:
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Factores de Riesgo Clave</div>", unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        dept_options = ["Todos"] + sorted(df["department"].unique())
        selected_dept = st.selectbox("Departamento", dept_options, index=0, key="risk_dept")
    with f2:
        period_options = ["Últimos 6 meses", "Últimos 12 meses"]
        selected_period = st.selectbox("Periodo", period_options, index=0, key="risk_period")

    df_f = df if selected_dept == "Todos" else df[df["department"] == selected_dept]

    score_general = df_f["risk_score"].mean() * 100
    prob_burnout = df_f["burnout"].mean() / 5 * 100
    high_risk = (df_f["risk_level"] == "Alto").sum()
    abs_rate = df_f["absenteeism"].mean() / 80 * 100

    risk_factors = [
        {
            "name": "Carga Laboral",
            "percent": 34,
            "desc": "Alto estrés debido a tareas excesivas y plazos ajustados",
            "color": "#ff6b81",
            "bg": "#fff3f4",
            "border": "#ffd3da",
            "metrics": [
                "Promedio de horas semanales: 52h",
                "Tareas simultáneas: 8-12 por empleado",
                "Plazos urgentes: 65% de proyectos",
            ],
        },
        {
            "name": "Falta de Apoyo",
            "percent": 25,
            "desc": "Baja percepción de apoyo gerencial y acompañamiento",
            "color": "#25b5e8",
            "bg": "#eff8ff",
            "border": "#cfe9ff",
            "metrics": [
                "Reuniones 1:1 con gerentes: 1 vez/mes",
                "Satisfacción con liderazgo: 52%",
                "Acceso a mentoría: 38% de empleados",
            ],
        },
        {
            "name": "Estigma/Preocupaciones",
            "percent": 18,
            "desc": "Miedo a consecuencias por discutir salud mental",
            "color": "#6aa6ff",
            "bg": "#f2f6ff",
            "border": "#dbe5ff",
            "metrics": [
                "Empleados que reportan problemas: 15%",
                "Percepción de confidencialidad: 48%",
                "Cultura de apertura: 3.2/10",
            ],
        },
        {
            "name": "Falta de Balance",
            "percent": 15,
            "desc": "Dificultad para mantener balance vida-trabajo",
            "color": "#16337b",
            "bg": "#eef2ff",
            "border": "#d4dcff",
            "metrics": [
                "Días de vacaciones usados: 42% del total",
                "Desconexión fuera de horario: 23%",
                "Satisfacción con balance: 4.8/10",
            ],
        },
    ]

    coverage = sum(item["percent"] for item in risk_factors)
    impacted_pct = (
        df_f["risk_level"].isin(["Medio", "Alto"]).mean() * 100 if not df_f.empty else 0
    )
    urgency = "Alto" if impacted_pct >= 60 else "Medio" if impacted_pct >= 40 else "Bajo"

    st.markdown(
        f"""
<div class='fr-summary'>
    <div class='fr-summary-title'>Resumen de Factores</div>
    <div class='fr-summary-text'>
        Los cuatro factores principales representan el {coverage}% del riesgo total identificado.
        La carga laboral sigue siendo el factor más crítico, con impacto directo en burnout y ausentismo.
    </div>
    <div class='fr-summary-grid'>
        <div class='fr-summary-item'>
            <div class='fr-summary-value'>{coverage}%</div>
            <div class='fr-summary-label'>Cobertura Total</div>
        </div>
        <div class='fr-summary-item'>
            <div class='fr-summary-value'>{len(risk_factors)}</div>
            <div class='fr-summary-label'>Factores Clave</div>
        </div>
        <div class='fr-summary-item'>
            <div class='fr-summary-value'>{impacted_pct:.0f}%</div>
            <div class='fr-summary-label'>Empleados Afectados</div>
        </div>
        <div class='fr-summary-item'>
            <div class='fr-summary-value'>{urgency}</div>
            <div class='fr-summary-label'>Nivel de Urgencia</div>
        </div>
    </div>
</div>
        """,
        unsafe_allow_html=True
    )

    for item in risk_factors:
        metrics_html = "".join([f"<div class='fr-metric-chip'>{m}</div>" for m in item["metrics"]])
        detail_html = f"""
<div class='fr-detail-card'>
    <div class='fr-detail-header' style='background:{item["bg"]}; border-left-color:{item["color"]};'>
        <div>
            <div class='fr-detail-title'>{item["name"]}</div>
            <div class='fr-detail-sub'>{item["desc"]}</div>
        </div>
        <div class='fr-detail-pct' style='color:{item["color"]};'>{item["percent"]}%</div>
    </div>
    <div class='fr-metrics'>
        {metrics_html}
    </div>
    <div class='fr-progress'>
        <div class='fr-progress-track'>
            <div class='fr-progress-bar' style='width:{item["percent"]}%; background:{item["color"]};'></div>
        </div>
        <div class='fr-progress-label'>
            <span>Impacto relativo en la organización</span>
            <span>{item["percent"]}%</span>
        </div>
    </div>
</div>
        """
        st.markdown(detail_html, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Evolución del Riesgo</div>", unsafe_allow_html=True)

    periods = 6 if selected_period == "Últimos 6 meses" else 12
    months = pd.date_range(end=pd.Timestamp.today(), periods=periods, freq="M")
    month_map = {
        "Jan": "Ene", "Feb": "Feb", "Mar": "Mar", "Apr": "Abr", "May": "May", "Jun": "Jun",
        "Jul": "Jul", "Aug": "Ago", "Sep": "Sep", "Oct": "Oct", "Nov": "Nov", "Dec": "Dic"
    }
    month_labels = [f"{month_map.get(m.strftime('%b'), m.strftime('%b'))} {m.strftime('%y')}" for m in months]

    rng = np.random.default_rng(7)

    def make_trend(base, noise=1.6):
        trend = np.linspace(base - 4, base + 4, periods) + rng.normal(0, noise, periods)
        return np.clip(trend, 0, 100)

    trend_df = pd.DataFrame({
        "Mes": month_labels,
        "Riesgo general": make_trend(score_general),
        "Estrés": make_trend(df_f["stress"].mean() / 5 * 100),
        "Burnout": make_trend(df_f["burnout"].mean() / 5 * 100),
        "Sobrecarga": make_trend(df_f["workload"].mean() / 150 * 100),
    })

    trend_long = trend_df.melt("Mes", var_name="Indicador", value_name="Indice")

    fig_trend = px.line(
        trend_long,
        x="Mes",
        y="Indice",
        color="Indicador",
        markers=True
    )

    fig_trend.update_traces(line_shape="spline")
    fig_trend.update_layout(
        height=420,
        xaxis=dict(title="Mes"),
        yaxis=dict(title="Índice (0-100)", range=[0, 100]),
        legend_title_text="Indicador",
        hovermode="x unified"
    )

    apply_plotly_style(fig_trend)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("<div class='section-title'>Distribución y Factores</div>", unsafe_allow_html=True)

    c_left, c_right = st.columns(2)

    with c_left:
        risk_dist = (
            df_f["risk_level"]
            .value_counts(normalize=True)
            .reindex(["Bajo", "Medio", "Alto"])
            .fillna(0)
            .reset_index()
        )
        risk_dist.columns = ["Nivel", "Porcentaje"]
        risk_dist["Porcentaje"] = risk_dist["Porcentaje"] * 100

        fig_donut = px.pie(
            risk_dist,
            names="Nivel",
            values="Porcentaje",
            hole=0.6,
            color="Nivel",
            color_discrete_map={"Bajo": "#25b5e8", "Medio": "#16337b", "Alto": "#ff6b81"}
        )
        fig_donut.update_traces(
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>"
        )
        fig_donut.update_layout(height=380, legend_title_text="Nivel de riesgo")
        apply_plotly_style(fig_donut)
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_right:
        factor_scores = pd.DataFrame({
            "Factor": ["Estrés", "Burnout", "Sobrecarga", "Ausentismo", "Ansiedad"],
            "Indice": [
                df_f["stress"].mean() / 5 * 100,
                df_f["burnout"].mean() / 5 * 100,
                df_f["workload"].mean() / 150 * 100,
                df_f["absenteeism"].mean() / 80 * 100,
                df_f["anxiety"].mean() / 5 * 100,
            ]
        }).round(1).sort_values("Indice", ascending=True)

        fig_factors = px.bar(
            factor_scores,
            x="Indice",
            y="Factor",
            orientation="h",
            text="Indice",
            color="Indice",
            color_continuous_scale=["#dbe3eb", "#25b5e8", "#16337b"]
        )
        fig_factors.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_factors.update_layout(
            height=380,
            xaxis=dict(title="Índice de riesgo (0-100)"),
            yaxis=dict(title="Factor", showgrid=False),
            coloraxis_showscale=False
        )
        apply_plotly_style(fig_factors)
        fig_factors.update_yaxes(showgrid=False)
        st.plotly_chart(fig_factors, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# 8. SEGMENTACIÓN
# ============================================

with tabs[4]:
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Segmentación por Áreas</div>", unsafe_allow_html=True)

    segment_depts = [
        {"name": "Operaciones", "employees": 285, "satisfaction": 5.2, "overtime": 12.5, "burnout": 38, "risk": 45},
        {"name": "Ventas", "employees": 220, "satisfaction": 6.8, "overtime": 8.2, "burnout": 25, "risk": 30},
        {"name": "IT", "employees": 195, "satisfaction": 7.5, "overtime": 6.5, "burnout": 18, "risk": 22},
        {"name": "RRHH", "employees": 85, "satisfaction": 8.2, "overtime": 3.8, "burnout": 10, "risk": 12},
        {"name": "Finanzas", "employees": 140, "satisfaction": 8.5, "overtime": 2.5, "burnout": 6, "risk": 8},
        {"name": "Marketing", "employees": 120, "satisfaction": 7.8, "overtime": 5.2, "burnout": 12, "risk": 15},
    ]

    subdivisions = [
        {"name": "Norte", "employees": 380, "capacity": 130, "risk": 42},
        {"name": "Sur", "employees": 285, "capacity": 75, "risk": 18},
        {"name": "Este", "employees": 320, "capacity": 95, "risk": 28},
        {"name": "Oeste", "employees": 260, "capacity": 85, "risk": 22},
    ]

    dept_names = [d["name"] for d in segment_depts]
    sub_names = [s["name"] for s in subdivisions]

    spacer, f_dept, f_sub = st.columns([2.4, 1.2, 1.2])
    with f_dept:
        selected_dept = st.selectbox(
            "Departamento",
            ["Todos los Departamentos"] + dept_names,
            index=0,
            key="seg_dept_v2"
        )
    with f_sub:
        selected_sub = st.selectbox(
            "Subdivisión",
            ["Todas las Subdivisiones"] + sub_names,
            index=0,
            key="seg_sub_v2"
        )

    dept_filtered = (
        segment_depts if selected_dept == "Todos los Departamentos"
        else [d for d in segment_depts if d["name"] == selected_dept]
    )
    sub_filtered = (
        subdivisions if selected_sub == "Todas las Subdivisiones"
        else [s for s in subdivisions if s["name"] == selected_sub]
    )

    risk_top = max(segment_depts, key=lambda d: d["risk"])
    sat_top = max(segment_depts, key=lambda d: d["satisfaction"])
    cap_top = max(subdivisions, key=lambda s: s["capacity"])

    h1, h2, h3 = st.columns(3)
    h1.markdown(
        f"""
<div class='seg-hero-card seg-risk'>
    <div class='seg-hero-label'>Mayor Riesgo</div>
    <div class='seg-hero-title'>{risk_top['name']}</div>
    <div class='seg-hero-sub'>{risk_top['risk']}% de riesgo con {risk_top['employees']} empleados afectados</div>
</div>
        """,
        unsafe_allow_html=True
    )
    h2.markdown(
        f"""
<div class='seg-hero-card seg-sat'>
    <div class='seg-hero-label'>Mayor Satisfacción</div>
    <div class='seg-hero-title'>{sat_top['name']}</div>
    <div class='seg-hero-sub'>{sat_top['satisfaction']}/10 de satisfacción promedio</div>
</div>
        """,
        unsafe_allow_html=True
    )
    h3.markdown(
        f"""
<div class='seg-hero-card seg-cap'>
    <div class='seg-hero-label'>Sobrecapacidad</div>
    <div class='seg-hero-title'>Subdivisión {cap_top['name']}</div>
    <div class='seg-hero-sub'>{cap_top['capacity']}% de capacidad · requiere balanceo</div>
</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='section-title'>Departamentos Clave</div>", unsafe_allow_html=True)

    def risk_class(value):
        if value >= 35:
            return "seg-risk-high"
        if value >= 20:
            return "seg-risk-mid"
        return "seg-risk-low"

    def initials(name):
        parts = name.split()
        initials_val = "".join([p[0] for p in parts if p])[:2]
        return initials_val.upper()

    for i in range(0, len(dept_filtered), 3):
        row = dept_filtered[i:i + 3]
        cols = st.columns(3)
        for idx, dept in enumerate(row):
            badge_class = risk_class(dept["risk"])
            card_html = f"""
<div class='seg-card'>
    <div class='seg-card-top'>
        <div class='seg-card-icon'>{initials(dept['name'])}</div>
        <div>
            <div class='seg-card-title'>{dept['name']}</div>
            <div class='seg-card-sub'>{dept['employees']} empleados</div>
        </div>
    </div>
    <div class='seg-risk-badge {badge_class}'>{dept['risk']}%</div>
    <div class='seg-metrics'>
        <div><span>Satisfacción</span><b>{dept['satisfaction']}/10</b></div>
        <div><span>Horas Extra</span><b>{dept['overtime']}h</b></div>
        <div><span>Burnout</span><b>{dept['burnout']}%</b></div>
        <div><span>Riesgo</span><b>{dept['risk']}%</b></div>
    </div>
</div>
            """
            cols[idx].markdown(card_html, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Distribución por Subdivisión</div>", unsafe_allow_html=True)

    sub_cols = st.columns(4)
    for idx, sub in enumerate(sub_filtered):
        cap_class = "seg-capacity-high" if sub["capacity"] >= 110 else "seg-capacity-ok"
        bar_class = risk_class(sub["risk"])
        bar_color = "#ff6b81" if bar_class == "seg-risk-high" else "#f2c94c" if bar_class == "seg-risk-mid" else "#25b5e8"
        card_html = f"""
<div class='seg-sub-card'>
    <div class='seg-sub-title'>{sub['name']}</div>
    <div class='seg-sub-row'><span>Empleados:</span><b>{sub['employees']}</b></div>
    <div class='seg-sub-row'><span>Capacidad:</span><span class='{cap_class}'>{sub['capacity']}%</span></div>
    <div class='seg-sub-row'><span>Riesgo:</span><b>{sub['risk']}%</b></div>
    <div class='seg-progress'>
        <div class='seg-progress-bar' style='width:{sub['risk']}%; background:{bar_color};'></div>
    </div>
</div>
        """
        sub_cols[idx % 4].markdown(card_html, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Comparación de Riesgo por Departamento</div>", unsafe_allow_html=True)

    chart_left, chart_right = st.columns(2)

    with chart_left:
        bar_df = pd.DataFrame(segment_depts)
        if selected_dept != "Todos los Departamentos":
            bar_df = bar_df[bar_df["name"] == selected_dept]
        fig_compare = px.bar(
            bar_df,
            x="name",
            y="risk",
            color="name",
            color_discrete_sequence=["#16337b", "#25b5e8", "#ff6b81", "#1f7a5c", "#6aa6ff", "#f2c94c"]
        )
        fig_compare.update_layout(
            height=360,
            xaxis=dict(title="Departamento", showgrid=False),
            yaxis=dict(title="Riesgo (%)", range=[0, 60]),
            showlegend=False
        )
        apply_plotly_style(fig_compare)
        st.plotly_chart(fig_compare, use_container_width=True)

    with chart_right:
        radar_categories = ["Carga Laboral", "Estrés", "Satisfacción", "Balance Vida-Trabajo", "Apoyo Gerencial"]
        radar_values = {
            "Operaciones": [78, 72, 48, 40, 55],
            "IT": [58, 50, 72, 68, 70],
            "RRHH": [45, 38, 80, 75, 78],
        }
        radar_colors = {
            "Operaciones": "#ff6b81",
            "IT": "#25b5e8",
            "RRHH": "#16337b",
        }
        fig_radar = go.Figure()
        for dept, values in radar_values.items():
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_categories,
                fill="toself",
                name=dept,
                line=dict(color=radar_colors.get(dept, "#16337b")),
                opacity=0.55
            ))
        fig_radar.update_layout(
            height=360,
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        apply_plotly_style(fig_radar)
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# 9. RECOMENDACIONES
# ============================================

with tabs[5]:
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Plan de Acción Inmediato</div>", unsafe_allow_html=True)

    recs = [
        {
            "title": "Implementar Programa de Gestión de Estrés en Operaciones",
            "priority": "Alta",
            "status": "En Progreso",
            "category": "Bienestar",
            "tags": ["Alta Prioridad", "En Progreso", "Correctiva", "Bienestar"],
            "summary": "Operaciones muestra niveles críticos de estrés (78/100).",
            "description": "Talleres de manejo de estrés, pausas activas y acceso a apoyo psicológico.",
            "target": "Líderes de Operaciones, Recursos Humanos",
            "impact": "Reducción del 25-30% en niveles de estrés en 3 meses",
            "owner": "María Torres",
            "date": "15 Jun 2026",
            "due": "30 Jun 2026",
        },
        {
            "title": "Redistribuir Carga de Trabajo en Subdivisión Norte",
            "priority": "Alta",
            "status": "Pendiente",
            "category": "Operativa",
            "tags": ["Alta Prioridad", "Pendiente", "Operativa"],
            "summary": "Sobrecarga laboral superior al 130% en múltiples equipos.",
            "description": "Rebalanceo de tareas, rotación de roles y ajuste de metas semanales.",
            "target": "Jefaturas de Equipo",
            "impact": "Reducción del 20% en carga laboral",
            "owner": "Carlos Ruiz",
            "date": "12 Jun 2026",
            "due": "26 Jun 2026",
        },
        {
            "title": "Programa de Prevención de Burnout para Alto Riesgo",
            "priority": "Alta",
            "status": "Pendiente",
            "category": "Bienestar",
            "tags": ["Alta Prioridad", "Pendiente", "Bienestar"],
            "summary": "Empleados con riesgo alto sostenido durante 3 meses consecutivos.",
            "description": "Intervenciones preventivas, soporte psicológico y ajustes de carga.",
            "target": "Empleados en riesgo alto",
            "impact": "Reducción del 30% en burnout",
            "owner": "Ana Martín",
            "date": "10 Jun 2026",
            "due": "24 Jun 2026",
        },
        {
            "title": "Capacitación en Gestión de Tiempo para Mandos Medios",
            "priority": "Media",
            "status": "En Progreso",
            "category": "Desarrollo",
            "tags": ["Media Prioridad", "En Progreso", "Desarrollo"],
            "summary": "Retrasos recurrentes en entregables clave por mala gestión del tiempo.",
            "description": "Capacitación con casos reales y planes de seguimiento individual.",
            "target": "Mandos medios",
            "impact": "Mejora del 15% en cumplimiento de plazos",
            "owner": "Javier López",
            "date": "05 Jun 2026",
            "due": "20 Jun 2026",
        },
        {
            "title": "Encuesta de Clima Psicosocial y Bienestar",
            "priority": "Media",
            "status": "Pendiente",
            "category": "Psicosocial",
            "tags": ["Media Prioridad", "Pendiente", "Psicosocial"],
            "summary": "Recolección de señales tempranas de riesgo psicosocial.",
            "description": "Encuesta trimestral con análisis por área y plan de acción.",
            "target": "Toda la organización",
            "impact": "Mejor detección temprana de riesgos",
            "owner": "Lucía Vega",
            "date": "02 Jun 2026",
            "due": "18 Jun 2026",
        },
        {
            "title": "Seguimiento Post-intervención de Alto Riesgo",
            "priority": "Baja",
            "status": "Completadas",
            "category": "Seguimiento",
            "tags": ["Baja Prioridad", "Completadas", "Seguimiento"],
            "summary": "Validación del impacto de intervenciones previas.",
            "description": "Revisión de indicadores y entrevistas de cierre.",
            "target": "Equipos intervenidos",
            "impact": "Asegurar continuidad y sostenibilidad",
            "owner": "Paula Ríos",
            "date": "28 May 2026",
            "due": "12 Jun 2026",
        },
    ]

    categories = ["Todas"] + sorted({r["category"] for r in recs})
    f1, f2, f3 = st.columns(3)
    status_filter = f1.selectbox("Estado", ["Todos", "Pendiente", "En Progreso", "Completadas"], index=0, key="rec_status")
    priority_filter = f2.selectbox("Prioridad", ["Todas", "Alta", "Media", "Baja"], index=0, key="rec_priority")
    category_filter = f3.selectbox("Categoría", categories, index=0, key="rec_category")

    recs_filtered = []
    for rec in recs:
        if status_filter != "Todos" and rec["status"] != status_filter:
            continue
        if priority_filter != "Todas" and rec["priority"] != priority_filter:
            continue
        if category_filter != "Todas" and rec["category"] != category_filter:
            continue
        recs_filtered.append(rec)

    rec_df = pd.DataFrame(recs_filtered) if recs_filtered else pd.DataFrame(columns=["status", "priority"])
    status_counts = rec_df["status"].value_counts().reindex(["Pendiente", "En Progreso", "Completadas"]).fillna(0)
    priority_counts = rec_df["priority"].value_counts().reindex(["Alta", "Media", "Baja"]).fillna(0)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        "<div class='rec-summary-box'>"
        f"<div class='rec-summary-value'>{len(recs_filtered)}</div>"
        "<div class='rec-summary-label'>Total Recomendaciones</div>"
        "</div>",
        unsafe_allow_html=True
    )
    c2.markdown(
        "<div class='rec-summary-box'>"
        f"<div class='rec-summary-value'>{int(status_counts.get('Pendiente', 0))}</div>"
        "<div class='rec-summary-label'>Pendientes</div>"
        "</div>",
        unsafe_allow_html=True
    )
    c3.markdown(
        "<div class='rec-summary-box'>"
        f"<div class='rec-summary-value'>{int(status_counts.get('En Progreso', 0))}</div>"
        "<div class='rec-summary-label'>En Progreso</div>"
        "</div>",
        unsafe_allow_html=True
    )
    c4.markdown(
        "<div class='rec-summary-box'>"
        f"<div class='rec-summary-value'>{int(status_counts.get('Completadas', 0))}</div>"
        "<div class='rec-summary-label'>Completadas</div>"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c_left, c_right = st.columns(2)
    with c_left:
        status_df = status_counts.reset_index()
        status_df.columns = ["Estado", "Cantidad"]
        fig_status = px.bar(
            status_df,
            x="Estado",
            y="Cantidad",
            color="Estado",
            color_discrete_map={"Pendiente": "#ff6b81", "En Progreso": "#25b5e8", "Completadas": "#16337b"}
        )
        fig_status.update_layout(
            height=320,
            xaxis=dict(title="Estado"),
            yaxis=dict(title="Recomendaciones")
        )
        apply_plotly_style(fig_status)
        st.plotly_chart(fig_status, use_container_width=True)

    with c_right:
        pri_df = priority_counts.reset_index()
        pri_df.columns = ["Prioridad", "Cantidad"]
        fig_pri = px.pie(
            pri_df,
            names="Prioridad",
            values="Cantidad",
            hole=0.55,
            color="Prioridad",
            color_discrete_map={"Alta": "#E74C3C", "Media": "#ff6b81", "Baja": "#25b5e8"}
        )
        fig_pri.update_traces(textinfo="percent+label")
        fig_pri.update_layout(height=320, legend_title_text="Prioridad")
        apply_plotly_style(fig_pri)
        st.plotly_chart(fig_pri, use_container_width=True)

    owner_status_map = {}
    if "employee_name" in df.columns and "active_status" in df.columns:
        owner_status_map = df.set_index("employee_name")["active_status"].to_dict()

    def resolve_owner_status(name):
        status = owner_status_map.get(name)
        if status:
            return status
        score = sum(ord(c) for c in name) % 100
        return "Activo" if score < 78 else "Inactivo"

    priority_class_map = {
        "Alta": "priority-high",
        "Media": "priority-medium",
        "Baja": "priority-low",
    }

    if not recs_filtered:
        st.info("No hay recomendaciones con los filtros seleccionados.")
    else:
        for rec in recs_filtered:
            priority_class = priority_class_map.get(rec["priority"], "priority-medium")
            owner_status = resolve_owner_status(rec["owner"])
            owner_status_class = "inactive" if owner_status == "Inactivo" else ""
            tags_html = "".join([f"<span class='rec-tag'>{tag}</span>" for tag in rec["tags"]])
            rec_html = f"""
<details class='rec-accordion'>
    <summary>
        <div class='rec-summary-left'>
            <div class='rec-title'>{rec['title']}</div>
            <div class='rec-summary-meta'>{rec['category']} · {rec['status']}</div>
        </div>
        <div class='rec-summary-right'>
            <span class='rec-priority {priority_class}'>{rec['priority']}</span>
        </div>
    </summary>
    <div class='rec-details'>
        <div class='rec-tags'>{tags_html}</div>
        <div class='rec-description'>{rec['summary']}</div>
        <div class='rec-description'>{rec['description']}</div>
        <div class='rec-meta-grid'>
            <div><b>Dirigido a:</b> {rec['target']}</div>
            <div><b>Impacto esperado:</b> {rec['impact']}</div>
            <div><b>Responsable:</b>
                <span class='owner-status'><span class='owner-dot {owner_status_class}'></span>
                {rec['owner']} ({owner_status})</span>
            </div>
            <div><b>Estado:</b> {rec['status']}</div>
            <div><b>Fecha de creación:</b> {rec['date']}</div>
            <div><b>Fecha límite:</b> {rec['due']}</div>
        </div>
    </div>
</details>
            """
            st.markdown(rec_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
