import numpy as np
import pandas as pd

np.random.seed(123)

n = 800

# -------------------------
# VARIABLES PSICOLÓGICAS
# -------------------------
stress = np.random.randint(1, 6, n)
burnout = np.random.randint(1, 6, n)
anxiety = np.random.randint(1, 6, n)
depression = np.random.randint(1, 6, n)
support_supervisor = np.random.randint(1, 6, n)
support_coworkers = np.random.randint(1, 6, n)
leave_difficulty = np.random.randint(1, 6, n)

# -------------------------
# VARIABLES HRIS
# -------------------------
age = np.random.randint(20, 60, n)
tenure = np.random.randint(0, 15, n)
absenteeism = np.random.randint(0, 80, n)
performance = np.random.randint(50, 100, n)
promotion = np.random.choice([0,1], n, p=[0.8, 0.2])

departments = ["Operaciones", "Ventas", "IT", "RRHH", "Finanzas"]
department = np.random.choice(departments, n)

# -------------------------
# VARIABLES OPERATIVAS
# -------------------------
workload = np.random.randint(60, 150, n)
task_completion = np.random.randint(50, 100, n)
error_rate = np.random.randint(0, 20, n)

# -------------------------
# RIESGO REALISTA + RUIDO CONTROLADO
# -------------------------
base_risk = (
    0.35 * (burnout/5) +
    0.25 * (stress/5) +
    0.20 * (workload/150) +
    0.10 * (absenteeism/80) +
    0.10 * (anxiety/5)
)

# Ruido controlado para evitar perfección
noise = np.random.normal(0, 0.05, n)

risk_score = np.clip(base_risk + noise, 0, 1)

risk_level = pd.cut(
    risk_score,
    bins=[0, 0.33, 0.66, 1],
    labels=["Bajo", "Medio", "Alto"]
)

# -------------------------
# DATAFRAME FINAL
# -------------------------
df = pd.DataFrame({
    "stress": stress,
    "burnout": burnout,
    "anxiety": anxiety,
    "depression": depression,
    "support_supervisor": support_supervisor,
    "support_coworkers": support_coworkers,
    "leave_difficulty": leave_difficulty,
    "age": age,
    "tenure": tenure,
    "absenteeism": absenteeism,
    "performance": performance,
    "promotion": promotion,
    "department": department,
    "workload": workload,
    "task_completion": task_completion,
    "error_rate": error_rate,
    "risk_score": risk_score,
    "risk_level": risk_level
})

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

X = df.drop(columns=["risk_level"])
y = df["risk_level"]

cat_cols = ["department"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(), cat_cols),
    ("num", "passthrough", num_cols)
])

X_processed = preprocess.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=123, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=600,
    max_features="sqrt",
    random_state=123
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Impulso — Bienestar y Riesgo", layout="wide")

st.title("Impulso — Bienestar y Riesgo")

# ===================== KPIs =====================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Riesgo Promedio", f"{df['risk_score'].mean():.2f}")
col2.metric("Riesgo Alto", f"{(df['risk_level']=='Alto').mean()*100:.1f}%")
col3.metric("Productividad Promedio", f"{df['performance'].mean():.1f}%")
col4.metric("Carga Laboral Promedio", f"{df['workload'].mean():.1f}%")

# ===================== RIESGO POR DEPARTAMENTO =====================
st.header("Riesgo por Departamento")

df_dep = df.groupby("department")["risk_score"].mean().reset_index()

fig_dep = px.bar(df_dep, x="department", y="risk_score", color="department")
st.plotly_chart(fig_dep, use_container_width=True)

# ===================== DISTRIBUCIÓN DE RIESGO =====================
st.header("Distribución del Riesgo")

fig_hist = px.histogram(df, x="risk_score", nbins=20)
st.plotly_chart(fig_hist, use_container_width=True)

# ===================== IMPORTANCIA DE VARIABLES =====================
st.header("Factores de Riesgo")

importances = rf.feature_importances_
feature_names = preprocess.get_feature_names_out()

imp_df = pd.DataFrame({
    "variable": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

fig_imp = px.bar(
    imp_df.head(20).sort_values("importance"),
    x="importance",
    y="variable",
    orientation="h"
)

st.plotly_chart(fig_imp, use_container_width=True)

# ===================== EMPLEADOS =====================
st.header("Empleados")

st.dataframe(df)
